'''
run_tree_model_training_and_test.py

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import os
import time
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import _format_eval_result

import logging
from logging import getLogger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src import env
from src import utils
(
    plot_parameters, okabe_ito_color_list, tol_bright_color_list
)    = utils.setup_plot(show_color_palette=False)

log = getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="config_tree.yaml",    
    )

def main(cfg: DictConfig):
    
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    if cfg.dataset.species == "om":
        species_dir = "omizunagidori"
    elif cfg.dataset.species == "um":
        species_dir = "umineko"
    
    if cfg.train.smote == False and cfg.train.smote_dir_name == "smote-true":
        raise Exception(f"check config file -> cfg.train.smote: {cfg.train.smote} | cfg.train.smote_dir_name: {cfg.train.smote_dir_name}")
    
    path = os.path.join(cfg.path.log.rootdir, "config.yaml")
    if os.path.exists(cfg.path.log.rootdir) == False:
        os.makedirs(cfg.path.log.rootdir) 
    OmegaConf.save(cfg, path)
    log.info(f"cfg.path.log.rootdir : {cfg.path.log.rootdir}")
    
    log.info(f"cfg.seed: {cfg.seed}")
    # np.random.seed(int(cfg.seed))
    env.reset_seed_ml(int(cfg.seed))

    n_features = len(cfg.features.features_list)
    log.info(f"n_features: {n_features}")
    features = cfg.features
    log.info(f"features: {features}")

    df, df_sampled = setup_data_ml(cfg)
    df = df.fillna(0) # "OM2103" ODBA_mean, ODBA_var, ODBA_max
    df_sampled = df_sampled.fillna(0) # "OM2103" ODBA_mean, ODBA_var, ODBA_max
    y_gt_all = []
    y_pred_all = []
    test_animal_id_all = []
    model_name = cfg.model.model_name
    test_animal_id_list_list = cfg.dataset.labelled.test_animal_id_list_list
    importances = None
    
    # run only one LOIO-CV fold for debugging
    if cfg.debug == True:
        if cfg.dataset.species == "om":
            test_animal_id_list = ["OM2101"]
        elif cfg.dataset.species == "um": 
            test_animal_id_list = ["UM1901"]
        test_animal_id_list_list = [test_animal_id_list]
        cfg.train.n_estimators = 100
    
    for test_animal_id_list in test_animal_id_list_list:
        log.info("------------------------------------------------------------------------------------------")
        log.info(f"test_animal_id_list: {test_animal_id_list}")
        test_animal_id = test_animal_id_list[0]
        (
            train_animal_id_list, 
            val_animal_id_list, 
            test_animal_id_list
        ) = utils.setup_train_val_test_animal_id_list(
            cfg, 
            test_animal_id_list, 
        )
        # train val split
        X_drop_col_list = ['animal_id', 'unixtime', 'label_id_int']
        label_target = 'label_id_int'
        (
            X_train, X_valid, X_test, 
            y_train, y_valid, y_test 
        ) = prep_train_val_test_data_ml(
            cfg,
            df, 
            df_sampled,
            X_drop_col_list, 
            label_target, 
            train_animal_id_list, 
            val_animal_id_list, 
            test_animal_id_list,
            smote=cfg.train.smote,
        )

        # Run Tree Model
        start_time = time.time()
        model = fit_tree_models(cfg, log, X_train, X_valid, y_train, y_valid)
        save_tree_model(cfg, log, model, test_animal_id)
        elapsed_time = time.time() - start_time
        log.info(f"Elapsed time: {elapsed_time:.3f} seconds")
        
        # feature importances
        if importances is None:
            importances = model.feature_importances_
        else:
            importances = np.concatenate([importances, model.feature_importances_])
        
        # Prediction over test data
        y_gt, y_pred = y_test, model.predict(X_test)
        y_gt_all.extend(y_gt)
        y_pred_all.extend(y_pred)
        test_animal_id_all.extend([test_animal_id_list]*len(y_gt))
    
    # Save model outputs
    fname1 = f"{cfg.model.model_name}"
    fname2 = f"{cfg.dataset.species}" # source = target in ML models
    fname3 = f"seed{cfg.seed}"
    if cfg.train.smote == True:
        cfg.train.smote_dir_name = "smote-true"
    else:
        cfg.train.smote_dir_name = "smote-false"
    file_name_base = f"{fname1}_{fname2}_feats_{cfg.features.features_set_name}_{cfg.train.smote_dir_name}_{fname3}".lower().replace("-", "_")
    
    test_type = "Test"
    test_animal_id = "LOIOCV_ALL"
    base_dir = f"/home/bob/protein/dl-wabc/data/test-results/{cfg.issue}/{cfg.ex}/{cfg.dataset.species}-{cfg.dataset.window_size}/{cfg.model.model_name}"
    results_base_dir = f"{base_dir}/{cfg.model.model_name}-feats-{cfg.features.features_set_name}-{cfg.train.smote_dir_name}"
    log.info(f"results_base_dir: {results_base_dir}")
    log.info(f"file_name_base: {file_name_base}")
    
    # Save importances
    log.info(f"N of LOIO-CV folds: {importances.shape[0]/len(features['features_list'])}")
    data_dict = {
        'features': list(features['features_list'])*int(importances.shape[0]/len(features['features_list'])),
        'importances': importances,
    }
    df_importance = pd.DataFrame(data_dict)
    importance_save_dir = f"{results_base_dir}/feature-importances"
    os.makedirs(importance_save_dir, exist_ok=True)
    importance_save_path = f"{importance_save_dir}/feature_importances_{file_name_base}.csv"
    df_importance.to_csv(importance_save_path, index=False)
    

    df_test_score_all = utils.generate_test_score_df(cfg, y_gt_all, y_pred_all, test_animal_id)
    log.info(f"df_f1_all\n{df_test_score_all}")

    data_dict = {
        'y_gt': y_gt_all,
        'y_pred': y_pred_all, 
        'test_id': test_animal_id_all,
    }
    df_gt_pred_all = pd.DataFrame(data=data_dict)
    # log.info(df_gt_pred_all)
    
    # test score
    save_dir = f"{results_base_dir}/test-score"
    os.makedirs(save_dir, exist_ok=True)
    df_test_score_all.to_csv(
        f"{save_dir}/test_score_{file_name_base}.csv", 
        index=False
    )
    
    # ground truth and prediction
    save_dir = f"{results_base_dir}/y-gt-y-pred"
    os.makedirs(save_dir, exist_ok=True)
    df_gt_pred_all.to_csv(
        f"{save_dir}/y_gt_y_pred_{file_name_base}.csv", 
        index=False
    )
    
    # confusion matrix
    cm_base_dir = f"{results_base_dir}/confusion-matrix/"
    log.info(f"cm_base_dir: {cm_base_dir}")
    os.makedirs(cm_base_dir, exist_ok=True)
    cm, df_cm, fig_cm = utils.plot_confusion_matrix(y_gt_all, y_pred_all, cfg)
    fig_cm.savefig(
        f"{cm_base_dir}/confusion_matrix_{file_name_base}.png", 
        dpi=350, 
        bbox_inches='tight', 
        pad_inches=0.15
    )

    log.info("Saved model outputs.")


def setup_data_ml(cfg):
    if cfg.dataset.species == "om":
        species_dir = "omizunagidori"
    elif cfg.dataset.species == "um":
        species_dir = "umineko"
    base_target = "/home/bob/protein/dl-wabc/data/datasets/logbot_data/feature_extraction/acc_features"
    target = f"{base_target}/{species_dir}/*.csv"
    features_data_path_list = sorted(glob.glob(target))
    df = pd.DataFrame()
    for features_data_path in features_data_path_list:
        df_tmp = pd.read_csv(features_data_path)
        df = pd.concat([df, df_tmp])
    label_species = cfg.dataset.species
    dataset_amount = 100
    log.info(f'Dataset amount: {dataset_amount}%')
    if dataset_amount == 100:
        df = df
        df_sampled = df
    else:
        raise Exception(f"dataset_amount: {dataset_amount} is not accepted.")

    log.info(f'N of instances (df): {len(df)}')
    log.info(f'N of instances (df_sampled): {len(df_sampled)}')
    log.info(f"df: \n{df['label_id_int'].value_counts().sort_index()}")
    
    usecols = ['animal_id', 'unixtime', 'label_id_int']
    features = cfg.features.features_list
    usecols = usecols + features
    df = df[usecols]
    df_sampled = df_sampled[usecols]
    
    return df, df_sampled


def prep_train_val_test_data_ml(cfg,
                                df, 
                                df_sampled,
                                X_drop_col_list, 
                                label_target, 
                                train_animal_id_list, 
                                val_animal_id_list, 
                                test_animal_id_list,
                                smote=True,
                               ):
    
    df_test = df[df["animal_id"].isin(test_animal_id_list)]
    X_test = df_test.drop(X_drop_col_list, axis=1)
    y_test = df_test[label_target]
    
    df_train_val = df_sampled[df_sampled["animal_id"].isin(train_animal_id_list)]
    log.info(f"len(df_train_val): {len(df_train_val)}")
    X_df = df_train_val.drop(X_drop_col_list, axis=1)
    y_s = df_train_val[label_target]
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_df, 
            y_s, 
            test_size=0.2, 
            random_state=cfg.seed, 
            stratify=y_s)
    log.info(f"y_train.shape: {y_train.shape}")
    log.info(f"y_train: \n{y_train.value_counts().sort_index()}")
    # log.info(f"y_valid: \n{y_valid.value_counts(sort=False)}")
    log.info(f'X_train.shape: {X_train.shape}')
    log.info(f'X_valid.shape: {X_valid.shape}')
    
    if smote == True:
        # SMOTE
        # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
        sm = SMOTE(
            random_state=cfg.seed,
            k_neighbors=cfg.train.smote_k_neighbors
        )
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log.info(f"| Applied SMOTE |")
        log.info(f"y_train (SMOTE): \n{y_train.value_counts().sort_index()}")
        log.info(f'X_train.shape: {X_train.shape}')
        log.info(f'X_valid.shape: {X_valid.shape}')
        # No need to apply SMOTE on validation datasets
        # X_valid, y_valid = sm.fit_resample(X_valid, y_valid) 
    elif smote == False:
        log.info(f'X_train.shape: {X_train.shape}')
        log.info(f'X_valid.shape: {X_valid.shape}')
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def fit_tree_models(cfg, logger, X_train, X_valid, y_train, y_valid):
    if cfg.model.model_name == "random-forest":
        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=500, 
            max_depth=6, # 4, 5, 6, 7, 8, 9, 10
            random_state=cfg.seed,
        )
        model.fit(X_train, y_train)
    elif cfg.model.model_name == "xgboost":
        # https://xgboost.readthedocs.io/en/stable/parameter.html
        params = {
            'objective': 'multi:softprob', # https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py#L1411-L1415
            'eval_metric': 'mlogloss',
            # 'num_class': cfg.dataset.n_classes, # https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py#L1393-L1400
            'booster': 'gbtree',
            'n_estimators': cfg.train.n_estimators,
            'early_stopping_rounds': 10,
            'callbacks': [XGBLogging(epoch_log_interval=100)],
            'learning_rate': 0.01, 
            'max_depth': 6, # 4, 5, 6, 7, 8, 9, 10
            'random_state': cfg.seed,
            'verbosity': 0,
            'tree_method': 'gpu_hist', # 'gpu_hist', 'gpu_exact' -> use GPU for fast calculation
            'gpu_id': cfg.train.cuda
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )
        results = model.evals_result()
        plt.figure(figsize=(6, 4))
        plt.plot(results["validation_0"]["mlogloss"], label="Training loss")
        plt.plot(results["validation_1"]["mlogloss"], label="Validation loss")
        plt.xlabel("Number of estimators")
        plt.ylabel("mlogloss")
        plt.legend()
        plt.show()
        plt.close()
    elif cfg.model.model_name == "lightgbm":
        lgb.register_logger(logger)
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        params = {
            'objective': 'multiclass',  
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',    
            'n_estimators': cfg.train.n_estimators,
            'learning_rate': 0.01,    
            'random_state': cfg.seed,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                # lgb.log_evaluation(1),
                lgb_log_evaluation(logger, period=50)
            ]
        )
        plt.rcParams["figure.figsize"] = (6, 4)
        lgb.plot_metric(model)
        plt.show()
        plt.close()
    else:
        raise Exception(f'cfg.model.model_name "{cfg.model.model_name}" is not appropriate.')
        
    return model


class XGBLogging(xgb.callback.TrainingCallback):
    """log train logs to file"""

    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.epoch_log_interval == 0:
            for data, metric in evals_log.items():
                metrics = list(metric.keys())
                metrics_str = ""
                for m_key in metrics:
                    metrics_str = metrics_str + f"{m_key}: {metric[m_key][-1]}"
                log.info(f"Epoch: {str(epoch).zfill(4)}, {data}: {metrics_str}")
        # False to indicate training should not stop.
        return False

# https://amalog.hateblo.jp/entry/lightgbm-logging-callback
def lgb_log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(str(env.iteration+1).zfill(3), result))
    _callback.order = 10
    return _callback


def save_tree_model(cfg, logger, model, test_animal_id):
    path = Path(
        cfg.path.log.rootdir,
        f"{test_animal_id}.pickle",
        )
    logger.debug(f"model pickle file path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode='wb') as f:
        pickle.dump(model, f, protocol=2)
        
        
        
if __name__ == '__main__':
    main()