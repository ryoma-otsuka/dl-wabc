'''
run_dl_model_training.py

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import os
import time
from logging import getLogger
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src import utils
from src.data_module import setup_dataloaders_supervised_learning
from src.trainer import setup_model, train
from src.env import reset_seed
from src.gpu import get_available_devices

CUDA_DEVICE_ID = get_available_devices() # Yoshimura's device selector

log = getLogger(__name__)

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="config_dl.yaml",
           )

def main(cfg: DictConfig):
    
    reset_seed(int(cfg.seed))
    
    if cfg.debug == True:
        cfg.train.n_epoch = 30
        cfg.k8s = False
    
    # Run Leave-One-ID-Out Cross Validation
    test_animal_id_list = [cfg.dataset.test_animal_id]
    
    # logger
    logdir = Path(cfg.path.log.rootdir)
    logdir.mkdir(parents=True, exist_ok=True)
    log.info(f"logdir={logdir}")
    
    # set manifold_mixup = False when mixup_alpha param = 0.0
    if cfg.train.mixup_alpha == 0.0:
        cfg.train.manifold_mixup = False
    
    # GPU
    if cfg.k8s == True:
        cfg.train.cuda = CUDA_DEVICE_ID
    # print(cfg.train.cuda)
    if 'cuda:' in str(cfg.train.cuda):
        CUDA = str(cfg.train.cuda)
    else:
        CUDA = 'cuda:' + str(cfg.train.cuda)
    DEVICE = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
    log.info(f"device: {DEVICE}")
    
    log.info(
        "------------------------------------------------------------------------"
    )
    log.info(f"test_animal_id_list: {test_animal_id_list}")

    # initialization of train, val, and test animal id list
    (
        train_animal_id_list, 
        val_animal_id_list, 
        test_animal_id_list
    ) = utils.setup_train_val_test_animal_id_list(
        cfg, 
        test_animal_id_list
    )

    # override cfg.dataset
    cfg.dataset.labelled.animal_id_list.train = train_animal_id_list
    cfg.dataset.labelled.animal_id_list.val = val_animal_id_list
    cfg.dataset.labelled.animal_id_list.test = test_animal_id_list

    # create directory for this test_animal_id
    path = os.path.join(logdir, "config.yaml")
    if os.path.exists(logdir) == False:
        os.makedirs(logdir) 
    OmegaConf.save(cfg, path)

    # load datasets and prepare dataloader
    (
        train_loader_balanced, 
        val_loader, 
        test_loader 
    ) = setup_dataloaders_supervised_learning(
        cfg, 
        train=True, 
        train_balanced=True
    )

    model = setup_model(cfg)
    model.to(DEVICE) # send model to GPU
    print(f'model:\n {model}')

    time.sleep(5)
    
    # initialize the optimizer and loss
    if cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            # lr=cfg.train.lr, 
            lr=0.01, 
            momentum=0.9,
            weight_decay=0
        )
    
    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run training
    best_model, df_log = train(
        model, 
        optimizer, 
        criterion, 
        train_loader_balanced, 
        val_loader, 
        log, 
        DEVICE, 
        cfg
    )

    # save the training log
    path = Path(cfg.path.log.rootdir, cfg.path.log.logfile.csv)
    df_log.to_csv(path, index=False)
    
    # model structure
    log.info(f'model:\n {model}')
    
if __name__ == '__main__':
    main()