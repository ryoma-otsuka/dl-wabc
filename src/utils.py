"""
utils

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

"""

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
import torch
import sklearn
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    precision_score,
    recall_score,
    jaccard_score
)
from sklearn.manifold import TSNE
import logging
from logging import getLogger

log = getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


def print_path_list_contents_with_index(path_list):
    if len(path_list) == 0:
        print(f"The input path list is empty.")
    else:
        for i, path in enumerate(path_list):
            print(f"{i:0=2}: {os.path.basename(path)}")
        

def setup_device(cfg):
    if 'cuda:' in str(cfg.train.cuda):
        CUDA = str(cfg.train.cuda)
    else:
        CUDA = 'cuda:' + str(cfg.train.cuda)
    DEVICE = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
    print(f"device={DEVICE}")
    return DEVICE


def _setup_device(cfg):
    DEVICE = torch.device('cuda:' + str(cfg.train.cuda)
                          if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE)
    return DEVICE


def return_species_jp_name(cfg):
    
    if cfg.dataset.species == "om":
        species_jp_name = "omizunagidori"
    elif cfg.dataset.species == "um":
        species_jp_name = "umineko"
    else:
        raise Exception(f"cfg.dataset: {cfg.dataset} is unknonw dataset.")
        
    return species_jp_name




def setup_train_val_test_animal_id_list(cfg, test_animal_id_list):
    train_animal_id_list = cfg.dataset.labelled.animal_id_list.all.copy()
    val_animal_id_list = ["DUMMY"]
    for i in range(len(val_animal_id_list)):
        if val_animal_id_list[i] in train_animal_id_list:
            train_animal_id_list.remove(val_animal_id_list[i])
    for i in range(len(test_animal_id_list)):
        if test_animal_id_list[i] in train_animal_id_list:
            train_animal_id_list.remove(test_animal_id_list[i])

    log.info(f"train_animal_id_list: {train_animal_id_list}")
    log.info(f"test_animal_id_list: {test_animal_id_list}")
    log.info(f"N of train animals: {len(train_animal_id_list)}")
    log.info(f"N of test animals: {len(test_animal_id_list)}")

    return train_animal_id_list, val_animal_id_list, test_animal_id_list


def convert_pandas_labels(df, label_species):
    y_list = list(df["label_id"].astype(int))
    y_list_int = []
    y_list_str = []

    if label_species == "om":
        for i in range(0, len(y_list)):
            if y_list[i] == 200:  # stationary
                y_list_int.append(0)
                y_list_str.append("0: Stationary")
            elif y_list[i] == 201:  # preening
                y_list_int.append(0)
                y_list_str.append("0: Stationary")
            elif y_list[i] == 300:  # bathing
                y_list_int.append(1)
                y_list_str.append("1: Bathing")
            elif y_list[i] == 400:  # flight_take_off
                y_list_int.append(2)
                y_list_str.append("2: Take-off")
            elif y_list[i] == 401:  # flight_crusing
                y_list_int.append(3)
                y_list_str.append("3: Cruising Flight")
            elif y_list[i] == 501:  # foraging_dive
                y_list_int.append(4)
                y_list_str.append("4: Foraging Dive")
            elif y_list[i] == 502:  # surface_seizing (dipping)
                y_list_int.append(5)
                y_list_str.append("5: Dipping")
            else:
                y_list_int.append(y_list[i])
                y_list_str.append(y_list[i])
    elif label_species == "um":
        for i in range(0, len(y_list)):
            if y_list[i] == 100:  # ground_stationary
                y_list_int.append(0)
                y_list_str.append("0: Stationary")
            elif y_list[i] == 101:  # ground_active
                y_list_int.append(1)
                y_list_str.append("1: Ground Active")
            elif y_list[i] == 200:  # stationary
                y_list_int.append(0)
                y_list_str.append("0: Stationary")
            elif y_list[i] == 201:  # preening
                y_list_int.append(0)
                y_list_str.append("0: Stationary")
            elif y_list[i] == 300:  # bathing
                y_list_int.append(2)
                y_list_str.append("2: Bathing")
            elif y_list[i] == 301:  # bathing_poss
                y_list_int.append(2)
                y_list_str.append("2: Bathing")
            elif y_list[i] == 400:  # flying_active
                y_list_int.append(3)
                y_list_str.append("3: Active Flight")
            elif y_list[i] == 401:  # flying_passive
                y_list_int.append(4)
                y_list_str.append("4: Passive Flight")
            elif y_list[i] == 500:  # foraging
                y_list_int.append(5)
                y_list_str.append("5: Foraging")
            elif y_list[i] == 501:  # foraging_poss
                y_list_int.append(5)
                y_list_str.append("5: Foraging")
            elif y_list[i] == 502:  # foraging_fish
                y_list_int.append(5)
                y_list_str.append("5: Foraging")
            elif y_list[i] == 503:  # foraging_fish_poss
                y_list_int.append(5)
                y_list_str.append("5: Foraging")
            # elif y_list[i] == 510:  # foraging_insect
            #     y_list_int.append(6)
            #     y_list_str.append("6: Foraging Insect")
            # elif y_list[i] == 511:  # foraging_insect_poss
            #     y_list_int.append(6)
            #     y_list_str.append("6: Foraging Insect")
            else:
                y_list_int.append(y_list[i])
                y_list_str.append(y_list[i])

    y_list_int = list(map(int, y_list_int))
    df["label_id_int"] = y_list_int
    df["label_id_str"] = y_list_str

    return df


def get_label_species(cfg):
    label_species = cfg.dataset.species
    return label_species


def convert_torch_labels(targets, label_species):

    if label_species == "om":
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 1, targets)  # bathing
        targets = torch.where(targets == 400, 2, targets)  # flight_take_off
        targets = torch.where(targets == 401, 3, targets)  # flight_cruising
        targets = torch.where(targets == 501, 4, targets)  # foraging_dive
        targets = torch.where(targets == 502, 5, targets)  # surface_seizing (dipping)
    elif label_species == "um":
        targets = torch.where(targets == 100, 0, targets)  # ground_stationary
        targets = torch.where(targets == 101, 1, targets)  # ground_active
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 2, targets)  # bathing
        targets = torch.where(targets == 301, 2, targets)  # bathing
        targets = torch.where(targets == 400, 3, targets)  # flying_active
        targets = torch.where(targets == 401, 4, targets)  # flying_passive
        targets = torch.where(targets == 500, 5, targets)  # foraging
        targets = torch.where(targets == 501, 5, targets)  # foraging_poss
        targets = torch.where(targets == 502, 5, targets)  # foraging_fish
        targets = torch.where(targets == 503, 5, targets)  # foraging_fish_poss
        # targets = torch.where(targets==510, 6, targets) # foraging_insect
        # targets = torch.where(targets==511, 6, targets) # foraging_insect_poss
        # targets = torch.where(targets==520, 7, targets) # foraging_something
        
    return targets


def convert_numpy_labels(labels, label_species):  # targets -> list?
    if label_species == "om":
        labels = np.where(labels == 200, 0, labels)  # stationary
        labels = np.where(labels == 201, 0, labels)  # preening
        labels = np.where(labels == 300, 1, labels)  # bathing
        labels = np.where(labels == 400, 2, labels)  # flight_take_off
        labels = np.where(labels == 401, 3, labels)  # flight_cruising
        labels = np.where(labels == 501, 4, labels)  # foraging_dive
        labels = np.where(labels == 502, 5, labels)  # surface_seizing (dipping)
    elif label_species == "um":
        labels = np.where(labels == 100, 0, labels)  # ground_stationary
        labels = np.where(labels == 101, 1, labels)  # ground_active
        labels = np.where(labels == 200, 0, labels)  # stationary
        labels = np.where(labels == 201, 0, labels)  # preening
        labels = np.where(labels == 300, 2, labels)  # bathing
        labels = np.where(labels == 301, 2, labels)  # bathing
        labels = np.where(labels == 400, 3, labels)  # flying_active
        labels = np.where(labels == 401, 4, labels)  # flying_passive
        labels = np.where(labels == 500, 5, labels)  # foraging
        labels = np.where(labels == 501, 5, labels)  # foraging_poss
        labels = np.where(labels == 502, 5, labels)  # foraging_fish
        labels = np.where(labels == 503, 5, labels)  # foraging_fish_poss
    return labels


# 1 data point 1 label (dense label) -> 1 window 1 label | ([batch_size,
# 1, window_size]) -> ([batch_size])
def get_last_labels(labels):
    batch_size = labels.shape[0]
    last_labels = []
    for i in range(0, batch_size):
        # use the last label of each window
        last_labels.append(labels[i, 0, -1].item())
        last_labels_array = np.array(last_labels)
    last_labels_tensor = torch.from_numpy(
        last_labels_array.astype(np.float32)).clone()
    last_labels_tensor = last_labels_tensor.long()  # long (uint64)
    return last_labels_tensor


def setup_plot(show_color_palette=False, style="ticks"):
    sns.set_style(style)
    
    # figure settings
    parameters = {
        'font.size': 12,
        'axes.labelsize': 12,
        'legend.fontsize': 13,
        'figure.titlesize': 14,
        "figure.facecolor": "white"
    }
    plt.rcParams.update(parameters)
    
    sns.set_context(context='notebook', font_scale=1, rc=None)

    # set your own color palette
    palette = sns.color_palette('deep')
    if show_color_palette:
        print("deep")
        sns.palplot(palette)
        plt.show()
        plt.close()

    okabe_ito_color_list = sns.color_palette(
        ['#E69F00', 
         '#56B4E9', 
         '#009E73', 
         '#F0E442', 
         '#0072B2', 
         '#D55E00', 
         '#CC79A7', 
         '#000000'])
    if show_color_palette:
        print("Okabe-Ito")
        sns.palplot(okabe_ito_color_list)
        plt.show()
        plt.close()

    tol_bright_color_list = sns.color_palette(
        ['#4477AA', 
         '#EE6677', 
         '#228833', 
         '#CCBB44', 
         '#66CCEE', 
         '#AA3377', 
         '#BBBBBB'])
    if show_color_palette:
        print("Tol Bright")
        sns.palplot(tol_bright_color_list)
        plt.show()
        plt.close()

    return parameters, okabe_ito_color_list, tol_bright_color_list


def show_color_palette_func(color_palette):
    sns.palplot(color_palette)
    plt.show()
    plt.close()
    

def plot_window(X, label, npz_file_name, figsize=(8, 3)):
    '''
    X: numpy array
    y: 
    '''
    acc_x = X[0].transpose(1, 0)[0]
    acc_y = X[0].transpose(1, 0)[1]
    acc_z = X[0].transpose(1, 0)[2]
    window_size = len(X[0])
    data_number = list(range(1, window_size + 1, 1))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # color_list = ['#EE6677', '#228833', '#4477AA']
    color_list = ['#D81B60', '#FFC107', '#1E88E5']
    ax = sns.lineplot(x=data_number, y=acc_x, label="x", color=color_list[0])
    ax = sns.lineplot(x=data_number, y=acc_y, label="y", color=color_list[1])
    ax = sns.lineplot(x=data_number, y=acc_z, label="z", color=color_list[2])
    if npz_file_name is None:
        print("No title")
    else: 
        if label is None:
            ax.set_title(f"{npz_file_name}", pad=10)
        else:
            ax.set_title(f"{npz_file_name} | label_id: {(int(label))}", pad=10)
    ax.set_xlabel("t")
    ax.set_ylabel("g")
    ax.set_xticks(np.arange(0, 51, 10), fontsize=18)
    ax.set_yticks(np.arange(-4.0, 4.2, 2.0))
    xticklabels = ['{:,.0f}'.format(x) for x in np.arange(0, 51, 10.0)]
    yticklabels = ['{:,.1f}'.format(x) for x in np.arange(-4.0, 4.1, 2.0)]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-3, 53)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(ncol=3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
    ax.grid(axis='both', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.5)
    plt.show()
    plt.close()

    return fig

def plot_window_ax(ax, X, label, npz_file_name):
    '''
    X: numpy array
    '''
    acc_x = X[0].transpose(1, 0)[0]
    acc_y = X[0].transpose(1, 0)[1]
    acc_z = X[0].transpose(1, 0)[2]
    window_size = len(X[0])
    data_number = list(range(1, window_size + 1, 1))
    # color_list = ['#EE6677', '#228833', '#4477AA']
    color_list = ['#D81B60', '#FFC107', '#1E88E5']
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_x, label="x", color=color_list[0])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_y, label="y", color=color_list[1])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_z, label="z", color=color_list[2])
    if npz_file_name is None:
        print("No title")
    else: 
        if label is None:
            ax.set_title(f"{npz_file_name}", pad=10)
        else:
            ax.set_title(f"{npz_file_name} | label_id: {(int(label))}", pad=10)
    ax.set_xlabel("t")
    ax.set_ylabel("g")
    ax.set_xticks(np.arange(0, 51, 10), fontsize=18)
    ax.set_yticks(np.arange(-4.0, 4.2, 2.0))
    xticklabels = ['{:,.0f}'.format(x) for x in np.arange(0, 51, 10.0)]
    yticklabels = ['{:,.1f}'.format(x) for x in np.arange(-4.0, 4.1, 2.0)]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-3, 53)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(ncol=3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
    ax.grid(axis='both', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.5)
    # plt.show()
    # plt.close()

    return ax


def generate_fig_title_species_name(species):
    if species == "omizunagidori":
        title_species_name = "Streaked shearwater"
    elif species == "umineko":
        title_species_name = "Black-tailed gull"
    return title_species_name


def generate_class_labels_for_vis(species):
    if species == "omizunagidori":
        class_label = [
            'Stationary',
            'Bathing',
            'Take-off',
            'Cruising Flight',
            'Foraging Dive',
            'Dipping'
        ]
    elif species == "umineko":
        class_label = [
            'Stationary',
            'Ground Active',
            'Bathing',
            'Active Flight',
            'Passive Flight',
            'Foraging'
        ]
    return class_label


def plot_confusion_matrix(y_gt, y_pred, cfg, figsize=(9, 7)):
    
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)
    
    labels_int = np.arange(0, len(class_labels), 1).tolist()
    
    cm = confusion_matrix(y_gt, y_pred, labels=labels_int)

    df_cm = pd.DataFrame(data=cm, index=class_labels, columns=class_labels)
    print(df_cm)
    
    fig = plt.figure(figsize=figsize)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_precision_scores = (df_cm / np.sum(df_cm)).values.flatten()
    group_percentages = ["{0:.2f}".format(value)
                         for value in group_precision_scores]
    annot_labels = [
        f"{v1}\n({v2})" for v1,
        v2 in zip(
            group_counts,
            group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(len(labels_int), len(labels_int))
    ax = sns.heatmap(
        df_cm / np.sum(df_cm),
        # df_cm,
        vmin=0, vmax=1.0,
        square=True, cbar=True, annot=annot_labels, fmt='',
        cmap='Blues'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Prediction", fontsize=14, rotation=0, labelpad=10)
    plt.ylabel("Ground Truth", fontsize=14, labelpad=10)
    ax.set_ylim(len(cm), 0)
    fig.tight_layout()
    plt.show()
    plt.close()

    fig_cm = fig

    return cm, df_cm, fig_cm


def get_f1_score_df(cfg, y_gt, y_pred, test_type, test_animal_id):
    model_name = cfg.model.model_name

    if cfg.dataset.species == "OM":
        category = [
            "macro",
            "weighted",
            "stationary",
            "bathing",
            "flight_take_off",
            "flight_crusing",
            "foraging_dive",
            "surface_seizing" # "dipping"
        ]
    elif cfg.dataset.species == "UM":
        category = [
            "macro",
            "weighted",
            "stationary",
            "ground_active",
            "bathing",
            "flying_active",
            "flying_passive",
            "foraging"]
    
    model = np.array([model_name]*len(category))
    test_id  = np.array([test_animal_id]*len(category))
    # accuracy_scores = np.array([np.nan]*len(category))
    source_species = np.array([cfg.dataset.species]*len(category))
    target_species = np.array([cfg.dataset.species]*len(category))

    # macro average
    precision_macro = precision_score(y_gt, y_pred, average="macro")
    recall_macro = recall_score(y_gt, y_pred, average="macro")
    f1_macro = f1_score(y_gt, y_pred, average="macro")
    # weighted average
    precision_weighted = precision_score(y_gt, y_pred, average="weighted")
    recall_weighted = recall_score(y_gt, y_pred, average="weighted")
    f1_weighted = f1_score(y_gt, y_pred, average="weighted")
    # scores for each class
    labels = [0, 1, 2, 3, 4, 5]
    precision_scores = precision_score(
        y_gt, y_pred, average=None, labels=labels)
    recall_scores = recall_score(y_gt, y_pred, average=None, labels=labels)
    f1_scores = f1_score(y_gt, y_pred, average=None, labels=labels)

    # df for storing results
    data_dict = {
        'Model': model,
        'Category': category,
        'Test_ID': test_id,
        'Source_species': source_species,
        'Target_species': target_species, 
        'Precision': np.append(
            np.append(
                precision_macro,
                precision_weighted),
            precision_scores),
        'Recall': np.append(
            np.append(
                recall_macro,
                recall_weighted),
            recall_scores),
        'F1': np.append(
            np.append(
                f1_macro,
                f1_weighted),
            f1_scores),
    }

    df = pd.DataFrame(data=data_dict)

    return df



def generate_test_score_df(cfg, y_gt, y_pred, test_animal_id):
    
    model_name = cfg.model.model_name
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)
    labels_int = np.arange(0, len(class_labels), 1).tolist()
    
    category = ["Macro", "Weighted"]
    category.extend(class_labels)
    
    model = np.array([model_name]*len(category))
    test_id  = np.array([test_animal_id]*len(category))
    
    species = np.array([cfg.dataset.species]*len(category))

    # macro average
    precision_macro = precision_score(y_gt, y_pred, average="macro")
    recall_macro = recall_score(y_gt, y_pred, average="macro")
    f1_macro = f1_score(y_gt, y_pred, average="macro")
    IoU_macro = jaccard_score(y_gt, y_pred, average="macro")
    
    # weighted average
    precision_weighted = precision_score(y_gt, y_pred, average="weighted")
    recall_weighted = recall_score(y_gt, y_pred, average="weighted")
    f1_weighted = f1_score(y_gt, y_pred, average="weighted")
    IoU_weighted = jaccard_score(y_gt, y_pred, average="weighted")
    
    # scores for each class
    precision_scores = precision_score(y_gt, y_pred, average=None, labels=labels_int)
    recall_scores = recall_score(y_gt, y_pred, average=None, labels=labels_int)
    f1_scores = f1_score(y_gt, y_pred, average=None, labels=labels_int)
    IoU_scores = jaccard_score(y_gt, y_pred, average=None, labels=labels_int)

    # df for storing results
    data_dict = {
        'Model': model,
        'Category': category,
        'Test_ID': test_id,
        'Species': species,
        'Precision': np.append(
            np.append(
                precision_macro,
                precision_weighted),
            precision_scores),
        'Recall': np.append(
            np.append(
                recall_macro,
                recall_weighted),
            recall_scores),
        'F1': np.append(
            np.append(
                f1_macro,
                f1_weighted),
            f1_scores),
        'IoU': np.append(
            np.append(
                IoU_macro,
                IoU_weighted),
            IoU_scores),
    }

    df = pd.DataFrame(data=data_dict)

    return df


def return_results_dir(issue, ex, dataset):
    if ex in ["ex-d10", "ex-d15", "ex-d16"]: # ex-d10, ex-d11, ...
        model_name = "dcl"
    elif ex in ["ex-d11"]:
        model_name = "dcl-sa"
    elif ex in ["ex-d17"]:
        model_name = "dcl-v3"
    elif ex in ["ex-d21"]: # ex-d20 ex-d21, ...
        model_name = "cnn-ae"
    elif ex in ["ex-d22"]: # ex-d22, ...
        model_name = "cnn-ae-wo"
    elif "ex-d7" in ex: # ex-d70, ex-d71, ...
        model_name = "dcl"
    # elif "ex-d91" in ex: # ex-d40, ex-d41, ...
    #     model_name = input("lightgbm or xgboost")
    results_dir = f"../../data/test-results/{issue}/{ex}/{dataset}/{model_name}/test-score"
    return results_dir


def return_test_score_path_list(results_dir, checkpoint, num_seeds=5):
    target_path = f"{results_dir}/test_score_*_{checkpoint}*csv"
    test_score_path_list = sorted(glob.glob(target_path))
    print(f"target_path: {target_path}")
    print(f"len(test_score_path_list): {len(test_score_path_list)}")
    print(f"{int(len(test_score_path_list)/num_seeds)} conditions/groups")
    return test_score_path_list



def return_results_dir_tree_for_ex_d30(issue, ex, dataset, model_name):
    results_dir = f"../../data/test-results/{issue}/{ex}/{dataset}/{model_name}/{model_name}-feats-119-smote-true/test-score"
    return results_dir


def return_results_dir_dl_model_for_ex_d30(issue, ex, dataset, model_name):
    results_dir = f"../../data/test-results/{issue}/{ex}/{dataset}/{model_name}/test-score"
    return results_dir


def return_results_dir_tree_for_ex_d91a(issue, ex, dataset, model_name):
    results_dir = f"../../data/test-results/{issue}/{ex}/{dataset}/{model_name}/{model_name}-feats-*-smote-true/test-score"
    return results_dir


def return_results_dir_tree_for_ex_d91b(issue, ex, dataset, model_name):
    results_dir = f"../../data/test-results/{issue}/{ex}/{dataset}/{model_name}/{model_name}-feats-119-smote-*/test-score"
    return results_dir


def return_test_score_path_list_tree(results_dir, num_seeds=10):
    target_path = f"{results_dir}/test_score_*csv"
    test_score_path_list = sorted(glob.glob(target_path))
    print(f"target_path: {target_path}")
    print(f"len(test_score_path_list): {len(test_score_path_list)}")
    print(f"{int(len(test_score_path_list)/num_seeds)} conditions/groups")
    return test_score_path_list


def change_pandas_label_name_for_vis(df, dataset):
    
    if "-" in dataset:
        species = dataset[:2]
    else:
        species = dataset
    
    if species == "om":
        label_name_dict = {
            'macro': 'Macro',
            'weighted': 'Weighted',
            'weigthed': 'Weighted', # spell mistake
            'stationary': 'Stationary',
            'bathing': 'Bathing',
            'flight_take_off': 'Take-off',
            'flight_cruising': 'Cruising Flight',
            'flight_crusing': 'Cruising Flight',
            'foraging_dive': 'Foraging Dive',
            # 'surface_seizing': 'Dipping',
            'dipping': 'Dipping',
        }
    elif species == "um":
        label_name_dict = {
            'macro': 'Macro',
            'weighted': 'Weighted',
            'weigthed': 'Weighted', # spell mistake
            'stationary': 'Stationary',
            'ground_active': 'Ground Active',
            'bathing': 'Bathing',
            'flying_active': 'Active Flight',
            'flying_passive': 'Passive Flight',
            'foraging': 'Foraging',
        }
        
    if label_name_dict is not None:
        df["Category"] = df["Category"].replace(label_name_dict)
    
    return df


def make_bar_label_list(
    df, 
    score_type,
    group_by_column, 
    group_list
):
    """
    Args:
        df (DataFrame):
        group_by_column (str):
        group_list (List):
    Return:
        
    """
    
    mean_sd = df.groupby([group_by_column])[score_type].agg(['mean', 'std'])
    mean_sd = mean_sd.reindex(index=group_list)
    
    mean_list = list(mean_sd['mean'])
    sd_list = list(mean_sd['std'])
    
    bar_label_list = []
    for i in range(len(group_list)):
        # print(mean_list[i])
        # print(type(mean_list[i]))
        if np.isnan(mean_list[i]) == True:
            bar_label = f""
        else:
            bar_label = f" {mean_list[i]:.3f} \n({sd_list[i]:.3f})"
        bar_label_list.append(bar_label)
    return bar_label_list


def plot_macro_score(df, ax, score_type, group_by_column, group_list, color_palette):
    
    df_macro = df[df.Category.isin(['Macro'])]
    bar_label_list = make_bar_label_list(df_macro, score_type, group_by_column, group_list)
    sns.barplot(ax=ax, 
                data=df_macro, 
                x=group_by_column, 
                y=score_type, 
                order=group_list, 
                palette=color_palette, 
                saturation=0.75,  # default 0.75
                errorbar='sd', 
                errwidth=2.5,
                capsize=0.0)
    return ax, bar_label_list


def plot_class_score(df, ax, score_type, group_by_column, group_list, color_palette):
    
    df_class = df[df.Category.isin(['Macro', 'Weighted']) == False]
    sns.barplot(ax=ax, 
                data=df_class, 
                x='Category', 
                y=score_type, 
                hue=group_by_column, 
                hue_order=group_list, 
                palette=color_palette, 
                saturation=0.75, # default 0.75
                errorbar='sd', 
                errwidth=2.5,
                capsize=0.0,
                dodge=True)
    
    return ax


def plot_macro_and_class_test_scores(
    df,
    # dataset,
    score_type, 
    group_by_column, 
    group_list, 
    color_palette, 
    GRIDSPEC_KW
):
    
    category_list = df["Category"].unique().tolist()
    category_list.remove("Macro")
    category_list.remove("Weighted")
    behaviour_list = category_list
    print(behaviour_list)
    
    if score_type == "F1":
        y_label_list = ["Macro F1-score", "Class F1-score"]
    elif score_type == "IoU":
        y_label_list = [f"Macro IoU score", "Class IoU score"]
    
    species = df["Species"].values[0]
    if species == "om":
        title_list = ["Streaked Shearwaters"] * 2
    elif species == "um":
        title_list = ["Black-tailed Gulls"] * 2
    title_x_list = [(len(group_list)-1)/2, (len(behaviour_list)-1)/2]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw=GRIDSPEC_KW)
    ax_list = list(axes.flatten())
    for i, ax in enumerate(ax_list):
        # Macro F1-score/IoU score
        if i == 0:
            ax, bar_label_list = plot_macro_score(df, ax, score_type, group_by_column, group_list, color_palette)
            text_color = 'white'
            ax.bar_label(
                ax.containers[0], labels=bar_label_list, 
                linespacing=0.9,
                fmt='%.3f', label_type='center', padding=0, 
                color=text_color, fontweight='bold', 
                fontsize=14, rotation=90
            )
            ax.tick_params(axis='x', labelsize=16, rotation=90)
        # Class F1-score/IoU score
        else:
            ax = plot_class_score(df, ax, score_type, group_by_column, group_list, color_palette)
            ax.get_legend().remove()
            ax.tick_params(axis='x', labelsize=16, rotation=30)
        
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
        ax.grid(axis='y', which='major', alpha=0.5)
        ax.grid(axis='y', which='minor', alpha=0.5)
        ax.tick_params(axis='y', labelsize=14)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label_list[i], fontsize=16, labelpad=10)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_ylim(-0.0, 1.05)
        ax.text(title_x_list[i], 1.1, s=title_list[i], 
                fontsize=16, va='center', ha='center')
    plt.show()
    plt.close()
    
    return fig


def plot_macro_and_class_test_scores_v2(
    df,
    score_type, 
    group_by_column, 
    group_list, 
    x_rotation_list,
    fig_label_x_list,
    color_palette,
    GRIDSPEC_KW
):
    
    behaviour_list_om = None
    behaviour_list_um = None
    
    df_om = df[df["Species"] == "om"]
    if len(df_om) > 0: 
        category_list = df_om["Category"].unique().tolist()
        category_list.remove("Macro")
        category_list.remove("Weighted")
        behaviour_list_om = category_list
        print(behaviour_list_om)
    
    df_um = df[df["Species"] == "um"]
    if len(df_um) > 0: 
        category_list = df_um["Category"].unique().tolist()
        category_list.remove("Macro")
        category_list.remove("Weighted")
        behaviour_list_um = category_list
        print(behaviour_list_um)
    
    num_classes = 6
    
    if score_type == "F1":
        y_label_list = ["Macro F1-score", "Class F1-score"] * 2
    elif score_type == "IoU":
        y_label_list = [f"Macro IoU score", "Class IoU score"] * 2

    title_list = ["Streaked Shearwaters"] * 2 + ["Black-tailed Gulls"] * 2
    title_x_list = [(len(group_list)-1)/2, (num_classes-1)/2] * 2
    
    if fig_label_x_list is not None:
        fig_label_list = ["a", "b", "c", "d"]
        # fig_label_x_list = [-0.95, -0.95] * 2
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw=GRIDSPEC_KW)
    ax_list = list(axes.flatten())
    for i, ax in enumerate(ax_list):
        
        if i == 0 or i == 1:
            df = df_om
            behaviour_list = behaviour_list_om
        else:
            df = df_um
            behaviour_list = behaviour_list_um
            
        if len(df) == 0:
            print("The data frame is empty.")
            continue
            
        # Macro F1-score/IoU score
        if i == 0 or i == 2:
            ax, bar_label_list = plot_macro_score(df, ax, score_type, group_by_column, group_list, color_palette)
            text_color = 'white'
            ax.bar_label(
                ax.containers[0], labels=bar_label_list, 
                linespacing=0.9,
                fmt='%.3f', label_type='center', padding=0, 
                color=text_color, fontweight='bold', 
                fontsize=14, rotation=90
            )
            ax.tick_params(axis='x', labelsize=16, rotation=x_rotation_list[0])
        # Class F1-score/IoU score
        else:
            ax = plot_class_score(df, ax, score_type, group_by_column, group_list, color_palette)
            ax.get_legend().remove()
            ax.tick_params(axis='x', labelsize=16, rotation=x_rotation_list[1])
        
        if fig_label_x_list is not None:
            ax.text(
                fig_label_x_list[i], 1.15, 
                s=fig_label_list[i], 
                fontsize=28, fontweight='bold', va='center', ha='center'
            )
        
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
        ax.grid(axis='y', which='major', alpha=0.5)
        ax.grid(axis='y', which='minor', alpha=0.5)
        ax.tick_params(axis='y', labelsize=16)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label_list[i], fontsize=16, labelpad=10)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_ylim(-0.0, 1.05)
        ax.text(title_x_list[i], 1.1, s=title_list[i], 
                fontsize=16, va='center', ha='center')
    plt.show()
    plt.close()
    
    return fig


def plot_macro_and_class_test_scores_v3(
    df,
    score_type, 
    group_by_column, 
    group_list, 
    x_rotation_list,
    fig_label_x_list,
    fig_title,
    color_palette,
    GRIDSPEC_KW
):
    
    behaviour_list_om = None
    behaviour_list_um = None
    
    df_om = df[df["Species"] == "om"]
    if len(df_om) > 0: 
        category_list = df_om["Category"].unique().tolist()
        category_list.remove("Macro")
        category_list.remove("Weighted")
        behaviour_list_om = category_list
        print(behaviour_list_om)
    
    df_um = df[df["Species"] == "um"]
    if len(df_um) > 0: 
        category_list = df_um["Category"].unique().tolist()
        category_list.remove("Macro")
        category_list.remove("Weighted")
        behaviour_list_um = category_list
        print(behaviour_list_um)
    
    num_classes = 6
    
    if score_type == "F1":
        y_label_list = ["Macro F1-score", "Class F1-score"] * 2
    elif score_type == "IoU":
        y_label_list = [f"Macro IoU score", "Class IoU score"] * 2

    title_list = ["Streaked Shearwaters"] * 2 + ["Black-tailed Gulls"] * 2
    title_x_list = [(len(group_list)-1)/2, (num_classes-1)/2] * 2
    
    if fig_label_x_list is not None:
        fig_label_list = ["a", "b", "c", "d"]
        # fig_label_x_list = [-0.95, -0.95] * 2
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw=GRIDSPEC_KW)
    ax_list = list(axes.flatten())
    for i, ax in enumerate(ax_list):
        
        if i == 0 or i == 1:
            df = df_om
            behaviour_list = behaviour_list_om
        else:
            df = df_um
            behaviour_list = behaviour_list_um
            
        if len(df) == 0:
            print("The data frame is empty.")
            continue
            
        # Macro F1-score/IoU score
        if i == 0 or i == 2:
            ax, bar_label_list = plot_macro_score(df, ax, score_type, group_by_column, group_list, color_palette)
            text_color = 'white'
            ax.bar_label(
                ax.containers[0], labels=bar_label_list, 
                linespacing=0.9,
                fmt='%.3f', label_type='center', padding=0, 
                color=text_color, fontweight='bold', 
                fontsize=14, rotation=90
            )
            ax.tick_params(axis='x', labelsize=16, rotation=x_rotation_list[0])
        # Class F1-score/IoU score
        else:
            ax = plot_class_score(df, ax, score_type, group_by_column, group_list, color_palette)
            ax.get_legend().remove()
            ax.tick_params(axis='x', labelsize=16, rotation=x_rotation_list[1])
        
        if fig_label_x_list is not None:
            ax.text(
                fig_label_x_list[i], 1.15, 
                s=fig_label_list[i], 
                fontsize=28, fontweight='bold', va='center', ha='center'
            )
        
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
        ax.grid(axis='y', which='major', alpha=0.5)
        ax.grid(axis='y', which='minor', alpha=0.5)
        ax.tick_params(axis='y', labelsize=16)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label_list[i], fontsize=16, labelpad=10)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_ylim(-0.0, 1.05)
        ax.text(title_x_list[i], 1.1, s=title_list[i], 
                fontsize=16, va='center', ha='center')
    
    fig.suptitle(fig_title, fontsize=28, fontweight="bold")
    
    plt.show()
    plt.close()
    
    return fig


def calc_mean_diff_between_two_conditions(df, group1, group2):

    df_1 = df[(df["group_name"]==group1) & (df['Category']=='Macro')]
    display(df_1[["group_name", "F1"]].groupby("group_name").describe().applymap('{:,.5f}'.format))

    df_2 = df[(df['group_name']==group2) & (df['Category']=='Macro')]
    display(df_2[["group_name", "F1"]].groupby("group_name").describe().applymap('{:,.5f}'.format))

    mean_1 = df_1[["group_name", "F1"]].groupby("group_name").mean().values[0][0]
    mean_2 = df_2[["group_name", "F1"]].groupby("group_name").mean().values[0][0]
    mean_diff = mean_2 - mean_1
    print(f"mean difference is: {mean_diff:.5f} ({mean_diff * 100:.1f} %)")
    return mean_diff


def calc_class_mean_diff_between_two_conditions(df, behaviour_class, group1, group2):

    df_1 = df[(df["group_name"]==group1) & (df['Category']==behaviour_class)]
    display(df_1[["group_name", "F1"]].groupby("group_name").describe().applymap('{:,.5f}'.format))

    df_2 = df[(df['group_name']==group2) & (df['Category']==behaviour_class)]
    display(df_2[["group_name", "F1"]].groupby("group_name").describe().applymap('{:,.5f}'.format))

    mean_1 = df_1[["group_name", "F1"]].groupby("group_name").mean().values[0][0]
    mean_2 = df_2[["group_name", "F1"]].groupby("group_name").mean().values[0][0]
    mean_diff = mean_2 - mean_1
    print(f"mean difference is: {mean_diff:.5f} ({mean_diff * 100:.1f} %)")
    return mean_diff


def check_num_seeds_of_test_score_df(df, num_seeds=10):
    
    print(np.unique(df["group_name"], return_counts=True))
    
    # om
    print("---------------------------------")
    print("om")
    counts_om = df[ (df["Species"]=="om") & (df["Category"]=="Macro") ]["group_name"].value_counts()
    print(counts_om)   
        
    # um
    print("---------------------------------")
    print("um")
    counts_um = df[ (df["Species"]=="um") & (df["Category"]=="Macro") ]["group_name"].value_counts()
    print(counts_um)
    
    for i in range(0, len(counts_om)):
        if counts_om[i] != num_seeds or counts_um[i] != num_seeds:  
            raise Exception("!!! Check the df !!!")
    
    print("---------------------------------")
    print("Check complete. No issue found.")
    print("---------------------------------")
    
    

def find_params_from_str(target_str, key1="ks", key2="_"):
    """
    Args:
        target_str (str):
        key1 (str):
        key2 (str):
    Returns:
        param: 
    """
    index1 = target_str.find(key1)+len(key1)
    index2 = target_str.find(key2)
    param = target_str[index1:index2]
    target_str_ = target_str[index2+1:]
    return param, target_str_


def add_dcl_sa_parameter_columns_for_ex_d6x(df):
    
    ks_list = []
    cl_list = []
    cf_list = []
    cd_list = []
    ll_list = []
    ah_list = []
    al_list = []
    
    for i in range(0, len(df)):
        a = df["architecture"].values[i] + "_"
        a_ = a
        
        param, a_ = find_params_from_str(a, key1="ks", key2="_")
        ks_list.append(param)

        param, a_ = find_params_from_str(a_, key1="cl", key2="_")
        cl_list.append(param)

        param, a_ = find_params_from_str(a_, key1="cf", key2="_")
        cf_list.append(param)

        param, a_ = find_params_from_str(a_, key1="cd_", key2="_")
        cd_list.append(param)

        param, a_ = find_params_from_str(a_, key1="ll", key2="_")
        ll_list.append(param)

        param, a_ = find_params_from_str(a_, key1="ah", key2="_")
        ah_list.append(param)

        param, a_ = find_params_from_str(a_, key1="al", key2="_")
        al_list.append(param)
    
    df2 = df.copy()
    df2.insert(loc=2, column="al", value=al_list)
    df2.insert(loc=2, column="ah", value=ah_list)
    df2.insert(loc=2, column="ll", value=ll_list)
    df2.insert(loc=2, column="cd", value=cd_list)
    df2.insert(loc=2, column="cf", value=cf_list)
    df2.insert(loc=2, column="cl", value=cl_list)
    df2.insert(loc=2, column="ks", value=ks_list)
    
    return df2


def add_cnn_ae_parameter_columns_for_ex_d6x(df):
    
    ks_list = []
    depth_list = []
    c1f_list = []
    c2f_list = []
    c3f_list = []
    double_list = []
    
    for i in range(0, len(df)):
        a = df["architecture"].values[i] + "_"
        a_ = a # ks5_depth2_c1f128_doublefalse
        print(a_)
        
        ks, a_ = find_params_from_str(a, key1="ks", key2="_")
        ks_list.append(ks)

        depth, a_ = find_params_from_str(a_, key1="depth", key2="_")
        depth = int(depth)
        depth_list.append(depth)

        c1f, a_ = find_params_from_str(a_, key1="c1f", key2="_")
        c1f = int(c1f)
        c1f_list.append(c1f)
        
        double, a_ = find_params_from_str(a_, key1="double", key2="_")
        double_list.append(double)
        
        if double == "true":
            c2f = c1f*2
            c3f = c1f*2*2
        else:
            c2f = c1f
            c3f = c1f
        
        if depth == 2:
            c3f = "NA"
        
        c2f_list.append(c2f)
        c3f_list.append(c3f)
        
    df2 = df.copy()
    df2.insert(loc=2, column="double", value=double_list)
    df2.insert(loc=2, column="c3f", value=c3f_list)
    df2.insert(loc=2, column="c2f", value=c2f_list)
    df2.insert(loc=2, column="c1f", value=c1f_list)
    df2.insert(loc=2, column="depth", value=depth_list)
    df2.insert(loc=2, column="ks", value=ks_list)
    
    return df2
    

def find_da_type_and_da_params_from_str(target_str, key1="dcl_", key2="_"):
    """
    Args:
        target_str (str):
        key1 (str):
        key2 (str):
    Returns:
        info: 
    """
    index1 = target_str.find(key1)+len(key1)
    index2 = target_str.find(key2)
    info = target_str[index1:index2]
    target_str_ = target_str[index2+1:]
    return info, target_str_


def parse_test_score_file_name_for_ex_d70(test_score_path):
    
    seed = os.path.basename(test_score_path).replace("_best_model_weights.csv", "")[-1:]

    # print(os.path.basename(test_score_path))
    target_str = os.path.basename(test_score_path).replace(f"test_score_", "")
    # print(seed)
    da_type, _ = find_da_type_and_da_params_from_str(target_str, key1="dcl_", key2="_p1")
    da_param1, _ = find_da_type_and_da_params_from_str(target_str, key1="p1_", key2="_p2")
    da_param2, _ = find_da_type_and_da_params_from_str(target_str, key1="p2_", key2="_seed")
    
    if da_type in ["random2_om", "random2_um"]:
        da_type = "random2"
    elif da_type == "t_warp":
        da_type = "t-warp"
            
    if da_type in ["none", "random", "random2_om", "random2_um"]:
        da_param1 = int(0)
    elif da_type in ["scaling", "jittering", "t_warp"]:
        da_param1 = float(da_param1)
    elif da_type in ["permutation", "rotation"]:
        da_param1 = int(da_param1)
        
    return seed, da_type, da_param1, da_param2
    

def tsne(latent, y_ground_truth, fig_size=(9, 7)):
    """
        Plot t-SNE embeddings of the features
    """
    # latent = latent.cpu().detach().numpy()
    # y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, 
                verbose=1, 
                perplexity=40, 
                n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(1, 1, 1)
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    ax = sns_plot = sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=y_ground_truth,
        hue_order=[0, 1, 2, 3, 4, 5],
        palette='deep',
        legend="full",
        alpha=0.65
    )
    plt.show()
    plt.close()

    return fig


def mixup(x, y, mixup_alpha=0.2):
    # print(f"mixup_alpha = {mixup_alpha}")
    batch_size = x.size()[0]
    y = y.to(torch.float32)

    lam = np.random.beta(mixup_alpha, mixup_alpha, batch_size)
    x_shuffled, y_shuffled = sklearn.utils.shuffle(x, y)
    # https://pytorch.org/docs/stable/generated/torch.Tensor.add.html
    # https://pytorch.org/docs/stable/generated/torch.mul.html#torch.mul

    x_mixed = torch.Tensor(x.shape)
    y_mixed = torch.Tensor(y.shape)
    for i in range(batch_size):
        x_mixed[i] = x[i].clone().mul(lam[i]).add(x_shuffled[i].clone(), alpha=1 - lam[i]) # x_mixed = lambda*x[i] + (1-lambda)*x_[i]
        y_mixed[i] = y[i].clone().mul(lam[i]).add(y_shuffled[i].clone(), alpha=1 - lam[i]) # y_mixed = lambda*y[i] + (1-lambda)*y_[i]
    return x_mixed, y_mixed



def mixup_process(x, y, lam):
    '''
    x: shape(BS, CH, T, 1)
    y (y_onehot): shape(BS, T, 1, n_classes)
    lam: shape()
    '''
    # print(f"mixup_alpha = {mixup_alpha}")
    batch_size = x.size()[0]
    y = y.to(torch.float32)
    indices = np.random.permutation(batch_size)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    # expand lambda
    # log.debug(f"x.shape: {x.shape}")
    # log.debug(f"x.dim(): {x.dim()}")
    # log.debug(f"y.shape: {y.shape}")
    # log.debug(f"y.dim(): {y.dim()}")
    x_last_dim_idx = int(x.dim() - 1)
    y_last_dim_idx = int(y.dim() - 1)
    lam_x = lam.expand_as(x.transpose(0, x_last_dim_idx))
    lam_x = lam_x.transpose(0, x_last_dim_idx)
    lam_y = lam.expand_as(y.transpose(0, y_last_dim_idx))
    lam_y = lam_y.transpose(0, y_last_dim_idx)
    # lam_x = lam.expand_as(x.transpose(0, 3))
    # lam_x = lam_x.transpose(0, 3)
    # lam_y = lam.expand_as(y.transpose(0, 3))
    # lam_y = lam_y.transpose(0, 3)

    # mixup
    x_mixed = x * lam_x + x_shuffled * (1 - lam_x)
    y_mixed = y * lam_y + y_shuffled * (1 - lam_y)

    return x_mixed, y_mixed


# https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321
def to_one_hot(y, n_classes):
    '''
    y: shape(BS, T, 1)
    y_onehot: shape(BS, T, 1, n_classes)
    '''
    y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes)
    return y_onehot


# https://imagingsolution.net/program/python/python-basic/elapsed_time_hhmmss/
def elapsed_time_str(seconds):
    """
    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
        hh:mm:ss
    """
    seconds = int(seconds + 0.5)
    h = seconds // 3600
    m = (seconds - h * 3600) // 60
    s = seconds - h * 3600 - m * 60

    return f"{h:02}:{m:02}:{s:02}" # hh:mm:ss


def start_time_counter():
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    return start_time_perf, start_time_process


def end_time(start_time_perf, start_time_process):
    end_time_perf = time.perf_counter()
    elapsed_time_perf = end_time_perf - start_time_perf
    end_time_process = time.process_time()
    elapsed_time_process = end_time_process - start_time_process
    print("elapsed_time_perf: ", elapsed_time_str(elapsed_time_perf))
    print("elapsed_time_process: ", elapsed_time_str(elapsed_time_process))

    return elapsed_time_perf, elapsed_time_process