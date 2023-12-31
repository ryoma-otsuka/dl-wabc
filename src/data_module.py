'''
Dataset and Dataloader classes used in the study below.

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import glob
import os
import random
from logging import getLogger
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from src import utils, augmentations

log = getLogger(__name__)

MAX_INSTS = 100000000


# ----------------------------------------------------------------
# Unsupervised Pre-training
# ----------------------------------------------------------------

# Load the data when the batch is called in the for loop
class DatasetLogbot(Dataset):
    def __init__(self, paths, augmentation=True):
        self.paths = paths
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        npz = np.load(self.paths[index], allow_pickle=True)
        sample = npz['X']
        target = npz['label_id']
        
        # apply random data augmentation during unsupervised pre-training
        if self.augmentation == True:
            
            aug_sample_array = None
            # sample.shape: (20, 3, 50, 1)
            for i in range(0, sample.shape[0]):
                sample_tmp = sample[i]
                # print(f"sample_tmp.shape: {sample_tmp.shape}")
                sample_tmp = np.reshape(sample_tmp, (1, 50, 3))
                # print(f"sample_tmp.shape: {sample_tmp.shape}")
                
                # Random Data Augmentation
                random_int = np.random.randint(0, 6)
                # print(random_int)
                
                if random_int == 0:
                    sample_tmp = sample_tmp
                elif random_int == 1:
                    sample_tmp = augmentations.gen_aug(sample_tmp, 'scale', None, None) # da_param1=None default argument
                elif random_int == 2:
                    sample_tmp = augmentations.gen_aug(sample_tmp, 'noise', None, None)
                elif random_int == 3:
                    sample_tmp = augmentations.gen_aug(sample_tmp, 'perm', None, None)
                elif random_int == 4:
                    sample_tmp = augmentations.gen_aug(sample_tmp, 't_warp', None, None)
                elif random_int == 5:
                    sample_tmp = augmentations.gen_aug(sample_tmp, 'rotation', None, None)
                
                if i == 0:
                    aug_sample_array = sample_tmp 
                else:
                    aug_sample_array = np.concatenate([aug_sample_array, sample_tmp], axis=0)
            sample = aug_sample_array
            # print(f"sample.shape: {sample.shape}")
        
        return sample, target
    
    def __len__(self):
        return len(self.paths)

    
class DataLoaderNpz(DataLoader):
    def __init__(self, 
                 dataset, 
                 batch_size=30, 
                 shuffle=False, 
                 drop_last=True):
        super(
            DataLoaderNpz, 
            self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last)
        self.shuffle=shuffle

    
class DataLoaderNpy(DataLoader):
    def __init__(self, 
                 dataset, 
                 batch_size=30, 
                 shuffle=False, 
                 drop_last=True):
        super(
            DataLoaderNpy, 
            self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last)
        self.shuffle=shuffle

        
def prep_dataloaders_for_unsupervised_learning(cfg: DictConfig):
    
    if cfg.dataset.species == 'om':
        species_dir = 'omizunagidori'
    elif cfg.dataset.species == 'um':
        species_dir = 'umineko'

    train_loader_paths = []
    val_loader_paths = []
    test_loader_paths = []

    npz_format_data_dir = Path(cfg.path.dataset.npz_format_data_dir.unlabelled_data)
    log.info(f"npz_format_data_dir: {npz_format_data_dir}")
    npz_search_path = Path(
        npz_format_data_dir,
        species_dir, 
        "**/*.npz"
    )
    log.info(f"npz_search_path: {npz_search_path}")
    npz_file_path_list = sorted(glob.glob(str(npz_search_path)))

    if cfg.debug == True:
        npz_file_path_list = random.sample(npz_file_path_list, 2000)

    # limit animals used for unsupervised pre-training
    for npz_file_path in npz_file_path_list:
        animal_id = os.path.basename(os.path.dirname(npz_file_path))
        # training
        if animal_id in cfg.dataset.un_sup_pretraining.animal_id_list.train:
            train_loader_paths.append(npz_file_path)
        # validation
        if animal_id in cfg.dataset.un_sup_pretraining.animal_id_list.val:
            val_loader_paths.append(npz_file_path)
        # test
        if animal_id in cfg.dataset.un_sup_pretraining.animal_id_list.test:
            test_loader_paths.append(npz_file_path)

    if cfg.train.train_val_split == True:
        train_val_loader_paths = train_loader_paths + val_loader_paths
        dataset_size = len(train_val_loader_paths)
        train_count = int(dataset_size * cfg.train.train_data_ratio)
        val_count = dataset_size - train_count
        log.info(f"train_count={train_count}, val_count={val_count}")
        train_loader_paths_, val_loader_paths_ = train_test_split(
            train_val_loader_paths,
            test_size=val_count,
            train_size=train_count,
            random_state=cfg.seed,
            shuffle=True,
            stratify=None,
        )
    else:
        train_loader_paths_ = train_loader_paths
        val_loader_paths_ = val_loader_paths

    # train
    train_dataset = DatasetLogbot(
        paths=train_loader_paths_,
        augmentation=cfg.train.data_augmentation,
    )
    train_loader = DataLoaderNpz(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True
    )

    # validation
    val_dataset = DatasetLogbot(
        paths=val_loader_paths_,
        augmentation=False,
    )
    val_loader = DataLoaderNpz(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True
    )

    # test
    dataset_test = DatasetLogbot(
        paths=test_loader_paths,
        augmentation=False,
    )
    test_loader = DataLoaderNpz(
        dataset_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

        

# ----------------------------------------------------------------
# Supervised Learning
# ----------------------------------------------------------------
        
class BaseDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def load(self):
        pass
    
    def unload(self):
        pass


class DatasetLogbot2(BaseDataset):
    def __init__(self, 
                 samples=None, 
                 labels=None, 
                 paths=None, 
                 augmentation=False, 
                 da_type='random', 
                 da_param1=None, # None -> default params
                 da_param2=None,
                 in_ch=3):
        super(DatasetLogbot2, self).__init__(samples, labels)
        self.paths = paths
        self.augmentation = augmentation
        self.da_type = da_type
        self.da_param1 = da_param1
        self.da_param2 = da_param2
        self.in_ch = in_ch

    def __getitem__(self, index):
        if self.samples is None:
            self.load()
        
        sample, target = self.samples[index], self.labels[index]

        # check the shape of sample
        # print(f"sample.shape: {sample.shape}") # sample.shape: (1, 50, 3)
        # print(target.shape)
        
        if self.augmentation == True:
            # receive da_param from config and set da params here
            # how to give da_param to this class? 
            # see -> prep_dataloaders_for_supervised_learning
            if self.da_type in [None, "None", "none"]:
                sample = sample
            elif self.da_type == 'scaling':
                # utils.augmentations.gen_aug(sample, da_type, da_param1=None, da_param2=None)
                sample = augmentations.gen_aug(sample, 'scale', self.da_param1, None) 
            elif self.da_type == 'jittering':
                sample = augmentations.gen_aug(sample, 'noise', self.da_param1, None)
            elif self.da_type == 'permutation':
                sample = augmentations.gen_aug(sample, 'perm', self.da_param1, None)
            elif self.da_type == 't_warp':
                sample = augmentations.gen_aug(sample, 't_warp', self.da_param1, self.da_param2)
            elif self.da_type == 'rotation':
                sample = augmentations.gen_aug(sample, 'rotation', self.da_param1, None)
            elif self.da_type == 'random':
                # Random Data Augmentation
                random_int = np.random.randint(0, 6)
                # print(random_int)
                if random_int == 0:
                    sample = sample
                elif random_int == 1:
                    sample = augmentations.gen_aug(sample, 'scale', None, None) # da_param1=None default argument
                elif random_int == 2:
                    sample = augmentations.gen_aug(sample, 'noise', None, None)
                elif random_int == 3:
                    sample = augmentations.gen_aug(sample, 'perm', None, None)
                elif random_int == 4:
                    sample = augmentations.gen_aug(sample, 't_warp', None, None)
                elif random_int == 5:
                    sample = augmentations.gen_aug(sample, 'rotation', None, None)
            elif self.da_type == "random2_om":
                # Random Data Augmentation
                random_int = np.random.randint(0, 6)
                # print(random_int)
                # set da params (the best one)
                if random_int == 0:
                    sample = sample
                elif random_int == 1:
                    sample = augmentations.gen_aug(sample, 'scale', 0.1, None)
                elif random_int == 2:
                    sample = augmentations.gen_aug(sample, 'noise', 0.05, None)
                elif random_int == 3:
                    sample = augmentations.gen_aug(sample, 'perm', 10, None)
                elif random_int == 4:
                    sample = augmentations.gen_aug(sample, 't_warp', 0.2, None)
                elif random_int == 5:
                    sample = augmentations.gen_aug(sample, 'rotation', 45, None)
            elif self.da_type == "random2_um":
                # Random Data Augmentation
                random_int = np.random.randint(0, 6)
                # print(random_int)
                # set da params (the best one)
                if random_int == 0:
                    sample = sample
                elif random_int == 1:
                    sample = augmentations.gen_aug(sample, 'scale', 0.8, None)
                elif random_int == 2:
                    sample = augmentations.gen_aug(sample, 'noise', 0.1, None)
                elif random_int == 3:
                    sample = augmentations.gen_aug(sample, 'perm', 15, None)
                elif random_int == 4:
                    sample = augmentations.gen_aug(sample, 't_warp', 0.2, None)
                elif random_int == 5:
                    sample = augmentations.gen_aug(sample, 'rotation', 45, None)
            else:
                raise Exception(
                    f'da_type "{self.da_type}" is not appropriate.')

                
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample)
                
            return sample, target
        
        else:
            
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample)
            
            return sample, target
    
    def __len__(self):
        return len(self.paths)
            
    def load(self, MAX_INSTS=MAX_INSTS):
        self.samples = []
        self.labels = []
        X_list = []
        label_id_list = []
        for npz_file_path in self.paths:
            npz = np.load(npz_file_path, allow_pickle=True)
            X = npz["X"]
            label_id = npz["label_id"]
            try:
                label_id_list.append(label_id)
                X_list.append(X)          
            except:
                # print("error in ", npz_file_path)
                exit()
            if len(X_list) > MAX_INSTS:
                break
        if len(self.paths) > 1:
            log.info(
                f".npz files: {len(self.paths)} |\
                samples: {len(X_list)} | labels: {len(label_id_list)}"
            )
        
        self.samples = X_list
        self.labels = label_id_list
        samples = np.array(X_list)
        labels = np.array(label_id_list)
        
        if self.augmentation == True:
            return DatasetLogbot2(samples=samples, 
                                  labels=labels, 
                                  paths=self.paths, 
                                  augmentation=True,
                                  da_type=self.da_type,
                                  da_param1=self.da_param1,
                                  da_param2=self.da_param2,
                                  in_ch=self.in_ch)
        else:
            return DatasetLogbot2(samples=samples, 
                                  labels=labels, 
                                  paths=self.paths, 
                                  augmentation=False,
                                  da_type=self.da_type,
                                  da_param1=self.da_param1,
                                  da_param2=self.da_param2,
                                  in_ch=self.in_ch)
    
    def unload(self):
        self.samples = None
        self.labels = None


class DataLoaderNpz2(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False, drop_last=True):
        super(
            DataLoaderNpz2, 
            self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last)
        self.shuffle=shuffle
        
    def load(self, MAX_INSTS=MAX_INSTS):
        dataset=self.dataset.load(MAX_INSTS)
        return DataLoaderNpz2(dataset, 
                              batch_size=self.batch_size, 
                              shuffle=self.shuffle, 
                              drop_last=self.drop_last)
    
    def unload(self):
        self.dataset.unload()
    
        
def prep_dataloaders_for_supervised_learning(cfg, test_only=False):
    
    train_animal_id_list = cfg.dataset.labelled.animal_id_list.train
    val_animal_id_list = cfg.dataset.labelled.animal_id_list.val
    test_animal_id_list = cfg.dataset.labelled.animal_id_list.test
    train_val_split = cfg.train.train_val_split
    train_data_ratio = cfg.train.train_data_ratio
    batch_size = cfg.train.batch_size
    shuffle = cfg.train.shuffle
    
    # Added for experiment 07 (Experiment S1) data augmentation parameter grid search
    da_param1 = cfg.dataset.da_param1
    da_param2 = cfg.dataset.da_param2
    da_param1 = None if da_param1 in ["None", "none", None, False, 0] else da_param1
    da_param2 = None if da_param2 in ["None", "none", None, False, 0] else da_param2    
    
    # if cfg.metadata.approach == "experiment-07-da-params" and cfg.metadata.task == "da-params":
    #     da_param1 = cfg.dataset.da_param1
    #     da_param2 = cfg.dataset.da_param2
    #     # ["None", False, 0] -> None
    #     da_param1 = None if da_param1 in ["None", False, 0] else da_param1
    #     da_param2 = None if da_param2 in ["None", False, 0] else da_param2
    # else: # Use default values (by setting to None) except ex07 (Expeiriment S1 in dl-wabc study)
    #     da_param1 = None # -> use default value
    #     da_param2 = None # -> use default value

    if cfg.dataset.species == 'om':
        species_dir = 'omizunagidori'
    elif cfg.dataset.species == 'um':
        species_dir = 'umineko'
    
    train_loader_paths = []
    val_loader_paths = []
    test_loader_paths = []

    npz_format_data_dir = Path(cfg.path.dataset.npz_format_data_dir.labelled_data)
    log.info(f"npz_format_data_dir: {npz_format_data_dir}")
    npz_search_path = Path(
        npz_format_data_dir, 
        species_dir, 
        "**/*.npz"
    )
    log.info(f"npz_search_path: {npz_search_path}")
    npz_file_path_list = sorted(glob.glob(str(npz_search_path)))
    
    if cfg.debug == True:
        npz_file_path_list = random.sample(npz_file_path_list, 2000)
    log.info(f'N of npz files (instances): {len(npz_file_path_list)}')
    
    
    # -----------
    # Test data
    # -----------
    for idx, npz_file_path in enumerate(npz_file_path_list):
        animal_id = os.path.basename(os.path.dirname(npz_file_path))
        if animal_id in test_animal_id_list:
            test_loader_paths.append(npz_file_path)
    
    # test dataset & dataloader
    test_dataset = DatasetLogbot2(
        samples=None, 
        labels=None, 
        paths=test_loader_paths, 
        augmentation=False, # Do not apply DA for test dataset
        da_type=cfg.dataset.da_type,
        da_param1=da_param1,
        da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    test_loader = DataLoaderNpz2(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    if test_only == True:
        return None, None, test_loader
    
    # ---------------------
    # Train and val data
    # ---------------------
    
    for idx, npz_file_path in enumerate(npz_file_path_list):
        animal_id = os.path.basename(os.path.dirname(npz_file_path))
        if animal_id in train_animal_id_list:
            train_loader_paths.append(npz_file_path)
        if animal_id in val_animal_id_list:
            val_loader_paths.append(npz_file_path)
    
    if train_val_split == True:
        train_val_loader_paths = train_loader_paths + val_loader_paths
        dataset_size = len(train_val_loader_paths)
        train_count = int(dataset_size * train_data_ratio)
        val_count = dataset_size - train_count
        train_loader_paths_, val_loader_paths_ = train_test_split(
            train_val_loader_paths, 
            test_size=val_count, 
            train_size=train_count, 
            random_state=cfg.seed, 
            shuffle=True, 
            stratify=None)
    else:
        train_loader_paths_ = train_loader_paths
        val_loader_paths_ = val_loader_paths
        
    # train dataset & dataloader
    train_dataset = DatasetLogbot2(
        samples=None, 
        labels=None, 
        paths=train_loader_paths_, 
        augmentation=cfg.train.data_augmentation,
        da_type=cfg.dataset.da_type,
        da_param1=da_param1,
        da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    train_loader = DataLoaderNpz2(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=True
    )
    
    # val dataset & dataloader
    val_dataset = DatasetLogbot2(
        samples=None, 
        labels=None, 
        paths=val_loader_paths_, 
        augmentation=False, # Do not apply DA for validation dataset
        da_type=cfg.dataset.da_type,
        da_param1=da_param1,
        da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    val_loader = DataLoaderNpz2(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def setup_balanced_dataloader(train_dataset, cfg):

    y_train_indices = range(0, len(train_dataset))
    
    # use last label
    y_train_ = np.array(
        [int(train_dataset.labels[i][0][-1]) for i in y_train_indices]
    ) 
    
    # convert the labels before calculating the weights
    label_species = utils.get_label_species(cfg)
    y_train_ = utils.convert_numpy_labels(y_train_, label_species)
    
    # calculate class weights
    y_train = y_train_.tolist()
    class_sample_count_list = []
    class_sample_ratio_list = []
    weight_list = []
    total = len(y_train)
    for i in range(0, cfg.dataset.n_classes):
        count = y_train.count(i)
        class_sample_count_list.append(count)
        class_sample_ratio_list.append(count/total)
        if count == 0:
            weight_list.append(0)
        else:
            weight_list.append(1/count)
    weights = np.array(weight_list)
    log.info(f"class_sample_count_list: {class_sample_count_list}")
    log.info(f"class_sample_ratio_list: {(np.round(class_sample_ratio_list, 4))}")
    log.info(f"weights: {weights}")
    
    # assign a class weight to each window
    sample_weights = np.array([weights[t] for t in y_train])    
    sample_weights = torch.from_numpy(sample_weights)
    
    # generate sampler and balanced dataloader
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights.type('torch.DoubleTensor'), 
        len(sample_weights)
    )
    train_loader_balanced = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        drop_last=True, 
        sampler=sampler
    )
    return train_loader_balanced



def setup_dataloaders_supervised_learning(cfg, 
                                          train=True, 
                                          train_balanced=True):
    (
        train_loader, val_loader, test_loader
    ) = prep_dataloaders_for_supervised_learning(cfg)
    # print("train_loader: ", len(train_loader))
    # print("val_loader: ", len(val_loader))
    # print("test_loader: ", len(test_loader))
    
    if train == True:
        if hasattr(train_loader, "load"): 
            # print("Loading train_loader: ")
            train_loader = train_loader.load()
        if hasattr(val_loader, "load"): 
            # print("Loading val_loader: ")
            val_loader = val_loader.load()
        if train_balanced == True:
            train_dataset = train_loader.dataset
            train_loader_balanced = setup_balanced_dataloader(train_dataset, cfg)
            return train_loader_balanced, val_loader, test_loader
        else:
            return train_loader, val_loader, test_loader
    else:
        if hasattr(test_loader, "load"): 
            # print("Loading test_loader: ")
            test_loader = test_loader.load()
        return train_loader, val_loader, test_loader