'''
run_ae_model_training.py

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import os
import time
import copy
from logging import getLogger
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src import utils
from src.data_module import (
    setup_dataloaders_supervised_learning, 
    prep_dataloaders_for_unsupervised_learning
)
from src.trainer import setup_model, train
from src.env import reset_seed
from src.gpu import get_available_devices

CUDA_DEVICE_ID = get_available_devices()

log = getLogger(__name__)

@hydra.main(version_base=None,
            config_path="./configs",
            # config_name="config_cnn_ae_pre.yaml",
            config_name="config_cnn_ae.yaml", # no-freeze, soft-freeze, hard-freeze
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

    # Make sure to cfg.model.pretrain == True
    if cfg.model.pretrain == True and cfg.metadata.task == "loiocv":
        log.debug(f"check the config file !")
        return
    
    if cfg.metadata.task == "pretraining":
        cfg.model.pretrain = True
        cfg.dataset.test_animal_id = "DUMMY"
    
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
    log.info(f"device={DEVICE}")
    
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
    if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
        (
            train_loader, 
            val_loader, 
            test_loader
        ) = prep_dataloaders_for_unsupervised_learning(cfg)
    else:
        (
            train_loader_balanced, 
            val_loader, 
            test_loader 
        ) = setup_dataloaders_supervised_learning(
            cfg, 
            train=True, 
            train_balanced=True
        )
        train_loader = train_loader_balanced
    

    model = setup_model(cfg)

    # Load pretrained weights and freeze parameters if hard freeze mode
    if cfg.model.model_name == 'cnn-ae' and cfg.model.pretrain == False:
        best_model = copy.deepcopy(model)
        best_model.to(DEVICE)
        dir_name = os.path.dirname(cfg.path.log.rootdir)
        dir_name = dir_name.replace(f"{cfg.ex}", "ex-d20")
        dir_name = dir_name.replace(f"{cfg.sub3}", "cnn-ae-pretrained-da-true")
        dir_name = dir_name.replace(cfg.dataset.test_animal_id, "DUMMY")
        path = os.path.join(dir_name, 
                            cfg.path.log.checkpoints.dir, 
                            cfg.path.log.checkpoints.fname)
        log.info(f'pretrained model path: {path}')
        best_model.load_state_dict(torch.load(path, map_location=DEVICE))

        # freeze
        log.info("freeze all model parameter")
        for param in best_model.parameters():
            param.requires_grad = False
        
        # unfreeze
        log.info("unfreeze some model parameter")
        for param in best_model.linear1.parameters():
                param.requires_grad = True  
        for param in best_model.out.parameters():
            param.requires_grad = True
        # hard freeze mode (until here)
        
        # soft or none freeze mode
        if cfg.model.freeze in ["soft-freeze", "no-freeze"]:
            for param in best_model.e_conv1.parameters():
                param.requires_grad = True
            for param in best_model.pool1.parameters():
                param.requires_grad = True
            for param in best_model.e_conv2.parameters():
                param.requires_grad = True
            for param in best_model.pool2.parameters():
                param.requires_grad = True
            for param in best_model.e_conv3.parameters():
                param.requires_grad = True
            for param in best_model.pool3.parameters():
                param.requires_grad = True
        
        model = best_model
    
    model.to(DEVICE) # send model to GPU
    print(f'model:\n {model}')

    time.sleep(5)

    # check if the parameters were freezed or not
    log.info("i: |(model param) name| |param.requires_grad|")
    for i, (name, param) in enumerate(model.named_parameters()):
        log.info(f'{i:0=2}: {name} {param.requires_grad}')

    # initialize the optimizer and loss
    if cfg.train.optimizer == "Adam":
        if cfg.model.model_name == 'cnn-ae' and cfg.model.freeze == "soft-freeze":
            lower_lr_for_conv = 1e-5
            log.info(f"soft-freeze -> using smaller learning rate: {lower_lr_for_conv}")
            optimizer = torch.optim.Adam(
                [
                    {'params': model.e_conv1.parameters(), 'lr': lower_lr_for_conv},
                    {'params': model.e_conv2.parameters(), 'lr': lower_lr_for_conv},
                    {'params': model.e_conv3.parameters(), 'lr': lower_lr_for_conv},
                    {'params': model.linear1.parameters()},
                    {'params': model.out.parameters()},
                ],
                lr=cfg.train.lr, 
                weight_decay=cfg.train.weight_decay
            )
        else: # hard or none freeze
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=cfg.train.lr, 
                weight_decay=cfg.train.weight_decay
            )         
    elif cfg.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=0.01, 
            momentum=0.9,
            weight_decay=0)
    
    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run training
    best_model, df_log = train(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
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