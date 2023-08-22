# Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers (dl-wabc)

<!-- Note for author.  
This repository for the development (not for publishing) is named as "cl-bbr", because this repository was originally created to develop contrastive learning models for bird behavior recognition (CL-BBR). -->

## Author
Ryoma Otsuka  
https://sites.google.com/view/ryomaotsuka/

## Reference
> Ryoma Otsuka, Naoya Yoshimura, Kei Tanigaki, Shiho Koyama, Yuichi Mizutani, Ken Yoda, Takuya Maekawa. YYYY (year) 'Exploring deep learning techniques for wild animal behaviour classification using animal-borne accelerometers', XXX (journal).


## Requirements
```bash
pip install -r requirements.txt
```

## Datasets
Please see the following supporting information of the study paper:    
Datasets: Table S2  
Behaviours: Table S3  

## Data Preparation Flow
1. raw data (.csv): raw data extracted using BioTagger2018 app.
2. preprocessed data (.csv): cleaned and resampled data.
3. npz file (.npz): extracted sliding windows.   
    A window data has 
    * sample: triaxial acceleration data (acc_x, acc_y, acc_z)
    * label_id: if a window is labelled window, each data point in the window has a label (int)
    * timestamp: unixtime
    * animal_id: animal_id such as "OM1901" or "UM2202".

[notebooks/run_dataset_preparation.ipynb]("notebooks/run_dataset_preparation.ipynb")  

[src/data_preprocess_logbot.py]("src/data_preprocess_logbot.py")

[src/data_module.py]("src/data_module.py")

Please modify the paths accordingly to fit your environment.

## Deep Learning Models
[src/models/dl_models.py]("src/models/dl_models.py")

## Data Augmentation
[src/augmentations.py]("src/augmentations.py")

## Model training
```bash
python run_dl_model_training.py -m train.cuda=0 k8s=false seed=0 test_animal_id="OM2101"
```
(use run_dl_model_training_ae.py for CNN-AE model)

[src/trainer.py]("src/trainer.py")

## Model evaluation
[notebooks/run_test_loiocv.ipynb]("notebooks/run_test_loiocv.ipynb")
<!-- [notebooks/20_test_loiocv.ipynb]("notebooks/20_test_loiocv.ipynb) -->

## LightGBM and XGBoost
```bash
python run_ml_model_training_test.py -m train.cuda=0 seed=0
```
(Note that we used GPU for XGBoost)

[src/run_tolling_cal.py]("src/run_tolling_cal.py")

[src/run_feature_extraction.py]("src/run_feature_extraction.py")

## Links
The code in this repository was written with reference to the following repositories:   

OpenPack: [open-pack/openpack-torch](https://github.com/open-pack/openpack-torch)   
   
CL-HAR: [Tian0426/CL-HAR](https://github.com/Tian0426/CL-HAR)  
  
DeepConvLSTM:  
Ordóñez and Roggen (2016) *Sensors*  
"Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"  
[Paper Link](https://www.mdpi.com/1424-8220/16/1/115/htm) / [GitHub Link](https://github.com/STRCWearlab/DeepConvLSTM)
  
DeepConvLSTM+SelfAttn:  
Singh et al., (2020) *IEEE Sensors Journal*  
"Deep ConvLSTM With Self-Attention for Human Activity Decoding Using Wearable Sensors"  
[Paper Link](https://ieeexplore.ieee.org/document/9296308) / [GitHub Link](https://github.com/isukrit/encodingHumanActivity)  


## Terminology in the source codes and paper
AE5 or AE6 -> CNN-AE  
RNLSA -> DCLSA-RN  

ex00 -> Experiment 3  
ex01 -> Experiment 1 Data Augmentation  
ex02 -> Experiment 1 Manifold Mixups  
ex05 -> Experiment 2 CNN-AE pre-training  
ex06 -> Supplemental Experiment S2  
ex07 -> Supplemental Experiment S1  