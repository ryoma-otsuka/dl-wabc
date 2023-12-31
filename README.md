# Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers (dl-wabc)

## Author
Ryoma Otsuka  
https://sites.google.com/view/ryomaotsuka/

## Reference

> Ryoma Otsuka, Naoya Yoshimura, Kei Tanigaki, Shiho Koyama, Yuichi Mizutani, Ken Yoda, Takuya Maekawa. (2024). Exploring deep learning techniques for wild animal behaviour classification using animal-borne accelerometers. Methods in Ecology and Evolution.


## Requirements

Ubuntu 18.04.6 LTS  
Python 3.10.8  
See [requirements.txt](requirements.txt)
  
<!-- <br> -->
  
## Data

### Datasets

Please see the following information in the paper.     
- Datasets: Materials and Methods 2-1. Datasets & Table S2    
- Behaviour classes: Table S3 in the Supporting Information file  

See Data Availability section to access the datasets.

The names of directories and files in the datasets often include the term "omizunagidori" or "umineko." These terms represent the Japanese names for streaked shearwaters (_Calonectris leucomelas_) and black-tailed gulls (_Larus crassirostris_) in alphabetical form, respectively. The bird names in both the source code and the paper begin with "OM" or "UM," representing the initial two characters of "omizunagidori" and "umineko," respectively.


### Data preparation for deep learning
1. raw data (.csv): raw data exported from our labelling application
2. preprocessed data (.csv): cleaned and resampled data
3. extracted time windows (.npz): an extracted window contains the following items: 
    * X (sample): tri-axial acceleration data (acc_x, acc_y, acc_z)
    * label_id (target): Each data point in the labelled window has a label as "int"
    * timestamp: unixtime
    * animal_id: animal_id such as "OM1901" or "UM2202".

The data preparation can be done using the Jupyter Notebook file.    
[notebooks/run_dataset_preparation.ipynb](notebooks/run_dataset_preparation.ipynb)

See also the following python files:  
[src/data_preprocess_logbot.py](src/data_preprocess_logbot.py)  
[src/data_module.py](src/data_module.py)
  
<!-- <br> -->
  
## Deep Learning

### Models
[src/models/](src/models/)

### Data Augmentation
[src/augmentations.py](src/augmentations.py)  
[src/data_module.py](src/data_module.py)

### Model training

The following is an example command for training a deep learning model.  
```bash
python run_dl_model_training.py -m dataset=om model=dcl debug=false k8s=false seed=0 train.cuda=0 
```
  
[run_dl_model_training.py](run_dl_model_training.py)  
[run_ae_model_training.py](run_ae_model_training.py)  
[src/trainer.py](src/trainer.py)  

### Model evaluation
The following Jupyter Notebook can be used to run test of deep learning models.  
[notebooks/run_test.ipynb](notebooks/run_test.ipynb)  
  
<!-- <br> -->
  
## LightGBM and XGBoost

The following is an example command for training and testing LightGBM or XGBoost models.    
```bash
python run_tree_model_training_and_test.py -m dataset=um model=xgboost debug=false seed=0 train.cuda=0 
```
  
[run_tree_model_training_and_test.py](run_tree_model_training_and_test.py)   

For extraction of handcrafted features, see the following files:   
[run_rolling_calc.py](run_rolling_calc.py)    
[run_feature_extraction.py](run_feature_extraction.py)   
[src/feature_extraction.py](src/feature_extraction.py)   

Once the preprocessed files are ready (step 2 in the Data preparation for deep learning),  
run ```python run_rolling_calc.py```, then, run ```python run_feature_extraction.py```.
  
<!-- <br> -->
  
## Terms in the source code and paper

### Model name
| Source code | Paper    |
| :---        | :----    |
| dcl-sa      | DCLSA    |
| resnet-l-sa | DCLSA-RN | 
| cnn_ae_v5   | CNN-AE   |
| cnn_ae_v6   | CNN-AE   |

### Experiment name
| Source code | Paper    |
| :---        | :----    |
| ex-d10      | Experiment 1 Data Augmentation on DCL    |
| ex-d11      | Experiment 1 Data Augmentation on DCLSA  |
| ex-d16      | Experiment 1 Manifold Mixup on DCL       |
| ex-d17      | Experiment 1 Manifold Mixup on DCL-V3 (mixup after LSTM layer)   |
| ex-d20      | Experiment 2 Unsupervised Pre-training of CNN-AE   |
| ex-d21      | Experiment 2 Unsupervised Pre-training of CNN-AE (no, soft, hard freeze)   |
| ex-d22      | Experiment 2 CNN-AE w/o pre-training    |
| ex-d30      | Experiment 3 Model Comparison   |
| ex-d60      | Supplementary Experiment S2 Hyperparameter tuning of DCLSA   |
| ex-d61      | Supplementary Experiment S2 Hyperparameter tuning of CNN-AE w/o   |
| ex-d70      | Supplementary Experiment S1 Hyperparameter tuning for each data augmentation type   |
  
<!-- <br> -->
  
## Links
The code in this repository was written with reference to the following repositories or papers:   

OpenPack: [GitHub Link](https://github.com/open-pack/openpack-torch)   
   
CL-HAR: [Paper Link 1](https://dl.acm.org/doi/10.1145/3534678.3539134)/[Paper Link 2](https://arxiv.org/abs/2202.05998)/[GitHub Link](https://github.com/Tian0426/CL-HAR)    
Qian et al., (2022) *KDD 2022*  
 "What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks?"  

  
DeepConvLSTM: [Paper Link](https://www.mdpi.com/1424-8220/16/1/115/htm)    
Ordóñez and Roggen, (2016) *Sensors*  
"Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition"  

  
DeepConvLSTM+SelfAttn: [Paper Link](https://ieeexplore.ieee.org/document/9296308)   
Singh et al., (2021) *IEEE Sensors Journal*  
"Deep ConvLSTM With Self-Attention for Human Activity Decoding Using Wearable Sensors"  