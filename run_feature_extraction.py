'''
run_feature_extraction.py

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from src import feature_extraction
from src import utils

# debug_test_mode = False
debug_test_mode = True # uncomment this line to run in debug test mode

# species_list = ["omizunagidori", "umineko"]
# species_list = ["omizunagidori"]
species_list = ["umineko"]

path = "configs/features/119.yaml"
config = OmegaConf.load(path)
features = config.features_list
print(f"Number of features: {len(features)}")

col_names = ['animal_id', 'unixtime', 'label', 'label_id']
col_names = col_names + features
print(col_names)

def main():
    
    print(f"-----------------------------------")
    print(f"running run_feature_extraction.py !")
    print(f"-----------------------------------")
    
    for i, species in enumerate(species_list):
        data_root_dir = "/home/bob/protein/dl-wabc/data/datasets/logbot_data"
        target = f"{data_root_dir}/feature_extraction/data_after_rolling_calc/{species}/*.csv"
        labelled_data_path_list = sorted(glob.glob(target))
        for idx, labelled_data_path in enumerate(labelled_data_path_list):
            start_time_perf, start_time_process = utils.start_time_counter()
            print(f"labelled_data_path: {labelled_data_path}")
            print("--------------------------------------")
            animal_id = os.path.basename(labelled_data_path).replace(".csv", "")
            # print(animal_id)
            print(f"loading csv file for {animal_id} ...")
            df_features = pd.DataFrame(columns=col_names)
            df = pd.read_csv(labelled_data_path, low_memory=False)
            df = df.reset_index(drop=True)
            # df = df.with_row_count(name='index')
            # display(df.head(3))
            print(f"length of the input df: {len(df)}")
            elapsed_time_perf, elapsed_time_process = utils.end_time(
                start_time_perf, start_time_process)
            
            # load unixtime list for this animal/session
            path = f"{data_root_dir}/npz_files_scan/{species}/df_scan_{animal_id}.csv"
            df_scan = pd.read_csv(path)
            scanned_unixtime_values = df_scan["unixtime"].values
            scanned_label_id_values = df_scan["label_id"].values        

            window_size = 50
            window_stepsize = 25
            window_counter = 0
            df_values = df.values
            data_array_list = []
            label_id_list = []
            
            scanned_unixtime_list = []
            scanned_label_id_list = []
            for i in range(0, scanned_label_id_values.size):
                if (i+1) % window_size == 0:
                    scanned_unixtime_list.append(scanned_unixtime_values[i])
                    scanned_label_id_list.append(scanned_label_id_values[i])
            
            start_time_perf, start_time_process = utils.start_time_counter()
            for i in tqdm(range(0, len(df)-window_size, window_stepsize)): # with progress bar
            # for i in range(0, len(df)-window_size, window_overlap): # without progress bar
                window_tmp = df_values[i:i+window_size]
                unixtime_tmp = window_tmp[:, 1].astype(np.float64)  # 'unixtime'
                unixtime_tmp_representative = np.float64(unixtime_tmp[-1])
                    
                unixtime_diff = np.max(unixtime_tmp) - np.min(unixtime_tmp) 

                if unixtime_diff < 2.0:
                    X_tmp = window_tmp[:, 2:5].astype(np.float64) # 'acc_x', 'acc_y', 'acc_z'
                    num_zeros_in_X = np.sum(X_tmp == 0)
                    label_id_tmp = window_tmp[:, 6].astype(np.float64) # 'label_id'
                    num_na_in_label_id = np.sum(np.isnan(label_id_tmp))
                    num_unique_label_id = len(np.unique(label_id_tmp))

                    if num_zeros_in_X < 5 and num_na_in_label_id == 0 and num_unique_label_id == 1: 
                        unixtime = unixtime_tmp[-1]
                        label = window_tmp[:, 5][-1]  # 'label'
                        label_id = int(label_id_tmp[-1])
                        
                        label_id_list.append(label_id) # just to check
                        
                        window_counter += 1
                        
                        if debug_test_mode == True:
                            continue
                        
                        # feature extraction
                        feature_list = feature_extraction.calc_features_for_one_sliding_window(
                            window_tmp
                        )
                        data_array_1 = np.array([animal_id, unixtime, label, label_id])
                        data_array_2 = np.array(feature_list)
                        # data_array_2 = np.array(feature_list, dtype=object)
                        
                        # print(data_array_1.shape)
                        # print(data_array_2.shape)
                        data_array_tmp = np.concatenate([data_array_1, data_array_2], axis=0)
                        # data_array_tmp = np.concatenate([data_array_1, data_array_2], axis=0).reshape(1, -1)
                        # print(data_array_tmp.dtype) # -> <U32
                        # https://numpy.org/devdocs/reference/arrays.dtypes.html

                        # print(data_array_tmp.shape)
                        # data_list_tmp = [animal_id, unixtime, label, label_id]
                        # data_list_tmp.extend(feature_list)
                        # data_array_tmp = np.array(data_list_tmp).reshape(1, -1)
                        
                        data_array_list.append(data_array_tmp)  
                        # if window_counter == 0:
                        #     # print(data_array_tmp.shape)
                        #     data_array = data_array_tmp
                        # else:
                        #     data_array = np.append(data_array, data_array_tmp, axis=0)
                        
                        # just to check the extracted data window exists in the data for dl 
                        # (identified by the unixtime of the last sample in the window) 
                        if unixtime_tmp_representative not in scanned_unixtime_list:
                            raise Exception(f"unixtime {unixtime_tmp_representative} not included in scanned_unixtime_values")
                else:
                    # print(i, "skip this window")
                    continue
            
            # check length
            print(f"window_counter: {window_counter}")
            print(f"len(label_id_list): {len(label_id_list)}")
            print(f"len(scanned_unixtime_list): {len(scanned_unixtime_list)}")
            print(f"len(scanned_label_id_list): {len(scanned_label_id_list)}")
            print(f"scanned_unixtime_values.shape: {scanned_unixtime_values.shape}")
            print(f"scanned_label_id_values.shape: {scanned_label_id_values.shape}")
            
            data_array = np.asarray(data_array_list)
            print(f"data_array.shape: {data_array.shape}")
            
            print("----- Check label_id and scanned label_id              -----")
            label_check_array = np.array(label_id_list) - np.array(scanned_label_id_list)
            print(f"np.sum(label_check_array): {np.sum(label_check_array)}")
            is_array_equal = np.array_equal(np.array(label_id_list), np.array(scanned_label_id_list))
            print(f"is_array_equal: {is_array_equal}")
            
            if np.array(label_id_list).shape[0] != np.array(scanned_label_id_list).shape[0]:
                raise Exception(f"extracted window size {np.array(label_id_list).shape[0]} and scanned sample size {np.array(scanned_label_id_list).shape[0]} are different!")
            
            if np.sum(label_check_array) > 0 :
                raise Exception(f"np.sum(label_check_array) should be 0, but {np.sum(label_check_array)}")
                
            if debug_test_mode:
                print("| debug test mode -> do not save extracted features |")
                continue
            
            df_features = pd.DataFrame(
                data=data_array,
                columns=col_names
            )

            df_features = df_features.reset_index(drop=True)
            # df_features = df_features.with_row_count(name='index')

            if species == "omizunagidori":
                label_species = "om"
            elif species == "umineko":
                label_species = "um"
            df_features = utils.convert_pandas_labels(
                df_features, label_species)

            print(f"N of extracted windows (len(df_features)): {len(df_features)}")
            print(f"N of extracted window counter: {window_counter}")
            print(df_features.head(3))

            base_dir = "/home/bob/protein/dl-wabc/data/datasets/logbot_data/feature_extraction/acc_features"
            save_path = f"{base_dir}/{species}/{animal_id}.csv"
            df_features.to_csv(save_path, index=False)
            print(f"df_features save at path: {save_path}")

            elapsed_time_perf, elapsed_time_process = utils.end_time(
                start_time_perf, start_time_process)
        
        print(f"All df_features for {species} were succecfully saved.")
            
if __name__ == '__main__':
    main()