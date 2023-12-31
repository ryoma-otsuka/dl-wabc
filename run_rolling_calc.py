'''
run_rolling_calc.py

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import os
import glob
import polars as pl
from src import feature_extraction
from src import utils

# species_list = ["omizunagidori", "umineko"]
# species_list = ["omizunagidori"]
species_list = ["umineko"]

debug_test_mode = True
# debug_test_mode = False

def main():
    
    print(f"--------------------------------")
    print(f"running run_rolling_calc.py !")
    print(f"--------------------------------")
            
    for idx, species in enumerate(species_list):

        base_dir = "/home/bob/protein/dl-wabc/data/datasets/logbot_data"
        input_target = f"{base_dir}/preprocessed_data/{species}/*.csv"
        preprocessed_data_path_list = sorted(glob.glob(input_target))
        print("N of individuals:", len(preprocessed_data_path_list))

        output_dir = f"{base_dir}/feature_extraction/data_after_rolling_calc/"

        # for test
        # preprocessed_data_path_list = preprocessed_data_path_list[:4]
        
        for preprocessed_data_path in preprocessed_data_path_list:
            start_time_perf, start_time_process = utils.start_time_counter()
            animal_id = os.path.basename(preprocessed_data_path).replace(".csv", "")
            print("-----------------------------------------------------------------")
            print(f"Loading data for: {animal_id}")
            # df = pd.read_csv(preprocessed_data_path, low_memory=False)
            df = pl.read_csv(preprocessed_data_path)
            n_unique_labels = len(df['label_id'].unique())
            if n_unique_labels > 1: # all nan -> 1, one or more labels -> =< 2
                df = feature_extraction.calc_static_and_dynamic_components(
                    df, 
                    sampling_rate=25, 
                    rolling_window_sec=2
                ) 
                df_labelled = df
                print(f"len(df_labelled): {len(df_labelled)}")
                df_labelled = df_labelled.to_pandas()
                print(df_labelled.head(3))
                df_save_dir = f"{output_dir}/{species}"
                os.makedirs(df_save_dir, exist_ok=True)
                df_save_path = f"{df_save_dir}/{animal_id}.csv"
                
                print(f"df_save_path: {df_save_path}")
                
                if debug_test_mode == True:
                    print(f"| debug test mode -> do not save data |")
                else:
                    df_labelled.to_csv(df_save_path, index=False)
                
            else:
                print("No labelled data")
                
            elapsed_time_perf, elapsed_time_process = utils.end_time(
                start_time_perf, start_time_process)
            
            print(f"--------------------------------")
            print(f"run_rolling_calc.py completed !")
            print(f"--------------------------------")
            
if __name__ == '__main__':
    main()