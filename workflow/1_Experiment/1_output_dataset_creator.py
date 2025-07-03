from datetime import date
import sys
import pandas as pd
import os
from pathlib import Path
import re
import pyarrow

# Get the directory of the current script
current_script_path = Path(__file__).resolve().parent
dir_executables = current_script_path / 'Executables'

sys.path.insert(0, dir_executables)
import local_dataset_creator_0

dir_futures = current_script_path / 'Experimental_Platform' / 'Futures'
sys.path.insert(0, dir_futures)
import local_dataset_creator_f

'Define control parameters:'
if __name__ == '__main__':
    run_for_first_time = True
    
    if run_for_first_time == True:
        local_dataset_creator_0.execute_local_dataset_creator_0_outputs(dir_executables)
        local_dataset_creator_f.execute_local_dataset_creator_f_outputs(dir_futures)
        local_dataset_creator_0.execute_local_dataset_creator_0_inputs(dir_executables)
        local_dataset_creator_f.execute_local_dataset_creator_f_inputs(dir_futures)
    
    ############################################################################################################
    output_dataset_0_path = dir_executables / 'output_dataset_0.csv'
    df_0_output = pd.read_csv(output_dataset_0_path, index_col=None, header=0, low_memory=False)
    df_0_output['Scen_fut'] = df_0_output['Strategy'].astype(str) + "_" + df_0_output['Future.ID'].astype(str)
    
    # output_dataset_f_path = dir_futures / 'output_dataset_f.csv'
    output_dataset_f_path = dir_futures / 'output_dataset_f.parquet'
    # df_f_output = pd.read_csv( output_dataset_f_path, index_col=None, header=0)
    df_f_output = pd.read_parquet(output_dataset_f_path, engine='pyarrow')

    # Reemplazar valores no finitos (NaN, inf, -inf) con un valor predeterminado
    df_0_output['YEAR'] = pd.to_numeric(df_0_output['YEAR'], errors='coerce')  # Convierte valores no válidos a NaN
    df_0_output['YEAR'] = df_0_output['YEAR'].fillna(0)  # Rellena NaN con 0 (puedes usar otro valor si prefieres)
    df_0_output['YEAR'] = df_0_output['YEAR'].astype(int)  # Convierte a int
    
    df_f_output['YEAR'] = pd.to_numeric(df_f_output['YEAR'], errors='coerce')  # Convierte valores no válidos a NaN
    df_f_output['YEAR'] = df_f_output['YEAR'].fillna(0)  # Rellena NaN con 0 (puedes usar otro valor si prefieres)
    df_f_output['YEAR'] = df_f_output['YEAR'].astype(int)  # Convierte a int


    # df_0_output['YEAR'] = df_0_output['YEAR'].astype(int)
    # df_f_output['YEAR'] = df_f_output['YEAR'].astype(int)
    li_output = [df_0_output, df_f_output]
    #
    df_output = pd.concat(li_output, axis=0, ignore_index=True)
    df_output.sort_values(by=[
        'Strategy', 'Future.ID', 'REGION', 'COMMODITY', 'TECHNOLOGY', 'EMISSION',
        'YEAR', 'TIMESLICE', 'MODE_OF_OPERATION', 'SEASON', 'DAYTYPE',
        'DAILYTIMEBRACKET', 'STORAGE', 'STORAGEINTRADAY', 'STORAGEINTRAYEAR', 'UDC'
    ], inplace=True)
    
    ############################################################################################################
    input_dataset_0_path = dir_executables / 'input_dataset_0.csv'
    df_0_input = pd.read_csv(input_dataset_0_path, index_col=None, header=0, low_memory=False)
    df_0_input['Scen_fut'] = df_0_input['Strategy'].astype(str) + "_" + df_0_input['Future.ID'].astype(str)
    
    # input_dataset_f_path = dir_futures / 'input_dataset_f.csv'
    input_dataset_f_path = dir_futures / 'input_dataset_f.parquet'
    # df_f_input = pd.read_csv(input_dataset_f_path, index_col=None, header=0, low_memory=False)
    df_f_input = pd.read_parquet(input_dataset_f_path, engine='pyarrow')
    li_intput = [df_0_input, df_f_input]
    #
    df_input = pd.concat(li_intput, axis=0, ignore_index=True)
    df_input.sort_values(by=[
        'Strategy', 'Future.ID', 'REGION', 'COMMODITY', 'TECHNOLOGY', 'EMISSION',
        'YEAR', 'TIMESLICE', 'MODE_OF_OPERATION', 'SEASON', 'DAYTYPE',
        'DAILYTIMEBRACKET', 'STORAGE', 'STORAGEINTRADAY', 'STORAGEINTRAYEAR', 'UDC'
    ], inplace=True)

    # Reemplazar valores no finitos (NaN, inf, -inf) con un valor predeterminado
    df_output['YEAR'] = pd.to_numeric(df_output['YEAR'], errors='coerce')  # Convierte valores no válidos a NaN
    df_output['YEAR'] = df_output['YEAR'].fillna(0)  # Rellena NaN con 0 (puedes usar otro valor si prefieres)
    df_output['YEAR'] = df_output['YEAR'].astype(int)  # Convierte a int
    
    df_input['YEAR'] = pd.to_numeric(df_input['YEAR'], errors='coerce')  # Convierte valores no válidos a NaN
    df_input['YEAR'] = df_input['YEAR'].fillna(0)  # Rellena NaN con 0 (puedes usar otro valor si prefieres)
    df_input['YEAR'] = df_input['YEAR'].astype(int)  # Convierte a int

    
    # df_output['YEAR'] = df_output['YEAR'].astype(int)
    # df_input['YEAR'] = df_input['YEAR'].astype(int)
    dfa_list = [ df_output, df_input ]
    
    today = date.today()
    #
    # Check if 'Results' folder exists, and create it if not
    if not os.path.exists('Results'):
        os.makedirs('Results')
        print("Folder 'Results' created.")
    else:
        print("Folder 'Results' already exists.")
    #
    df_output = dfa_list[0]
    output_path = Path('Results') / 'OSEMOSYS_Energy_Output.csv'
    df_output.to_csv ( output_path, index = None, header=True)
    #
    df_input = dfa_list[1]
    input_path = Path('Results') / 'OSEMOSYS_Energy_Input.csv'
    df_input.to_csv ( input_path, index = None, header=True)
