import os
import sys

import pandas as pd

# Get the directory of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))
dir_executables = os.path.join(current_script_path, 'Executables')

sys.path.insert(0, dir_executables)
import local_dataset_creator_0

dir_futures = os.path.join(current_script_path, 'Experimental_Platform', 'Futures')
sys.path.insert(0, dir_futures)
import local_dataset_creator_f


def coerce_year_to_int(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure YEAR is an integer column, coercing invalid values to zero."""
    dataframe['YEAR'] = pd.to_numeric(dataframe['YEAR'], errors='coerce')
    dataframe['YEAR'] = dataframe['YEAR'].fillna(0).astype(int)
    return dataframe


if __name__ == '__main__':
    run_for_first_time = True
    
    if run_for_first_time:
        local_dataset_creator_0.execute_local_dataset_creator_0_outputs(dir_executables)
        local_dataset_creator_f.execute_local_dataset_creator_f_outputs(dir_futures)
        local_dataset_creator_0.execute_local_dataset_creator_0_inputs(dir_executables)
        local_dataset_creator_f.execute_local_dataset_creator_f_inputs(dir_futures)
    
    ############################################################################################################
    output_dataset_0_path = os.path.join(dir_executables, 'output_dataset_0.csv')
    df_0_output = pd.read_csv(output_dataset_0_path, index_col=None, header=0, low_memory=False)
    df_0_output['Scen_fut'] = (
        df_0_output['Strategy'].astype(str) + "_" + df_0_output['Future.ID'].astype(str)
    )
    
    output_dataset_f_path = os.path.join(dir_futures, 'output_dataset_f.parquet')
    df_f_output = pd.read_parquet(output_dataset_f_path, engine='pyarrow')

    df_0_output = coerce_year_to_int(df_0_output)
    df_f_output = coerce_year_to_int(df_f_output)
    output_frames = [df_0_output, df_f_output]
    df_output = pd.concat(output_frames, axis=0, ignore_index=True)
    df_output.sort_values(by=[
        'Strategy', 'Future.ID', 'REGION', 'COMMODITY', 'TECHNOLOGY', 'EMISSION',
        'YEAR', 'TIMESLICE', 'MODE_OF_OPERATION', 'SEASON', 'DAYTYPE',
        'DAILYTIMEBRACKET', 'STORAGE', 'STORAGEINTRADAY', 'STORAGEINTRAYEAR', 'UDC'
    ], inplace=True)
    
    ############################################################################################################
    input_dataset_0_path = os.path.join(dir_executables, 'input_dataset_0.csv')
    df_0_input = pd.read_csv(input_dataset_0_path, index_col=None, header=0, low_memory=False)
    df_0_input['Scen_fut'] = (
        df_0_input['Strategy'].astype(str) + "_" + df_0_input['Future.ID'].astype(str)
    )
    
    input_dataset_f_path = os.path.join(dir_futures, 'input_dataset_f.parquet')
    df_f_input = pd.read_parquet(input_dataset_f_path, engine='pyarrow')
    input_frames = [df_0_input, df_f_input]
    df_input = pd.concat(input_frames, axis=0, ignore_index=True)
    df_input.sort_values(by=[
        'Strategy', 'Future.ID', 'REGION', 'COMMODITY', 'TECHNOLOGY', 'EMISSION',
        'YEAR', 'TIMESLICE', 'MODE_OF_OPERATION', 'SEASON', 'DAYTYPE',
        'DAILYTIMEBRACKET', 'STORAGE', 'STORAGEINTRADAY', 'STORAGEINTRAYEAR', 'UDC'
    ], inplace=True)

    df_output = coerce_year_to_int(df_output)
    df_input = coerce_year_to_int(df_input)

    #
    if not os.path.exists('Results'):
        os.makedirs('Results')
        print("Folder 'Results' created.")
    else:
        print("Folder 'Results' already exists.")
    #
    output_path = os.path.join('Results', 'OSEMOSYS_Energy_Output.csv')
    df_output.to_csv(output_path, index=None, header=True)
    # Also write under 1_Experiment/Results for convenience
    exp_results_dir = os.path.join('workflow', '1_Experiment', 'Results')
    if not os.path.exists(exp_results_dir):
        os.makedirs(exp_results_dir)
    df_output.to_csv(
        os.path.join(exp_results_dir, 'OSEMOSYS_Energy_Output.csv'),
        index=None,
        header=True,
    )
    #
    input_path = os.path.join('Results', 'OSEMOSYS_Energy_Input.csv')
    df_input.to_csv(input_path, index=None, header=True)
    # Mirror input under 1_Experiment/Results as well
    df_input.to_csv(
        os.path.join(exp_results_dir, 'OSEMOSYS_Energy_Input.csv'),
        index=None,
        header=True,
    )
