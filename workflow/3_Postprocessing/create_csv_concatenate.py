# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:37:25 2024

@author: andreysava
"""

import os
from pathlib import Path
import pandas as pd
import yaml
from copy import deepcopy
import sys
import shutil
import numpy as np

def get_config_main_path(full_path, base_folder='3_Postprocessing'):
    # Split the path into parts
    parts = full_path.split(os.sep)
    
    # Find the index of the target directory 'src'
    target_index = parts.index('src') if 'src' in parts else None
    
    # If the directory is found, reconstruct the path up to that point
    if target_index is not None:
        base_path = os.sep.join(parts[:target_index + 1])
    else:
        base_path = full_path  # If not found, return the original path
    
    # Append the specified directory to the base path
    appended_path = os.path.join(base_path, base_folder) + os.sep
    
    return appended_path

def load_and_process_yaml(path):
    """
    Load a YAML file and replace the specific placeholder '${year_apply_discount_rate}' with the year specified in the file.
    
    Args:
    path (str): The path to the YAML file.
    
    Returns:
    dict: The updated data from the YAML file where the specific placeholder is replaced.
    """
    with open(path, 'r') as file:
        # Load the YAML content into 'params'
        params = yaml.safe_load(file)

    # Retrieve the reference year from the YAML file and convert it to string for replacement
    reference_year = str(params['year_apply_discount_rate'])

    # Function to recursively update strings containing the placeholder
    def update_strings(obj):
        if isinstance(obj, dict):
            return {k: update_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [update_strings(element) for element in obj]
        elif isinstance(obj, str):
            # Replace the specific placeholder with the reference year
            return obj.replace('${year_apply_discount_rate}', reference_year)
        else:
            return obj

    # Update all string values in the loaded YAML structure
    updated_params = update_strings(params)

    return updated_params


def calculate_npv(dataframe, new_column_name, parametervalue_column, rod, year_column="YEAR", output_csv_r=0, output_csv_year=0):
    # Verify that the necessary columns exist in the DataFrame
    if parametervalue_column not in dataframe.columns or year_column not in dataframe.columns:
        raise ValueError("The specified columns do not exist in the DataFrame")
    
    # Verify that the value column contains numeric data
    if not np.issubdtype(dataframe[parametervalue_column].dtype, np.number):
        raise ValueError(f"The column {parametervalue_column} does not contain numeric values")

    # Calculate the NPV and add the new column to the DataFrame
    dataframe[new_column_name] = dataframe.apply(
        lambda row: round(row[parametervalue_column] / ((1 + output_csv_r / 100) ** (float(row[year_column]) - output_csv_year)), rod),
        axis=1
    )


def calculate_npv_filtered(dataframe, new_column_name, parametervalue_column, rod, year_column="YEAR", filter_dict=None, output_csv_r=0, output_csv_year=0):
    # Initialize the new column with null values
    dataframe[new_column_name] = np.nan

    # Verify that the necessary columns exist in the DataFrame
    if parametervalue_column not in dataframe.columns or year_column not in dataframe.columns:
        raise ValueError("The specified columns do not exist in the DataFrame")
    
    # Verify that the value column contains numeric data
    if not np.issubdtype(dataframe[parametervalue_column].dtype, np.number):
        raise ValueError(f"The column {parametervalue_column} does not contain numeric values")
    
    # Verify that filter_dict has only one key and get its value
    if filter_dict and len(filter_dict) == 1:
        filter_column = list(filter_dict.keys())[0]
        filter_values = filter_dict[filter_column]
        
        # Verify that the filter column exists in the DataFrame
        if filter_column not in dataframe.columns:
            raise ValueError("The specified filter column does not exist in the DataFrame")
        
        # Iterate over the rows of the DataFrame
        for index, row in dataframe.iterrows():
            # Verify if the value in the filter column is in the filter values
            if row[filter_column] in filter_values:
                # Verify if the value in parametervalue_column is numeric
                if pd.notna(row[parametervalue_column]):
                    npv = row[parametervalue_column] / ((1 + output_csv_r / 100) ** (float(row[year_column]) - output_csv_year))
                    dataframe.at[index, new_column_name] = round(npv, rod)

def delete_files(file, solver):
    # Delete files
    if file:
        shutil.os.remove(file)
    
    if solver == 'glpk':
        shutil.os.remove(file.replace('sol', 'glp'))        
    else:
        shutil.os.remove(file.replace('sol', 'lp'))
def text_exists_in_file(filename, text):
    if not os.path.exists(filename):
        return False
    with open(filename, 'r') as file:
        content = file.read()
        return text in content

if __name__ == '__main__':    
    
    # Take the case from the model (B1 or experiment_manager)
    main_path = sys.argv
    # Take scenario and future from the main_path
    # example
    # this_case[0] = 'BAU_15.txt'
    case = main_path[1].replace('.txt', '')
    scen = main_path[1]
    scen = scen[:3]
    tier_by_path = main_path[2]
    solver = main_path[3]
    list_scenarios = main_path[4]
    
    
    # case = 'BAU_0'
    # scen = 'BAU'
    # tier_by_path = str(1)
    
    
    
    # Define option to write the file with the detail of solution of each file
    option = str()
    
    # Read yaml file with parameterization
    file_config_address = get_config_main_path(os.path.abspath(''))

    params = load_and_process_yaml(os.path.join(file_config_address,'config_concatenate.yaml'))
        
    sets_corrects = deepcopy(params['sets_otoole'])
    sets_corrects.insert(0,'Parameter')
    sets_corrects.append('VALUE')
    
    sets_csv = ['YEAR', 'TECHNOLOGY', 'FUEL', 'EMISSION']
    sets_csv_temp = deepcopy(sets_csv)
    sets_csv_temp.insert(0,'Parameter')
    sets_csv_temp.append('VALUE')
    set_no_needed = [item for item in params['sets_otoole'] if item not in sets_csv]
    
    
    
    
    if params['model']=='MOMF':
        dict_scen_folder_unique = {}
        
        # Select folder path
        tier1_dir = Path(params['tier1_dir'].replace('\\', os.sep))
        tier3a_dir = Path(params['tier3a_dir'].replace('\\', os.sep))
        outputs_dir = params['outputs'].strip('/\\')

        if tier_by_path=='1':
            tier_dir = tier1_dir / str(case) / outputs_dir
            output_filename = tier1_dir.parent / 'status_of_each_future'
        elif tier_by_path=='3a':
            tier_dir = tier3a_dir / str(scen) / str(case) / outputs_dir
            output_filename = tier3a_dir.parent / 'status_of_each_future'

        tier_dir = Path(get_config_main_path(os.path.abspath(''), str(tier_dir))).resolve()
        output_filename = Path(get_config_main_path(os.path.abspath(''), str(output_filename))).with_suffix('.txt').resolve()
        
        # Define the number of first case for tier 3a
        if params['execute_scenarios'][-1] == 'All':
            first_scen = list_scenarios[0]
            first_case_num = 1
        else:
            first_scen = params['execute_scenarios'][0]
            first_case_num = params['execute_futures'][0] -1
            

        if case == 'BAU_0' and tier_by_path=='1':
            option = 'w'
        elif tier_by_path=='3a':
            option = 'a'
        else:
            option = 'a'
            
        
        with open(output_filename, option) as file_status:
            if scen == first_scen and tier_by_path == '1':
                file_status.write('Status of solution of each future.\nWrite in order of solution.')
                file_status.write(f'\n\n\n################################# {scen} #################################\n\n\n')
            elif scen != first_scen and tier_by_path == '1':
                file_status.write(f'\n\n\n################################# {scen} #################################\n\n\n')
            elif scen != first_scen and tier_by_path == '3a':
                text_to_write = f'\n\n\n################################# {scen} #################################\n\n\n'
                if not text_exists_in_file(output_filename, text_to_write):
                    file_status.seek(0, 2)  # Move to the end of the file
                    file_status.write(text_to_write)
            
            
            out_quick = params['outputs'].strip('/\\')
            file_df_dir = Path(tier_dir).parent
        
        
            if os.path.exists(tier_dir):
                csv_file_list = os.listdir(tier_dir)
                
                # Search for the .sol file in the specified directory
                sol_file = None
                sol_folder = Path(tier_dir).parent
                for file_name in os.listdir(sol_folder):
                    if file_name.endswith('.sol'):
                        sol_file = sol_folder / file_name
                        break
                
                # If a .sol file is found, read its content
                sol_status_line = str()
                if sol_file:
                    with open(sol_file, 'r') as file:
                        # Load the content of the file
                        if solver == 'cbc':
                            sol_status_line = file.readline().strip()
                        # elif solver == 'glpk' and not params['glpk_option'] == 'old':
                        #     for current_line_number in range(5):
                        #         line = file.readline().strip()
                        #         if current_line_number == 5 - 1:
                        #             # Remove the last character from the specified line
                        #             sol_status_line = line[14:]
                        elif solver == 'cplex':
                            for current_line_number in range(9):
                                line = file.readline().strip()
                                if current_line_number == 9 - 1:
                                    # Remove the last character from the specified line
                                    sol_status_line = line[22:29]
                
                if (solver and sol_status_line[0:7] == 'Optimal') or (solver == 'glpk' and sol_status_line == 'OPTIMAL') or (solver == 'cplex' and sol_status_line == 'optimal'):
                    file_status.write(f'\n{case}: Optimal solution.')
                
                    df_list = []
                    
                    parameter_list = []
                    parameter_dict = {}
                    
                    if params['vis_dir'] in csv_file_list:
                        csv_file_list.remove(f'{params["vis_dir"]}')
                    
                    for f in csv_file_list:
                        
                        local_df = pd.read_csv(os.path.join(tier_dir,f))
                        
                        
                        # Delete columns of sets do not use in otoole config yaml
                        columns_check = [column for column in local_df.columns if column in sets_corrects]
                        local_df = local_df[columns_check]
    
                        
                        local_df['Parameter'] = f.split('.')[0]
                        parameter_list.append(f.split('.')[0])
                        parameter_dict.update({parameter_list[-1]: local_df})
                        
                        df_list.append(local_df)
                    columns_check.insert(0,'Parameter')
                    df_all = pd.concat(df_list, ignore_index=True, sort=False)
                    df_all = df_all[ sets_csv_temp ]
                    file_df_dir = Path(params["excel_data_file_dir"].replace('..'+os.sep, '').replace('\\', os.sep))
                    
                    # df_all.to_csv(f'Data_plots_{case}.csv')
                    
                    # 3rd try
                    # Assuming parameter_list and parameter_dict are defined
                    # Initialize df_all_2 with the first DataFrame to ensure the dimension columns are set
                    first_param = parameter_list[0]
                    df_all_3 = pd.DataFrame()
                    df_all_3 = df_all[df_all['Parameter'] == first_param]
                    df_all_3 = df_all_3.rename(columns={'VALUE': first_param})
                    df_all_3 = df_all_3.drop('Parameter', axis=1)
                    df_all_3 = df_all_3.drop(columns=[col for col in df_all_3.columns if col in set_no_needed], errors='ignore')
                    df_all_3 = df_all_3.assign(**{col: 'nan' for col in sets_csv if col not in df_all_3.columns})
    
                    
                    
                    # Iterate over the remaining parameters and merge their respective DataFrames on the dimension columns
                    for p in parameter_list[1:]:  # Skip the first parameter since it's already added
                        local_df_3 = df_all[df_all['Parameter'] == p]
                        local_df_3 = local_df_3.rename(columns={'VALUE': p})
                        local_df_3 = local_df_3.drop('Parameter', axis=1)
                        local_df_3 = local_df_3.drop(columns=[col for col in local_df_3.columns if col in set_no_needed], errors='ignore')
                        local_df_3 = local_df_3.assign(**{col: 'nan' for col in sets_csv if col not in local_df_3.columns})
    
                        df_all_3 = pd.merge(df_all_3, local_df_3, on=sets_csv, how='outer')
                    

                    # Add NPV columns
                    parameters_reference = params['parameters_reference']
                    parameters_news = params['parameters_news']
                    
                    # for k in range(len(parameters_reference)):
                    #     if parameters_reference[k] == 'AnnualTechnologyEmissionPenaltyByEmission':
                    #         parameter_filter = {'EMISSION':params['this_combina']}
                    #         calculate_npv_filtered(df_all_3, parameters_news[k], parameters_reference[k], params['round_#'], 'YEAR', parameter_filter, output_csv_r=params['disc_rate']*100, output_csv_year=params['year_apply_discount_rate'])
                    #     else:
                    #         calculate_npv(df_all_3, parameters_news[k], parameters_reference[k], params['round_#'], 'YEAR', output_csv_r=params['disc_rate']*100, output_csv_year=params['year_apply_discount_rate'])
                                        
                    
                    # The 'outer' join ensures that all combinations of dimension values are included, filling missing values with NaN
                    # df_all_3.to_csv(f'{file_df_dir}/Data_Output_{case[-1]}.csv')

                    df_all_3.to_csv(str(Path(sol_folder) / f"{case}_Output.csv"))
    
                    # Delete Outputs folder with otoole csvs files
                    if params['del_files']:
                        outputs_otoole_csvs = Path(sol_folder) / out_quick
                        if os.path.exists(outputs_otoole_csvs):
                            shutil.rmtree(outputs_otoole_csvs)
                    
                        # Delete glp, lp, txt and sol files
                        if params['del_files'] and (solver == 'cplex' or solver == 'cbc'):
                            delete_files(sol_file, params['solver'])

                else:
                    # Delete Outputs folder with otoole csvs files
                    if params['del_files']:
                        outputs_otoole_csvs = file_df_dir / out_quick
                        if os.path.exists(outputs_otoole_csvs):
                            shutil.rmtree(outputs_otoole_csvs)
                    
                        # Delete glp, lp, txt and sol files
                        if params['del_files'] and (solver == 'cplex' or solver == 'cbc'):
                            delete_files(sol_file, solver)
                    file_status.write(f'\n{case}: Infeasible solution.')