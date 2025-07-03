import os
from pathlib import Path
import re
import csv
import pandas as pd
import pyarrow
#
def test1():
    print( 'hello world 2' )
############################################################################################################################
def execute_local_dataset_creator_f_outputs(file_adress):
    # List scenarios in the directory, excluding unnecessary files
    file_adress = Path(file_adress)
    scenario_list_raw = os.listdir(file_adress)
    scenario_list = [
        e for e in scenario_list_raw 
        if ('.py' not in e) and ('.csv' not in e) and ('.parquet' not in e) and ('__pycache__' not in e)
    ]
    
    li = []  # List to store DataFrames
    
    for s in range(len(scenario_list)):
        # Construct the path for each scenario
        case_list_raw_path = file_adress / scenario_list[s]
        # List cases in the scenario directory, excluding unnecessary files
        case_list_raw = os.listdir(case_list_raw_path)
        case_list = [
            e for e in case_list_raw 
            if ('.py' not in e) and ('.csv' not in e) and ('.parquet' not in e) and ('__pycache__' not in e)
        ]
        
        for n in range(len(case_list)):
            # Construct the Parquet file path
            filename = file_adress / scenario_list[s] / case_list[n] / f"{case_list[n]}_Output.parquet"
            
            if os.path.exists(filename):
                # print(f"Reading file: {filename}")
                
                # Read the Parquet file
                df = pd.read_parquet(filename, engine='pyarrow')
                
                # Add a column to provide context
                df['Scen_fut'] = case_list[n]
                
                # Append the DataFrame to the list
                li.append(df)
    
    # Concatenate all DataFrames
    frame = pd.concat(li, axis=0, ignore_index=True)
    
    # Export the concatenated DataFrame as a Parquet file
    parquet_path = file_adress / 'output_dataset_f.parquet'
    frame.to_parquet(parquet_path, engine='pyarrow', index=False)
    # print(f"File saved at: {parquet_path}")



############################################################################################################################
def execute_local_dataset_creator_f_inputs(file_adress):
    # List scenarios in the directory, excluding unnecessary files
    file_adress = Path(file_adress)
    scenario_list_raw = os.listdir(file_adress)
    scenario_list = [
        e for e in scenario_list_raw 
        if ('.py' not in e) and ('.csv' not in e) and ('.parquet' not in e) and ('__pycache__' not in e)
    ]
    
    li = []  # List to store DataFrames
    
    for scenario in scenario_list:
        # List cases in the scenario directory
        case_list_raw_path = file_adress / scenario
        case_list_raw = os.listdir(case_list_raw_path)
        case_list = [
            e for e in case_list_raw 
            if ('.py' not in e) and ('.csv' not in e) and ('.parquet' not in e) and ('__pycache__' not in e)
        ]
        
        for case in case_list:
            # Construct the Parquet file path
            filename = file_adress / scenario / case / f"{case}_Input.parquet"
            
            if os.path.exists(filename):
                # print(f"Reading file: {filename}")
                
                # Read the Parquet file
                df = pd.read_parquet(filename, engine='pyarrow')
                
                # Add a column to provide context
                df['Scen_fut'] = case
                
                # Append the DataFrame to the list
                li.append(df)
    
    # Concatenate all DataFrames
    frame = pd.concat(li, axis=0, ignore_index=True)
    
    # Export the concatenated DataFrame as a Parquet file
    parquet_path = file_adress / 'input_dataset_f.parquet'
    frame.to_parquet(parquet_path, engine='pyarrow', index=False)
    # print(f"File saved at: {parquet_path}")

############################################################################################################################
def execute_local_dataset_creator_f_prices(file_adress):
    # file_aboslute_address = os.path.abspath("local_dataset_creator_f.py")
    # file_adress = re.escape( file_aboslute_address.replace( 'local_dataset_creator_f.py', '' ) ).replace( '\:', ':' )
    # file_adress += 'Experimental_Platform\\Futures\\'
    #
    file_adress = Path(file_adress)
    scenario_list_raw = os.listdir(file_adress)
    scenario_list = [e for e in scenario_list_raw if ('.py' not in e ) and ('.csv' not in e ) and ('__pycache__' not in e) ]
    #
    li = []
    #
    for s in range( len( scenario_list ) ):
        #
        case_list_raw_path = file_adress / scenario_list[s]
        case_list_raw = os.listdir( case_list_raw_path )
        case_list = [e for e in case_list_raw if ('.py' not in e ) and ('.csv' not in e ) and ('__pycache__' not in e) ]
        #
        for n in range( len( case_list ) ):
            #
            case_path = file_adress / scenario_list[s] / case_list[n]
            x = os.listdir( case_path  )
            #
            if len(x) == 5:
                #
                filename = file_adress / scenario_list[n] / case_list[n] / f"{case_list[n]}_Prices.csv"
                #
                line_count = 0
                with open( filename ) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        line_count += 1
                if line_count > 1:
                    df = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
                    li.append(df)
                else:
                    pass
            #
            else:
                print(case_list[n])
    print('###')
    #
    frame = pd.concat(li, axis=0, ignore_index=True)
    csv_path = file_adress / 'prices_dataset_f.csv'
    export_csv = frame.to_csv ( csv_path, index = None, header=True)
############################################################################################################################
def execute_local_dataset_creator_f_distribution(file_adress):
    # file_aboslute_address = os.path.abspath("local_dataset_creator_f.py")
    # file_adress = re.escape( file_aboslute_address.replace( 'local_dataset_creator_f.py', '' ) ).replace( '\:', ':' )
    # file_adress += 'Experimental_Platform\\Futures\\'
    #
    file_adress = Path(file_adress)
    scenario_list_raw = os.listdir(file_adress)
    scenario_list = [e for e in scenario_list_raw if ('.py' not in e ) and ('.csv' not in e ) and ('__pycache__' not in e) ]
    #
    li = []
    #
    for s in range( len( scenario_list ) ):
        #
        case_list_raw_path = file_adress / scenario_list[s]
        case_list_raw = os.listdir( case_list_raw_path )
        case_list = [e for e in case_list_raw if ('.py' not in e ) and ('.csv' not in e ) and ('__pycache__' not in e) ]
        #
        for n in range( len( case_list ) ):
            #
            case_path = file_adress / scenario_list[s] / case_list[n]
            x = os.listdir( case_path  )
            #
            if len(x) == 5:
                #
                filename = file_adress / scenario_list[n] / case_list[n] / f"{case_list[n]}_Distribution.csv"
                #
                line_count = 0
                with open( filename ) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        line_count += 1
                if line_count > 1:
                    df = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
                    li.append(df)
                else:
                    pass
            #
            else:
                print(case_list[n])
    print('###')
    #
    frame = pd.concat(li, axis=0, ignore_index=True)
    csv_path = file_adress / 'distribution_dataset_f.csv'
    export_csv = frame.to_csv(csv_path, index=None, header=True)
############################################################################################################################