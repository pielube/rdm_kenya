' Libraries'
import os
import shutil
import sys
import time
from pathlib import Path
import pandas as pd

'Auxiliar code'
from workflow import z_auxiliar_code as AUX

start = time.time()

'Read names of scenarios'

list_scenarios=os.listdir('./workflow/0_Scenarios')

'Read timeslices numbers'
book=pd.ExcelFile('Interface_RDM.xlsx')
setup_table = book.parse( 'Setup' , 0)
num_time_slices_SDP = int( setup_table.loc[ 0 ,'Timeslices_model'] )

'If you want to run base future the next variable as "Yes", if do not put "No"'
run_base_future = str( setup_table.loc[ 0 ,'Run_Base_Future'] )

'If you want to run RDM experiment the next variable as "Yes", if do not put "No"'
run_RDM = str( setup_table.loc[ 0 ,'Run_RDM'] )

'Solver use to solve the problem'
solver = str( setup_table.loc[ 0 ,'Solver'] )

'Name of the OSeMOSYS model use to solve problem'
osemosys_model = str( setup_table.loc[ 0 ,'OSeMOSYS_Model_Name'] )

'Parameters to print'
parameters_to_print = book.parse( 'To_Print' , 0)

if run_base_future == 'Yes':
    # ' Step 1: Delete ResultsPath param'
    
    for i in range(len(list_scenarios)):
        lines = []
        scen_path = Path('workflow/0_Scenarios') / list_scenarios[i]
        # read file
        with open(scen_path, 'r') as fp:
            # read an store all lines into list
            lines = fp.readlines()
        # Write file
        with open(scen_path, 'w') as fp:
            # iterate each line
            for number, line in enumerate(lines):
                if 'ResultsPath' not in line:
                    fp.write(line)
                    
    print('Step 1 finished')
    
    
    ' Step 2: Clean folders in ./workflow/1_Experiment/0_From_Confection/'
    
    dir = './workflow/1_Experiment/0_From_Confection/'
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    
    print('Step 2 finished')
    
    
    ' Step 3: Obtain ModelStructure & DefaultParams'
    """Note: all scenarios must have the same sets. We only use one of the scenario
    files to obtain the model structure and default parameters"""
    
    dic_sets = AUX.obtain_structure_file(
        Path('workflow/0_Scenarios') / list_scenarios[0],
        Path('workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx'),
        Path('workflow/2_Miscellaneous/OSeMOSYS_Structure.xlsx'),
        num_time_slices_SDP
    )
    
    print('Step 3 finished')
    
    ' Step 4: Clean ./workflow/1_Experiment/Executables folder except .py file'
    
    # Clean folders
    dir = Path('workflow/1_Experiment/Executables')
    for files in os.listdir(dir):
        path = dir / files
        if '.py' not in files:
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    
    print('Step 4 finished')
    
    ' Step 5: Clean ./workflow/1_Experiment/Experimental_Platform/Futures folder except .py file'
    
    # Clean folders
    dir = Path('workflow/1_Experiment/Experimental_Platform/Futures')
    for files in os.listdir(dir):
        path = dir / files
        if '.py' not in files:
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
                
    print('Step 5 finished')
    
    ''' Step 6: Create folders in ./workflow/1_Experiment/Executables and ./workflow/1_Experiment/Experimental_Platform/Futures folders 
    to stores future 0 and multiple futures respectively'''
    
    # Create a folder for each scenario
    for i in range(len(list_scenarios)):
        newpath = Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0"
        newpath.mkdir(parents=True, exist_ok=True)

    # Create a folder for each scenario
    for i in range(len(list_scenarios)):
        newpath = Path('workflow/1_Experiment/Experimental_Platform/Futures') / list_scenarios[i].replace('.txt','')
        newpath.mkdir(parents=True, exist_ok=True)
    
    print('Step 6 finished')
    
    ' Step 7: Paste Scenarios future 0 TXT files in ./workflow/1_Experiment/Executables/'
    
    for i in range(len(list_scenarios)):
        source_folder = Path('workflow/0_Scenarios')
        destination_folder = Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0"
        # construct full file path
        source = source_folder / list_scenarios[i]
        destination = destination_folder / f"{list_scenarios[i].replace('.txt','')}_0.txt"
        # copy files and write with timeslices quantity of the RDM_Interface.xlsx
        AUX.process_timeslices(source, num_time_slices_SDP, destination)
    
            
    print('Step 7 finished')
    
    ' Step 8: Store data with scenarios data'
    
    # Store data from executable file
    for i in range(len(list_scenarios)):
                
        # Isolate params in subfiles
        data_per_param, special_sets = AUX.isolate_params(Path('workflow/0_Scenarios') / list_scenarios[i])
        
        
        
        # Generate CSV parameters files for each scenario
        list_dataframes, dict_dataframes, parameters_without_values = AUX.generate_df_per_param(list_scenarios[i].replace('.txt',''), 
                                            data_per_param,
                                            num_time_slices_SDP)
        
        # Create future 0 input dataset'
        AUX.create_input_dataset_future_0(
            list_dataframes,
            list_scenarios[i].replace('.txt',''),
            Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/"
        )
        
    print('Step 8 finished')
    

    'Step 9: Create future 0 output dataset'
    
    # Output
    start1 = time.time()
    for i in range(len(list_scenarios)):
        # Run OSeMOSYS for each scenario
        AUX.run_osemosys(
            solver,
            Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/",
            Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/{list_scenarios[i].replace('.txt','')}_0.txt",
            Path('workflow') / osemosys_model,
            Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/{list_scenarios[i].replace('.txt','')}"
        )
        
        print('Step 9.Input finished')
        
        print('Step 9.Output generated. Star long function')
        # if solver == 'glpk':
        #     first_list=[list_scenarios[i].replace('.txt','_0')]
        #     time_range_vector=list()
        #     for j in range(2015,2071):
        #         time_range_vector.append(j)
            
            
        #    AUX.create_output_dataset_future_0(0, time_range_vector, first_list,'./workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx')
        if solver == 'cbc' or solver == 'cplex':
            AUX.data_processor_new(
                Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/{list_scenarios[i].replace('.txt','')}_0_Output.sol",
                Path('workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx'),
                list_scenarios[i].replace('.txt',''),
                str(0),
                solver,
                parameters_to_print,
                'csv'
            )
        elif solver == 'glpk':
            AUX.data_processor_new(
                Path('workflow/1_Experiment/Executables') / f"{list_scenarios[i].replace('.txt','')}_0/{list_scenarios[i].replace('.txt','')}_0_Output.txt",
                Path('workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx'),
                list_scenarios[i].replace('.txt',''),
                str(0),
                solver,
                parameters_to_print,
                'csv'
            )
        print('Step 9.Output finished')    
    print('Step 9 finished')
    end_1 = time.time()
    time_elapsed_1 = -start1 + end_1
    print('   The total time producing outputs and storing data of base futures have been: ' + str( time_elapsed_1 ) + ' seconds' )

if run_RDM == 'Yes':
    'Step 10: Execute RDM experiment'
    print('Start RDM experiment\n')
    AUX.run_scripts(
        str(Path('workflow/1_Experiment/0_experiment_manager.py')),
        solver,
        osemosys_model,
        str(Path('Interface_RDM.xlsx').resolve()),
        shape_file=str(Path('workflow/2_Miscellaneous/shape_of_demand.csv').resolve())
    )
    
    print('Step 10 finished\n')
    
    
    'Step 11: Execute RDM experiment'
    start3 = time.time()
    print('Start Output Dataset Creator\n')
    AUX.run_scripts(str(Path('workflow/1_Experiment/1_output_dataset_creator.py')))
    
    print('Step 11 finished\n')
    end_3 = time.time()
    time_elapsed_3 = -start3 + end_3
    print('   The total time producing storing data of the experiment has been: ' + str( time_elapsed_3 ) + ' seconds' )
    
if solver == 'cplex':
    for file in ['cplex.log', 'clone1.log', 'clone2.log']:
        if os.path.exists(file):
            os.remove(file)

print('#####################################')
print('Processing completed successfully.')
print('#####################################')

end = time.time()
time_elapsed = -start + end
print('   The total time of the workflow has been: ' + str( time_elapsed ) + ' seconds' )

