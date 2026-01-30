# -*- coding: utf-8 -*-
"""Experiment manager for the OSeMOSYS workflow.

Author: luisf
"""

import errno
import math
import os
import pickle
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy
from pyDOE import lhs  # SOURCE: https://pypi.org/project/lhsmdu/.

# Save a copy of the original sys.path
original_sys_path = sys.path.copy()

# Get the path of the "workflow" folder
workflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Temporarily add "workflow" to sys.path
sys.path.append(workflow_path)

try:
    # Import the module
    import z_auxiliar_code as AUX
finally:
    # Restore the original sys.path
    sys.path = original_sys_path

"""
We implement OSeMOSYS in a procedural code.
The main features are:
- inherited_scenarios: implemented in procedural code
- function_C_mathprog_parallel: runs the solver in parallel
- interpolation: implemented in helper functions for linear or non-linear time series
"""


def set_first_list(Executed_Scenario):
    """Populate the global list of future folders for a scenario."""
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    futures_dir = os.path.join(
        current_script_path, "Experimental_Platform", "Futures", Executed_Scenario
    )
    first_list_raw = os.listdir(futures_dir)

    global first_list
    first_list = [
        entry
        for entry in first_list_raw
        if ".csv" not in entry
        and "Table" not in entry
        and ".py" not in entry
        and "__pycache__" not in entry
    ]


def main_executer(
    n1, Executed_Scenario, time_vector, scenario_list, solver, osemosys_model, parameters_to_print
):
    """Run a single model execution and post-process output for a future."""
    print("# " + str(n1 + 1) + " of " + Executed_Scenario)
    set_first_list(Executed_Scenario)
    file_address = os.path.dirname(os.path.abspath(__file__))

    case_address = os.path.join(
        file_address, "Experimental_Platform", "Futures", Executed_Scenario, first_list[n1]
    )
    str_scen_fut = first_list[n1].split("_")
    str_scen, str_fut = str_scen_fut[0], str_scen_fut[-1]

    if str_scen in scenario_list:
        this_case = [entry for entry in os.listdir(case_address) if ".txt" in entry]

        str_start = "start /B start cmd.exe @cmd /k cd " + file_address

        data_file = case_address.replace("./", "").replace("/", "\\") + "\\" + str(
            this_case[0]
        )
        output_file = case_address.replace("./", "").replace("/", "\\") + "\\" + str(
            this_case[0]
        ).replace(".txt", "") + "_Output"

        model_file = os.path.join(file_address.replace("1_Experiment", ""), osemosys_model)

        if solver == "glpk":
            str_solve = (
                "glpsol -m "
                + str(model_file)
                + " -d "
                + str(data_file)
                + " -o "
                + str(output_file)
                + ".txt"
            )
        else:
            str_matrix = (
                "glpsol -m "
                + str(model_file)
                + " -d "
                + str(data_file)
                + " --wlp "
                + str(output_file)
                + ".lp --check"
            )
            os.system(str_start and str_matrix)

            if solver == "cbc":
                str_solve = (
                    "cbc " + str(output_file) + ".lp solve -solu " + str(output_file) + ".sol"
                )

            elif solver == "cplex":
                if os.path.exists(output_file + ".sol"):
                    shutil.os.remove(output_file + ".sol")
                str_solve = (
                    'cplex -c "read '
                    + str(output_file)
                    + '.lp" "set threads 2" "optimize" "write '
                    + str(output_file)
                    + '.sol"'
                )
        os.system(str_start and str_solve)
        time.sleep(1)

        if solver in {"cbc", "cplex"}:
            AUX.data_processor_new(
                output_file + ".sol",
                "./workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx",
                str_scen,
                str_fut,
                solver,
                parameters_to_print,
                "parquet",
            )
        elif solver == "glpk":
            AUX.data_processor_new(
                output_file + ".txt",
                "./workflow/1_Experiment/0_From_Confection/B1_Model_Structure.xlsx",
                str_scen,
                str_fut,
                solver,
                parameters_to_print,
                "parquet",
            )
    else:
        print("!!! At execution, we skip: future ", str_fut, " and scenario ", str_scen, " !!!")
def function_C_mathprog_parallel(
    fut_index, scen, inherited_scenarios, unpackaged_useful_elements, num_time_slices_SDP
):
    """Write OSeMOSYS input files and input parquet data for a scenario future."""
    scenario_list = unpackaged_useful_elements[0]
    S_DICT_sets_structure = unpackaged_useful_elements[1]
    S_DICT_params_structure = unpackaged_useful_elements[2]
    list_param_default_value = unpackaged_useful_elements[3]
    print_address = unpackaged_useful_elements[4]
    all_futures = unpackaged_useful_elements[5]
    parameters_in_the_model = unpackaged_useful_elements[7]
    parameters_without_values = unpackaged_useful_elements[8]
    special_sets = unpackaged_useful_elements[9]

    list_param_default_value_params = list(list_param_default_value["Parameter"])
    list_param_default_value_value = list(list_param_default_value["Default_Value"])

    header_indices = [
        "Scenario",
        "Parameter",
        "r",
        "t",
        "f",
        "e",
        "m",
        "l",
        "y",
        "ls",
        "ld",
        "lh",
        "s",
        "sd",
        "sy",
        "value",
    ]

    fut = all_futures[fut_index - scen * len(all_futures)]

    print("# This is future:", fut, " and scenario ", scenario_list[scen])

    try:
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        scen_file_dir = os.path.join(
            current_script_path,
            print_address,
            str(scenario_list[scen]),
            f"{scenario_list[scen]}_{fut}",
        )
        os.mkdir(scen_file_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    this_scenario_data = inherited_scenarios[scenario_list[scen]][fut]

    g_path = os.path.join(
        current_script_path,
        print_address,
        str(scenario_list[scen]),
        f"{scenario_list[scen]}_{fut}",
        f"{scenario_list[scen]}_{fut}.txt",
    )
    g = open(g_path, "w+")
    g.write("###############\n#    Sets     #\n###############\n#\n")
    for n1 in range( len( S_DICT_sets_structure['set'] ) ):
        if S_DICT_sets_structure['number_of_elements'][n1] != 0:
            g.write( 'set ' + S_DICT_sets_structure['set'][n1] + ' := ' )
            #
            for n2 in range( S_DICT_sets_structure['number_of_elements'][n1] ):
                if S_DICT_sets_structure['set'][n1] == 'YEAR' or S_DICT_sets_structure['set'][n1] == 'MODE_OF_OPERATION':
                    g.write( str( int( S_DICT_sets_structure['elements_list'][n1][n2] ) ) + ' ' )
                else:
                    g.write( str( S_DICT_sets_structure['elements_list'][n1][n2] ) + ' ' )
            g.write( ';\n' )
    #
    g.write( '\n' )
    g.write( '###############\n#    Parameters     #\n###############\n#\n' )
    #
    for p in range( len( list( this_scenario_data.keys() ) ) ):
        #
        this_param = list( this_scenario_data.keys() )[p]
        #
        default_value_list_params_index = list_param_default_value_params.index( this_param )
        default_value = float( list_param_default_value_value[ default_value_list_params_index ] )
        #
        this_param_index = S_DICT_params_structure['parameter'].index( this_param )
        this_param_keys = S_DICT_params_structure['index_list'][this_param_index]
        #
        if len( this_scenario_data[ this_param ]['value'] ) != 0:
            #
            if len(this_param_keys) != 2:
                g.write( 'param ' + this_param + ' default ' + str( default_value ) + ' :=\n' )
            else:
                g.write( 'param ' + this_param + ' default ' + str( default_value ) + ' :\n' )
            #
            #-----------------------------------------#
            if len(this_param_keys) == 1: #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                second_last_set_element = this_scenario_data[ this_param ][ this_param_keys[0] ] # header_indices.index( this_param_keys[-2] ) ]
                second_last_set_element_unique = [] # list( set( second_last_set_element ) )
                for u in range( len( second_last_set_element ) ):
                    if second_last_set_element[u] not in second_last_set_element_unique:
                        second_last_set_element_unique.append( second_last_set_element[u] )

                #
                for s in range( len( second_last_set_element_unique ) ):
                    g.write( second_last_set_element_unique[s] + ' ' )
                    value_indices = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[0] ] ) if x == str( second_last_set_element_unique[s] ) ]
                    these_values = []
                    for val in range( len( value_indices ) ):
                        these_values.append( this_scenario_data[ this_param ]['value'][ value_indices[val] ] )
                    for val in range( len( these_values ) ):
                        g.write( str( these_values[val] ) + ' ' )
                    g.write('\n') #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #-----------------------------------------#
            elif len(this_param_keys) == 2: #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # get the last and second last parameters of the list:
                last_set_element = this_scenario_data[ this_param ][ this_param_keys[-1] ] # header_indices.index( this_param_keys[-1] ) ]
                last_set_element_unique = [] # list( set( last_set_element ) )
                for u in range( len( last_set_element ) ):
                    if last_set_element[u] not in last_set_element_unique:
                        last_set_element_unique.append( last_set_element[u] )
                #
                for y in range( len( last_set_element_unique ) ):
                    g.write( str( last_set_element_unique[y] ) + ' ')
                g.write(':=\n')
                #
                second_last_set_element = this_scenario_data[ this_param ][ this_param_keys[-2] ] # header_indices.index( this_param_keys[-2] ) ]
                second_last_set_element_unique = [] # list( set( second_last_set_element ) )
                for u in range( len( second_last_set_element ) ):
                    if second_last_set_element[u] not in second_last_set_element_unique:
                        second_last_set_element_unique.append( second_last_set_element[u] )
                if this_param_keys[-2] == 'l':
                    second_last_set_element_unique_temp = second_last_set_element_unique
                    second_last_set_element_unique = []
                    for sdp in range(num_time_slices_SDP):
                        second_last_set_element_unique.append(second_last_set_element_unique_temp[sdp])
                #
                for s in range( len( second_last_set_element_unique ) ):
                    g.write( second_last_set_element_unique[s] + ' ' )
                    value_indices = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[-2] ] ) if x == str( second_last_set_element_unique[s] ) ]
                    these_values = []
                    for val in range( len( value_indices ) ):
                        these_values.append( this_scenario_data[ this_param ]['value'][ value_indices[val] ] )
                    for val in range( len( these_values ) ):
                        g.write( str( these_values[val] ) + ' ' )
                    g.write('\n') #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #%%%
            elif len(this_param_keys) == 3:
                this_set_element_unique_all = []
                for pkey in range( len(this_param_keys)-2 ):
                    for i in range( 2, len(header_indices)-1 ):
                        if header_indices[i] == this_param_keys[pkey]:
                            this_set_element = this_scenario_data[ this_param ][ header_indices[i] ]
                    this_set_element_unique_all.append( list( set( this_set_element ) ) )
                #
                this_set_element_unique_1 = deepcopy( this_set_element_unique_all[0] )
                #
                for n1 in range( len( this_set_element_unique_1 ) ): #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    g.write( '[' + str( this_set_element_unique_1[n1] ) + ',*,*]:\n' )
                    # get the last and second last parameters of the list:
                    last_set_element = this_scenario_data[ this_param ][ this_param_keys[-1] ] # header_indices.index( this_param_keys[-1] ) ]
                    last_set_element_unique = [] # list( set( last_set_element ) )
                    for u in range( len( last_set_element ) ):
                        if last_set_element[u] not in last_set_element_unique:
                            last_set_element_unique.append( last_set_element[u] )
                    #
                    for y in range( len( last_set_element_unique ) ):
                        g.write( str( last_set_element_unique[y] ) + ' ')
                    g.write(':=\n')
                    #
                    second_last_set_element = this_scenario_data[ this_param ][ this_param_keys[-2] ] #header_indices.index( this_param_keys[-2] ) ]
                    second_last_set_element_unique = [] # list( set( second_last_set_element ) )
                    for u in range( len( second_last_set_element ) ):
                        if second_last_set_element[u] not in second_last_set_element_unique:
                            second_last_set_element_unique.append( second_last_set_element[u] )
                    #
                    for s in range( len( second_last_set_element_unique ) ):
                        g.write( second_last_set_element_unique[s] + ' ' )
                        #
                        value_indices_s = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[-2] ] ) if x == str( second_last_set_element_unique[s] ) ]
                        value_indices_n1 = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[0] ] ) if x == str( this_set_element_unique_1[n1] ) ]
                        #
                        r_index = set(value_indices_s) & set(value_indices_n1)
                        #
                        value_indices = list( r_index )
                        value_indices.sort()
                        #
                        these_values = []
                        for val in range( len( value_indices ) ):
                            these_values.append( this_scenario_data[ this_param ]['value'][ value_indices[val] ] )
                        for val in range( len( these_values ) ):
                            g.write( str( these_values[val] ) + ' ' )
                        g.write('\n') #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            #%%%
            elif len(this_param_keys) == 4:
                this_set_element_unique_all = []
                for pkey in range( len(this_param_keys)-2 ):
                    for i in range( 2, len(header_indices)-1 ):
                        if header_indices[i] == this_param_keys[pkey]:
                            this_set_element = this_scenario_data[ this_param ][ header_indices[i] ]
                            this_set_element_unique_all.append( list( set( this_set_element ) ) )
                #
                this_set_element_unique_1 = deepcopy( this_set_element_unique_all[0] )
                this_set_element_unique_2 = deepcopy( this_set_element_unique_all[1] )
                this_set_element_unique_2.sort()
                #
                for n1 in range( len( this_set_element_unique_1 ) ):
                    count_storage = 0
                    for n2 in range( len( this_set_element_unique_2 ) ): #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                        g.write( '[' + str( this_set_element_unique_1[n1] ) + ',' + str( this_set_element_unique_2[n2] ) + ',*,*]:\n' )
                        # get the last and second last parameters of the list:
                        last_set_element = this_scenario_data[ this_param ][ this_param_keys[-1] ] # header_indices.index( this_param_keys[-1] ) ]
                        last_set_element_unique = [] # list( set( last_set_element ) )
                        #
                        second_last_set_element = this_scenario_data[ this_param ][ this_param_keys[-2] ] # header_indices.index( this_param_keys[-2] ) ]
                        second_last_set_element_unique = [] # list( set( second_last_set_element ) )
                        
                        for u in range( len( last_set_element ) ):
                            if last_set_element[u] not in last_set_element_unique:
                                last_set_element_unique.append( last_set_element[u] )
                        
                        for u in range( len( second_last_set_element ) ):
                            if second_last_set_element[u] not in second_last_set_element_unique:
                                second_last_set_element_unique.append( second_last_set_element[u] )
                        
                        #
                        if this_param == 'TechnologyToStorage' or this_param == 'TechnologyFromStorage':
                            last_set_element = this_scenario_data[ this_param ][ this_param_keys[-2] ]
                            last_set_element_unique = []
                            second_last_set_element = this_scenario_data[ this_param ][ this_param_keys[-1] ]
                            second_last_set_element_unique = [] 
                            for u in range( len( last_set_element ) ):
                                if last_set_element[u] not in last_set_element_unique:
                                    last_set_element_unique.append( last_set_element[u] )
                            
                            for u in range( len( second_last_set_element ) ):
                                if second_last_set_element[u] not in second_last_set_element_unique:
                                    second_last_set_element_unique.append( second_last_set_element[u] )
                                    
                                    
                        for y in range( len( last_set_element_unique ) ):
                            g.write( str( last_set_element_unique[y] ) + ' ')
                        if this_param != 'TechnologyToStorage' or this_param != 'TechnologyFromStorage':
                            g.write(':=\n')
                        
                        if this_param == 'TechnologyToStorage' or this_param == 'TechnologyFromStorage':
                            second_last_set_element_unique_iter = [this_set_element_unique_2[n2]]
                        else:
                            second_last_set_element_unique_iter = second_last_set_element_unique
                        
                        if this_param_keys[-2] == 'l':
                            second_last_set_element_unique_iter_temp = second_last_set_element_unique_iter
                            second_last_set_element_unique_iter = []
                            for sdp in range(num_time_slices_SDP):
                                second_last_set_element_unique_iter.append(second_last_set_element_unique_iter_temp[sdp])
                        
                        #
                        for s in range( len( second_last_set_element_unique_iter ) ):                                  
                            if (this_param == 'TechnologyToStorage' or this_param == 'TechnologyFromStorage'):
                                for p in range(len(second_last_set_element_unique_iter)):
                                    value_indices_s = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[-2] ] ) if x == str( second_last_set_element_unique[s] ) ]
                                    value_indices_n1 = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[0] ] ) if x == str( this_set_element_unique_1[n1] ) ]
                                    value_indices_n2 = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[1] ] ) if x == str( this_set_element_unique_2[n2] ) ]
                                    r_index = set(value_indices_n1) & set(value_indices_n2)
                                    value_indices = list( r_index )
                                    value_indices.sort()
                                    #
                                    these_values = []
                                    for val in range( len( value_indices ) ):
                                        these_values.append( this_scenario_data[ this_param ]['value'][ value_indices[val] ] )
                                    if these_values != []:
                                        if p == 0:
                                            g.write( second_last_set_element_unique[count_storage] + ' ' )
                                    for val in range( len( these_values ) ):
                                        g.write( str( these_values[val] ) + ' ' )
                                count_storage += 1
                            #
                            else:
                                value_indices_s = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[-2] ] ) if x == str( second_last_set_element_unique[s] ) ]
                                value_indices_n1 = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[0] ] ) if x == str( this_set_element_unique_1[n1] ) ]
                                value_indices_n2 = [ i for i, x in enumerate( this_scenario_data[ this_param ][ this_param_keys[1] ] ) if x == str( this_set_element_unique_2[n2] ) ]
                                r_index = set(value_indices_s) & set(value_indices_n1) & set(value_indices_n2)
                                value_indices = list( r_index )
                                value_indices.sort()
                                #
                                these_values = []
                                for val in range( len( value_indices ) ):
                                    these_values.append( this_scenario_data[ this_param ]['value'][ value_indices[val] ] )
                                if these_values != []:
                                    g.write( second_last_set_element_unique[s] + ' ' )                                    
                                for val in range( len( these_values ) ):
                                    g.write( str( these_values[val] ) + ' ' )
                            if these_values != []:
                                g.write('\n')

            #%%%
            if len(this_param_keys) == 5:
                this_set_element_unique_all = []
                last_set_element_unique = []
                second_last_set_element_unique = []
                for pkey in range(len(this_param_keys)-2):
                    for i in range(2, len(header_indices)-1):
                        if header_indices[i] == this_param_keys[pkey]:
                            this_set_element = this_scenario_data[this_param][header_indices[i]]
                            this_set_element_unique_all.append(list(set(this_set_element)))
                #
                this_set_element_unique_1 = deepcopy(this_set_element_unique_all[0])
                this_set_element_unique_2 = deepcopy(this_set_element_unique_all[1])

                last_set_element = np.array(this_scenario_data[this_param][this_param_keys[-1]])
                last_set_element_unique = np.unique(last_set_element)

                second_last_set_element = np.array(this_scenario_data[this_param][this_param_keys[-2]])
                second_last_set_element_unique = np.unique(second_last_set_element)

                long_list1 = this_scenario_data[this_param][this_param_keys[1]]
                long_list2 = this_scenario_data[this_param][this_param_keys[2]]
                concat_result = list(map(lambda x, y: x + '-' + y, long_list1, long_list2))
                concat_result_set = list(set(concat_result))
                nx_temp = 0.11
                
                for n1 in range(len(this_set_element_unique_1)):
                    for nx in range(len(concat_result_set)):
                        n1_faster = concat_result_set[nx].split('-')[0]
                        n2_faster = concat_result_set[nx].split('-')[1]

                        for s in range(len(second_last_set_element_unique)):
                            value_indices_s = [i for i, x in enumerate(this_scenario_data[this_param][this_param_keys[-2]]) if x == str(second_last_set_element_unique[s])]
                            value_indices_n1 = [i for i, x in enumerate(this_scenario_data[this_param][this_param_keys[0]]) if x == str(this_set_element_unique_1[n1])]
                            value_indices_n2 = [i for i, x in enumerate(this_scenario_data[this_param][this_param_keys[1]]) if x == str(n1_faster)]
                            value_indices_n3 = [i for i, x in enumerate(this_scenario_data[this_param][this_param_keys[2]]) if x == str(n2_faster)]

                            r_index = set(value_indices_s) & set(value_indices_n1) & set(value_indices_n2) & set(value_indices_n3)
                            value_indices = list(r_index)
                            value_indices.sort()
                            

                            if len(value_indices) != 0:
                                if nx != nx_temp:    
                                    g.write('[' + str(this_set_element_unique_1[n1]) + ',' + str(n1_faster) + ',' + str(n2_faster) + ',*,*]:\n')

                                    for y in range(len(last_set_element_unique)):
                                        g.write(str(last_set_element_unique[y]) + ' ')
                                    g.write(':=\n')

                                g.write(second_last_set_element_unique[s] + ' ')

                                these_values = []
                                for val in range(len(value_indices)):
                                    these_values.append(this_scenario_data[this_param]['value'][value_indices[val]])
                                for val in range(len(these_values)):
                                    g.write(str(these_values[val]) + ' ')
                                
                                nx_temp = nx
                                    
                                g.write('\n')


            #-----------------------------------------#
            g.write( ';\n\n' )

    #
    # remember the default values for printing:
    for param_without_values in parameters_without_values:
        this_param_default_value = list_param_default_value.loc[list_param_default_value["Parameter"] == param_without_values, "Default_Value"].values[0]
        g.write(f'param {param_without_values} default {this_param_default_value} :=\n;\n')
    for new_final_set in special_sets:
        g.write(new_final_set)
    g.write('#\n' + 'end;\n')
    g.close()
    #
    # Print inputs separately for faster deployment of the input matrix.
    basic_header_elements = [
        'Future.ID',
        'Strategy.ID',
        'Strategy',
        'Commodity',
        'Technology',
        'Emission',
        'TimeSlice',
        'Year',
    ]
    parameters_to_print = parameters_in_the_model
    #
    input_params_table_headers = basic_header_elements + parameters_to_print
    all_data_row = []
    all_data_row_partial = []
    #
    combination_list = []
    synthesized_all_data_row = []
    #
    # memory elements:
    f_unique_list, f_counter, f_counter_list, f_unique_counter_list = [], 1, [], []
    t_unique_list, t_counter, t_counter_list, t_unique_counter_list = [], 1, [], []
    e_unique_list, e_counter, e_counter_list, e_unique_counter_list = [], 1, [], []
    l_unique_list, l_counter, l_counter_list, l_unique_counter_list = [], 1, [], []
    y_unique_list, y_counter, y_counter_list, y_unique_counter_list = [], 1, [], []
    #
    for p in range( len( parameters_to_print ) ):
        #
        this_p_index = S_DICT_params_structure[ 'parameter' ].index( parameters_to_print[p] )
        this_p_index_list = S_DICT_params_structure[ 'index_list' ][ this_p_index ]
        for n in range( 0, len( this_scenario_data[ parameters_to_print[p] ][ 'value' ] ) ):
            #
            single_data_row = []
            single_data_row_partial = []
            #
            single_data_row.append( fut )
            single_data_row.append( scen )
            single_data_row.append( scenario_list[scen] )
            #
            strcode = ''
            #
            if 'f' in this_p_index_list:
                single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 'f' ][n] ) # Filling FUEL if necessary
                if single_data_row[-1] not in f_unique_list:
                    f_unique_list.append( single_data_row[-1] )
                    f_counter_list.append( f_counter )
                    f_unique_counter_list.append( f_counter )
                    f_counter += 1
                else:
                    f_counter_list.append( f_unique_counter_list[ f_unique_list.index( single_data_row[-1] ) ] )
                strcode += str(f_counter_list[-1])
            else:
                single_data_row.append( '' )
                strcode += '0'
            #
            if 't' in this_p_index_list:
                single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 't' ][n] ) # Filling TECHNOLOGY if necessary
                if single_data_row[-1] not in t_unique_list:
                    t_unique_list.append( single_data_row[-1] )
                    t_counter_list.append( t_counter )
                    t_unique_counter_list.append( t_counter )
                    t_counter += 1
                else:
                    t_counter_list.append( t_unique_counter_list[ t_unique_list.index( single_data_row[-1] ) ] )
                strcode += str(t_counter_list[-1])
            else:
                single_data_row.append( '' )
                strcode += '0'
            #
            if 'e' in this_p_index_list:
                single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 'e' ][n] ) # Filling EMISSION if necessary
                if single_data_row[-1] not in e_unique_list:
                    e_unique_list.append( single_data_row[-1] )
                    e_counter_list.append( e_counter )
                    e_unique_counter_list.append( e_counter )
                    e_counter += 1
                else:
                    e_counter_list.append( e_unique_counter_list[ e_unique_list.index( single_data_row[-1] ) ] )
                strcode += str(e_counter_list[-1])
            else:
                single_data_row.append( '' )
                strcode += '0'
            #
            if 'l' in this_p_index_list:
                single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 'l' ][n] ) # Filling SEASON if necessary
                if single_data_row[-1] not in l_unique_list:
                    l_unique_list.append( single_data_row[-1] )
                    l_counter_list.append( l_counter )
                    l_unique_counter_list.append( l_counter )
                    l_counter += 1
                else:
                    l_counter_list.append( l_unique_counter_list[ l_unique_list.index( single_data_row[-1] ) ] )
                strcode += str(l_counter_list[-1])
            else:
                single_data_row.append( '' )
                strcode += '000' # this is done to avoid repeated characters
            #
            if 'y' in this_p_index_list:
                single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 'y' ][n] ) # Filling YEAR if necessary
                if single_data_row[-1] not in y_unique_list:
                    y_unique_list.append( single_data_row[-1] )
                    y_counter_list.append( y_counter )
                    y_unique_counter_list.append( y_counter )
                    y_counter += 1
                else:
                    y_counter_list.append( y_unique_counter_list[ y_unique_list.index( single_data_row[-1] ) ] )
                strcode += str(y_counter_list[-1])
            else:
                single_data_row.append( '' )
                strcode += '0'
            #
            this_combination_str = str(1) + strcode # deepcopy( single_data_row )
            this_combination = int( this_combination_str )
            #
            for aux_p in range( len(basic_header_elements), len(basic_header_elements) + len( parameters_to_print ) ):
                if aux_p == p + len(basic_header_elements):
                    single_data_row.append( this_scenario_data[ parameters_to_print[p] ][ 'value' ][n] ) # Filling the correct data point
                    single_data_row_partial.append( this_scenario_data[ parameters_to_print[p] ][ 'value' ][n] )
                else:
                    single_data_row.append( '' )
                    single_data_row_partial.append( '' )
            #
            all_data_row.append( single_data_row )
            all_data_row_partial.append( single_data_row_partial )
            #
            if this_combination not in combination_list:
                combination_list.append( this_combination )
                synthesized_all_data_row.append( single_data_row )
            else:
                ref_combination_index = combination_list.index( this_combination )
                ref_parameter_index = input_params_table_headers.index( parameters_to_print[p] )
                synthesized_all_data_row[ ref_combination_index ][ ref_parameter_index ] = deepcopy( single_data_row_partial[ ref_parameter_index-len( basic_header_elements ) ] )
                #
            #
        #
    #
    ###########################################################################################################################

    # Make a DataFrame with the data
    param_parquet_path = os.path.join(current_script_path, 'Experimental_Platform', 'Futures', scenario_list[scen], str( scenario_list[scen] ) + '_' + str( fut ), str( scenario_list[scen] ) + '_' + str( fut ) + '_Input.parquet')
    data_dict = {header: [row[i] for row in synthesized_all_data_row] for i, header in enumerate(input_params_table_headers)}
    df = pd.DataFrame(data_dict)

    # Define the columns to exclude in lowercase
    columns_to_exclude = [
        'strategy', 'future.id', 'region', 'commodity', 'technology', 'emission',
        'timeslice', 'mode_of_operation', 'season', 'daytype',
        'dailytimebracket', 'storage', 'storageintraday', 'storageintrayear',
        'udc', 'scen_fut'
    ]
    
    # Loop through all columns, normalize names before comparing
    for col in df.columns:
        if col.lower() not in columns_to_exclude:  # Case-insensitive check
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, NaNs for invalids
                df[col] = df[col].fillna(0)  # Replace NaNs
                df = df.dropna(subset=[col])  # Drop rows with remaining NaNs
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {e}")



    
    column_rename_map = {
        'Commodity': 'COMMODITY',
        'Technology': 'TECHNOLOGY',
        'Emission': 'EMISSION',
        'Season': 'SEASON',
        'Year': 'YEAR',
        'TimeSlice': 'TIMESLICE',
        'Region': 'REGION',
    }
    df = df.rename(columns=column_rename_map)
    df.to_parquet(param_parquet_path, engine='pyarrow', index=False)
    df.to_csv(param_parquet_path.replace('parquet', 'csv'), index=False, sep=';')

if __name__ == '__main__':
    
    # Take the solver from the script call
    main_path = sys.argv
    solver = main_path[1]
    osemosys_model = main_path[2]
    Interface_RDM = main_path[3]
    shape_file = main_path[4]
    def load_interface_tables(interface_path):
        """Load interface tables from a directory of CSVs or from an Excel file."""
        interface_path = Path(interface_path)
        if interface_path.is_dir():
            return {
                'Setup': pd.read_csv(interface_path / 'Setup.csv'),
                'To_Print': pd.read_csv(interface_path / 'To_Print.csv'),
                'Params_Sets_Vari': pd.read_csv(interface_path / 'Params_Sets_Vari.csv'),
                'Uncertainty_Table': pd.read_csv(interface_path / 'Uncertainty_Table.csv'),
            }
        if interface_path.suffix.lower() == '.xlsx':
            book = pd.ExcelFile(interface_path)
            return {
                'Setup': book.parse('Setup', 0),
                'To_Print': book.parse('To_Print', 0),
                'Params_Sets_Vari': book.parse('Params_Sets_Vari'),
                'Uncertainty_Table': book.parse('Uncertainty_Table'),
            }
        raise ValueError(f'Interface_RDM must be a directory of CSVs or an .xlsx file: {interface_path}')

    interface_tables = load_interface_tables(Interface_RDM)
    parameters_to_print = interface_tables['To_Print']
    
    generator_or_executor = 'Both'
    parallel_or_linear = 'Parallel'
    


    # 1.A) Extract the structure setup of the model based on "B1_Model_Structure.xlsx".
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    structure_filename = os.path.join(
        current_script_path, "0_From_Confection", "B1_Model_Structure.xlsx"
    )
    structure_file = pd.ExcelFile(structure_filename)
    structure_sheetnames = structure_file.sheet_names  # see all sheet names
    sheet_sets_structure = pd.read_excel(
        open(structure_filename, "rb"), header=None, sheet_name=structure_sheetnames[0]
    )
    sheet_params_structure = pd.read_excel(
        open(structure_filename, "rb"), header=None, sheet_name=structure_sheetnames[1]
    )
    sheet_vars_structure = pd.read_excel(
        open(structure_filename, "rb"), header=None, sheet_name=structure_sheetnames[2]
    )

    S_DICT_sets_structure = {
        "set": [],
        "initial": [],
        "number_of_elements": [],
        "elements_list": [],
    }
    for col in range(1, len(sheet_sets_structure.iloc[1, 1:].tolist()) + 1):
        S_DICT_sets_structure["set"].append(sheet_sets_structure.iat[0, col])
        S_DICT_sets_structure["initial"].append(sheet_sets_structure.iat[1, col])
        S_DICT_sets_structure["number_of_elements"].append(
            int(sheet_sets_structure.iat[2, col])
        )
        element_number = int(sheet_sets_structure.iat[2, col])
        this_elements_list = []
        if element_number > 0:
            for n in range(1, element_number + 1):
                this_elements_list.append(sheet_sets_structure.iat[2 + n, col])
        S_DICT_sets_structure["elements_list"].append(this_elements_list)

    S_DICT_params_structure = {
        "category": [],
        "parameter": [],
        "number_of_elements": [],
        "index_list": [],
    }
    param_category_list = []
    for col in range(1, len(sheet_params_structure.iloc[1, 1:].tolist()) + 1):
        if str(sheet_params_structure.iat[0, col]) != "":
            param_category_list.append(sheet_params_structure.iat[0, col])
        S_DICT_params_structure["category"].append(param_category_list[-1])
        S_DICT_params_structure["parameter"].append(sheet_params_structure.iat[1, col])
        S_DICT_params_structure["number_of_elements"].append(
            int(sheet_params_structure.iat[2, col])
        )
        index_number = int(sheet_params_structure.iat[2, col])
        this_index_list = []
        for n in range(1, index_number + 1):
            this_index_list.append(sheet_params_structure.iat[2 + n, col])
        S_DICT_params_structure["index_list"].append(this_index_list)

    S_DICT_vars_structure = {
        "category": [],
        "variable": [],
        "number_of_elements": [],
        "index_list": [],
    }
    var_category_list = []
    for col in range(1, len(sheet_vars_structure.iloc[1, 1:].tolist()) + 1):
        if str(sheet_vars_structure.iat[0, col]) != "":
            var_category_list.append(sheet_vars_structure.iat[0, col])
        S_DICT_vars_structure["category"].append(var_category_list[-1])
        S_DICT_vars_structure["variable"].append(sheet_vars_structure.iat[1, col])
        S_DICT_vars_structure["number_of_elements"].append(
            int(sheet_vars_structure.iat[2, col])
        )
        index_number = int(sheet_vars_structure.iat[2, col])
        this_index_list = []
        for n in range(1, index_number + 1):
            this_index_list.append(sheet_vars_structure.iat[2 + n, col])
        S_DICT_vars_structure["index_list"].append(this_index_list)

    global time_range_vector  # This variable manages time throughout the experiment.
    time_range_vector = [int(i) for i in S_DICT_sets_structure["elements_list"][0]]
    
    global final_year
    final_year = time_range_vector[-1]
    global initial_year
    initial_year = time_range_vector[0]
    
    # Read user-defined scenarios in future 0 (Base_Runs_Generator.py). These
    # parameters define the baseline for subsequent uncertainty perturbations.
    
    setup_table = interface_tables['Setup']
    scenarios_to_reproduce = str( setup_table.loc[ 0 ,'Scenario_to_Reproduce'] )
    df_Params_Sets_Vari = interface_tables['Params_Sets_Vari']
    # Step 1: Remove the 'parameter' column and store its values to use as index
    new_index = df_Params_Sets_Vari['parameter'].reset_index(drop=True)
    df_Params_Sets_Vari = df_Params_Sets_Vari.drop(columns='parameter')
    
    # Step 2: Replace NaN values in the index (empty cells) with meaningful labels
    new_index = new_index.fillna('')  # Replace NaN with empty strings
    new_index.iloc[0] = 'Number'      # Rename first row index
    new_index.iloc[1] = 'Set1'        # Rename second row index
    new_index.iloc[2] = 'Set2'        # Rename third row index
    new_index.iloc[3] = 'Set3'        # Rename fourth row index
    
    # Step 3: Assign the updated labels as the new index of the DataFrame
    df_Params_Sets_Vari.index = new_index
    
    # Step 4: Extract the mapping from the existing dictionary
    initials = S_DICT_sets_structure['initial']
    full_names = S_DICT_sets_structure['set']
    
    # Step 5: Create a dictionary for easy replacement: {'y': 'YEAR', ...}
    replacement_dict = dict(zip(initials, full_names))
    
    # Step 6: Define the rows (index values) that need to be processed
    rows_to_replace = ['Set1', 'Set2', 'Set3']
    
    # Step 7: Replace the values in those rows using the mapping
    for row in rows_to_replace:
        df_Params_Sets_Vari.loc[row] = df_Params_Sets_Vari.loc[row].replace(replacement_dict)

    print(
        "1: I start by reading the Uncertainty Table and systematically perturbing the parameters."
    )
    uncertainty_table = interface_tables['Uncertainty_Table']
    np.random.seed(555)
    P = len(uncertainty_table.index)  # variables to vary
    N = int(setup_table.loc[0, 'Number_of_Runs'])  # number of samples

    # Here we need to define the number of elements that need to be included in the hypercube
    ignore_indices = []
    subtracter = 0
    col_idx = {}
    for p in range(P):
        if p in ignore_indices:
            subtracter += 1
            col_idx.update({p: 'none'})
        else:
            col_idx.update({p: p - subtracter})

    hypercube = lhs(P - subtracter, samples=N)
    # hypercube[p] gives values of variable p across the N futures.
    experiment_dictionary = {}

    # Loop over futures.
    for n in range(N):
        this_future_X_change = []  # relative to baseline
        X_Num_unique = []
        X_Num = []
        X_Cat = []
        # Loop over uncertainty lines.
        for p in range(P):
            math_type = str(uncertainty_table.loc[p, 'X_Mathematical_Type'])
            Explored_Parameter_of_X = str(uncertainty_table.loc[p, 'Explored_Parameter_of_X'])

            Involved_Scenarios = (
                str(uncertainty_table.loc[p, 'Involved_Scenarios']).replace(" ", "").split(";")
            )
            Involved_First_Sets_in_Osemosys = (
                str(uncertainty_table.loc[p, 'Involved_First_Sets_in_Osemosys'])
                .replace(" ", "")
                .split(";")
            )
            Involved_Second_Sets_in_Osemosys = (
                str(uncertainty_table.loc[p, 'Involved_Second_Sets_in_Osemosys'])
                .replace(" ", "")
                .split(";")
            )
            Involved_Third_Sets_in_Osemosys = (
                str(uncertainty_table.loc[p, 'Involved_Third_Sets_in_Osemosys'])
                .replace(" ", "")
                .split(";")
            )
            Exact_Parameters_Involved_in_Osemosys = (
                str(uncertainty_table.loc[p, 'Exact_Parameters_Involved_in_Osemosys'])
                .replace(" ", "")
                .split(";")
            )
            Exact_X = str(uncertainty_table.loc[p, 'X_Plain_English_Description'])
            Initial_Year_of_Uncertainty_EP = int(
                uncertainty_table.loc[p, 'Initial_Year_of_Uncertainty']
            )

            X_Num.append(int(uncertainty_table.loc[p, 'X_Num']))
            X_Cat.append(str(uncertainty_table.loc[p, 'X_Category']))
            this_min = uncertainty_table.loc[p, 'Min_Value']
            this_max = uncertainty_table.loc[p, 'Max_Value']
            this_loc = this_min
            this_loc_scale = this_max - this_min

            hyper_col_idx = col_idx[p]
            if hyper_col_idx != 'none':
                evaluation_value_preliminary = hypercube[n].item(hyper_col_idx)
            else:
                evaluation_value_preliminary = 1

            evaluation_value = scipy.stats.uniform.ppf(
                evaluation_value_preliminary, this_loc, this_loc_scale
            )

            # Adjust the sampling range to preserve a consistent direction.
            if evaluation_value > 1:
                this_loc_scale = 0.5 * (this_max - this_min)
            elif evaluation_value < 1:
                this_loc = this_min + 0.5 * (this_max - this_min)

            evaluation_value = scipy.stats.uniform.ppf(evaluation_value_preliminary, this_loc, this_loc_scale)
            #
            #######################################################################
            this_future_X_change.append(evaluation_value)
            #######################################################################
            # Store information for each future in a dictionary.
            if n == 0:  # The dictionary is created only when the first future appears.
                if int(uncertainty_table.loc[p, 'X_Num']) not in X_Num_unique:
                    X_Num_unique.append(int(uncertainty_table.loc[p, 'X_Num']))
                    experiment_dictionary.update(
                        {X_Num_unique[-1]: {'Category': X_Cat[-1], 'Math_Type': math_type, 'Exact_X': Exact_X}}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Involved_Scenarios': Involved_Scenarios}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Involved_First_Sets_in_Osemosys': Involved_First_Sets_in_Osemosys}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Involved_Second_Sets_in_Osemosys': Involved_Second_Sets_in_Osemosys}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Involved_Third_Sets_in_Osemosys': Involved_Third_Sets_in_Osemosys}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Exact_Parameters_Involved_in_Osemosys': Exact_Parameters_Involved_in_Osemosys}
                    )

                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Initial_Year_of_Uncertainty': Initial_Year_of_Uncertainty_EP}
                    )
                    experiment_dictionary[X_Num_unique[-1]].update(
                        {'Futures': [x for x in range(1, N + 1)]}
                    )
                    if math_type in ['Time_Series', 'Discrete_Investments', 'Mult_Adoption_Curve', 'Mult_Restriction', 'Mult_Restriction_Start', 'Mult_Restriction_End', 'Timeslices_Curve', 'Constant', 'Logistic', 'Linear']:
                        experiment_dictionary[ X_Num_unique[-1] ].update({ 'Explored_Parameter_of_X':Explored_Parameter_of_X } )
                        experiment_dictionary[ X_Num_unique[-1] ].update({ 'Values':[0.0 for x in range( 1, N+1 ) ] })
                        experiment_dictionary[ X_Num_unique[-1] ].update({ 'Emission_Years':[0.0 for x in range( 1, N+1 ) ] })
                        experiment_dictionary[int(uncertainty_table.loc[p, 'X_Num'])][
                            'Values'
                        ][0] = this_future_X_change[-1]
            ###################################################################################################################################
            else:
                ###################################################################################################################################
                if int( uncertainty_table.loc[ p ,'X_Num'] ) not in X_Num_unique:
                    #
                    X_Num_unique.append( int( uncertainty_table.loc[ p ,'X_Num'] ) )
                    #
                    if math_type in [
                        'Time_Series',
                        'Discrete_Investments',
                        'Mult_Adoption_Curve',
                        'Mult_Restriction',
                        'Mult_Restriction_Start',
                        'Mult_Restriction_End',
                        'Timeslices_Curve',
                        'Constant',
                        'Logistic',
                        'Linear',
                    ]:
                        experiment_dictionary[int(uncertainty_table.loc[p, 'X_Num'])]['Values'][
                            n
                        ] = this_future_X_change[-1]
                    #

                u = int( uncertainty_table.loc[ p ,'X_Num'] )
                print(u, len(experiment_dictionary[u].keys()), experiment_dictionary[u]['Category'])

    print('2: That is done. Now I initialize some key structural data.')
    header_row = [
        'PARAMETER',
        'Scenario',
        'REGION',
        'TECHNOLOGY',
        'COMMODITY',
        'EMISSION',
        'MODE_OF_OPERATION',
        'TIMESLICE',
        'YEAR',
        'SEASON',
        'DAYTYPE',
        'DAILYTIMEBRACKET',
        'STORAGE',
        'STORAGEINTRADAY',
        'STORAGEINTRAYEAR',
        'UDC',
        'Value',
    ]
    scenario_list = []
    if scenarios_to_reproduce == 'Experiment':
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        dir_executables = os.path.join(current_script_path, 'Executables')
        scenario_list = [
            f.replace('_0', '')
            for f in os.listdir(dir_executables)
            if not (f.endswith('.py') or f.endswith('.csv') or f.endswith('__pycache__'))
        ]
    elif scenarios_to_reproduce != 'All' and scenarios_to_reproduce != 'Experiment':
        scenario_list.append(scenarios_to_reproduce)

    # Define the dictionary for calibrated database:
    stable_scenarios = {}
    dict_special_sets = {}
    dict_rows = {}
    dict_S_DICT_params_structure = {}
    dict_S_DICT_sets_structure = {}
    dict_parameters_without_values = {}
    dict_parameters_in_the_model = {}
    for scen in scenario_list:
        stable_scenarios.update({scen.replace('_0', ''): {}})
    #
    for scen in range( len( scenario_list ) ):
        #
        if scenario_list[scen] != "__pycache__" and not scenario_list[scen].endswith('.csv'):
            scen_file = os.path.join(dir_executables, scenario_list[scen] + '_0/', scenario_list[scen] + '_0.txt')
        data_per_param, special_sets = AUX.isolate_params(scen_file)
        dict_special_sets[scenario_list[scen]] = special_sets
        
        rows = []
        
        for key, lines in data_per_param.items():
            if not lines:
                continue

            first_line = lines[0]
            
            parts = first_line.strip().split()

            try:
                param_index = parts.index("param") + 1
                default_index = parts.index("default") + 1

                parameter = parts[param_index]
                default_value = parts[default_index]

                rows.append({"Parameter": parameter, "Default_Value": default_value})

            except (ValueError, IndexError):
                continue
        dict_rows[scenario_list[scen]] = pd.DataFrame(rows)
        
        num_time_slices_SDP = int(setup_table.loc[0, 'Timeslices_model'])

        list_dataframes, dict_dataframes, parameters_without_values = AUX.generate_df_per_param(
            scenario_list[scen], data_per_param, num_time_slices_SDP
        )
        
        parameters_without_values.sort()
        parameters_without_values = list(dict.fromkeys(parameters_without_values))
        dict_parameters_without_values[scenario_list[scen]] = parameters_without_values
        parameters_in_the_model = list(dict_dataframes.keys())
        dict_parameters_in_the_model[scenario_list[scen]] = parameters_in_the_model
        for param in parameters_in_the_model:
            stable_scenarios[ scenario_list[scen] ].update( { param:{} } )
            # To extract the parameter input data:
            all_params_list_index = S_DICT_params_structure['parameter'].index(param)
            this_number_of_elements = S_DICT_params_structure['number_of_elements'][all_params_list_index]
            this_index_list = S_DICT_params_structure['index_list'][all_params_list_index]
            #
            for k in range(this_number_of_elements):
                stable_scenarios[ scenario_list[scen] ][ param ].update({this_index_list[k]:[]})
            stable_scenarios[ scenario_list[scen] ][ param ].update({'value':[]})
        #
        for param, value in dict_dataframes.items():            
            # Extract data:
            for index, row_df in value.iterrows():
                row = row_df.to_dict()
                row['Value'] = row.pop(param)
                if row[ header_row[-1] ] != None and row[ header_row[-1] ] != '':
                    #
                    for h in range( 2, len(header_row)-1 ):
                        if row[ header_row[h] ] != None and row[ header_row[h] ] != '':
                            set_index  = S_DICT_sets_structure['set'].index( header_row[h] )
                            set_initial = S_DICT_sets_structure['initial'][ set_index ]
                            stable_scenarios[ scenario_list[scen] ][ param ][ set_initial ].append( row[ header_row[h] ] )
                    stable_scenarios[ scenario_list[scen] ][ param ][ 'value' ].append( row[ header_row[-1] ] )
        
        dict_S_DICT_params_structure[scenario_list[scen]] = S_DICT_params_structure
        dict_S_DICT_sets_structure[scenario_list[scen]] = S_DICT_sets_structure
                    
    # PART 3: Perturb the system by reapplying uncertainty across parameters.
    all_futures = [n for n in range(1, N + 1)]
    inherited_scenarios = {}
    for n1 in range(len(scenario_list)):
        inherited_scenarios.update({scenario_list[n1]: {}})
        for n2 in range(len(all_futures)):
            copy_stable_dictionary = deepcopy(stable_scenarios[scenario_list[n1]])
            inherited_scenarios[scenario_list[n1]].update(
                {all_futures[n2]: copy_stable_dictionary}
            )

    print('3: That is done. Now I systematically perturb the model parameters.')
    # Broadly speaking, we must perform the same calculation across futures, as
    # all futures are independent.

    # Loop over scenarios (e.g., S1, S2, S3).
    for s in range(len(scenario_list)):
        fut_id = 0
        S_DICT_params_structure = dict_S_DICT_params_structure[scenario_list[s]]
        S_DICT_sets_structure = dict_S_DICT_sets_structure[scenario_list[s]]

        timeslice_index = S_DICT_sets_structure['set'].index('TIMESLICE')
        all_timeslices = S_DICT_sets_structure['elements_list'][timeslice_index]

        # Loop over futures.
        for f in range(1, len(all_futures) + 1):
            # Loop over uncertainties.
            for u in range(1, len(experiment_dictionary) + 1):

                Initial_Year_of_Uncertainty = experiment_dictionary[u]['Initial_Year_of_Uncertainty']
                # Extract crucial sets and parameters to be manipulated in the model.
                Parameters_Involved = experiment_dictionary[u]['Exact_Parameters_Involved_in_Osemosys']
                First_Involved = deepcopy(experiment_dictionary[u]['Involved_First_Sets_in_Osemosys'])
                Second_Involved = experiment_dictionary[u]['Involved_Second_Sets_in_Osemosys']
                Third_Involved = experiment_dictionary[u]['Involved_Third_Sets_in_Osemosys']

                Scenarios_Involved = experiment_dictionary[u]['Involved_Scenarios']
                # Extract crucial identifiers.
                Explored_Parameter_of_X = experiment_dictionary[u]['Explored_Parameter_of_X']
                Math_Type = experiment_dictionary[u]['Math_Type']
                # Extract the values.
                Values_per_Future = experiment_dictionary[u]['Values']

                # Last year of the analysis.
                last_year_analysis = time_range_vector[-1]
                round_cs = 10

                if str( scenario_list[s] ) in Scenarios_Involved:
                    # We iterate over the involved parameters of the model here:
                    for p in range( len( Parameters_Involved ) ):
                        this_parameter = Parameters_Involved[p]
                        
                        #------------------------------------------------------------------------------------------------------------------------------------------#
                        if Math_Type in ['Time_Series', 'Constant', 'Logistic', 'Linear'] and Explored_Parameter_of_X=='Final_Value':
                                                        
                            number_sets_by_param = df_Params_Sets_Vari.loc['Number', this_parameter]
                              
                            if number_sets_by_param == 0:
                                
                                # No need to iterate over sets as we are working with a global param                                
                                # No need to "extract time" as we are working with a global param

                                # extracting value:
                                value_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'])
                                value_list = [ float( value_list[j] ) for j in range( len( value_list ) ) ]
                                # assign new value
                                new_value_list = [xx*float(Values_per_Future[fut_id]) for xx in value_list]
                                new_value_list_rounded = [ round(elem, 3) for elem in new_value_list ]
                                # Assign parameters back: for these subset of uncertainties
                                inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'] = deepcopy(new_value_list_rounded)

                            elif number_sets_by_param == 1:                                    
                            
                                set1_by_param = df_Params_Sets_Vari.loc['Set1', this_parameter]
                                
                                # if user select all timeslices
                                if set1_by_param == 'TIMESLICE' and any(option in First_Involved for option in ['All', 'all', 'ALL']):
                                    First_Involved = all_timeslices
                                
                                for f_set in range( len( First_Involved ) ):
                                    tsfirst = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set1_by_param) ]
                                    this_set_first = First_Involved[f_set]
                                    this_set_range_indices_first = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tsfirst ] ) if x == str( this_set_first ) ]
                                    #
                                    # find elements in common
                                    this_set_range_indices = this_set_range_indices_first
                                    this_set_range_indices.sort()
                                    
                                    # for each index we extract the time and value in a list:
                                    # extracting time:
                                    time_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['y'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                    time_list = [ int( time_list[j] ) for j in range( len( time_list ) ) ]
                                    # extracting value:
                                    value_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                    value_list = [ float( value_list[j] ) for j in range( len( value_list ) ) ]
                                    #--------------------------------------------------------------------#
                                    if this_parameter == 'TotalTechnologyAnnualActivityLowerLimit':
                                        this_set_range_indices_upper = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ 'TotalTechnologyAnnualActivityUpperLimit' ][ tsfirst ] ) if x == str( this_set_first ) ]
                                        # find elements in common
                                        this_set_range_indices_upper.sort()
                                        if this_set_range_indices_upper != []:
                                            value_list_upper = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ 'TotalTechnologyAnnualActivityUpperLimit' ]['value'][ this_set_range_indices_upper[0]:this_set_range_indices_upper[-1]+1 ] )
                                            # Get gap between Upper and Lower Limit
                                            gap_list = [upper - lower for upper, lower in zip(value_list_upper, value_list)]
                                    #--------------------------------------------------------------------#
                                    # now that the value is extracted, we must manipulate the result and assign back
                                    if Math_Type == 'Time_Series':
                                        new_value_list = deepcopy(AUX.interpolation_non_linear_final(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                    elif Math_Type == 'Constant':
                                        new_value_list = deepcopy(AUX.interpolation_constant_trajectory(time_list, value_list, Initial_Year_of_Uncertainty))
                                    elif Math_Type == 'Linear':
                                        new_value_list = deepcopy(AUX.interpolation_linear(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                    elif Math_Type == 'Logistic':
                                        new_value_list = deepcopy(AUX.interpolation_logistic_trajectory(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                    new_value_list_rounded = [ round(elem, round_cs) for elem in new_value_list ]
                                    #--------------------------------------------------------------------#
                                    
                                    if this_parameter == 'TotalTechnologyAnnualActivityLowerLimit':
                                        if this_set_range_indices_upper != []:
                                            # Compute new_gap_list
                                            new_gap_list = [upper - lower for upper, lower in zip(value_list_upper, new_value_list_rounded)]
                                            
                                            # Check if there are any negative values
                                            if any(val < 0 for val in new_gap_list):
                                                # Create a corrected list by adding the original gap
                                                corrected_new_value_list = [rounded + gap for rounded, gap in zip(new_value_list_rounded, gap_list)]
                                                
                                                # Save new Upper Limit
                                                inherited_scenarios[scenario_list[s]][f]['TotalTechnologyAnnualActivityUpperLimit']['value'][
                                                    this_set_range_indices_upper[0]:this_set_range_indices_upper[-1] + 1
                                                ] = deepcopy(corrected_new_value_list)
                                    #--------------------------------------------------------------------#
                                    # Assign parameters back: for these subset of uncertainties
                                    inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] = deepcopy(new_value_list_rounded)
                            
                            elif number_sets_by_param == 2:
                                set1_by_param = df_Params_Sets_Vari.loc['Set1', this_parameter]
                                set2_by_param = df_Params_Sets_Vari.loc['Set2', this_parameter]
                                
                                # if user select all timeslices
                                if set1_by_param == 'TIMESLICE' and any(option in First_Involved for option in ['All', 'all', 'ALL']):
                                    First_Involved = all_timeslices
                                elif set2_by_param == 'TIMESLICE' and any(option in Second_Involved for option in ['All', 'all', 'ALL']):
                                    Second_Involved = all_timeslices
                                
                                for f_set in range( len( First_Involved ) ):
                                    tsfirst = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set1_by_param) ]
                                    this_set_first = First_Involved[f_set]
                                    this_set_range_indices_first = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tsfirst ] ) if x == str( this_set_first ) ]
                                    #
                                    for s_set in range( len( Second_Involved ) ):
                                        tssecond = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set2_by_param) ]
                                        this_set_second = Second_Involved[s_set]
                                        if this_set_second == '1.0': # pietro temp patch
                                            this_set_second = '1'
                                        if this_set_second == '2.0': # pietro temp patch
                                            this_set_second = '2'
                                        this_set_range_indices_second = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tssecond ] ) if x == str( this_set_second ) ]
                                        #
                                        # find elements in common
                                        this_set_range_indices = list(set(this_set_range_indices_first) & set(this_set_range_indices_second))
                                        
                                        if this_set_range_indices != []:
                                            this_set_range_indices.sort()
                                            # for each index we extract the time and value in a list:
                                            # extracting time:
                                            time_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['y'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                            time_list = [ int( time_list[j] ) for j in range( len( time_list ) ) ]
                                            # extracting value:
                                            value_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                            value_list = [ float( value_list[j] ) for j in range( len( value_list ) ) ]
                                            #--------------------------------------------------------------------#
                                            # now that the value is extracted, we must manipulate the result and assign back
                                            if Math_Type == 'Time_Series':
                                                new_value_list = deepcopy(AUX.interpolation_non_linear_final(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                            elif Math_Type == 'Constant':
                                                new_value_list = deepcopy(AUX.interpolation_constant_trajectory(time_list, value_list, Initial_Year_of_Uncertainty))
                                            elif Math_Type == 'Logistic':
                                                new_value_list = deepcopy(AUX.interpolation_logistic_trajectory(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                            #
                                            new_value_list_rounded = [ round(elem, round_cs) for elem in new_value_list ]
                                            #--------------------------------------------------------------------#``
                                            # Assign parameters back: for these subset of uncertainties
                                            inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] = deepcopy(new_value_list_rounded)
                                        else:
                                            print(f'Combination of {set1_by_param}:{this_set_first} and {set2_by_param}:{this_set_second} to {this_parameter} does not have any values.')
                                        
                            elif number_sets_by_param == 3:  
                                set1_by_param = df_Params_Sets_Vari.loc['Set1', this_parameter]
                                set2_by_param = df_Params_Sets_Vari.loc['Set2', this_parameter]
                                set3_by_param = df_Params_Sets_Vari.loc['Set3', this_parameter]
                                
                                # if user select all timeslices
                                if set1_by_param == 'TIMESLICE' and any(option in First_Involved for option in ['All', 'all', 'ALL']):
                                    First_Involved = all_timeslices
                                elif set2_by_param == 'TIMESLICE' and any(option in Second_Involved for option in ['All', 'all', 'ALL']):
                                    Second_Involved = all_timeslices
                                elif set3_by_param == 'TIMESLICE' and any(option in Third_Involved for option in ['All', 'all', 'ALL']):
                                    Third_Involved = all_timeslices
                                
                                for f_set in range( len( First_Involved ) ):
                                    tsfirst = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set1_by_param) ]
                                    this_set_first = First_Involved[f_set]
                                    this_set_range_indices_first = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tsfirst ] ) if x == str( this_set_first ) ]
                                    #
                                    for s_set in range( len( Second_Involved ) ):
                                        tssecond = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set2_by_param) ]
                                        this_set_second = Second_Involved[s_set]
                                        this_set_range_indices_second = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tssecond ] ) if x == str( this_set_second ) ]
                                        #
                                        for t_set in range( len( Third_Involved ) ):
                                            tsthird = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set3_by_param) ]
                                            this_set_third = Third_Involved[t_set]
                                            this_set_range_indices_third = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tsthird ] ) if x == str( this_set_third ) ]
                                            #
                                            # find elements in common
                                            this_set_range_indices = list(set(this_set_range_indices_first) & set(this_set_range_indices_second) & set(this_set_range_indices_third))
                                            if this_set_range_indices != []:
                                                this_set_range_indices.sort()
                                                # for each index we extract the time and value in a list:
                                                # extracting time:
                                                time_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['y'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                                time_list = [ int( time_list[j] ) for j in range( len( time_list ) ) ]
                                                # extracting value:
                                                value_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                                value_list = [ float( value_list[j] ) for j in range( len( value_list ) ) ]
                                                #--------------------------------------------------------------------#
                                                # now that the value is extracted, we must manipulate the result and assign back
                                                if Math_Type == 'Time_Series':
                                                    new_value_list = deepcopy(AUX.interpolation_non_linear_final(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                                elif Math_Type == 'Constant':
                                                    new_value_list = deepcopy(AUX.interpolation_constant_trajectory(time_list, value_list, Initial_Year_of_Uncertainty))
                                                elif Math_Type == 'Logistic':
                                                    new_value_list = deepcopy(AUX.interpolation_logistic_trajectory(time_list, value_list, float(Values_per_Future[fut_id]), last_year_analysis, Initial_Year_of_Uncertainty))
                                                #
                                                new_value_list_rounded = [ round(elem, round_cs) for elem in new_value_list ]
                                                #--------------------------------------------------------------------#``
                                                # Assign parameters back: for these subset of uncertainties
                                                inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] = deepcopy(new_value_list_rounded)
                                            else:
                                                print(f'Combination of {set1_by_param}:{this_set_first}, {set2_by_param}:{this_set_second} and {set3_by_param}:{this_set_third} to {this_parameter} does not have any values.')                            
                                                        
                                                        
                        
                        if Math_Type=='Timeslices_Curve' and Explored_Parameter_of_X=='Change_Curve':
                            set1_by_param = df_Params_Sets_Vari.loc['Set1', this_parameter]
                            set2_by_param = df_Params_Sets_Vari.loc['Set2', this_parameter]
                            for f_set in range( len( First_Involved ) ):
                                tsfirst = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set1_by_param) ]
                                #
                                this_set_first = First_Involved[f_set]
                                # Read "shape_of_demand.xlsx"
                                df_shapes = pd.read_csv(shape_file)
                                ts_min_value = uncertainty_table.loc[uncertainty_table['X_Num'] == u, 'Min_Value'].values
                                ts_min_value = float(ts_min_value[0])
                                ts_max_value = uncertainty_table.loc[uncertainty_table['X_Num'] == u, 'Max_Value'].values
                                ts_max_value = float(ts_max_value[0])
                                
                                curve_columns = df_shapes.columns.tolist()
                                num_curves = len(curve_columns)
                                
                                # Ensure valueLHS is within the defined range
                                valueLHS = max(min(Values_per_Future[0], ts_max_value), ts_min_value)
                                
                                # Calculate the width of each interval
                                interval_width = (ts_max_value - ts_min_value) / num_curves
                                
                                # Determine which interval valueLHS falls into
                                selected_index = int((valueLHS - ts_min_value) / interval_width)
                                
                                # Handle edge case when valueLHS == tsMaxValue
                                if selected_index == num_curves:
                                    selected_index -= 1
                                
                                # Get the name and data of the selected curve
                                selected_curve_name = curve_columns[selected_index]
                                selected_curve = df_shapes[selected_curve_name].tolist()
                                
                                # if user select all timeslices
                                if any(option in Second_Involved for option in ['All', 'all', 'ALL']):
                                    Second_Involved = all_timeslices
                                
                                if this_parameter == 'SpecifiedDemandProfile':
                                    for s_set in range( len( Second_Involved ) ):
                                        #
                                        this_set_second = Second_Involved[s_set]
                                        #
                                        tssecond = S_DICT_sets_structure['initial'][ S_DICT_sets_structure['set'].index(set2_by_param) ]
                                        this_set_range_indices_fuel = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tsfirst ] ) if x == str( this_set_first ) ]
                                        this_set_range_indices_ts = [ i for i, x in enumerate( inherited_scenarios[ scenario_list[ s ] ][ f ][ this_parameter ][ tssecond ] ) if x == str( this_set_second ) ]
                                        
                                        # find elements in common
                                        this_set_range_indices = list(set(this_set_range_indices_fuel) & set(this_set_range_indices_ts))
                                        this_set_range_indices.sort()
                                        # for each index we extract the time and value in a list:
                                        # extracting time:
                                        time_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['y'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                        time_list = [ int( time_list[j] ) for j in range( len( time_list ) ) ]
                                        # extracting value:
                                        value_list = deepcopy( inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] )
                                        value_list = [ float( value_list[j] ) for j in range( len( value_list ) ) ]
                                        #--------------------------------------------------------------------#
                                        # now that the value is extracted, we must manipulate the result and assign back
                                        new_value_list = [selected_curve[int(this_set_second) - 1]] * len(value_list)
                                        new_value_list_rounded = [ round(elem, round_cs) for elem in new_value_list ]
                                        #--------------------------------------------------------------------#``
                                        # Assign parameters back: for these subset of uncertainties
                                        inherited_scenarios[ scenario_list[s] ][ f ][ this_parameter ]['value'][ this_set_range_indices[0]:this_set_range_indices[-1]+1 ] = deepcopy(new_value_list_rounded)
                                
                        #--------------------------------------------------------------------#
                    #--------------------------------------------------------------------#
            fut_id += 1

    print( '    We have finished the experiment and inheritance' )
    #
    time_list = []
    #
    scenario_list_print = [scen for scen in scenario_list if scen != '__pycache__']

    # Before printing the experiment dictionary, be sure to add future 0:
    experiment_dictionary[1]['Futures'] = [0] + experiment_dictionary[1]['Futures']
    experiment_dictionary[1]['Values'] = [3] + experiment_dictionary[1]['Values']
    
    # Save the dictionary in chunks by scenario and future into Pickle files
    directory_path = "data_inherited_scenarios"
    max_x_per_iter = int(setup_table.loc[0, 'Parallel_Use'])  # Get the number of scenarios per file
    items = list(inherited_scenarios.items())  # Convert inherited_scenarios to a list of items (scenarios)
    total_items = len(items)  # Get the total number of items (scenarios)
    num_files = math.ceil(total_items / max_x_per_iter)  # Calculate the number of files needed
    
    # Iterate through the chunks of scenarios and save each part into a separate file
    for idx in range(num_files):
        start_idx = idx * max_x_per_iter  # Calculate the start index for the chunk of scenarios
        end_idx = min(start_idx + max_x_per_iter, total_items)  # Calculate the end index for the chunk of scenarios
        chunk = dict(items[start_idx:end_idx])  # Create a chunk of the dictionary for the scenarios
        
        # Now we separate each scenario into its futures and save them
        for scenario, futures_dict in chunk.items():  # Iterate over each scenario
            # Convert futures_dict into items (futures)
            items_2 = list(futures_dict.items())  # Convert the futures of each scenario into a list of items
            total_futures = len(items_2)  # Get the total number of futures
            num_futures_files = math.ceil(total_futures / max_x_per_iter)  # Calculate how many files are needed for futures
            
            # Iterate through the chunks of futures for this scenario and save each part into a separate file
            for f_idx in range(num_futures_files):
                start_f_idx = f_idx * max_x_per_iter  # Calculate the start index for the chunk of futures
                end_f_idx = min(start_f_idx + max_x_per_iter, total_futures)  # Calculate the end index for the chunk of futures
                futures_chunk = {scenario:dict(items_2[start_f_idx:end_f_idx])}  # Create a chunk of the dictionary for the futures
                
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                    print(f"Directory created: {directory_path}")
                # Save the chunk of futures into a Pickle file
                file_name = f"data_inherited_scenarios/{scenario}_futures_part_{f_idx + 1}.pkl"  # Define the file name with the scenario and future part
                with open(file_name, 'wb') as f:
                    pickle.dump(futures_chunk, f)  # Save the futures chunk into the file
                print(f"Saved: {file_name}")

    # Empty the original dictionary to free memory
    inherited_scenarios.clear()  # Clear the original dictionary to free memory
    print("The 'inherited_scenarios' dictionary has been cleared.")
               


    if generator_or_executor == 'Generator' or generator_or_executor == 'Both':
        print('4: We will now print the input .txt files of diverse future scenarios.')
    
        print_address = './Experimental_Platform/Futures/'
    
        
    
        if parallel_or_linear == 'Parallel':
            print('Entered Parallelization')
            
            for fut_id_new in range(len(scenario_list_print)):
                # Collect the elements needed for parallelization
                # Create a DataFrame
                list_param_default_value = pd.DataFrame(dict_rows[scenario_list[fut_id_new]])
                S_DICT_params_structure = dict_S_DICT_params_structure[scenario_list[fut_id_new]]
                S_DICT_sets_structure = dict_S_DICT_sets_structure[scenario_list[fut_id_new]]
                packaged_useful_elements = [
                    scenario_list_print,
                    S_DICT_sets_structure,
                    S_DICT_params_structure,
                    list_param_default_value,
                    print_address,
                    all_futures,
                    time_range_vector,
                    dict_parameters_in_the_model[scenario_list_print[fut_id_new]],
                    dict_parameters_without_values[scenario_list_print[fut_id_new]],
                    dict_special_sets[scenario_list_print[fut_id_new]],
                ]

                x = len(all_futures)
                max_x_per_iter = int(setup_table.loc[0, 'Parallel_Use'])  # Number of futures per file
                y = x / max_x_per_iter
                y_ceil = math.ceil(y)
        
                # Iterate over the chunks based on the number of scenarios
                for n in range(0, y_ceil):
                    n_ini = n * max_x_per_iter  # Start index for each chunk
                    processes = []
                    start1 = time.time()
        
                    if n_ini + max_x_per_iter <= x:
                        max_iter = n_ini + max_x_per_iter
                    else:
                        max_iter = x
        
                    # Process each future within the current chunk
                    for n2 in range(n_ini, max_iter):
                        fut_index = n2
                        fut = all_futures[fut_index]
        
                        # Load the appropriate chunk of the 'inherited_scenarios' based on scenario and future
                        file_name = (
                            f"data_inherited_scenarios/{scenario_list_print[fut_id_new]}"
                            f"_futures_part_{(fut_index // max_x_per_iter) + 1}.pkl"
                        )
                        with open(file_name, 'rb') as f:
                            inherited_scenarios = pickle.load(f)  # Load the required part of the scenarios and futures

                        # Call the function to process each future and scenario
                        if scenario_list_print[fut_id_new] in scenario_list:
                            p = mp.Process(target=function_C_mathprog_parallel, args=(n2, fut_id_new, inherited_scenarios, packaged_useful_elements, num_time_slices_SDP))
                            processes.append(p)
                            p.start()
                        else:
                            print(f'!!! At generation, we skip: future {fut} and scenario {scenario_list[fut_id_new]} !!!')
        
                    # Wait for all processes to finish
                    for process in processes:
                        process.join()
        
                    end_1 = time.time()
                    time_elapsed_1 = end_1 - start1
                    print(f"Time elapsed for chunk {n+1}: {time_elapsed_1} seconds")
                    time_list.append(time_elapsed_1)
    
            print(f"The total time for printing input .txt files: {sum(time_list)} seconds")
    
        if parallel_or_linear == 'Linear':
            print('Started Linear Runs')
            for fut_id_new in range(len(scenario_list_print)):
                x = len(all_futures)
                for n in range(x):
                    function_C_mathprog_parallel(n, fut_id_new, inherited_scenarios, packaged_useful_elements, num_time_slices_SDP)


    #########################################################################################
    #
    if generator_or_executor == 'Executor' or generator_or_executor == 'Both':
        #
        # Empty the original dictionary to free memory
        inherited_scenarios.clear()  # Clear the original dictionary to free memory
        print("The 'inherited_scenarios' dictionary has been cleared.")
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            shutil.rmtree(directory_path)
            print(f"Directory deleted: {directory_path}")
        #
        print('5: We will produce the outputs and store the data.')
        #
        for a_scen in range( len( scenario_list_print ) ):
            #
            Executed_Scenario = scenario_list_print[ a_scen ]
            set_first_list(Executed_Scenario)
            #
            x = len(first_list)
            #
            y = x / max_x_per_iter
            y_ceil = math.ceil( y )
            
            for n in range(0,y_ceil):
                print('###')
                n_ini = n*max_x_per_iter
                processes = []
                #
                start1 = time.time()
                #
                if n_ini + max_x_per_iter <= x:
                    max_iter = n_ini + max_x_per_iter
                else:
                    max_iter = x
                #
                for n2 in range( n_ini , max_iter ):

                    p = mp.Process(target=main_executer, args=(n2,Executed_Scenario,time_range_vector,scenario_list_print,solver,osemosys_model,parameters_to_print) )
                    processes.append(p)
                    p.start()
                #
                for process in processes:
                    process.join()
                #
                end_1 = time.time()   
                time_elapsed_1 = -start1 + end_1
                print( str( time_elapsed_1 ) + ' seconds' )
                time_list.append( time_elapsed_1 )
                #
            #
        #
    #
    print('   The total time producing outputs and storing data has been: ' + str( sum( time_list ) ) + ' seconds')
    print( 'For all effects, this has been the end. It all took: ' + str( sum( time_list ) ) + ' seconds')
    
