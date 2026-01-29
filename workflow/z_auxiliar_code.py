# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:00:45 2024

@author: IgnacioAlfaroCorrale
"""

from copy import deepcopy
import csv
import gc
import linecache
import math
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow
import xlsxwriter


def interpolation_multiplier(
    time_list, value_list, new_relative_final_value, Initial_Year_of_Uncertainty
):
    """Scale a time series so the multiplier increases linearly to 2050."""
    new_value_list = []
    initial_year_index = time_list.index(Initial_Year_of_Uncertainty)
    target_2050_increment = 2050 - Initial_Year_of_Uncertainty
    total_2050_increment = new_relative_final_value - 1
    delta_increment = total_2050_increment / target_2050_increment
    multiplier_list = [1] * len(time_list)

    for n in range(len(time_list)):
        if n > initial_year_index and time_list[n] < 2050:
            multiplier_list[n] = delta_increment + multiplier_list[n - 1]
        elif time_list[n] >= 2050:
            multiplier_list[n] = new_relative_final_value

    for n in range(len(time_list)):
        new_value_list.append(float(value_list[n]) * multiplier_list[n])

    return new_value_list

def intersection(lst1, lst2):
    """Return the intersection of two lists while preserving lst1 order."""
    return [value for value in lst1 if value in lst2]

def interpolation_non_linear_final(
    time_list, value_list, new_relative_final_value, finyear, Initial_Year_of_Uncertainty
):
    """Adjust a time series to reach a new final value with non-linear scaling."""
    old_relative_final_value = 1
    new_value_list = []
    initial_year_index = time_list.index(Initial_Year_of_Uncertainty)
    fraction_time_list = time_list[initial_year_index:]
    fraction_value_list = value_list[initial_year_index:]

    # Subtract the time between the last year and the finyear.
    diff_yrs = time_list[-1] - finyear

    xdata = [
        fraction_time_list[i] - fraction_time_list[0]
        for i in range(len(fraction_time_list) - diff_yrs)
    ]
    ydata = [
        float(fraction_value_list[i])
        for i in range(len(fraction_value_list) - diff_yrs)
    ]
    ydata_whole = [float(value) for value in fraction_value_list]
    delta_ydata = [
        ydata_whole[i] - ydata_whole[i - 1] for i in range(1, len(ydata_whole))
    ]
    m_original = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    m_new = (
        ydata[-1] * (new_relative_final_value / old_relative_final_value) - ydata[0]
    ) / (xdata[-1] - xdata[0])

    if int(m_original) == 0:
        delta_ydata_new = [m_new for _ in range(len(ydata_whole))]
    else:
        delta_ydata_new = [
            (m_new / m_original) * (ydata_whole[i] - ydata_whole[i - 1])
            for i in range(1, len(ydata_whole))
        ]
        delta_ydata_new = [0] + delta_ydata_new

    ydata_new = [0 for _ in range(len(ydata_whole))]
    list_apply_delta_ydata_new = []

    for i in range(len(delta_ydata) + 1):
        if time_list[i + initial_year_index] <= finyear:
            apply_delta_ydata_new = delta_ydata_new[i]
        else:
            apply_delta_ydata_new = sum(delta_ydata_new) / len(delta_ydata_new)
        list_apply_delta_ydata_new.append(apply_delta_ydata_new)

        if i == 0:
            ydata_new[i] = ydata_whole[0] + apply_delta_ydata_new
        else:
            ydata_new[i] = ydata_new[i - 1] + apply_delta_ydata_new

    fraction_list_counter = 0
    for n in range(len(time_list)):
        if time_list[n] >= Initial_Year_of_Uncertainty:
            new_value_list.append(ydata_new[fraction_list_counter])
            fraction_list_counter += 1
        else:
            new_value_list.append(float(value_list[n]))

    return new_value_list


def interpolation_non_linear_initial(time_list, value_list, new_relative_initial_value):
    """Adjust a time series to a new initial value with non-linear scaling."""
    old_relative_final_value = 1
    xdata = [time_list[i] - time_list[0] for i in range(len(time_list))]
    ydata = value_list
    delta_ydata = [ydata[i] - ydata[i - 1] for i in range(1, len(ydata))]
    m_original = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    m_new = (
        ydata[-1] - ydata[0] * (new_relative_initial_value / old_relative_final_value)
    ) / (xdata[-1] - xdata[0])

    if float(m_original) == 0.0:
        delta_ydata_new = [m_new for _ in range(1, len(ydata))]
    else:
        delta_ydata_new = [
            (m_new / m_original) * (ydata[i] - ydata[i - 1])
            for i in range(1, len(ydata))
        ]

    ydata_new = [0 for _ in range(len(ydata))]
    ydata_new[0] = ydata[0] * new_relative_initial_value
    for i in range(len(delta_ydata)):
        ydata_new[i + 1] = ydata_new[i] + delta_ydata_new[i]

    new_value_list = ydata_new
    return new_value_list

def interpolation_constant_trajectory(time_list, value_list, initial_year_of_uncertainty):
    """
    This function generates a new list of values where, starting from the
    'initial_year_of_uncertainty', all values are held constant at the level
    of that specific year.

    Parameters:
    - time_list (list of int): List of years.
    - value_list (list of float): Corresponding list of values.
    - initial_year_of_uncertainty (int): Year from which values will remain constant.

    Returns:
    - new_value_list (list of float): Modified list with constant values after the threshold year.
    """
    new_value_list = []
    constant_value = None

    for year, value in zip(time_list, value_list):
        if year < initial_year_of_uncertainty:
            new_value_list.append(value)
        else:
            if constant_value is None:
                constant_value = value  # Freeze the value at the uncertainty year
            new_value_list.append(constant_value)

    return new_value_list

def interpolation_logistic_trajectory(
    time_list,
    value_list,
    Values_per_Future,
    last_year_analysis,
    Initial_Year_of_Uncertainty
):
    """
    Applies a logistic interpolation to the value_list starting from Initial_Year_of_Uncertainty
    until last_year_analysis. Earlier values remain unchanged.

    Parameters:
    - time_list: list of years (integers)
    - value_list: list of float values corresponding to time_list
    - Values_per_Future: dictionary with float multipliers for each future
    - last_year_analysis: final year of the analysis
    - Initial_Year_of_Uncertainty: year where the logistic behavior starts

    Returns:
    - new_value_list: updated list with logistic interpolation applied
    """

    # Check if last_year_analysis is in time_list
    if last_year_analysis not in time_list:
        raise ValueError("last_year_analysis must be present in time_list.")

    # Create a copy of value_list to modify
    new_value_list = value_list.copy()

    # Get indices for interpolation range
    start_index = time_list.index(Initial_Year_of_Uncertainty)
    end_index = time_list.index(last_year_analysis)

    # Initial and final values for logistic curve
    initial_value = value_list[start_index]
    final_value = value_list[-1] * Values_per_Future

    # Length of the logistic portion
    num_steps = end_index - start_index + 1

    # Define logistic parameters
    L = final_value
    k = 1.0  # growth rate (can be adjusted if needed)
    x0 = num_steps / 2  # midpoint of the curve

    # Generate logistic growth curve values
    for i in range(num_steps):
        t = i
        logistic_factor = 1 / (1 + math.exp(-k * (t - x0)))
        interpolated_value = initial_value + (L - initial_value) * logistic_factor
        new_value_list[start_index + i] = interpolated_value

    return new_value_list

def interpolation_linear(time_list, value_list, new_relative_final_value, finyear, Initial_Year_of_Uncertainty):
    """
    Interpolates values linearly from Initial_Year_of_Uncertainty to finyear, keeping
    all previous values intact. Ensures a straight-line behavior to the new final value.

    Parameters:
    - time_list: List of years
    - value_list: List of original float values
    - new_relative_final_value: Multiplier to adjust the final value
    - finyear: Final year for the linear interpolation
    - Initial_Year_of_Uncertainty: Year where the linear behavior starts

    Returns:
    - new_value_list: List with linear interpolation applied
    """

    # Validate that finyear is in the time_list
    if finyear not in time_list:
        raise ValueError("finyear must be present in time_list.")

    # Copy the input list
    new_value_list = []

    # Find index boundaries
    start_index = time_list.index(Initial_Year_of_Uncertainty)
    end_index = time_list.index(finyear)

    # Get initial and final values
    initial_value = value_list[start_index]
    final_value = value_list[-1] * new_relative_final_value

    # Total steps for linear interpolation
    num_steps = end_index - start_index

    # Compute linear interpolated values
    for i, year in enumerate(time_list):
        if i < start_index:
            new_value_list.append(value_list[i])
        elif i <= end_index:
            # Linear interpolation formula
            t = i - start_index
            interpolated = initial_value + (final_value - initial_value) * (t / num_steps)
            new_value_list.append(interpolated)
        else:
            # From finyear onwards, keep value constant
            new_value_list.append(final_value)

    return new_value_list


def time_series_shift(time_list, value_list, new_relative_initial_value):
    """Shift an entire time series by adjusting the initial value."""
    new_value_list = []
    new_initial_value = value_list[0] * new_relative_initial_value
    shift_value = new_initial_value - value_list[0]

    for n in range(len(time_list)):
        new_value_list.append(value_list[n] + shift_value)

    return new_value_list


def dc_shift(time_list, value_list, new_relative_initial_value):
    """Apply a proportional shift to non-zero values in a time series."""
    new_value_list = []

    for t in range(len(time_list)):
        if float(value_list[t]) == 0.0:
            new_value_list.append(0.0)
        else:
            new_value_list.append(round(value_list[t] * new_relative_initial_value, 4))

    return new_value_list


def year_when_reaches_zero(time_list, value_list, ywrz, Initial_Year_of_Uncertainty):
    """Interpolate a technology trajectory to reach zero in year ywrz."""
    new_value_list = []
    x_coord_tofill = []
    xp_coord_known = []
    fp_coord_known = []

    original_shares = [100 * value_list[n] / value_list[0] for n in range(len(value_list))]
    original_shares_add = []
    for n in range(len(original_shares)):
        if time_list[n] <= Initial_Year_of_Uncertainty:
            fp_coord_known.append(original_shares[n])
            original_shares_add.append(original_shares[n])
    fp_coord_known.append(0)

    years_with_value_different_from_zero = [
        n for n in range(time_list[0], int(ywrz) + 1)
    ]
    for n in range(len(years_with_value_different_from_zero)):
        if (
            years_with_value_different_from_zero[n]
            <= Initial_Year_of_Uncertainty
            or years_with_value_different_from_zero[n] == ywrz
        ):
            xp_coord_known.append(n)
        else:
            x_coord_tofill.append(n)

    y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, fp_coord_known))
    percentage_list = original_shares_add + y_coord_filled + [0]

    for n in range(len(time_list)):
        if time_list[n] <= ywrz:
            new_value_list.append((percentage_list[n] / 100) * value_list[0])
        else:
            new_value_list.append(0.0)

    return new_value_list


def generalized_logistic_curve(x, L, Q, k, M):
    """Evaluate a generalized logistic curve."""
    return L / (1 + Q * math.exp(-k * (x - M)))


def logistic_curve_controlled(L, xM, C, xo, x):
    """Logistic curve with an inflection point controlled by xM."""
    k = np.log(L / C - 1) / (xo - xM)
    return L / (1 + math.exp(-k * (x - xo)))


def interpolation_blend(start_blend_point, final_blend_point, value_list, time_range_vector):
    """Blend values between two points using a linear interpolation of shares."""
    start_blend_year, start_blend_value = start_blend_point[0], start_blend_point[1] / 100
    final_blend_year, final_blend_value = final_blend_point[0], final_blend_point[1] / 100
    x_coord_tofill = []
    xp_coord_known = []
    fp_coord_known = []

    for t in range(len(time_range_vector)):
        needs_interpolation = False

        if time_range_vector[t] < start_blend_year:
            fp_coord_known.append(0.0)

        if time_range_vector[t] == start_blend_year:
            fp_coord_known.append(start_blend_value)

        if start_blend_year < time_range_vector[t] < final_blend_year:
            needs_interpolation = True

        if time_range_vector[t] >= final_blend_year:
            fp_coord_known.append(final_blend_value)

        if needs_interpolation:
            x_coord_tofill.append(t)
        else:
            xp_coord_known.append(t)

        y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, fp_coord_known))
        interpolated_values = []
        for coord in range(len(time_range_vector)):
            if coord in xp_coord_known:
                value_index = xp_coord_known.index(coord)
                interpolated_values.append(float(fp_coord_known[value_index]))
            elif coord in x_coord_tofill:
                value_index = x_coord_tofill.index(coord)
                interpolated_values.append(float(y_coord_filled[value_index]))

    new_value_list = []
    for n in range(len(value_list)):
        new_value_list.append(value_list[n] * (1 - interpolated_values[n]))
    new_value_list_rounded = [round(elem, 4) for elem in new_value_list]
    biofuel_shares = [round(elem, 4) for elem in interpolated_values]

    return new_value_list_rounded, biofuel_shares

def obtain_structure_file(
    input_scanario_file_name,
    output_structure_model_name,
    params_variables_file,
    num_time_slices_SDP,
):
    """Extract model structure from a scenario file and write it to Excel."""
    workbook = pd.ExcelFile(params_variables_file)
    sheet_primary = workbook.parse(workbook.sheet_names[0])
    sheet_secondary = workbook.parse(workbook.sheet_names[1])

    lines = []
    with open(input_scanario_file_name) as file_handle:
        for line in file_handle:
            if 'REGION' in line and line.strip().endswith(';'):
                stripped_line = line.strip()
                if not stripped_line.endswith(' ;'):
                    line = stripped_line.rstrip(';').rstrip() + ' ;'
            lines.append(line)

    subset_set_lines = [line for line in lines if 'set ' in line]
    set_map = {}
    for line in subset_set_lines:
        aux_list = line.split()
        for item in aux_list[3:-1]:
            set_map.setdefault(aux_list[1], []).append(item)

    sets = [
        'YEAR',
        'TECHNOLOGY',
        'TIMESLICE',
        'COMMODITY',
        'EMISSION',
        'MODE_OF_OPERATION',
        'REGION',
        'SEASON',
        'DAYTYPE',
        'DAILYTIMEBRACKET',
        'STORAGE',
        'STORAGEINTRADAY',
        'STORAGEINTRAYEAR',
        'UDC',
    ]
    sets_index = ['y', 't', 'l', 'f', 'e', 'm', 'r', 'ls', 'ld', 'lh', 's', 'sd', 'sy', 'u']

    for set_name in sets:
        set_map.setdefault(set_name, [])

    workbook_out = xlsxwriter.Workbook(output_structure_model_name)
    sheet = workbook_out.add_worksheet('Sets')
    sheet.write_column(0, 0, ['set', 'index', 'number'])
    sheet.write_row(0, 1, sets)
    sheet.write_row(1, 1, sets_index)

    for i, set_name in enumerate(sets):
        if set_name != 'TIMESLICE':
            sheet.write(2, i + 1, len(set_map[set_name]))
        else:
            sheet.write(2, i + 1, int(num_time_slices_SDP))

        for j, value in enumerate(set_map[set_name]):
            if set_name != 'TIMESLICE' and i == 0:
                sheet.write(3 + j, i + 1, int(value))
            elif set_name == 'TIMESLICE' and (j + 1) <= num_time_slices_SDP:
                sheet.write(3 + j, i + 1, value)
            elif set_name != 'TIMESLICE':
                sheet.write(3 + j, i + 1, value)

    sheet_primary_out = workbook_out.add_worksheet(workbook.sheet_names[0])
    headers = list(sheet_primary)
    for i, header in enumerate(headers):
        if 'Unnamed' not in header:
            sheet_primary_out.write(0, i, header)
        column = list(sheet_primary[header])
        for j, cell in enumerate(column):
            try:
                sheet_primary_out.write(j + 1, i, cell)
            except Exception:
                pass

    sheet_secondary_out = workbook_out.add_worksheet(workbook.sheet_names[1])
    headers = list(sheet_secondary)
    for i, header in enumerate(headers):
        if 'Unnamed' not in header:
            sheet_secondary_out.write(0, i, header)
        column = list(sheet_secondary[header])
        for j, cell in enumerate(column):
            try:
                sheet_secondary_out.write(j + 1, i, cell)
            except Exception:
                pass

    workbook_out.close()

    return set_map
    
    
def find_default_values(scenario_file_name, output_file_name, structure_default_file_name):
    matrix = []
    with open(scenario_file_name) as f:
        for line in f:
            matrix.append(line)

    name_params = []
    default_params = []
    for i in range(len(matrix)):
        if 'param' in matrix[i] and 'ResultsPath' not in matrix[i]:
            row = matrix[i].strip().split()
            row = matrix[i].split(" ")
            name_params.append(row[1])
            default_params.append(float(row[3]))

    default_params_by_scenario = dict(zip(name_params, default_params))

    workbook = pd.ExcelFile(structure_default_file_name)
    sheet_primary = workbook.parse(workbook.sheet_names[0])
    headers = list(sheet_primary)
    col1 = list(sheet_primary[headers[0]])
    col2 = list(sheet_primary[headers[1]])

    default_values = dict(zip(col1, col2))
    keys = list(default_values.keys())

    for i in range(len(keys)):
        if keys[i] in list(default_params_by_scenario.keys()):
            default_values[keys[i]] = default_params_by_scenario[keys[i]]

    wb = xlsxwriter.Workbook(output_file_name)
    sheet = wb.add_worksheet('Default_Values')
    sheet.write_row(0, 0, headers)
    for i in range(len(keys)):
        sheet.write(i + 1, 0, keys[i])
        sheet.write(i + 1, 1, round(default_values[keys[i]], 4))
    wb.close()


def isolate_params(scenario_file_name):
    data_per_param = {}
    matrix = []
    special_sets = []
    with open(scenario_file_name) as f:
        for line in f:
            matrix.append(line)
    name_params = []
    for i in range(len(matrix)):
        if 'param' in matrix[i] and 'ResultsPath' not in matrix[i]:
            row = matrix[i].split(" ")
            name_params.append(row[1])
        if 'set MODEx' in matrix[i] or 'set MODEper' in matrix[i]:
            special_sets.append(matrix[i])

    for i in range(len(name_params)):
        subset_matrix = []
        in_block = False
        for j in range(len(matrix)):
            if name_params[i] in matrix[j]:
                in_block = True
            if in_block and ';' not in matrix[j]:
                subset_matrix.append(matrix[j])
            
            if in_block and ';' in matrix[j]:
                break

        data_per_param[name_params[i]] = subset_matrix

    return data_per_param, special_sets
                
                
def generate_df_per_param(scenario_code_name, data_per_param, num_time_slices_SDP):
    # print(scenario_code_name)
    parameter_headers=['PARAMETER',
                      'Scenario',
                      'REGION',
                      'TECHNOLOGY',
                      'COMMODITY',
                      'EMISSION',
                      'MODE_OF_OPERATION',
                      'YEAR',
                      'TIMESLICE',
                      'SEASON',
                      'DAYTYPE',
                      'DAILYTIMEBRACKET',
                      'STORAGE',
                      'STORAGEINTRADAY',
                      'STORAGEINTRAYEAR',
                      'UDC',
                      'Value']
    # Store DataFrames
    list_dataframes = []
    dict_dataframes = {}
    parameters_without_values = []

    for param, value in data_per_param.items():
        matrix = value
        
        if len(matrix) == 1:
            line = matrix[0].strip()
            if line.startswith('param'):
                part = line.split()
                if len(part) > 1:
                    param_name = part[1]
                    parameters_without_values.append(param_name)
        else:

            
            ##################################
            # 1 ##############################
            ##################################
            if 'AccumulatedAnnualDemand' in param:

                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                fuel=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    fuel.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['AccumulatedAnnualDemand',scenario_code_name,region,'',fuel[j],'','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
            ##################################
            # 2 ##############################
            ##################################
            if 'AnnualEmissionLimit' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                emissions=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    emissions.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                            time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['AnnualEmissionLimit',scenario_code_name,region,'','',emissions[j],'',years[k],'','','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 3 ##############################
            ##################################
            if 'AvailabilityFactor' in param:
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['AvailabilityFactor',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                        
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
   
            ##################################
            # 4 ##############################
            ##################################
            if 'CapacityFactor' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_timeslices=list()
                separated_time_series=list()
                timeslice_mode_map={}
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        tech=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(tech)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_timeslices.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                        
                        if tech not in timeslice_mode_map:
                            timeslice_mode_map[tech] = {}
                        key = str(matrix[j].split(' ')[0])
                        timeslice_mode_map[tech][key]=aux
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        tech=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(tech)
                output_rows=list()
                num_time_slices_SDP_count=0
                for j in range(len(separated_techs)):
                    num_tech_time_slices = len(timeslice_mode_map[separated_techs[j]])
                    separated_timeslice_map = timeslice_mode_map[separated_techs[j]]
                    for separated_timeslice, data_series_temp in separated_timeslice_map.items():
                        for l in range(len(years)):
                            output_rows.append(['CapacityFactor',scenario_code_name,separated_regions[j],separated_techs[j],'','','',years[l],separated_timeslice,'','','','','','','',data_series_temp[l]])
                    num_time_slices_SDP_count += num_tech_time_slices
                
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 5 ##############################
            ##################################
            if 'CapacityOfOneTechnologyUnit' in param:
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['CapacityOfOneTechnologyUnit',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                        
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
            ##################################
            # 6 ##############################
            ##################################
            if 'CapacityToActivityUnit' in param:
    
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[2].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                            time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['CapacityToActivityUnit',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 7 ##############################
            ##################################
            if 'CapitalCost' == param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]

                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['CapitalCost',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 8 ##############################
            ##################################
            if 'DiscountRate' == param:
    
                region=list()
                time_series=list()
                for j in range(1,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['DiscountRate',scenario_code_name,region[j],'','','','','','','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 9 ##############################
            ##################################                   
            if 'DiscountRateIdv' == param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]

                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['DiscountRateIdv',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 10 #############################
            ##################################
            if 'EmissionActivityRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_emissions=list()
                separated_modes=list()
                separated_time_series=list()
                separated_mode_map={}
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1]
                        emission=matrix[j].split(',')[2]
                        separated_techs.append(technology)
                        separated_regions.append(region)
                        separated_emissions.append(emission)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                        if f'{technology}/{emission}' not in separated_mode_map:
                            separated_mode_map[f'{technology}/{emission}'] = {}
                        key = str(matrix[j].split(' ')[0])
                        separated_mode_map[f'{technology}/{emission}'][key]=aux
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1]
                        emission=matrix[j].split(',')[2]
                        separated_techs.append(technology)
                        separated_regions.append(region)
                        separated_emissions.append(emission)
                        a=1
                output_rows=list()
                for j in range(len(separated_techs)):
                    separated_modes_temp = separated_mode_map[f'{separated_techs[j]}/{separated_emissions[j]}']
                    for separated_mode_temp, data_series_temp in separated_modes_temp.items():
                        for l in range(len(years)):
                            output_rows.append(['EmissionActivityRatio',scenario_code_name,separated_regions[j],separated_techs[j],'',separated_emissions[j],separated_mode_temp,years[l],'','','','','','','','',data_series_temp[l]])

                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 11 #############################
            ##################################
            if 'EmissionsPenalty' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                emissions=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    emissions.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                            time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['EmissionsPenalty',scenario_code_name,region,'','',emissions[j],'',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
            ##################################
            # 12 #############################
            ##################################
            if 'FixedCost' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                        
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['FixedCost',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
    
            ##################################
            # 13 #############################
            ##################################
            if 'InputActivityRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_fuels=list()
                separated_modes=list()
                separated_time_series=list()
                separated_mode_map={}
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        separated_mode_map[f'{technology}/{fuel}']={}
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                        key = str(matrix[j].split(' ')[0])
                        separated_mode_map[f'{technology}/{fuel}'][key]=aux
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        separated_mode_map[f'{technology}/{fuel}']={}
                        a=1
                output_rows=list()
                for j in range(len(separated_techs)):
                    separated_modes_temp = separated_mode_map[f'{separated_techs[j]}/{separated_fuels[j]}']
                    for separated_mode_temp, data_series_temp in separated_modes_temp.items():
                        for l in range(len(years)):                   
                            output_rows.append(['InputActivityRatio',scenario_code_name,separated_regions[j],separated_techs[j],separated_fuels[j],'',separated_mode_temp,years[l],'','','','','','','','',data_series_temp[l]])
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df 

            ##################################
            # 14 #############################
            ##################################
            if 'InputToNewCapacityRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_fuels=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                            
                        if aux[len(aux)-1].replace('\n', '') == '':
                            aux.pop(len(aux)-1)
                        separated_time_series.append(aux)

                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        a=1
                output_rows=list()
                count_mode=0
                tech_temp = []
                for j in range(len(separated_techs)):
                    # for k in range(len(separated_fuels)):                            
                    for l in range(len(years)):
                        try:
                            output_rows.append(['InputToNewCapacityRatio',scenario_code_name,separated_regions[j],separated_techs[j],separated_fuels[j],'','',years[l],'','','','','','','','',separated_time_series[j][l]])
                        except IndexError as e:
                            pass
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df                    

            ##################################
            # 15 #############################
            ##################################
            if 'InputToTotalCapacityRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_fuels=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                            
                        if aux[len(aux)-1].replace('\n', '') == '':
                            aux.pop(len(aux)-1)
                        separated_time_series.append(aux)

                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        a=1
                output_rows=list()
                count_mode=0
                tech_temp = []
                for j in range(len(separated_techs)):
                    # for k in range(len(separated_fuels)):                            
                    for l in range(len(years)):
                        try:
                            output_rows.append(['InputToTotalCapacityRatio',scenario_code_name,separated_regions[j],separated_techs[j],separated_fuels[j],'','',years[l],'','','','','','','','',separated_time_series[j][l]])
                        except IndexError as e:
                            pass
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df     
    
            ##################################
            # 16 #############################
            ##################################
            if 'ModelPeriodEmissionLimit' in param:
                
                emision=matrix[1].split(' ')
                if emision[len(emision)-1]==':=\n':
                    emision=emision[0:len(matrix[1].split(' '))-1]
                emision[len(emision)-1]=emision[len(emision)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['ModelPeriodEmissionLimit',scenario_code_name,region[j],'','',emision[k],'','','','','','','','','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                    
            ##################################
            # 17 #############################
            ##################################
            if 'ModelPeriodExogenousEmission' in param:
                
                emision=matrix[1].split(' ')
                if emision[len(emision)-1]==':=\n':
                    emision=emision[0:len(matrix[1].split(' '))-1]
                emision[len(emision)-1]=emision[len(emision)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['ModelPeriodExogenousEmission',scenario_code_name,region[j],'','',emision[k],'','','','','','','','','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
    
            ##################################
            # 18 #############################
            ##################################
            if 'OperationalLife' == param:
                
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[1].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['OperationalLife',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
                        
            ##################################
            # 19 #############################
            ##################################
            if 'OutputActivityRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_fuels=list()
                separated_modes=list()
                separated_mode_map={}
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        separated_mode_map[f'{technology}/{fuel}']=[]
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        separated_mode_map[f'{technology}/{fuel}'].append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        separated_mode_map[f'{technology}/{fuel}']=[]
                        a=1
                output_rows=list()
                for j in range(len(separated_techs)):
                    separated_mode_temp = separated_mode_map[f'{separated_techs[j]}/{separated_fuels[j]}']
                    for k in range(len(separated_mode_temp)):
                        for l in range(len(years)):
                            output_rows.append(['OutputActivityRatio',scenario_code_name,separated_regions[j],separated_techs[j],separated_fuels[j],'',separated_mode_temp[k],years[l],'','','','','','','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
                        
            ##################################
            # 20 #############################
            ##################################
            if 'ReserveMargin' == param.replace('.txt',''):
                
                years=matrix[1].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[1].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                reg=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    reg.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                            time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['ReserveMargin',scenario_code_name,reg[j],'','','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 21 #############################
            ##################################
            if 'ReserveMarginTagFuel' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                
                fuel=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    fuel.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                            time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['ReserveMarginTagFuel',scenario_code_name,region,'',fuel[j],'','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
            ##################################
            # 22 #############################
            ##################################
            if 'ReserveMarginTagTechnology' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['ReserveMarginTagTechnology',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
    
            ##################################
            # 23 #############################
            ##################################
            if 'ResidualCapacity' in param:
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['ResidualCapacity',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
    
            ##################################
            # 24 #############################
            ##################################
            if 'SpecifiedAnnualDemand' in param:
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                fuel=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    fuel.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['SpecifiedAnnualDemand',scenario_code_name,region,'',fuel[j],'','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
                        
            ##################################
            # 25 #############################
            ##################################
            if 'SpecifiedDemandProfile' in param:
                
                separated_fuels=list()
                separated_regions=list()
                separated_timeslices=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        fuel=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_fuels.append(fuel)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_timeslices.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        fuel=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_fuels.append(fuel)
                output_rows=list()
                num_time_slices_SDP_count=0
                for j in range(len(separated_fuels)):
                    for k in range(num_time_slices_SDP_count,num_time_slices_SDP+num_time_slices_SDP_count):
                        for l in range(len(years)):
                            output_rows.append(['SpecifiedDemandProfile',scenario_code_name,separated_regions[j],'',separated_fuels[j],'','',years[l],separated_timeslices[k],'','','','','','','',separated_time_series[k][l]])
                    num_time_slices_SDP_count += num_time_slices_SDP
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 26 #############################
            ##################################
            if 'TechnologyActivityByModeLowerLimit' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(years)):
                            output_rows.append(['TechnologyActivityByModeLowerLimit',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],years[l],'','','','','','','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 27 #############################
            ##################################
            if 'TechnologyActivityByModeUpperLimit' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(years)):
                            output_rows.append(['TechnologyActivityByModeUpperLimit',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],years[l],'','','','','','','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                    
            ##################################
            # 28 #############################
            ##################################
            if 'TechnologyActivityDecreaseByModeLimit' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(years)):
                            output_rows.append(['TechnologyActivityDecreaseByModeLimit',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],years[l],'','','','','','','','',separated_time_series[(1*j)+k][l]])

            ##################################
            # 29 #############################
            ##################################
            if 'TechnologyActivityIncreaseByModeLimit' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(years)):
                            output_rows.append(['TechnologyActivityIncreaseByModeLimit',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],years[l],'','','','','','','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                  
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 30 #############################
            ##################################
            if 'TotalAnnualMaxCapacity' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalAnnualMaxCapacity',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
    
            ##################################
            # 31 #############################
            ##################################
            if 'TotalAnnualMaxCapacityInvestment' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalAnnualMaxCapacityInvestment',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 32 #############################
            ##################################
            if 'TotalAnnualMinCapacity' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalAnnualMinCapacity',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 33 #############################
            ##################################
            if 'TotalAnnualMinCapacityInvestment' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalAnnualMaxCapacityInvestment',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
   
            ##################################
            # 34 #############################
            ##################################
            if 'TotalTechnologyAnnualActivityLowerLimit' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalTechnologyAnnualActivityLowerLimit',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
                        
            ##################################
            # 35 #############################
            ##################################
            if 'TotalTechnologyAnnualActivityUpperLimit' == param.replace('.txt',''):
                
                region=matrix[1].split(',')[0].replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                tech=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    tech.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalTechnologyAnnualActivityUpperLimit',scenario_code_name,region,tech[j],'','','',years[k],'','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                        
            ##################################
            # 36 #############################
            ##################################
            if 'TotalTechnologyModelPeriodActivityLowerLimit' in param:
    
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[2].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalTechnologyModelPeriodActivityLowerLimit',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                    
            ##################################
            # 37 #############################
            ##################################
            if 'TotalTechnologyModelPeriodActivityUpperLimit' in param:
    
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[2].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['TotalTechnologyModelPeriodActivityUpperLimit',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 38 #############################
            ##################################
            # if 'TradeRoute' in param:
    
            ##################################
            # 39 #############################
            ##################################
            if 'VariableCost' in param:
            	
            	separated_techs=list()
            	separated_regions=list()
            	separated_modes=list()
            	separated_time_series=list()
            	separated_mode_map={}
            	a=0
            	for j in range(1,len(matrix)):
            		if '[' in matrix[j] and a==0:
            			a=1
            			region=matrix[j].split(',')[0].replace(',','').replace('[','')
            			technology=matrix[j].split(',')[1].replace(',','')
            			separated_regions.append(region)
            			separated_techs.append(technology)
            		elif a==1:
            			years=matrix[j].split(' ')
            			if years[len(years)-1]==':=\n':
            				years=years[0:len(matrix[j].split(' '))-1]
            			years[len(years)-1]=years[len(years)-1]
            			a=2
            		elif a==2 and '[' not in matrix[j]:
            			separated_modes.append(matrix[j].split(' ')[0])
            			aux=matrix[j].split(' ')[1:]
            			aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
            			separated_time_series.append(aux)
            			if technology not in separated_mode_map:
            				separated_mode_map[technology] = {}
            			key = str(matrix[j].split(' ')[0])
            			separated_mode_map[technology][key]=aux
            		elif a==2 and '[' in matrix[j]:
            			region=matrix[j].split(',')[0].replace(',','').replace('[','')
            			technology=matrix[j].split(',')[1].replace(',','')
            			a=1
            			separated_regions.append(region)
            			separated_techs.append(technology)
            	output_rows=list()
            	for j in range(len(separated_techs)):
            		separated_modes_temp = separated_mode_map[separated_techs[j]]
            		for separated_mode_temp, data_series_temp in separated_modes_temp.items():
            		# for k in range(0,1):
            			for l in range(len(years)):
            				output_rows.append(['VariableCost',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_mode_temp,years[l],'','','','','','','','',data_series_temp[l]])
            
            	# Store data
            	if output_rows:
            		df = pd.DataFrame(output_rows, columns=parameter_headers)
            		df=df.rename(columns={'Value': param})
            		list_dataframes.append(df)
            		dict_dataframes[f'{param}'] = df
                    
            ##################################
            # 40 #############################
            ##################################
            if 'YearSplit' in param:
                
                years=matrix[1].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                TS=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    TS.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['YearSplit',scenario_code_name,'','','','','',years[k],TS[j],'','','','','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 41 #############################
            ##################################
            if 'EmissionToActivityChangeRatio' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_fuels=list()
                separated_modes=list()
                separated_time_series=list()
                separated_mode_tech={}
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                    elif a==1:
                        years=matrix[j].split(' ')
                        if years[len(years)-1]==':=\n':
                            years=years[0:len(matrix[j].split(' '))-1]
                        years[len(years)-1]=years[len(years)-1]
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                            
                        if aux[len(aux)-1].replace('\n', '') == '':
                            aux.pop(len(aux)-1)
                        separated_time_series.append(aux)

                        if technology not in separated_mode_tech:
                            separated_mode_tech[technology] = [matrix[j].split(' ')[0]]
                        else:
                            separated_mode_tech[technology].append(matrix[j].split(' ')[0])
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        fuel=matrix[j].split(',')[2].replace(',','').replace('[','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                        separated_fuels.append(fuel)
                        a=1
                output_rows=list()
                count_mode=0
                tech_temp = []
                for j in range(len(separated_techs)):
                    mode_per_tech = separated_mode_tech[separated_techs[j]]
                    for k in range(len(mode_per_tech)):
                        if k > 0 and len(tech_temp) < len(mode_per_tech)-1:
                            count_mode+=1
                            tech_temp.append(separated_techs[j])
                            
                        for l in range(len(years)):
                            try:
                                output_rows.append(['EmissionToActivityChangeRatio',scenario_code_name,separated_regions[j],separated_techs[j],separated_fuels[j],'',mode_per_tech[k],years[l],'','','','','','','','',separated_time_series[j+count_mode][l]])
                            except IndexError as e:
                                pass
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 42 #############################
            #################################
            if 'UDCMultiplierTotalCapacity' in param:
                separated_techs = list()
                separated_regions = list()
                separated_udc = list()
                separated_time_series = list()
                separated_udc_map = {}
                a = 0
            
                for j in range(1, len(matrix)):
                    if '[' in matrix[j] and a == 0:
                        a = 1
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
            
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                    elif a == 1:
                        years = matrix[j].split(' ')
                        if years[-1] == ':=\n':
                            years = years[:-1]
                        years[-1] = years[-1]
                        a = 2
            
                    elif a == 2 and '[' not in matrix[j]:
                        separated_udc.append(matrix[j].split(' ')[0])
                        aux = matrix[j].split(' ')[1:]
                        aux[-1] = aux[-1].replace('\n', '')
            
                        separated_time_series.append(aux)
            
                        if technology not in separated_udc_map:
                            separated_udc_map[technology] = {}
            
                        key = str(matrix[j].split(' ')[0])
                        separated_udc_map[technology][key] = aux
            
                    elif a == 2 and '[' in matrix[j]:
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
                        a = 1
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                output_rows = list()
                separated_techs=list(set(separated_techs))

                for j in range(len(separated_techs)):
                    separated_udcs_temp = separated_udc_map[separated_techs[j]]
                    for separated_udc_temp, data_series_temp in separated_udcs_temp.items():
                        for l in range(len(years)):
                            output_rows.append([
                                'UDCMultiplierTotalCapacity',
                                scenario_code_name,
                                separated_regions[j],
                                separated_techs[j],
                                '', '', '',
                                years[l],
                                '', '', '', '', '', '', '',
                                separated_udc_temp,
                                data_series_temp[l]
                            ])
            
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df = df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                    
            ##################################
            # Bonus ##########################
            #################################
            
            if 'UDCMultiplierDemand' in param:
                # Parse blocks of the form:
                #   [REGION, COMMODITY] ...
                #   <years> :=
                #   <UDC> <v1> <v2> ...
                # ensuring COMMODITY goes into the COMMODITY column (not TECHNOLOGY),
                # and preserving the (REGION, COMMODITY) pairing.

                a = 0
                current_key = None  # (region, commodity)
                years = []
                separated_udc_map = {}  # {(region, commodity): {udc: [values...] } }

                for j in range(1, len(matrix)):
                    line = matrix[j]

                    # Header line with indices
                    if '[' in line and a == 0:
                        a = 1
                        region = line.split(',')[0].replace(',', '').replace('[', '').strip()
                        fuel = line.split(',')[1].replace(',', '').replace(']', '').strip()
                        current_key = (region, fuel)
                        if current_key not in separated_udc_map:
                            separated_udc_map[current_key] = {}

                    # Years line
                    elif a == 1:
                        years = line.split(' ')
                        if years and years[-1] == ':=\n':
                            years = years[:-1]
                        a = 2

                    # Data lines (until next header)
                    elif a == 2 and '[' not in line:
                        parts = line.split(' ')
                        udc_name = str(parts[0]).strip()
                        aux = parts[1:]
                        if aux:
                            aux[-1] = aux[-1].replace('\n', '')
                        if current_key is not None:
                            separated_udc_map[current_key][udc_name] = aux

                    # Next header starts a new block
                    elif a == 2 and '[' in line:
                        region = line.split(',')[0].replace(',', '').replace('[', '').strip()
                        fuel = line.split(',')[1].replace(',', '').replace(']', '').strip()
                        current_key = (region, fuel)
                        if current_key not in separated_udc_map:
                            separated_udc_map[current_key] = {}
                        a = 1

                output_rows = []

                # Write rows: TECHNOLOGY blank, COMMODITY filled with fuel
                for (region, fuel), udcs_dict in separated_udc_map.items():
                    for udc_name, data_series_temp in udcs_dict.items():
                        for l in range(len(years)):
                            output_rows.append([
                                'UDCMultiplierDemand',
                                scenario_code_name,
                                region,
                                '',            # TECHNOLOGY (blank)
                                fuel,          # COMMODITY
                                '',            # EMISSION
                                '',            # MODE_OF_OPERATION
                                years[l],       # YEAR
                                '', '', '', '', '', '', '',  # TIMESLICE..STORAGEINTRAYEAR
                                udc_name,       # UDC
                                data_series_temp[l]  # Value
                            ])

                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df = df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df


            ##################################
            # 43 #############################
            ##################################
            if 'UDCMultiplierNewCapacity' in param:
                separated_techs = list()
                separated_regions = list()
                separated_udc = list()
                separated_time_series = list()
                separated_udc_map = {}
                a = 0
            
                for j in range(1, len(matrix)):
                    if '[' in matrix[j] and a == 0:
                        a = 1
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
            
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                    elif a == 1:
                        years = matrix[j].split(' ')
                        if years[-1] == ':=\n':
                            years = years[:-1]
                        years[-1] = years[-1]
                        a = 2
            
                    elif a == 2 and '[' not in matrix[j]:
                        separated_udc.append(matrix[j].split(' ')[0])
                        aux = matrix[j].split(' ')[1:]
                        aux[-1] = aux[-1].replace('\n', '')
            
                        separated_time_series.append(aux)
            
                        if technology not in separated_udc_map:
                            separated_udc_map[technology] = {}
            
                        key = str(matrix[j].split(' ')[0])
                        separated_udc_map[technology][key] = aux
            
                    elif a == 2 and '[' in matrix[j]:
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
                        a = 1
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                output_rows = list()
                separated_techs=list(set(separated_techs))
            
                for j in range(len(separated_techs)):
                    separated_udcs_temp = separated_udc_map[separated_techs[j]]
                    for separated_udc_temp, data_series_temp in separated_udcs_temp.items():
                        for l in range(len(years)):
                            output_rows.append([
                                'UDCMultiplierNewCapacity',
                                scenario_code_name,
                                separated_regions[j],
                                separated_techs[j],
                                '', '', '',
                                years[l],
                                '', '', '', '', '', '', '',
                                separated_udc_temp,
                                data_series_temp[l]
                            ])
            
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df = df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 44 #############################
            ##################################
            if 'UDCMultiplierActivity' in param:
                separated_techs = list()
                separated_regions = list()
                separated_udc = list()
                separated_time_series = list()
                separated_udc_map = {}
                a = 0
            
                for j in range(1, len(matrix)):
                    if '[' in matrix[j] and a == 0:
                        a = 1
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
            
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                    elif a == 1:
                        years = matrix[j].split(' ')
                        if years[-1] == ':=\n':
                            years = years[:-1]
                        years[-1] = years[-1]
                        a = 2
            
                    elif a == 2 and '[' not in matrix[j]:
                        separated_udc.append(matrix[j].split(' ')[0])
                        aux = matrix[j].split(' ')[1:]
                        aux[-1] = aux[-1].replace('\n', '')
            
                        separated_time_series.append(aux)
            
                        if technology not in separated_udc_map:
                            separated_udc_map[technology] = {}
            
                        key = str(matrix[j].split(' ')[0])
                        separated_udc_map[technology][key] = aux
            
                    elif a == 2 and '[' in matrix[j]:
                        region = matrix[j].split(',')[0].replace(',', '').replace('[', '')
                        technology = matrix[j].split(',')[1].replace(',', '')
                        a = 1
                        separated_regions.append(region)
                        separated_techs.append(technology)
            
                output_rows = list()
                separated_techs=list(set(separated_techs))
            
                for j in range(len(separated_techs)):
                    separated_udcs_temp = separated_udc_map[separated_techs[j]]
                    for separated_udc_temp, data_series_temp in separated_udcs_temp.items():
                        for l in range(len(years)):
                            output_rows.append([
                                'UDCMultiplierActivity',
                                scenario_code_name,
                                separated_regions[j],
                                separated_techs[j],
                                '', '', '',
                                years[l],
                                '', '', '', '', '', '', '',
                                separated_udc_temp,
                                data_series_temp[l]
                            ])
            
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df = df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            #45 ##############################
            ##################################
            if 'UDCConstant' == param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]

                
                udc=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    udc.append(matrix[j].split(' ')[0])
                    
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)

                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['UDCConstant',scenario_code_name,region,'','','','',years[k],'','','','','','','',udc[j],time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                       
            ##################################
            # 46 #############################
            ##################################
            if 'UDCTag' == param:
                
                udc=matrix[1].split(' ')
                if udc[len(udc)-1]==':=\n':
                    udc=udc[0:len(matrix[1].split(' '))-1]
                udc[len(udc)-1]=udc[len(udc)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['UDCTag',scenario_code_name,region[j],'','','','','','','','','','','','',udc[k],data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 47 #############################
            ##################################
            if 'CapitalRecoveryFactor' == param:
                
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[1].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['CapitalRecoveryFactor',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 48 #############################
            ##################################
            if 'PvAnnuity' == param:
                
                tech=matrix[1].split(' ')
                if tech[len(tech)-1]==':=\n':
                    tech=tech[0:len(matrix[1].split(' '))-1]
                tech[len(tech)-1]=tech[len(tech)-1].replace(':=\n','')
                
                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                            data_series[j].pop(len(data_series[j])-1)
    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['PvAnnuity',scenario_code_name,region[j],tech[k],'','','','','','','','','','','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 49 #############################
            ##################################
            if 'OperationalLifeStorage' in param:
                
                storage=matrix[1].split(' ')
                if storage[len(storage)-1]==':=\n':
                    storage=storage[0:len(matrix[1].split(' '))-1]
                storage[len(storage)-1]=storage[len(storage)-1].replace(':=\n','')

                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                        data_series[j].pop(len(data_series[j])-1)    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['OperationalLifeStorage',scenario_code_name,region[j],'','','','','','','','','',storage[k],'','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df) 
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 50 #############################
            ##################################
            if 'CapitalCostStorage' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                storage=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    storage.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                        
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['CapitalCostStorage',scenario_code_name,region,'','','','',years[k],'','','','',storage[j],'','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 51 #############################
            ##################################
            if 'ResidualStorageCapacity' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                storage=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    storage.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                        
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['ResidualStorageCapacity',scenario_code_name,region,'','','','',years[k],'','','','',storage[j],'','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 52 #############################
            ##################################
            if 'TechnologyToStorage' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        storage=matrix[j].split(' ')
                        if storage[len(storage)-1]==':=\n':
                            storage=storage[0:len(matrix[j].split(' '))-1]
                        storage[len(storage)-1]=storage[len(storage)-1].replace(':=\n','')
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(storage)):
                            output_rows.append(['TechnologyToStorage',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],'','','','','',storage[l],'','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 53 #############################
            ##################################
            if 'TechnologyFromStorage' in param:
                
                separated_techs=list()
                separated_regions=list()
                separated_modes=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if '[' in matrix[j] and a==0:
                        a=1
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        separated_regions.append(region)
                        separated_techs.append(technology)
                    elif a==1:
                        storage=matrix[j].split(' ')
                        if storage[len(storage)-1]==':=\n':
                            storage=storage[0:len(matrix[j].split(' '))-1]
                        storage[len(storage)-1]=storage[len(storage)-1].replace(':=\n','')
                        a=2
                    elif a==2 and '[' not in matrix[j]:
                        separated_modes.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                    elif a==2 and '[' in matrix[j]:
                        region=matrix[j].split(',')[0].replace(',','').replace('[','')
                        technology=matrix[j].split(',')[1].replace(',','')
                        a=1
                        separated_regions.append(region)
                        separated_techs.append(technology)
                output_rows=list()
                for j in range(len(separated_techs)):
                    for k in range(0,1):
                        for l in range(len(storage)):
                            output_rows.append(['TechnologyFromStorage',scenario_code_name,separated_regions[j],separated_techs[j],'','',separated_modes[j],'','','','','',storage[l],'','','',separated_time_series[(1*j)+k][l]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 54 #############################
            ##################################
            if 'StorageLevelStart' in param:
                
                storage=matrix[1].split(' ')
                if storage[len(storage)-1]==':=\n':
                    storage=storage[0:len(matrix[1].split(' '))-1]
                storage[len(storage)-1]=storage[len(storage)-1].replace(':=\n','')

                region=list()
                data_series=list()
                for j in range(2,len(matrix)):
                    region.append(matrix[j].split(' ')[0])
                    data_series.append(matrix[j].split(' ')[1:])
                for j in range(len(data_series)):
                    data_series[j][len(data_series[j])-1]=data_series[j][len(data_series[j])-1].replace('\n','')
                    if data_series[j][len(data_series[j])-1].replace('\n', '') == '':
                        data_series[j].pop(len(data_series[j])-1)    
                
                output_rows=list()
                for j in range(len(data_series)):
                    for k in range(len(data_series[j])):
                        output_rows.append(['StorageLevelStart',scenario_code_name,region[j],'','','','','','','','','',storage[k],'','','',data_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df) 
                    dict_dataframes[f'{param}'] = df

            ##################################
            # 55 #############################
            ##################################
            if 'MinStorageCharge' in param:
                
                region=matrix[1].split(',')[0].replace(',','').replace('[','')
                years=matrix[2].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                storage=list()
                time_series=list()
                for j in range(3,len(matrix)):
                    storage.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                        
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['MinStorageCharge',scenario_code_name,region,'','','','',years[k],'','','','',storage[j],'','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
                    
            ##################################
            # 56 #############################
            ##################################
            if 'Conversionls' in param:
                separated_timeslices=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if a==0:
                        season=matrix[j].split(' ')
                        if season[len(season)-1]==':=\n':
                            season=season[0:len(matrix[j].split(' '))-1]
                        season[len(season)-1]=season[len(season)-1].replace(':=\n','')
                        a=1
                    elif a==1 and '[' not in matrix[j]:
                        separated_timeslices.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                output_rows=list()

                for j in range(len(separated_timeslices)):
                    for l in range(len(season)):
                        output_rows.append(['Conversionls',scenario_code_name,'','','','','','',separated_timeslices[j],season[l],'','','','','','',separated_time_series[j][l]])

                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
 
                
            ##################################
            # 57 #############################
            ##################################
            if 'Conversionld' in param:
                separated_timeslices=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if a==0:
                        dayt=matrix[j].split(' ')
                        if dayt[len(dayt)-1]==':=\n':
                            dayt=dayt[0:len(matrix[j].split(' '))-1]
                        dayt[len(dayt)-1]=dayt[len(dayt)-1].replace(':=\n','')
                        a=1
                    elif a==1 and '[' not in matrix[j]:
                        separated_timeslices.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                output_rows=list()
                for j in range(len(separated_timeslices)):
                    for l in range(len(dayt)):
                        output_rows.append(['Conversionld',scenario_code_name,'','','','','','',separated_timeslices[j],'',dayt[l],'','','','','',separated_time_series[j][l]])

                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df
    
            ##################################
            # 58 #############################
            ##################################
            if 'Conversionlh' in param:
                separated_timeslices=list()
                separated_time_series=list()
                a=0
                for j in range(1,len(matrix)):
                    if a==0:
                        dailytb=matrix[j].split(' ')
                        if dailytb[len(dailytb)-1]==':=\n':
                            dailytb=dailytb[0:len(matrix[j].split(' '))-1]
                        dailytb[len(dailytb)-1]=dailytb[len(dailytb)-1].replace(':=\n','')
                        a=1
                    elif a==1 and '[' not in matrix[j]:
                        separated_timeslices.append(matrix[j].split(' ')[0])
                        aux=matrix[j].split(' ')[1:]
                        aux[len(aux)-1]=aux[len(aux)-1].replace('\n','')
                        separated_time_series.append(aux)
                output_rows=list()
                for j in range(len(separated_timeslices)):
                    for l in range(len(dailytb)):
                        output_rows.append(['Conversionlh',scenario_code_name,'','','','','','',separated_timeslices[j],'','',dailytb[l],'','','','',separated_time_series[j][l]])

                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df


            ##################################
            # 59 #############################
            ##################################
            if 'DaySplit' in param:
                
                years=matrix[1].split(' ')
                if years[len(years)-1]==':=\n':
                    years=years[0:len(matrix[2].split(' '))-1]
                years[len(years)-1]=years[len(years)-1]
                
                dailytb=list()
                time_series=list()
                for j in range(2,len(matrix)):
                    dailytb.append(matrix[j].split(' ')[0])
                    time_series.append(matrix[j].split(' ')[1:])
                for j in range(len(time_series)):
                    time_series[j][len(time_series[j])-1]=time_series[j][len(time_series[j])-1].replace('\n','')
                    if time_series[j][len(time_series[j])-1].replace('\n', '') == '':
                        time_series[j].pop(len(time_series[j])-1)
                
                output_rows=list()
                for j in range(len(time_series)):
                    for k in range(len(time_series[j])):
                        output_rows.append(['DaySplit',scenario_code_name,'','','','','',years[k],'','','',dailytb[j],'','','','',time_series[j][k]])
                    
                # Store data
                if output_rows:
                    df = pd.DataFrame(output_rows, columns=parameter_headers)
                    df=df.rename(columns={'Value': param})
                    list_dataframes.append(df)
                    dict_dataframes[f'{param}'] = df

    return list_dataframes, dict_dataframes, parameters_without_values

            
def create_input_dataset_future_0(list_dataframes, scenario_name, output_dataset_path):
    column_map = {'Scenario': 'Strategy'}
    dataframe_list = list_dataframes
    df_complete = pd.DataFrame()
    for i in range(len(dataframe_list)):
        if i == 0:
            df_complete = dataframe_list[i]
        else:
            df_complete = pd.concat([df_complete, dataframe_list[i]], axis=0)

    df_complete = df_complete.assign(FutureID=0)
    df_complete = df_complete.assign(StrategyID=0)
    df_complete = df_complete.rename(columns={'FutureID': 'Future.ID'})
    df_complete = df_complete.rename(columns={'StrategyID': 'Strategy.ID'})
    cols = list(df_complete.columns.values)
    cols = cols[0:len(cols) - 2]
    cols.insert(0, 'Strategy.ID')
    cols.insert(0, 'Future.ID')

    df_complete = df_complete[cols]
    df_complete = df_complete.drop('PARAMETER', axis=1)
    df_complete = df_complete.rename(columns=column_map)
    df_complete.to_csv(
        output_dataset_path + scenario_name + '_0_Input.csv', index=None, header=True
    )
    
def create_output_dataset_future_0(case, time_range_vector, first_list, structure_file_path_name):

    # 1 - Always call the structure of the model:
    #-------------------------------------------#
    structure_filename = structure_file_path_name 
    structure_file = pd.ExcelFile(structure_filename)
    structure_sheetnames = structure_file.sheet_names  # see all sheet names
    sheet_sets_structure = pd.read_excel(open(structure_filename, 'rb'),
                                         header=None,
                                         sheet_name=structure_sheetnames[0])
    sheet_params_structure = pd.read_excel(open(structure_filename, 'rb'),
                                           header=None,
                                           sheet_name=structure_sheetnames[1])
    sheet_vars_structure = pd.read_excel(open(structure_filename, 'rb'),
                                         header=None,
                                         sheet_name=structure_sheetnames[2])

    S_DICT_sets_structure = {'set':[],'initial':[],'number_of_elements':[],'elements_list':[]}
    for col in range(1,11+1):  # 11 columns
        S_DICT_sets_structure['set'].append(sheet_sets_structure.iat[0, col])
        S_DICT_sets_structure['initial'].append(sheet_sets_structure.iat[1, col])
        S_DICT_sets_structure['number_of_elements'].append(int(sheet_sets_structure.iat[2, col]))
        #
        element_number = int(sheet_sets_structure.iat[2, col])
        this_elements_list = []
        if element_number > 0:
            for n in range(1, element_number+1):
                this_elements_list.append(sheet_sets_structure.iat[2+n, col])
        S_DICT_sets_structure['elements_list'].append(this_elements_list)
    #
    S_DICT_params_structure = {'category':[],'parameter':[],'number_of_elements':[],'index_list':[]}
    param_category_list = []
    for col in range(1,30+1):  # 30 columns
        if str(sheet_params_structure.iat[0, col]) != '':
            param_category_list.append(sheet_params_structure.iat[0, col])
        S_DICT_params_structure['category'].append(param_category_list[-1])
        S_DICT_params_structure['parameter'].append(sheet_params_structure.iat[1, col])
        S_DICT_params_structure['number_of_elements'].append(int(sheet_params_structure.iat[2, col]))
        #
        index_number = int(sheet_params_structure.iat[2, col])
        this_index_list = []
        for n in range(1, index_number+1):
            this_index_list.append(sheet_params_structure.iat[2+n, col])
        S_DICT_params_structure['index_list'].append(this_index_list)
    #
    S_DICT_vars_structure = {'category':[],'variable':[],'number_of_elements':[],'index_list':[]}
    var_category_list = []
    for col in range(1,43+1):  # 43 columns
        if str(sheet_vars_structure.iat[0, col]) != '':
            var_category_list.append(sheet_vars_structure.iat[0, col])
        S_DICT_vars_structure['category'].append(var_category_list[-1])
        S_DICT_vars_structure['variable'].append(sheet_vars_structure.iat[1, col])
        S_DICT_vars_structure['number_of_elements'].append(int(sheet_vars_structure.iat[2, col]))
        #
        index_number = int(sheet_vars_structure.iat[2, col])
        this_index_list = []
        for n in range(1, index_number+1):
            this_index_list.append(sheet_vars_structure.iat[2+n, col])
        S_DICT_vars_structure['index_list'].append(this_index_list)
    #-------------------------------------------#    
    all_vars = ['RateOfDemand', #1
                'Demand', #2
                'NumberOfNewTechnologyUnits', #3
                'NewCapacity', #4
                'AccumulatedNewCapacity', #5
                'TotalCapacityAnnual', #6
                'RateOfActivity', #7
                'RateOfTotalActivity', #8
                'TotalTechnologyAnnualActivity',	#9
                'TotalAnnualTechnologyActivityByMode', #10
                'TotalTechnologyModelPeriodActivity', #11
                'RateOfProductionByTechnologyByMode', #12
                'RateOfProductionByTechnology', #13
                'ProductionByTechnology', #14
                'ProductionByTechnologyAnnual', #15
                'RateOfProduction', #16
                'Production', #17
                'RateOfUseByTechnologyByMode', #18
                'RateOfUseByTechnology', #19
                'UseByTechnologyAnnual', #20
                'UseByTechnology', #21
                'UseAnnual', #22
                'CapitalInvestment', #23
                'DiscountedCapitalInvestment', #24
                'SalvageValue', #25
                'DiscountedSalvageValue', #26
                'OperatingCost', #27
                'DiscountedOperatingCost', #28
                'AnnualVariableOperatingCost', #29
                'AnnualFixedOperatingCost', #30
                'TotalDiscountedCostByTechnology', #31
                'TotalDiscountedCost', #32
                'TotalCapacityInReserveMargin', #33
                'DemandNeedingReserveMargin', #34
                'TotalREProductionAnnual', #35
                'RETotalProductionOfTargetFuelAnnual', #36
                'AnnualTechnologyEmissionByMode', #37
                'AnnualTechnologyEmission', #38
                'AnnualTechnologyEmissionPenaltyByEmission', #39
                'AnnualTechnologyEmissionsPenalty', #40
                'DiscountedTechnologyEmissionsPenalty', #41
                'AnnualEmissions', #42
                'ModelPeriodEmissions'] #43


    all_vars_output_dict = [{} for e in range(len(first_list))]
    #
    output_header = ['Strategy', 'Future.ID', 'Commodity', 'Technology', 'Emission', 'Year', 'TimeSlice']
    #-------------------------------------------------------#
    for v in range(len(all_vars)):
        output_header.append(all_vars[v])

    #-------------------------------------------------------#
    this_strategy = first_list[case].split('_')[0] 
    this_future   = first_list[case].split('_')[1]
    #-------------------------------------------------------#
    #
    vars_as_appear = []

    data_name = str('./workflow/1_Experiment/Executables/' + first_list[case]) + '/' + str(first_list[case]) + '_Output.txt'
    #
    n = 0
    break_this_while = False
    print('Start long loop')
    while break_this_while == False:
        n += 1
        structure_line_raw = linecache.getline(data_name, n)
        if 'No. Column name  St   Activity     Lower bound   Upper bound    Marginal' in structure_line_raw:
            ini_line = deepcopy(n+2)
        if 'Karush-Kuhn-Tucker' in structure_line_raw:
            end_line = deepcopy(n-1)
            break_this_while = True
            break
    print('Finish long loop')
    #
    #print('###########################')
    #print(data_name)
    #print('###########################')
    for n in range(ini_line, end_line, 2):
        structure_line_raw = linecache.getline(data_name, n)
        structure_list_raw = structure_line_raw.split(' ')
        #
        structure_list_raw_2 = [s_line for s_line in structure_list_raw if s_line != '']
        structure_line = structure_list_raw_2[1]
        structure_list = structure_line.split('[')
        the_variable = structure_list[0]
        #
        if the_variable in all_vars:
            set_list = structure_list[1].replace(']','').replace('\n','').split(',')
            #--%
            index = S_DICT_vars_structure['variable'].index(the_variable)
            this_variable_indices = S_DICT_vars_structure['index_list'][index]
            #
            #--%
            if 'y' in this_variable_indices:
                data_line = linecache.getline(data_name, n+1)
                data_line_list_raw = data_line.split(' ')
                data_line_list = [data_cell for data_cell in data_line_list_raw if data_cell != '']
                useful_data_cell = data_line_list[1]
                #--%
                if useful_data_cell != '0':
                    #
                    if the_variable not in vars_as_appear:
                        vars_as_appear.append(the_variable)
                        all_vars_output_dict[case].update({ the_variable:{} })
                        all_vars_output_dict[case][the_variable].update({ the_variable:[] })
                        #
                        for n in range(len(this_variable_indices)):
                            all_vars_output_dict[case][the_variable].update({ this_variable_indices[n]:[] })
                    #--%
                    this_variable = vars_as_appear[-1]
                    all_vars_output_dict[case][this_variable][this_variable].append(useful_data_cell)
                    for n in range(len(this_variable_indices)):
                        all_vars_output_dict[case][the_variable][this_variable_indices[n]].append(set_list[n])
                #
            #
            elif 'y' not in this_variable_indices:
                data_line = linecache.getline(data_name, n+1)
                data_line_list_raw = data_line.split(' ')
                data_line_list = [data_cell for data_cell in data_line_list_raw if data_cell != '']
                useful_data_cell = data_line_list[1]
                #--%
                if useful_data_cell != '0':
                    #
                    if the_variable not in vars_as_appear:
                        vars_as_appear.append(the_variable)
                        all_vars_output_dict[case].update({ the_variable:{} })
                        all_vars_output_dict[case][the_variable].update({ the_variable:[] })
                        #
                        for n in range(len(this_variable_indices)):
                            all_vars_output_dict[case][the_variable].update({ this_variable_indices[n]:[] })
                    #--%
                    this_variable = vars_as_appear[-1]
                    all_vars_output_dict[case][this_variable][this_variable].append(useful_data_cell)
                    for n in range(len(this_variable_indices)):
                        all_vars_output_dict[case][the_variable][this_variable_indices[n]].append(set_list[n])
        #--%
        else:
            pass
    #
    linecache.clearcache()
    #%%
    #-----------------------------------------------------------------------------------------------------------%
    output_adress = './workflow/1_Experiment/Executables/' + str(first_list[case])
    combination_list = [] # [fuel, technology, emission, year]
    data_row_list = []
    for var in range(len(vars_as_appear)):
        this_variable = vars_as_appear[var]
        this_var_dict = all_vars_output_dict[case][this_variable]
        #--%
        index = S_DICT_vars_structure['variable'].index(this_variable)
        this_variable_indices = S_DICT_vars_structure['index_list'][index]
        #--------------------------------------#
        for k in range(len(this_var_dict[this_variable])):
            this_combination = []
            #
            if 'f' in this_variable_indices:
                this_combination.append(this_var_dict['f'][k])
            else:
                this_combination.append('')
            #
            if 't' in this_variable_indices:
                this_combination.append(this_var_dict['t'][k])
            else:
                this_combination.append('')
            #
            if 'e' in this_variable_indices:
                this_combination.append(this_var_dict['e'][k])
            else:
                this_combination.append('')
            #
            if 'l' in this_variable_indices:
                #this_combination.append('')
                this_combination.append(this_var_dict['l'][k])
            else:
                this_combination.append('')
            #
            if 'y' in this_variable_indices:
                this_combination.append(this_var_dict['y'][k])
            else:
                this_combination.append('')
            #
            if this_combination not in combination_list:
                combination_list.append(this_combination)
                data_row = ['' for n in range(len(output_header))]
                # print('check', len(data_row), len(run_id))
                data_row[0] = this_strategy
                data_row[1] = this_future
                data_row[2] = this_combination[0] # Fuel
                data_row[3] = this_combination[1] # Technology
                data_row[4] = this_combination[2] # Emission
                #data_row[7] = this_combination[3] 
                data_row[5] = this_combination[4] # Year
                data_row[6] = this_combination[3] # TimeSlice
                #
                var_position_index = output_header.index(this_variable)
                data_row[var_position_index] = this_var_dict[this_variable][k]
                data_row_list.append(data_row)
            else:
                ref_index = combination_list.index(this_combination)
                this_data_row = deepcopy(data_row_list[ref_index])
                #
                var_position_index = output_header.index(this_variable)
                #
                if 'l' in this_variable_indices: 
                    #
                    if str(this_data_row[var_position_index]) != '' and str(this_var_dict[this_variable][k]) != '' and ('Rate' not in this_variable):
                        this_data_row[var_position_index] = str( float(this_data_row[var_position_index]) + float(this_var_dict[this_variable][k]))
                    elif str(this_data_row[var_position_index]) == '' and str(this_var_dict[this_variable][k]) != '':
                        this_data_row[var_position_index] = str(float(this_var_dict[this_variable][k]))
                    elif str(this_data_row[var_position_index]) != '' and str(this_var_dict[this_variable][k]) == '':
                        pass
                else:
                    this_data_row[var_position_index] = this_var_dict[this_variable][k]
                #
                data_row_list[ref_index]  = deepcopy(this_data_row)
                #
            ''' $ (end) $ '''
            #
        #
    #
    non_year_combination_list = []
    non_year_combination_list_years = []
    for n in range(len(combination_list)):
        this_combination = combination_list[n]
        this_non_year_combination = [this_combination[0], this_combination[1], this_combination[2], this_combination[3]]
        if this_combination[4] != '' and this_non_year_combination not in non_year_combination_list:
            non_year_combination_list.append(this_non_year_combination)
            non_year_combination_list_years.append([this_combination[4]])
        elif this_combination[4] != '' and this_non_year_combination in non_year_combination_list:
            non_year_combination_list_years[non_year_combination_list.index(this_non_year_combination)].append(this_combination[4])
    #
    for n in range(len(non_year_combination_list)):
        if len(non_year_combination_list_years[n]) != len(time_range_vector):
            #
            this_existing_combination = non_year_combination_list[n]
            # print('flag 1', this_existing_combination)
            #this_existing_combination.append('')
            # print('flag 2', this_existing_combination)
            this_existing_combination.append(non_year_combination_list_years[n][0])
            # print('flag 3', this_existing_combination)
            
            ref_index = combination_list.index(this_existing_combination)
            this_existing_data_row = deepcopy(data_row_list[ref_index])
            #
            for n2 in range(len(time_range_vector)):
                #
                if time_range_vector[n2] not in non_year_combination_list_years[n]:
                    #
                    data_row = ['' for n in range(len(output_header))]
                    data_row[0] = this_strategy
                    data_row[1] = this_future
                    data_row[2] = non_year_combination_list[n][0]
                    data_row[3] = non_year_combination_list[n][1]
                    data_row[4] = non_year_combination_list[n][2]
                    data_row[5] = time_range_vector[n2]
                    data_row[6] = non_year_combination_list[n][3]
                    #
                    for n3 in range(len(vars_as_appear)):
                        this_variable = vars_as_appear[n3]
                        this_var_dict = all_vars_output_dict[case][this_variable]
                        index = S_DICT_vars_structure['variable'].index(this_variable)
                        this_variable_indices = S_DICT_vars_structure['index_list'][index]
                        #
                        var_position_index = output_header.index(this_variable)
                        #
                        print_true = False
                        if ('f' in this_variable_indices and str(non_year_combination_list[n][0]) != ''): # or ('f' not in this_variable_indices and str(non_year_combination_list[n][0]) == ''):
                            print_true = True
                        else:
                            pass
                        #
                        if ('t' in this_variable_indices and str(non_year_combination_list[n][1]) != ''): # or ('t' not in this_variable_indices and str(non_year_combination_list[n][1]) == ''):
                            print_true = True
                        else:
                            pass
                        #
                        if ('e' in this_variable_indices and str(non_year_combination_list[n][2]) != ''): # or ('e' not in this_variable_indices and str(non_year_combination_list[n][2]) == ''):
                            print_true = True
                        else:
                            pass
                        if ('l' in this_variable_indices and str(non_year_combination_list[n][3]) != ''): # or ('e' not in this_variable_indices and str(non_year_combination_list[n][2]) == ''):
                            print_true = True
                        else:
                            pass
                        #
                        if 'y' in this_variable_indices and (str(this_existing_data_row[var_position_index]) != '') and print_true == True:
                            data_row[var_position_index] = '0'
                            #
                        else:
                            pass
                    #
                    data_row_list.append(data_row)
    #--------------------------------------#
    with open(output_adress + '/' + str(first_list[case]) + '_Output' + '.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(output_header)
        for n in range(len(data_row_list)):
            csvwriter.writerow(data_row_list[n])
    #-----------------------------------------------------------------------------------------------------------%
    shutil.os.remove(data_name) #-----------------------------------------------------------------------------------------------------------%
    gc.collect(generation=2)
    time.sleep(0.05)
    #-----------------------------------------------------------------------------------------------------------%
    #print( 'We finished with printing the outputs: ' + str(first_list[case]))
    
    
def run_osemosys( solver, scenario_dir, data_file, model_file, output_file ):
    
    file_aboslute_address = os.path.abspath("z_auxiliar_code.py")
    file_adress = re.escape( file_aboslute_address.replace( 'z_auxiliar_code.py', '' ) )
    file_config_address = get_config_main_path(os.path.abspath(''),os.path.join('workflow','3_Postprocessing'))
    
    str_start = "start /B start cmd.exe @cmd /k cd " + file_adress

    output_file = data_file.replace('.txt','') + '_Output'
    #
    if solver == 'glpk':
        str_solve = 'glpsol -m '+ str( model_file ) +' -d ' + str( data_file )  +  " -o " + str( output_file ) + '.txt'
    else:
        str_matrix = 'glpsol -m ' + str( model_file ) + ' -d ' + str( data_file ) + ' --wlp ' + str( output_file ) + '.lp --check'
        os.system( str_start and str_matrix )
        
        if solver == 'cbc':
            str_solve = 'cbc ' + str( output_file ) + '.lp solve -solu ' + str( output_file ) + '.sol'
    
        elif solver == 'cplex':
            if os.path.exists(output_file + '.sol'):
                shutil.os.remove(output_file + '.sol')
            str_solve = 'cplex -c "read ' + str( output_file ) + '.lp" "optimize" "write ' + str( output_file ) + '.sol"'
    os.system( str_start and str_solve )
    
    time.sleep(1)
    
    
def run_scripts(script, solver=None, osemosys_model=None, Interface_RDM=None, shape_file=None):
    if solver is None:
        str_scripts = 'python -u ' + script
    else:
        str_scripts = 'python -u ' + script + ' ' + solver + ' ' + osemosys_model + ' ' + Interface_RDM + ' ' + shape_file
    subprocess.run(str_scripts, shell=True, check=True)
    
def process_timeslices(input_file_path, num_time_slices_SDP_read, output_file_path):
    """Trim TIMESLICE definitions in a data file to the requested count."""
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    timeslices = []
    new_timeslices = []
    delete_timeslices = []
    timeslice_line_index = -1
    
    # Step 1: Locate the "set TIMESLICE :=" line.
    for i, line in enumerate(lines):
        if "set TIMESLICE :=" in line:
            timeslice_line_index = i
            # Extract the portion after "set TIMESLICE :=".
            start_index = line.find("set TIMESLICE :=") + len("set TIMESLICE :=")
            end_index = line.find(";", start_index)
            timeslice_str = line[start_index:end_index].strip()
            timeslices = timeslice_str.split()
            break

    if timeslice_line_index == -1:
        raise ValueError("Could not find 'set TIMESLICE :=' in the file.")

    # Step 2: Split the timeslices into those to keep and those to drop.
    new_timeslices = timeslices[:num_time_slices_SDP_read]
    delete_timeslices = timeslices[num_time_slices_SDP_read:]

    # Step 3: Overwrite the "set TIMESLICE :=" line with the new timeslices.
    new_timeslice_str = "set TIMESLICE := " + " ".join(new_timeslices) + " ;\n"
    lines[timeslice_line_index] = new_timeslice_str

    # Step 4: Remove any lines containing deleted timeslices.
    lines = [line for line in lines if not any(ts in line for ts in delete_timeslices)]

    # Step 5: Save the updated file.
    with open(output_file_path, 'w') as file:
        file.writelines(lines)
        
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

# Function : Postprocessing data

def parse_cbc_sol_file(file_path, parameters_to_print):
    data = []
    interest_vars = get_selected_parameters(parameters_to_print)

    with open(file_path, 'r') as file:
        for line in file:
            # Skip objective line and empty lines
            if line.strip() == "" or "Optimal - objective" in line:
                continue
            
            # Extract values using regex without including the ID column
            match = re.match(r'\s*\d+\s+([^\s]+)\s+([\d\.\-eE]+)\s+(\d+)', line)
            if match:
                full_variable = match.group(1)  # Full variable with parentheses
                variable_value = float(match.group(2))
                
                # Split the variable into two parts
                variable_match = re.match(r'([^\(]+)\(([^)]+)\)', full_variable)
                if variable_match:
                    variable_name = variable_match.group(1).strip()  # Before parentheses
                    variable_details = variable_match.group(2).strip()  # Inside parentheses
                else:
                    variable_name = full_variable
                    variable_details = ""
                
                # Append data excluding the ID column
                if variable_name in interest_vars:
                    data.append((variable_name, variable_details, variable_value))
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Variable", "Details", "Value"])
    return df

def parse_cplex_sol_file(file_path, parameters_to_print):
    """
    Parses a .sol file output from CPLEX to extract variable data.

    Parameters:
        file_path (str): Path to the .sol file.
        parameters_to_print (DatFrame): DataFrame with data to print.

    Returns:
        pd.DataFrame: DataFrame containing columns ['Variable', 'Details', 'Value'].
    """
    data = []
    inside_variables = False

    interest_vars = get_selected_parameters(parameters_to_print)

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for start and end of <variables> section
            if line == "<variables>":
                inside_variables = True
                continue
            elif line == "</variables>":
                inside_variables = False
                break

            # Process lines within the <variables> section
            if inside_variables:
                # Match the variable lines
                match = re.match(r'<variable name="([^"]+)".*value="([^"]+)"', line)
                if match:
                    full_name = match.group(1)  # Full variable name (e.g., NewStorageCapacity(REG,PHS001,2019))
                    value = float(match.group(2))  # Extract the value

                    # Split full_name into Variable and Details
                    variable_match = re.match(r'([^(]+)\(([^)]+)\)', full_name)
                    if variable_match:
                        variable_name = variable_match.group(1).strip()  # Before the parenthesis
                        details = variable_match.group(2).strip()  # Inside the parenthesis
                    else:
                        variable_name = full_name
                        details = ""
                    # Append to data
                    if variable_name in interest_vars:
                        data.append((variable_name, details, value))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Variable", "Details", "Value"])
    return df

def parse_glpk_sol_file(file_path, parameters_to_print):
    """
    Parses a GLPK solution file (.sol) to extract variable data.
    
    Parameters:
        file_path (str): Path to the .sol file.
        parameters_to_print (DataFrame): DataFrame with data to print.

    Returns:
        pd.DataFrame: DataFrame containing columns ['Variable', 'Details', 'Value'].
    """
    data = []
    inside_variables = False
    
    # Get the variables of interest
    interest_vars = get_selected_parameters(parameters_to_print)
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Detect the start of the variables section
            if line.startswith("No. Column name"):
                inside_variables = True
                continue
            
            if inside_variables:
                # Extract the variable name and details
                match = re.match(r'\d+\s+([^\[]+)\[([^\]]+)\]', line)
                if match:
                    variable_name = match.group(1).strip()  # Before brackets
                    details = match.group(2).strip()  # Inside brackets
                    
                    # Read the next line for the value
                    next_line = next(file).strip()
                    value_match = re.search(r'([+-]?\d*\.\d+|\d+)', next_line)
                    if value_match:
                        value = float(value_match.group(0))
                    else:
                        value = 0.0  # Default if no value found
                    
                    # Append data if variable is in interest_vars
                    if variable_name in interest_vars:
                        data.append((variable_name, details, value))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Variable", "Details", "Value"])
    return df

def get_selected_parameters(parameters_to_print):
    """
    Reads the 'To_Print' sheet from the Excel file, filters selected parameters, and returns a list of selected parameters.
    
    :param parameters_to_print: DataFrame with data to print.
    :return: List of selected parameters.
    """
    # Read the "To_Print" sheet into a DataFrame
    df = parameters_to_print
    
    # Normalize the "Select" column by converting all values to uppercase and stripping spaces
    df["Select"] = df["Select"].astype(str).str.strip().str.upper()
    
    # Filter parameters where the "Select" column contains 'X'
    selected_parameters = df.loc[df["Select"] == "X", "Parameter"].tolist()
    
    return selected_parameters


def process_structure_file(file_path):
    """
    Processes an Excel file with 'sets' and 'variables' structures.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        dict: A dictionary with processed DataFrames:
              - 'sets': DataFrame with updated set indices.
              - 'variables': Cleaned variables DataFrame.
    """
    # Read the Excel file
    structure_file = pd.ExcelFile(file_path)
    
    # Read the sets sheet
    sheet_sets_structure = pd.read_excel(file_path, header=None, sheet_name=structure_file.sheet_names[0])
    
    # Read the variables sheet
    sheet_vars_structure = pd.read_excel(file_path, header=None, sheet_name=structure_file.sheet_names[2])

    ###################### Sets Acronyms ###########################
    # Filter rows containing only 'sets' or 'index' in column 0
    sheet_sets_structure = sheet_sets_structure[sheet_sets_structure[0].isin(["set", "index"])]
    
    # Extract the 'sets' row for column names
    column_names = sheet_sets_structure.iloc[0, 1:].values  # Exclude column 0 and take names
    
    # Remove the "sets" row and reset the index
    sheet_sets_structure = sheet_sets_structure[sheet_sets_structure[0] != "set"]
    sheet_sets_structure = sheet_sets_structure.set_index(0)  # Assign column 0 as the index
    
    # Replace "index" with "acronym" in the index
    sheet_sets_structure.index = sheet_sets_structure.index.str.replace("index", "acronym")
    
    # Update column names
    sheet_sets_structure.columns = column_names

    ###################### Sets-Variables ##########################
    # Remove the first row
    sheet_vars_structure = sheet_vars_structure.drop(0, axis=0).reset_index(drop=True)
    
    # Assign column 0 as the index
    sheet_vars_structure.index = sheet_vars_structure[0].fillna("")  # Initialize with column 0
    
    # Replace NaN values in the index with "set#"
    counter = 1
    new_index = []
    for value in sheet_vars_structure.index:
        if value == "" or pd.isna(value):
            new_index.append(f"set{counter}")
            counter += 1
        else:
            new_index.append(value)
    sheet_vars_structure.index = new_index

    # Extract the 'variable' row for column names
    column_names = sheet_vars_structure.loc["variable"].values  # Take values from the "variable" row
    
    # Remove the 'variable' row from the DataFrame
    sheet_vars_structure = sheet_vars_structure.drop("variable", axis=0)
    
    # Assign new column names
    sheet_vars_structure.columns = column_names
    
    # Remove the "variable" column
    sheet_vars_structure = sheet_vars_structure.drop(columns="variable")

    # Return the results as a dictionary
    return sheet_sets_structure, sheet_vars_structure

def transform_output_sol_optimized(df, sheet_vars_structure, sheet_sets_structure, strategy, fut_id):
    """
    Transforms the DataFrame 'df' into the requested format with optimized structure.

    Parameters:
        df (DataFrame): Original DataFrame generated by 'parse_sol_file'.
        sheet_vars_structure (DataFrame): DataFrame mapping variables to sets.
        sheet_sets_structure (DataFrame): DataFrame mapping acronyms to column names.
        strategy (str): Value to assign in the 'Strategy' column.
        fut_id (str): Value to assign in the 'Future.ID' column.

    Returns:
        DataFrame: Transformed DataFrame with the requested structure.
    """
    # Define the output header with uppercase columns
    output_header = [
        'Strategy', 'Future.ID', 'REGION', 'COMMODITY', 'TECHNOLOGY', 'EMISSION',
        'YEAR', 'TIMESLICE', 'MODE_OF_OPERATION', 'SEASON', 'DAYTYPE',
        'DAILYTIMEBRACKET', 'STORAGE', 'STORAGEINTRADAY', 'STORAGEINTRAYEAR', 'UDC'
    ]
    
    # Add unique values from the "Variable" column to the header
    variable_columns = list(set(df["Variable"].unique()))
    output_header.extend(variable_columns)

    # Preprocess sheet_vars_structure to quickly map variables to letters
    var_to_sets = {}
    for col in sheet_vars_structure.columns:
        set_values = sheet_vars_structure[col].dropna()
        var_to_sets[col] = set_values[set_values.index.str.startswith("set")].to_list()

    # Preprocess sheet_sets_structure to map letters to column names
    acronym_to_column = sheet_sets_structure.loc["acronym"].to_dict()
    
    # List to accumulate rows
    output_rows = []

    # Iterate over each row in 'df'
    for _, row in df.iterrows():
        details = row["Details"].split(",")  # Split 'Details' values by commas
        variable_name = row["Variable"]      # Variable name
        variable_value = row["Value"]        # Variable value

        # Get the letters associated with the variable from var_to_sets
        set_letters = var_to_sets.get(variable_name, [])

        # Create a dictionary for the current row
        row_data = {
            "Strategy": strategy,
            "Future.ID": fut_id
        }

        # Map letters to columns and assign values
        for i, letter in enumerate(set_letters):
            if i < len(details):  # Ensure it does not exceed details length
                for col, acronym in acronym_to_column.items():
                    if letter == acronym:  # Match the letter
                        column_name = col.upper()
                        row_data[column_name] = details[i]
                        break

        # Assign the variable's value to the corresponding column
        row_data[variable_name] = variable_value

        # Add the dictionary to the list
        output_rows.append(row_data)

    # Create the final DataFrame from the list of rows
    df_output_sol = pd.DataFrame(output_rows, columns=output_header).fillna("")

    return df_output_sol



def data_processor_new(output_file, model_structure, strategy, fut_id, solver, parameters_to_print, output_file_type):
    # Parse the .sol file
    if solver == 'cbc':
        df = parse_cbc_sol_file(output_file, parameters_to_print)
    elif solver == 'cplex':
        df = parse_cplex_sol_file(output_file, parameters_to_print)
    elif solver == 'glpk':
        df = parse_glpk_sol_file(output_file, parameters_to_print)
    
    # Call the function
    sheet_sets_structure, sheet_vars_structure = process_structure_file(model_structure)
    
    # Transform the data
    df_output_sol = transform_output_sol_optimized(df, sheet_vars_structure, sheet_sets_structure, strategy, fut_id)

    if output_file_type == 'csv':
        # Change the output name for the CSV
        if solver == 'cplex' or solver == 'cbc':
            output_csv_file = output_file.replace('sol', 'csv')
        elif solver == 'glpk':
            output_csv_file = output_file.replace('txt', 'csv')
        # Save the result to a CSV file
        df_output_sol.to_csv(output_csv_file, index=False)
    
    elif output_file_type == 'parquet':    
        # Ensure consistent column types
        for col in df_output_sol.columns:
            if df_output_sol[col].dtype == "object":  # Check for object (potential mixed types)
                try:
                    # Attempt to convert to numeric
                    df_output_sol[col] = pd.to_numeric(df_output_sol[col])
                except ValueError:
                    # If conversion fails, leave as string
                    # print(f"Column '{col}' contains non-numeric values. Converting to string.")
                    df_output_sol[col] = df_output_sol[col].astype(str)
        
        # Change the output name for the Parquet file
        if solver == 'cplex' or solver == 'cbc':
            case_out_path = output_file.replace('sol', 'parquet')
        elif solver == 'glpk':
            case_out_path = output_file.replace('txt', 'parquet')
        # Save as Parquet file
        df_output_sol.to_parquet(case_out_path, engine='pyarrow', index=False)
        

    
    # Optionally delete the solver solution file
    if solver == 'cplex' or solver == 'cbc':
        shutil.os.remove(output_file)
        shutil.os.remove(output_file.replace('sol', 'lp'))
    elif solver == 'glpk':
        shutil.os.remove(output_file)
