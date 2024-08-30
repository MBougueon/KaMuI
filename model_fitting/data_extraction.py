#python 3.10
"""Extract data from the simulation for parameters inference"""

import os
import pathlib
import re

import pandas as pd

def get_dataframe(filepath):  # Pierre
    """
    Open a csv file and return a pandas dataframe.

    Parameters:
    -----------
    filepath: string
        Filepath of the file to be opened minus the '.csv' suffix.
    Return:
    -------
    A pandas Dataframe object or None if the file doesn't exist.
    """

    with open(filepath, 'r'):
        return pd.read_csv(
            filepath, encoding='utf-8', skipinitialspace=True,
            skiprows=[0, 1],  # Skip lines for each problem
        )

def parse_dataframes(file_path, var_v_float, variables, timing, exp_id):
    """ Extract the wanted values for each simulation files

    Parameters
    ----------
    filepath: string,
        path of the folder where the simulations file are
    var_v_float: list,
        tested parameters values extract from the file name
    variables: list,
        tested parameters
    timing: list,
        Time of simulation that need to be extract
    exp_id: int
        Id of the simulation

    Return
    ------
    df_list: list,
        concatenate list of the values extracted
    col_name: list
        name of the column of the df
    """
    df_list = []
    for path in pathlib.Path(file_path).iterdir():
        # list copy, whitout it both list will be incremented
        df = pd.DataFrame()
        if path.is_file():
            dframe = get_dataframe(path)
            dframe['[T]'] = dframe['[T]'].astype('int')
            col_name = dframe.columns
            #extract the wanted timing
            for time in timing:
                df = pd.concat([df, dframe.loc[dframe['[T]'] == time]])
                if not time in dframe['[T]']:
                    print(f' {path}')
        df.columns = col_name
        for i in range(0, len(variables)):
            var = []
            var += len(df) * [var_v_float[i]]
            df.insert(1, variables[i], var, allow_duplicates=True)
        df.insert(1, 'exp_sim', exp_id, allow_duplicates=True)
        col_name = df.columns            
        df_list += df.values.tolist()
    return(df_list, col_name)

def get_data(folder, variables, timing):
    """
    Parse folder to extract the wanted values for each simulation files

    Parameters:
    -----------
    folder: string
        Path were the simulations are store
    variables: list
        List of variable that are tests and use a check
    timing: list,
        Time of simulation that need to be extract

    Return:
    -------
    df: panda dataframe
        dataframe stroring the extracted values
    """

    df_list = []
    col_name = []
    exp_id = 1
    for subfold in os.listdir(folder):
        file_path = f"{folder}{subfold}/"#concatenattion to form the path were the files are
        var_v = []
        var_v_float = []
        for var in variables:
            #some variable have number in their name, exclude them
            var_temp = re.findall(str(var) + r"_\d+[,]*\d*_", str(subfold))
            if (len(var_temp) > 0):
                var_v.append(re.findall(r"\d+[,]*\d*", "".join(var_temp))[0])
                var_v = [v.replace(',','.') for v in var_v]
        if (len(var_v) == len(variables)):
            for item in var_v:
                var_v_float.append(float(item))
            d_list, col_name = parse_dataframes(file_path, var_v_float, variables, timing, exp_id)
            df_list += d_list
        exp_id +=1

    #fuse the mean data from of each experiments
    dfram = pd.DataFrame(df_list, columns=col_name)
    return dfram
