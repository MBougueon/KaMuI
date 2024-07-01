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

def get_data(folder, variables, timing ):
    """"""

    df_list = []
    col_name = []
    time_to_extract = []#timing.copy()
    nb_subfold = 0

    for subfold in os.listdir(folder):
    
        file_path = f"{folder}{subfold}/"#concatenattion to form the path were the files are
        stimulation_time = []
        time_to_extract = []#timing.copy()
        var_v = []
        var_v_float = []
        for var in variables:
            var_temp = re.findall(str(var) + r"_\d+[,]*\d*_", str(subfold))#some variable have number in their name, exclude them
            if var_temp is not None:
                var_v.append(re.findall(r"\d+[,]*\d*", "".join(var_temp))[0])
                var_v = [v.replace(',','.') for v in var_v]

        for item in var_v:
            print(item)
            var_v_float.append(float(item))
        for path in pathlib.Path(file_path).iterdir():
            # list copy, whitout it both list will be incremented
            df = pd.DataFrame()
            if path.is_file():
                dframe = get_dataframe(path)
                dframe['[T]'] = dframe['[T]'].astype('int')
                col_name = dframe.columns
                #extract the wanted timing
                for time in timing:
                    df = df._append(dframe.loc[dframe['[T]'] == time])
                    if not time in dframe['[T]']:
                        print(f' {path}')
                        #TODO PARFOIS DES SIMULATIONS S'ARRETTENT AVANT LA FIN CAUSANT UNE ERREUR LORS DE L'EXTRACTION, REGARDER POURQUOI ET COMMENT LE RESOUDRE
        
        df.columns = col_name
        #mean the replicated experiments
        dfram = pd.DataFrame(df.groupby(['[T]']).mean())

        for i in range(0, len(variables)):#add the variables into the table for futur plot
            var = []
            var += len(dfram) * [var_v_float[i]]
            dfram.insert(1, variables[i], var, allow_duplicates=True)
            col_name = dfram.columns
        
        df_list += dfram.values.tolist()
        nb_subfold +=1
    #fuse the mean data from of each experiments
    df = pd.DataFrame(df_list, columns=col_name)
    print(df)
    df.insert(0, "[T]", (time_to_extract*nb_subfold), allow_duplicates=True)
    return df
def data_ponderation():
    """"""