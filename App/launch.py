"""
Alow the launching of mulltiple Kappa simulations using an INI file for the parameters
"""
# -*- coding: utf-8 -*-
import itertools
import os
import random
import subprocess

from multiprocessing import Pool
from functools import partial
from pathlib import Path
import random


def combinaisons_making(varia):
    """
    Define and concatenate all the possible combination from a given lis

    Parameters:
    -----------
    varia: dictionnary
        contain the list of variables that will be modified and their values

    Returns:
    ------
    combination: list,
        contain of the combinaison for all the variables
    """
    value = []
    for var in varia:  # browse the list of variable
        value.append(varia[var])
    combination = (list(itertools.product(*value)))
    # concatenate all the possible combination for the all the variable
    return combination

def launch_kasim(kasim,time,varia,input_file,output,log_folder,repeat,combination):
    """
    Lauch KaSim

    Parameters:
    -----------
    kasim: string
        path for KaSim exec
    time: int
        simulation time to stop
    varia: dictionary
        contain the list of variables that will be modified and their values
    input_file: string
        folder where the .ka script are
    output_file: string
        folder where the .csv files will be create
    log_folder: string
        folder where the log files will be store
    repeat: int
        number of time a each simulation must be launch,
        used for measuring the stochastic impact on the simulation
    combination: list
        contain of the combinaison of all the variable
    """
    i = 0
    var_com = ""
    output_name = ""
    nb_repeat = str(random.randint(1, 1000000))
    time_simu = int(time)

    if not isinstance(varia, str): # create a file name from the variables modified, if not variables are modified used the default name
        for key in varia:
            # loop writting the part of the command use to specified the variables modified
            var_com = var_com + f" -var {key} {combination[i]}"
            output_name = output_name + f"{key}_{combination[i]}_"
            i += 1
        output_name = output_name.replace(".", ",")
    else:
        output_name = varia

    if ((len(combination) > 1) or (repeat > 1)):
        # In case of numerous test with repetition, the creation of numerous folder for each combination is advices
        output_file = f'{output}{output_name}/'
        if not os.path.isdir(output):
            os.mkdir(output)

    output_name = output_name + f"_{str(nb_repeat)}"
    command = f"{kasim} {input_file} {var_com} -l {time_simu} -d {log_folder} -o {output_file}{output_name}.csv"
    subprocess.run(command, shell=True, check=True)  # execute the KaSim command

def parallelized_launch(kasim, time, variables_test, input_file, output_file, log_folder, nb_para_job=4, repeat =1, ):
    """
    Launch the KaSim simulation parallelized

    Parameters:
    -----------
    kasim: string
        path for KaSim exec
    time: int
        simulation time to stop
    varia: dictionary
        contain the list of variables that will be modified and their values
    input_file: string
        folder where the .ka script are
    output_file: string
        folder where the .csv files will be create
    log_folder: string
        folder where the log files will be store
    nb_para_job: int
        number of parallel job you want to launch
    repeat: int
        number of time a each simulation must be launch,
        used for measuring the stochastic impact on the simulation
    """
    with Pool(processes=nb_para_job) as pool:
        # define the number of parallel works that will be launch simultinaly

        if not isinstance(variables_test, str):#test if there is variables that are modified to past the value to the next function
            combinations = combinaisons_making(variables_test)
        else:
            combinations = [variables_test]

        combinations_repeats = combinations * repeat

        func = partial(launch_kasim,
            kasim,
            time,
            variables_test,
            input_file,
            output_file,
            log_folder,
            repeat)
        # store and fuse all the parameter of the function launch_KaSim
        pool.map(func, combinations_repeats)
        # launch the parallelisation and does it iteratly by doing it for all the combinaition
        pool.close()
        pool.join()
        progressbar.destroy()

    