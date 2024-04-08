"""
Alow the graph generation from KaSim simulations
"""


import os
import pathlib
import re
import shutil


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import log as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio   
import random

import utils

from collections import Counter

#When export graph in pdf a loading message can be print, this line is to prevent it.
pio.kaleido.scope.mathjax = None

def evnv(subfold, variables):
    """Extract variable name and value
    
    arameters:output
    -----------
    Folder: file_path
        folder where csv files are stocked
    """    
    stimulation_time = []
    var_v = []
    var_v_float = []
    for var in variables:
        
        var_temp = re.findall(str(var) + r"_\d+_", str(subfold))#some variable have number in their name, exclude them
        var_v.append(re.findall(r"\d+", "".join(var_temp)))
        if not var_temp:  # Some time the TGFB1 value can be a float so a test must be done
            var_temp = re.findall(str(var) + r"_\d+\,\d+_", str(subfold))
            var_v.append(re.findall(r"\d+,\d+", "".join(var_temp)))

        if var == 'nb_iteration' or var == 'interval_py' or var == 'second_input':
            stimulation_time.append(re.findall(r"\d+", "".join(var_temp)))
    for item in var_v:
        if len(item) > 1: #TGFB1 as a int into it's name, value extraction can't make the diff
            #this condition check if the value extract is good or if a part of the name of the value was extract to
            var_v_float.append(float(item[1]))

    return (var_v_float)

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

def dataframe_opening(folder, timing, variables, first_input_time):
    """
    Open all the csv file in the folder and open it to extract wanted information

    Parameters:output
    -----------
    Folder: string
        folder where csv files are stocked
    timing: list of integer
        timing where the agents occurences will be studied
    """
    
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
            var_temp = re.findall(str(var) + r"_\d+_", str(subfold))#some variable have number in their name, exclude them
            var_v.append(re.findall(r"\d+", "".join(var_temp)))
            if not var_temp:  # Some time the dT value can be a float so a test must be done
                var_temp = re.findall(str(var) + r"_\d+\.\d+_", str(subfold))
                var_v.append(re.findall(r"\d+.\d+", "".join(var_temp)))

            if var == 'nb_iteration' or var == 'interval_py' or var == 'second_input':
                stimulation_time.append(re.findall(r"\d+", "".join(var_temp)))
        for item in var_v:
            if len(item) > 1: #TGFB1 as a int into it's name, value extraction can't make the diff
                #this condition check if the value extract is good or if a part of the name of the value was extract to
                var_v_float.append(float(item[1]))
            else:
                var_v_float.append(float(item[0]))
        if ("second_input" in variables):
            for t in timing:
                time_to_extract.append(float(first_input_time) + (float(stimulation_time[2][0]) + (2*(float(stimulation_time[0][0])*(float(stimulation_time[1][0])/24))) + float(t)))
        else:

            for t in timing:
                time_to_extract.append(int(first_input_time) + (float(stimulation_time[0][0])*float(stimulation_time[1][0])/24)+ int(t))

        for path in pathlib.Path(file_path).iterdir():
            # list copy, whitout it both list will be incremented
            df = pd.DataFrame()
            if path.is_file():
                dframe = get_dataframe(path)
                dframe['[T]'] = dframe['[T]'].astype('int')
                col_name = dframe.columns
                #extract the wanted timing
                for time in time_to_extract:
                    df = df.append(dframe.loc[dframe['[T]'] == time])
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

def graph_simu(folder, output):
    """
    Make graph from the outpu file generate by KaSim

    Parameters:
    -----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    """

    # make graph for all the csv files of the folder
    for path in pathlib.Path(folder).iterdir():
        if path.is_file():
            dframe = get_dataframe(path)
            time = dframe["[T]"]  # selection the column time
            # fig = plt.figure(0.25, figsize=(30, 18))
            for col in dframe.columns:
                compt = dframe[col]
                # if col !=  "[T]" and col not in variables:
                if col == "\'COL1 concentration\'":
                    plt.plot(time, compt, label=col, linestyle="-", )
            # extract the name of the file from the input path
            filename = os.path.basename(path)
            # extract the name of the csv file to used it for the graph
            filename = filename.split('.')[0]
            plt.xlabel("Time (s)", fontsize=18)
            plt.ylabel("Ocurrences", fontsize=18)
            plt.title(filename, fontsize=18)
            plt.margins(0.005)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            filename = f"{output}{filename}.png"
            plt.savefig(filename, dpi=200)
            plt.close()
            plt.clf()

def graph_4d(folder, output, file_name, timing, varia, first_input_time):
    """
    Generate 4D graph

    Parameters:
    ----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    file_name: string
        general name for the graph file, variables values are add to this name
    timing: list integer
        timing that will be studied
    varia: list string
        contain the list of variables that will be modified and their values
    """

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    dframe = dataframe_opening(folder, timing,first_input_time )

    for col in dframe.columns:
        # graph_name = file_name
        if not (col in varia):
            fig = px.scatter_3d(
                dframe, x='[T]', y=varia[0], z=varia[1],
                color=col, size=variables[2],
                size_max=30, opacity=0.6,
                width=900, height=780
            )

            graph_name = f"{output}{str(col)}.{file_name}"
            camera = dict(
                eye=dict(x=1, y=2.1, z=1)  # change the view of the graph
            )
            fig.update_layout(scene_camera=camera, margin=dict(l=0, r=0, b=0, t=0))

            if file_name == "html":
                plotly.offline.plot(fig, filename=graph_name, auto_open=False)  # dynamic graph
            else:
                fig.write_image(graph_name, scale=1)

def surface_graph(folder, output, timing, variables, data_measure, first_input_time, graph_format):
    """Generate surface 4D graph

    Parameters:
    ----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    timing: list integer
        timing that will be studied
    variable: list string
        contain the list of variables that will be modified and their values
    data_measure: TODO
    Unit: TODO
    """
    df = dataframe_opening(folder, timing, variables, first_input_time)
    #extract the values timing by timing to make separated graph


    for nb_time in range(0, len(timing)):
        values_timing = []
        for i in range(nb_time, len(df), len(timing)):
            values_timing.append(i)
        dfram = pd.DataFrame(df.iloc[values_timing], columns=df.columns)
        # means the values for each parameter CHECK IF ITS WORK WITH TIME !!!!!
        graph_title = f"{data_measure} Surface representation {timing[nb_time]} days after the last input of TGFB1"
        graph_name = f"{folder}sufrace_graph_{data_measure}_{timing[nb_time]}_d_after_last_input_of_TGFB1.{graph_format}"
        # Stock the value for the X,y,z axis
        X = dfram[variables[0]]
        Y = dfram[variables[1]]
        Z = dfram[variables[2]]


        #creation of the 4D surface plot
        fig = go.Figure(data = [go.Mesh3d(x=X,
                        y=Y,
                        z=Z,
                        intensity=dfram[data_measure],
                        alphahull=0 #change the algotrithm used for data reprsentation, 0 for convex.
        )])
        fig.update_layout(scene = dict(
                    xaxis_title=variables[0],
                    yaxis_title=variables[1],
                    zaxis_title=variables[2]))
          

        #fig.show()
        if graph_format == "html":
            plotly.offline.plot(fig, filename=graph_name, auto_open=False)  # dynamic graph
        else:
            fig.write_image(graph_name, scale=1)

def mean_graph(folder, output, nb_iteration, interval, input_value,graph_format):
    """
    Open all the csv file in the folder and to extract the mean,
    the max and min for each values of each species

    Parameters:
    -----------
    folder: string
        folder where the .csv files will be create
    output: string:
        folder where the graph will be store
    nb_iteration: integer
        number of input in the simulation
    interval: integer
        number of time between each input during the simulation
    input_value: interger
        number of TGFB1 by cells
    """
    #TODO Ajouter variable de temps pour arreter le graph sur un timing souhaité

    if not os.path.isdir(output):
        os.mkdir(output)

    frames = []
    # line_col = 0
    # extraction of the mean, var, min and max from outputs
    for path in pathlib.Path(folder).iterdir():
        if path.is_file():
            dframe = get_dataframe(path)
            # delete the exacte timing of input causing the appearance of spikes
            dframe['[T]'] = dframe['[T]'].astype('int')
            frames.append(dframe)
    result = pd.concat(frames, ignore_index=True)
    # result_one_by_one = pd.DataFrame(result.groupby(['[T]']))
    result_min =  pd.DataFrame(result.groupby(['[T]']).min())
    result_max =  pd.DataFrame(result.groupby(['[T]']).max())
    result_mean = pd.DataFrame(result.groupby(['[T]']).mean())
    result_var = pd.DataFrame(result.groupby(['[T]']).var())


    # result_min = result_min1.head(190)
    # result_max = result_max1.head(190)
    # result_mean = result_mean1.head(190)
    # result_var = result_var1.head(190)
    time = result_mean.index  # selection the column time

    # mean and var graph section
    solo_graph = {'Mean': result_mean, 'Variance': result_var}
    for key in solo_graph:
        fig = go.Figure()
        for col in solo_graph[key].columns:
            occurences = solo_graph[key][col]
            fig.add_trace(go.Scatter(
                x=time, y=occurences,
                mode='lines',
                name=col
            ))
            fig_title = f"{key} of the simulations with {input_value} TGFB by input, {nb_iteration} inputs spaced by {interval} hours"
            fig.update_layout(
                xaxis_title='Time (day)',
                yaxis_title='number of cell occurences (10^3)',
                font=dict(
                    family="Courier New, monospace",
                    size=48,
                )
            )
        # fig.update_layout(yaxis_range=[0,17000])
        fig_name = f"{key}_graph_of_{input_value}_TGFB_{nb_iteration}_input_{interval}h_interval.{graph_format}"
        fig_save = f"{output}{fig_name}"
        fig.write_image(fig_save, width=1920, height=1080)


    for col in result.columns:
        r = random.randrange(1, 240)
        g = random.randrange(1, 240)
        b = random.randrange(1, 240)
        color = f"rgb({r},{g},{b})"
        if col != "[T]":
            if col =="\'COL1 tot\'":
                y_axis_title = f"COL1 arbitrary unit"
            else:
                y_axis_title = f"{col} occurences"

            min_occurences = result_min[col]
            max_occurences = result_max[col]
            mean_occurences = result_mean[col]
            
            fig = go.Figure([
                go.Scatter(
                    name=col,
                    x=result_mean.index,
                    y=mean_occurences,
                    mode='lines',
                    line=dict(color=color),
                ),
                go.Scatter(
                    name='Upper Bound',
                    x=result_max.index,
                    y=max_occurences,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='Lower Bound',
                    x=result_min.index,
                    y=min_occurences,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
            # if col =="\'COL1 tot\'":
            #     fig.update_layout(yaxis_range=[0,8])
            fig.update_layout(
                font=dict(
                    family="Courier New, monospace",
                    size=48,  # Set the font size here
                ),
                legend=dict( yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99),
                yaxis_title=y_axis_title,
                xaxis_title='Time (day)',
                #title=f"{col} of the simulations with {input_value} TGFB by input, {nb_iteration} inputs spaced by {interval} hours",
                hovermode="x",
                title=dict(text=f"{col} dynamics over 50 simulations",
                        font=dict(size=28))
            )
            # fig.update_layout(yaxis=dict(type='log'))

            fig_name = f"{col}_graph_of_{input_value}_TGFB_{nb_iteration}_input_{interval}h_interval.{graph_format}"
            fig_save = f"{output}{fig_name}"
            fig.write_image(fig_save, width=1920, height=1080, scale =2)

def sensibility_graph(folder, output, data_measure, graph_format):
    """Generate surface 2D graph of a wanted variable by means the simulation

    Parameters:
    ----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    data_measure: string
        Name of the species that should be study
    graph_format: string
        format of the graph output: png,jpg,pdf,html...
    """
    if not os.path.isdir(output):
        os.mkdir(output)

    # extraction of the mean
    df = pd.DataFrame()
    for subfold in os.listdir(folder):#go into sub folders
        frames = []
        if subfold != "graph":#Pass folder with graph data
            file_path = f"{folder}{subfold}/"#concatenattion to form the path were the files are 
            for path in sorted(pathlib.Path(file_path).iterdir()):#go into files in the subfolder
                if str(path) != f"{file_path}graph":
                    dframe = get_dataframe(path)
                    # delete the exacte timing of input causing the appearance of spikes
                    dframe['[T]'] = dframe['[T]'].astype('int')
                    frames.append(dframe)
                    result = pd.concat(frames, ignore_index=True)
                    result_mean = pd.DataFrame(result.groupby(['[T]']).mean())
        #extract the wanted measure for the graph
            df[subfold] = result_mean[data_measure]

    df = df.rename(columns={'nb_iteration_16_interval_py_84_': 'Kisseleva 16 injections separated by 84h'})
    #df.set_index(result_mean.index)
    #df.set_index(result_mean.iloc[0:243].index)

    time = df.index
    fig = go.Figure()
    #color for the trace
    list_color = ['#6d4cb4','#12085e','#6d0908','#532217',
                '#3797cb','#7fdafc','#e89b01','#5f811f',
                '#195e2b','#5f244f','#537a75', '#005e80',
                '#748b04','#695480','#88b8eb','#eacb01']
    i = 0
    for col in sorted(df.columns):
        random_col = list_color[i]
        occurences = df[col]
        fig.add_trace(go.Scatter(
            x=time, y=occurences,
            mode='lines',
            name=col,
            line=dict(color=random_col)
        ))
        fig.update_layout(
            #title=fig_title,
            xaxis_title='Time (day) ',
            yaxis_title='Arbitary unit',
            font=dict(
                family="Courier New, monospace",
                size=28),
            title=dict(text=f"Sensitivity analysis of the model by modifying the TGFB1 input parameters,{data_measure}",
                        font=dict(size=28))
        )
        list_color.remove(random_col)
        fig.update_yaxes(range=[0, 8])
    fig_name = f"Sensitivity analysis of the model by modifying the TGFB1 input parameters {data_measure}.{graph_format}"
    fig_save = f"{output}{fig_name}"
    fig.write_image(fig_save, width=1920, height=1080)

def sensibility_3d_graph(folder, output, data_measure, graph_format, variables):
    """Generate surface 3D graph of a wanted variable by means the simulation

    Parameters:
    ----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    data_measure: string
        Name of the species that should be study
    graph_format: string
        format of the graph output: png,jpg,pdf,html...
    variables: string
        name of the variable that will be plot
    """
    #checl if output exist, otherwise create it
    if not os.path.isdir(output):
        os.mkdir(output)

    # extraction of the mean
    df = pd.DataFrame()
    frame = []
    list_color = ['#901960','#77a8fd','#5d2b51','#295a0c',
                '#f5a004','#1f1090']
    
    for subfold in os.listdir(folder):#go into sub folders
        frames = []
        if subfold != "graph":#Pass folder with graph data
            file_path = f"{folder}{subfold}/"#concatenattion to form the path were the files are 
            for path in pathlib.Path(file_path).iterdir():#go into files in the subfolder
                if str(path) != f"{file_path}graph":
                    dframe = get_dataframe(path)
                    # delete the exacte timing of input causing the appearance of spikes
                    dframe['[T]'] = dframe['[T]'].astype('int')
                    frames.append(dframe)
                    result = pd.concat(frames, ignore_index=True)
                    result_mean = pd.DataFrame(result.groupby(['[T]']).mean())
                    
                    #the following block extract the value of the number of input and TGFB1 concentration from the name of the folder
                    var_temp = re.findall("nb_iteration" + r"_\d+_", str(subfold))#some variable have number in their name, exclude them
                    nb_iteration = re.findall(r"\d+", "".join(var_temp))             
                    # var_temp = re.findall("inactivation_pourcentage_react" + r"_\d+_", str(subfold))#some variable have number in their name, exclude them
                    var_temp = re.findall("TGFB1_pool_by_cell" + r"_\d+_", str(subfold))#some variable have number in their name, exclude them
                    if not var_temp:  # Some time the TGFB1 value can be a float so a test must be done
                        var_temp = re.findall("TGFB1_pool_by_cell" + r"_\d+\,\d+_", str(subfold))
                        # var_temp = re.findall("inactivation_pourcentage_react" + r"_\d+\,\d+_", str(subfold))
                        tgfb1_concentration_temp = re.findall(r"\d+,\d+", "".join(var_temp))
                    else:
                        tgfb1_concentration_temp = re.findall(r"\d+", "".join(var_temp))
                    if len(tgfb1_concentration_temp) > 1: #TGFB1 as a int into it's name, value extraction can't make the diff
                        #this condition check if the value extract is good or if a part of the name of the value was extract to
                        #there is a factor 10000 in the model for the number of TGFB1 molecules
                        # tgfb1_concentration = float(tgfb1_concentration_temp[1])*10000
                        tgfb1_concentration = float(tgfb1_concentration_temp[1])
                    else:
                        tgfb1_concentration = tgfb1_concentration_temp[0]
                        tgfb1_concentration = tgfb1_concentration.replace(",", "." )
                        # tgfb1_concentration = float(tgfb1_concentration) * 10000
                        tgfb1_concentration = float(tgfb1_concentration)

                    #Add column about the number of input and the concentration and time    
                    result_mean['nb_iteration']= len(result_mean)*[int(nb_iteration[0])]
                    # result_mean['inactivation_pourcentage_react'] = len(result_mean)*[tgfb1_concentration]
                    result_mean['TGFB1_by_cells'] = len(result_mean)*[tgfb1_concentration]
                    result_mean['Time'] = result_mean.index
                    #result_mean['condition']= len(result_mean)*[f"{nb_iteration[0]} Input, {tgfb1_concentration} TGFB1 by cells (*10 000)"]
        #extract the wanted measure for the graph
            frame.append(result_mean[[data_measure, 'nb_iteration', 'TGFB1_by_cells', 'Time']])
            # frame.append(result_mean[[data_measure, 'nb_iteration', 'inactivation_pourcentage_react', 'Time']])

    df = pd.concat(frame)
    fig = go.Figure()
    #To make the 3D graph, each combination between nb_iteration and TGFB1_by_cell must be add one at a time
    #Otherwise plotly link them making the graph unreadable
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=25)

    list_color = ['#901960','#77a8fd','#5d2b51','#295a0c',
                '#f5a004','#1f1090']
    concentrat = df['TGFB1_by_cells'].unique()
    # concentrat = df['inactivation_pourcentage_react'].unique()
    concentrat.sort()
    df2 = pd.DataFrame(df[df["Time"] <= 200])

    # for val in concentrat:
    #     concentration.append(float(val))
    #     col = list_color[i]
    #     # df_graph = pd.DataFrame(df2[df2['TGFB1_by_cells'] == val])
    #     df_graph = pd.DataFrame(df2[df2['inactivation_pourcentage_react'] == val])
        
    #     # df = df[df['TGFB1_by_cells'] != val]

    #     # ax.plot(np.log(df_graph["TGFB1_by_cells"]),df_graph["Time"], df_graph[data_measure], color=col)
    #     ax.plot(df_graph["inactivation_pourcentage_react"],df_graph["Time"], df_graph[data_measure], color=col)
    #     # verts = [list(zip(np.log(df_graph["TGFB1_by_cells"]),df_graph["Time"], df_graph[data_measure]))]
    #     verts = [list(zip(df_graph["inactivation_pourcentage_react"],df_graph["Time"], df_graph[data_measure]))]
    #     ax.add_collection3d(Poly3DCollection(verts, color = col, alpha=.3))
    #     i+=1
    xtick_coor= []
    for i in range(len(concentrat)):
        
        val = concentrat[i]
        # concentration.append(float(val))
        col = list_color[i]
        df_graph = pd.DataFrame(df2[df2['TGFB1_by_cells'] == val])
        # df_graph = pd.DataFrame(df2[df2['inactivation_pourcentage_react'] == val])
        
        xaxix = [i-0.8]*len(df_graph[data_measure])
        # df = df[df['TGFB1_by_cells'] != val]

        # ax.plot(np.log(df_graph["TGFB1_by_cells"]),df_graph["Time"], df_graph[data_measure], color=col)
        ax.plot(xaxix,df_graph["Time"], df_graph[data_measure], color=col)
        # verts = [list(zip(np.log(df_graph["TGFB1_by_cells"]),df_graph["Time"], df_graph[data_measure]))]
        verts = [list(zip(xaxix,df_graph["Time"], df_graph[data_measure]))]
        ax.add_collection3d(Poly3DCollection(verts, color = col, alpha=.3))
        # xtick_coor.append(log(val))
   # ax.set_xticks(np.log(concentration))
    # tick_val =[]
    # for i in concentration:
        # tick_val.append(f"{i/1000}")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # for i in range(1,len(labels)-1,1):
    #     print(xtick_coor[(i-1)])
    #     labels[i]=xtick_coor[(i-1)]

    ax.set_xticks(list(range(len(concentrat))))
    # # ax.set_xticklabels(tick_val, rotation = -10)
    ax.set_xticklabels(concentrat, rotation = -25,verticalalignment='center',
                    horizontalalignment='left')

    time_ticks = np.arange(0, 200, 50)
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_ticks)
    data_measure_ticks = np.arange(0, 8, 1)#to change for other species
    ax.set_zticks(data_measure_ticks)
    ax.set_zticklabels(data_measure_ticks)

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    ax.xaxis.labelpad = 20
        # Show the plot.
    ax.invert_xaxis()
    ax.set_xlabel('TGFB1 concentration \n (molecules/cell).10^3', fontsize=12 )
    # ax.set_xlabel('Proportion of react_MFB Inactivated', fontsize=10 )
    ax.set_ylabel('Time in days', fontsize=12 )
    ax.set_zlabel(f"{data_measure} occurences", fontsize=12 )
    # plt.show()

    fig_name_tiff = f"3D_sensitivity_analysis_of_the_model_by_modifying_the_TGFB1_input_parameters_{data_measure}.tiff"
    fig_name = f"3D_sensitivity_analysis_of_the_model_by_modifying_the_TGFB1_input_parameters_{data_measure}.{graph_format}"
    fig_save_tiff = f"{output}{fig_name_tiff}"
    fig_save = f"{output}{fig_name}"
    plt.savefig(fig_save_tiff, dpi=1200, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(fig_save, dpi=1200, format=graph_format)











    #     fig.add_trace(go.Scatter3d(x=df_graph["TGFB1_by_cells"], 
    #                             y=df_graph["Time"], 
    #                             z=df_graph[data_measure],
    #                             mode='lines',
    #                             #legendgroup='TGFB1_by_cells',  # this can be any string, not just "group"
    #                             name=val,
    #                             # surfaceaxis=0,
    #                             # line=dict(color=random_color),
    #                             )
    #                 )
        
    # fig.update_traces(line=dict(width=4))
    # fig.update_scenes(xaxis_autorange="reversed",yaxis_autorange="reversed")           

    # fig.update_layout(scene = dict(
    #                 xaxis_title="TGFB1 concentration (log(molecules/cell))",
    #                 yaxis_title="Time in days",
    #                 zaxis_title=f"{data_measure} occurences",
    #                 xaxis=dict(dtick=1, type='log')),
    #                 legend_traceorder="normal",
    #                 )
    # fig_name = f"3D_sensitivity_analysis_of_the_model_by_modifying_the_TGFB1_input_parameters_{data_measure}.{graph_format}"
    # fig_save = f"{output}{fig_name}"
    # fig.show()
    #fig.write_html(fig_save)
    # fig.write_image(fig_save, width=1920, height=1080, scale=2)

def graph_4d_v2(folder, output, timing, first_input_time, data_measure,graph_format):
    """
    Generate 4D graph

    Parameters:
    ----------
    folder: string
        path, folder containing the csv generate by KaSIm
    output: string
        path, folder where the graph will be store
    file_name: string
        general name for the graph file, variables values are add to this name
    timing: list integer
        timing that will be studied
    varia: list string
        contain the list of variables that will be modified and their values
    """
    time_to_extract = []#timing.copy()
    frames_all = []
    nb_input = []
    delta_T= []
    periods = []
    if not os.path.isdir(output):
        os.mkdir(output)

    for subfold in os.listdir(folder):#go into sub folders
        if subfold != "graph":#Pass folder with graph data
            period = re.findall(r'\d+', "".join(subfold))
            subfold_path = f"{folder}{subfold}/"
            for subsubfold in os.listdir(subfold_path):
                time_to_extract = []#timing.copy()
                frames =[]         
                file_path = f"{folder}{subfold}/{subsubfold}/"#files path
                
                var_temp = re.findall("nb_iteration" + r"_\d+_", str(subsubfold))#some variable have number in their name, exclude them
                nb_iteration = re.findall(r"\d+", "".join(var_temp))                                                    
                var_temp = re.findall("TGFB1_pool_by_cell" + r"_\d+_", str(subsubfold))#some variable have number in their name, exclude them

                nb_input.append(int(nb_iteration[0]))
                # delta_T.append(float(period[0])/int(nb_iteration[0]))
                #periods.append(float(period[0]))           
                periods.append(float(period[0])/100)           
                if not var_temp:  # Some time the TGFB1 value can be a float so a test must be done
                    var_temp = re.findall("TGFB1_pool_by_cell" + r"_\d+\,\d+_", str(subsubfold))
                    tgfb1_concentration_temp = re.findall(r"\d+,\d+", "".join(var_temp))
                else:
                    tgfb1_concentration_temp = re.findall(r"\d+", "".join(var_temp))
                
                if len(tgfb1_concentration_temp) > 1: #TGFB1 as a int into it's name, value extraction can't make the diff
                                                    #this condition check if the value extract is good or if a part of the name of the value was extract to
                                                    #there is a factor 10000 in the model for the number of TGFB1 molecules
                    tgfb1_concentration = float(tgfb1_concentration_temp[1])*10000
                else:
                    tgfb1_concentration = tgfb1_concentration_temp[0]
                    tgfb1_concentration = tgfb1_concentration.replace(",", "." )
                    tgfb1_concentration = float(tgfb1_concentration) * 10000


                for path in pathlib.Path(file_path).iterdir():#go into files in the subfolder
                    if str(path) != f"{file_path}graph":
                        dframe = get_dataframe(path)
                        time_to_extract = []#timing.copy()
                        # delete the exacte timing of input causing the appearance of spikes
                        dframe['[T]'] = dframe['[T]'].astype('int')
                        frames.append(dframe)
                        result = pd.concat(frames, ignore_index=True)
                        result_mean = pd.DataFrame(result.groupby(['[T]']).mean())
                        df = pd.DataFrame()
                        
                        for t in timing:
                            # stimulation_t = (float(period[0])/int(nb_iteration[0]))*(int(nb_iteration[0])-1)
                            stimulation_t = (60/int(nb_iteration[0]))*(int(nb_iteration[0])-1)
                            
                            time_to_extract.append(int(first_input_time) + stimulation_t + int(t))

                        for time in time_to_extract:
                            df = df.append(result_mean.iloc[[time]])
                df = df[[data_measure]]
                df['nb_iteration']= len(df)*[int(nb_iteration[0])]
                df['TGFB1_by_cells'] = len(df)*[tgfb1_concentration]
                df['period']= len(df)*[int(period[0])/100]
                # df['period']= len(df)*[int(period[0])]
                df['dT'] = len(df)*[int(period[0])/int(nb_iteration[0])]
                df['T'] = timing
                frames_all.append(df)
    df_final = pd.concat(frames_all, ignore_index=True)
    maxrange = df_final[data_measure].max()


    for time in timing:
        fig = px.scatter_3d(df_final.loc[df_final['T'] == time], x='nb_iteration', y='period', z='TGFB1_by_cells',
                color=data_measure, opacity=0.7, range_color=(0,maxrange),
                color_continuous_scale="burgyl")

        delta_T= list(Counter(delta_T))
        nb_input = list(Counter(nb_input))
        periods = list(Counter(periods))

        camera = dict(
                eye=dict(x=1.4, y=2, z=0.7)  # change the view of the graph
            )
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                    scene = dict(
                    xaxis_title="Number of TGFB1 inputs",
                    zaxis_title="TGFB1 concentration Log(molecules/cell)",
                    # yaxis_title="Treatment time (days)",
                    yaxis_title="% of react_MFB inactivated ",
                    # yaxis_title="% of iHSC reverting to qHSC ",
                    zaxis=dict(gridcolor="gray", dtick=1, type='log',title_font=dict(size=18)),
                    yaxis=dict(gridcolor="gray",title_font=dict(size=18), tickmode = 'array', tickvals= periods, tickformat= '.0%'),
                    xaxis=dict(gridcolor="gray",title_font=dict(size=18), tickmode = 'array',tickvals =nb_input)),
                    legend_traceorder="normal",
                    font=dict(size= 12),
                    scene_camera=camera)
        fig.update_traces(marker=dict(size=10))
        fig.update_scenes(xaxis_autorange="reversed",yaxis_autorange="reversed")           
                    
        
        fig_name = f"3D_sensitivity_analysis_of_the_model_by_modifying_the_TGFB1_input_parameters_{data_measure}_{time}_days_after_last_input.{graph_format}"
        fig_save = f"{output}{fig_name}"
        fig.write_image(fig_save, width=1920, height=1080, scale=2)
        
        #fig.show()





if __name__ == '__main__':
    INI_FILE = "graph_kaSim.ini"
    parameters = utils.import_ini(INI_FILE)
    variables = parameters['variables'].split(',')


    # graph_4d_v2(parameters['input_graph'], 
    #             parameters['output_graph'],
    #             parameters['timing'].split(','),
    #             parameters['first_input_time'],
    #             parameters['variable_analysed'],
    #             str(parameters['graph_format'])
    #             )
    # sensibility_graph(str(parameters['input_graph']),
    #                     parameters['output_graph'],
    #                     parameters['variable_analysed'],
    #                     str(parameters['graph_format']))

    sensibility_3d_graph(str(parameters['input_graph']),
                        parameters['output_graph'],
                        parameters['variable_analysed'],
                        str(parameters['graph_format']),
                        variables)

    # mean_graph(parameters['input_graph'], 
    #     parameters['output_graph'],
    #     int(parameters['nb_iteration']), 
    #     int(parameters['interval_py']),
    #     parameters['tgfb1_pool_by_cell'],
    #     parameters['graph_format'])

    # for f in os.listdir(parameters['input_graph']):
    #     output = f"{parameters['input_graph']}{f}/graph/"
    #     if not os.path.isdir(output):
    #         os.mkdir(output)
    # surface_graph(str(parameters['input_graph']),
    #               parameters['output_graph'], 
    #               parameters['timing'].split(','), 
    #               variables,
    #               parameters['surface_plot_variable_analysed'], 
    #               parameters['first_input_time'],
    #               str(parameters['graph_format'])

    

    # if (parameters['curves_graph']) == 'False':
    #     if len(variables) == 3:
    #         graph_4d(str(parameters['input_graph']),str(parameters['output_graph']),
    #                   str(parameters['graph_format']), parameters['timing'].split(','), variables)
    #     else:
    #         print("The number of variables for this graph must be 3")
    # else:
    #graph_simu(str(parameters['input_graph']), str(parameters['output_graph']))
    # data = dataframe_opening(str(parameters['input_graph']),parameters['timing'].split(','))


