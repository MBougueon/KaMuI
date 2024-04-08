# -*- coding: utf-8 -*-
import configparser as cp
import os
import pathlib

import cairosvg


def import_ini(ini_file):
    """
    Import the parameters of the .ini file

    This function will extract the parameters of the .ini file for a use into this script.

    Parameters
    ----------
    ini_file : str
        the path of the .ini file were the files path used are store.
    """

    # Import the .ini file and its parameters
    config = cp.ConfigParser()
    config.read(ini_file)
    parameters = {}
    # section_name =[]

    for section in config.sections():  # browse ini file section
        # section_name.append(section)
        for key in config[section]:  # browse key in the section
            parameters[key] = config[section][key]  # extract the value and the key

    # Return the parameters
    return parameters
    # ,section_name


def svg_to_png(folder):
    """
    Convert all the svg file of a folder to png

    Parameters
    ----------
    folder: str
        path of the folder containing the svg
    """

    for path in pathlib.Path(folder).iterdir():
        if path.is_file():
            file_name = os.path.basename(path)  # extract the name of the file from the input path
            file_name = file_name.split('.')[0]  # extract the name of the csv file to used it for the graph
            svg_file = str(path)
            cairosvg.svg2png(url=svg_file, write_to=folder + file_name + '.png')
