"""
GUI controller programme
python 3.10

"""

import tkinter as tk

import queue
import threading
from view_gui import *
from launch import *




def get_parameter_values(frame):
    """Extract the value in entry fro the parameter section and 
    check if there is any default value

    Parameter:
    ----------
    frame: Class tkinter.Frame"""
    parameters_test = {}
    frame.lop[frame.parameter_1_name_entry] = frame.parameter_1_value_entry
    for cle, valeur in frame.lop.items():
        if valeur.entry.get() == 'Default value':
            coor = valeur.entry.grid_info()
            warning_message_f2 = ctk.CTkLabel(frame,
                                           text= "*",
                                           font=('comic',48,'bold'),
                                           text_color="#0087F2")
            warning_message_f2.grid(column= (int(coor["column"])+1),
                                    row=int(coor["row"]))
        else:
            parameters_test[cle.entry.get()]= valeur.entry.get()

    if len(frame.lop) == len(parameters_test):
        # try:
        #     warning_message_f2.destroy()
        # except:
        #     pass
        return(parameters_test)

def get_path_values(frame):
    """Extract the value in entry for the path frame and
    check if there is any default value

    Parameter:
    ----------
    frame: Class tkinter.Frame
    """
    simulation_value = []
    simulation_parameter = [frame.kasim_path_entry.entry,
                      frame.model_path_entry.entry,
                      frame.output_path_entry.entry,
                      frame.log_path_entry.entry,
                      frame.job_number_entry.entry,
                      frame.replicat_entry.entry,
                      frame.simulation_time_entry.entry
                      ]
    for entry in simulation_parameter:
        if entry.get() == "Default value":
            coor = entry.grid_info()
            warning_message_f1 = ctk.CTkLabel(frame,
                                           text= "*",
                                           font=('comic',48,'bold'),
                                           text_color="#0087F2")
            warning_message_f1.grid(column= 2, row=coor["row"])
        else:
            simulation_value.append(entry.get())

    if len(simulation_parameter) == len(simulation_value):
        try:
            warning_message_f1.destroy()
            print('ull')
        except:
            pass
        return(simulation_value)

def tb_click(app, frame1, frame2):
    """function launching the thread class
    
    Parameter
    app: tk window class
    
    frame1: ctk frame class
    
    frame2: ctk frame class"""
    app.progbar()
    # app.prog_bar.start()
    app.queue = queue.Queue()
    ThreadedTask(app.queue, app, frame1, frame2).start()
    # app.after(100, app.process_queue)



class ThreadedTask(threading.Thread):
    """Class threading.thread  to launch the simulation
    
    The simulations are launch in a different thread than
    the one for the GUI, allowing other action in the GUI
    
    Attributes
    
    queue: class queue

    app: class App

    frame1 : class Frame1

    frame2 class Frame2
    """
    def __init__(self, queue, app, frame1, frame2):
        super().__init__()
        self.queue = queue
        self.app = app
        self.frame1 = frame1
        self.frame2 = frame2
        
    def run(self):
        parameters_simu = get_path_values(self.frame1)
        parameter_estimates = get_parameter_values(self.frame2)

        if parameters_simu is None or parameter_estimates is None:
            warning_gl = ctk.CTkLabel(self.app, text="Warning, parameter(s) not Fill",
                                    font=('comic',24))
            warning_gl.grid(column = 2, row = 2 )
        else:
            parallelized_launch(str(parameters_simu[0]), #KaSim
                        int(parameters_simu[6]), #Time
                        parameter_estimates, #variables 
                        parameters_simu[1], #Model
                        parameters_simu[2], #output
                        parameters_simu[3],#log
                        int(parameters_simu[4]), #job
                        int(parameters_simu[5])) #repeat

        self.queue.put("Task finished")   

   
