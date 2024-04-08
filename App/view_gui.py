"""
GUI interface programme
python 3.10
"""

# import tkinter.font as tkFont
import customtkinter as ctk
from tkinter import filedialog

import queue
import controller_gui as cg
from PIL import Image

loadimage_plus = ctk.CTkImage(Image.open("button/plus-circle.png"))
loadimage_minus = ctk.CTkImage(Image.open("button/minus-circle.png"))


class Entry():
    """
    A tkinter Entry to generalized the visual of all entries

    This class is corresponding to the entry used in the gui with commun feature

    Attibutes:

    Entry: customtkinter Entry"""
    def __init__(self, container, text, coory, coorx, entry_width, sticky="w"):
        self.entry = ctk.CTkEntry(container,
                                  text_color="grey",
                                  font=("comic",14),
                                  width=(entry_width*5),
                                  corner_radius=4)
        self.entry.insert(0, text)
        self.entry.grid(column=coory, row = coorx, pady= 20, sticky=sticky)
        super().__init__()

class FolderButton():
    """
    A tkinter Buttom to generalized the visual of all button asking for path

    This class is corresponding to the buttom used to ask a path
    
    Attibutes:
    
    Entry: customtkinter button"""
    def __init__(self, container, entry):
        self.loadimage_folder = ctk.CTkImage(Image.open("button/folder.png"))
        self.fb = ctk.CTkButton(container,image=self.loadimage_folder,
                                                 text="",
                                                 fg_color="transparent",
                                                 width = 8,
                                                 command=lambda: container.get_path(entry))

class Frame1(ctk.CTkFrame):
    """
    A customtkinter frame containing the variables concerning the paths

    This class is corresponding the frame containing information
    about path needed to launch simulation

    Attributes:
    titre_simu: customtkinter label

    kasim_path: customtkinter label, naming the entry section for the path of KaSim excecutable
    kasim_path_entry: Class Entry for the path of KaSim excecutable
    folder_button_kasim: Class FoldeButtom, asked the path for KaSim excecutable

    model_path: customtkinter label, naming the entry section for the path of the model
    model_path_entry: Class Entry entry for the path of the model
    folder_button_model: Class FoldeButtom, asked the path for the model

    output_path: customtkinter label naming the entry section for the path of the output file
    output_path_entry: Class Entry entry for the path of the output file
    folder_button_output: Class FoldeButtom, asked the path for the simultation output

    log_path: customtkinter label naming the entry section for the path of the log file
    log_path_entry: Class Entry entry for the path of the log file
    folder_button_log: Class FoldeButtom, asked the path for the simulation log

    job_number: customtkinter label naming the entry section for the number of parallelized job
    job_number_entry: Class Entry entry for the path of the number of parallelized job
    decrease_job_button: customtkinter button, increase the number of job
    increase_job_button: customtkinter button, decreased the number of job

    replicat: tkinter customtkinter, naming the entry section for the number of replicat wanted
    replicat_entry: Class Entry for the the number of replicat wanted
    decrease_rep_button: customtkinter button, increase the number of replicat
    increase_rep_button: customtkinter button, decreased the number of replicat

    simulation_time: customtkinter label naming the entry section for the simulation time
    simulation_time_entry: Class Entry entry for the simulation time

    """
    def __init__(self, container):
        super().__init__(container)
        self.create_widgets()
        self.grid(column = 1, row = 1, sticky='ne')
        self.grid_rowconfigure(4, minsize=10)

    def get_path(self, entry_section):
        """
        Open a window to select a wanted path
        Overwrite the default value with the wanted paht
        
        Parameter
        ---------
        entry_section: Class Entry: entry were the path will be store"""
        if entry_section == self.model_path_entry:
            folder_selected = filedialog.askopenfilename()
        else:
            folder_selected = filedialog.askdirectory()
        if len(folder_selected) > 0:
            entry_section.entry.delete(0,"end")
            entry_section.entry.insert(0, folder_selected)
            entry_section.entry.configure(width=(len(folder_selected)*10))

    def increased_job(self):
        """
        Increased the value for the number of job to be parallelized
        """
        job_value = int(self.job_number_entry.entry.get())
        if 1 < job_value < 8:
            job_value = job_value + 1
            self.job_number_entry.entry.delete(0)
            self.job_number_entry.entry.insert(0,str(job_value))
        elif job_value == 1:
            self.mini_job_label.destroy()
            job_value = job_value + 1
            self.job_number_entry.entry.delete(0)
            self.job_number_entry.entry.insert(0,str(job_value))       
        else:
            self.mini_job_label = ctk.CTkLabel(self.frame_job,
                                               text= "Maximal number of job reach",
                                               text_color="#0087F2",
                                               font=('comic', 14,'bold')
                                               )
            self.mini_job_label.grid(column = 3, row=0)

    def decreased_job(self):
        """
        Decreased the value for the number of job to be parallelized
        """
        job_value = int(self.job_number_entry.entry.get())
        if 1 < job_value < 8 :
            job_value = job_value - 1
            self.job_number_entry.entry.delete(0)
            self.job_number_entry.entry.insert(0,str(job_value))
        elif job_value == 8:
            self.mini_job_label.destroy()
            job_value = job_value - 1
            self.job_number_entry.entry.delete(0)
            self.job_number_entry.entry.insert(0,str(job_value))        
        else:
            self.mini_job_label = ctk.CTkLabel(self.frame_job,
                                               text= "Minimal number of job reach",
                                               text_color="#0087F2",
                                               font=('comic', 14,'bold'))
            self.mini_job_label.grid(column = 3, row=0)
    
    def increased_rep(self):
        """
        Increased the value for the number of repeted simulation, used to get the profil of the model (SSA)
        """
        rep_value = int(self.replicat_entry.entry.get())
        if rep_value >= 10:
            rep_value = rep_value + 5
            self.replicat_entry.entry.delete(0,"end")
            self.replicat_entry.entry.insert(0,str(rep_value))
        elif rep_value ==1:
            self.mini_rep_label.destroy()
            rep_value = rep_value + 1
            self.replicat_entry.entry.delete(0,"end")
            self.replicat_entry.entry.insert(0,str(rep_value))
        else:
            rep_value = rep_value + 1
            self.replicat_entry.entry.delete(0,"end")
            self.replicat_entry.entry.insert(0,str(rep_value))
        
        self.replicat_entry.entry.configure(width=(len(str(rep_value))*20))
    
    def decreased_rep(self):
        """
        Decreased the value for the number of repeted simulation, used to get the profil of the model (SSA)
        """
        rep_value = int(self.replicat_entry.entry.get())
        if rep_value > 10:
            rep_value = rep_value - 5
            self.replicat_entry.entry.delete(0,"end")
            self.replicat_entry.entry.insert(0,str(rep_value))
        elif rep_value <= 10 and rep_value > 1:
            rep_value = rep_value - 1
            self.replicat_entry.entry.delete(0,"end")
            self.replicat_entry.entry.insert(0,str(rep_value))
        elif  rep_value == 1:
            self.mini_rep_label = ctk.CTkLabel(self.frame_rep,
                                               text= "Minimal number of repetitions reach",
                                               text_color="#0087F2",
                                               font=('comic', 14,'bold'))
            self.mini_rep_label.grid(column = 3, row=1)

        self.replicat_entry.entry.configure(width=(len(str(rep_value))*20))

    def create_widgets(self):
        """Create widgets include in the frame
        """
        self.configure(fg_color="transparent")
        self.loadimage_folder = ctk.CTkImage(Image.open("button/folder.png"))
        self.loadimage_plus = ctk.CTkImage(Image.open("button/plus-circle.png"))
        self.loadimage_minus = ctk.CTkImage(Image.open("button/minus-circle.png"))


        self.titre_simu = ctk.CTkLabel(self,
                                       text='Simulation',
                                       font=('comic',36,'bold'),
                                    #    text_color="#0087F2"
                                    )
        self.titre_simu.grid(column=0, row = 0, pady = 20)

        self.kasim_path = ctk.CTkLabel(self,
                                       text='Path of KaSim executable',
                                       font=('comic',24))
        self.kasim_path.grid(column=0, row = 2, sticky='w')
        self.kasim_path_entry = Entry(self,
                                      "/Tools/KappaTools-master/bin/KaSim",
                                      0,
                                      3,
                                      75)

        self.folder_button_kasim = FolderButton(self, self.kasim_path_entry)
        self.folder_button_kasim.fb.grid(column=1, row=3)


        self.model_path = ctk.CTkLabel(self,
                                       text='Path of the model',
                                       font=('comic',24))
        self.model_path.grid(column=0, row = 4, sticky='w')
        self.model_path_entry = Entry(self,
                                      "/home/palantir/these_TGF_B/models/code/These_model/ECM_HSC_model/final_model/test_KaMi/HSC_dynamics_model.ka",
                                      0,
                                      5,
                                      75)
        self.folder_button_model = FolderButton(self, self.model_path_entry)
        self.folder_button_model.fb.grid(column=1, row=5)

        self.output_path = ctk.CTkLabel(self,
                                        text='Path of the output',
                                        font=('comic',24))
        self.output_path.grid(column=0, row = 6, sticky='w')
        self.output_path_entry = Entry(self,
                                        "/home/palantir/these_TGF_B/models/code/These_model/ECM_HSC_model/final_model/test_KaMi/",
                                        0,
                                        7,
                                        75)
        self.folder_button_output = FolderButton(self, self.output_path_entry)
        self.folder_button_output.fb.grid(column=1, row=7)

        self.log_path = ctk.CTkLabel(self,
                                     text='Path of the log file',
                                     font=('comic',24))
        self.log_path.grid(column=0, row = 8, sticky='w')
        self.log_path_entry = Entry(self,
                                    "/home/palantir/these_TGF_B/models/code/These_model/ECM_HSC_model/final_model/test_KaMi/",
                                    0,
                                    9,
                                    75)
        self.folder_button_log = FolderButton(self, self.log_path_entry)
        self.folder_button_log.fb.grid(column=1, row=9)

        self.job_number = ctk.CTkLabel(self,
                                       text='Number of jobs parallelized',
                                       font=('comic',24))
        self.job_number.grid(column=0, row = 10, sticky='w')
        self.frame_job = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_job.grid(column = 0, row = 11,sticky='w')
        self.decrease_job_button = ctk.CTkButton(self.frame_job,
                                             image=loadimage_minus,
                                             text="",
                                             fg_color="transparent",
                                             width = 8,
                                             command=lambda: self.decreased_job())
        self.decrease_job_button["border"] = "0"
        self.decrease_job_button.grid(column = 0, row = 0, )
        self.job_number_entry = Entry(self.frame_job, "4", 1,0, 1)
        self.increase_job_button = ctk.CTkButton(self.frame_job,image=loadimage_plus,
                                            text="",
                                            fg_color="transparent",
                                             width = 8,
                                            command=lambda: self.increased_job())
        self.increase_job_button.grid(column = 2, row = 0, sticky="e")
        self.increase_job_button["border"] = "0"

        self.frame_rep = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_rep.grid(column = 0, row = 12,sticky='w')
        self.replicat = ctk.CTkLabel(self.frame_rep,
                                     text='Replicat',
                                     font=('comic',24))
        self.replicat.grid(column=0, row = 0, sticky='w')

        self.decrease_rep_button = ctk.CTkButton(self.frame_rep,
                                             image=loadimage_minus,
                                             text="",
                                             fg_color="transparent",
                                             width = 4,
                                             command=lambda: self.decreased_rep())
        self.decrease_rep_button["border"] = "0"
        self.decrease_rep_button.grid(column = 0, row = 1, sticky="w" )
        self.increase_rep_button = ctk.CTkButton(self.frame_rep,
                                             image=loadimage_plus,
                                             text="",
                                             fg_color="transparent",
                                             width = 4,
                                             command=lambda: self.increased_rep())
        self.increase_rep_button["border"] = "0"
        self.increase_rep_button.grid(column = 0, row = 1, sticky="e" )


        self.replicat_entry = Entry(self.frame_rep,
                                      "5",
                                      0,
                                      1,
                                      6,
                                      "")
        self.simulation_time = ctk.CTkLabel(self,
                                            text='Simultion time',
                                            font=('comic',24))
        self.simulation_time.grid(column=0, row = 13, sticky='w')
        self.simulation_time_entry = Entry(self, "10000", 0,14, len("10000  "))

class Frame2(ctk.CTkFrame):
    """A customtkinter frame containing the variables concerning the parameters estimated for the model

    This class is corresponding the frame containing information about parameters
    that are estimated in the model and that need to be confirmed.

    Attributes
    parameter_n: customtkinter label, naming the entry section for the name of additional paramters,
                n have the value of the number of parameters add by the user
    parameter_name: Class Entry for the name of the parameter estimated

    parameter_v: customtkinter label, naming the entry section for value of additional paramters,
                n have the value of the number of parameters add by the user
    parameter_val: Class Entry for the value of the parameter estimated

    add_parameter_button: customtkinter button, a button to add extra parameters to estimate

    titre_variables: customtkinter label, naming section

    frame_para1: customtkinter frame, frame storing the first parameters name and value

    parameter_1_name: customtkinter label, naming the entry section for the name of the first paramter,
    parameter_1_name_entry: Class Entry for the name of the first paramter
    parameter_1_value: customtkinter label, naming the entry section for the value of the first paramter
    parameter_1_value_entry: Class Entry for the the value of the first paramter
    """
    lop = {}

    def __init__(self, container):
        super().__init__(container)
        self.grid(column = 3, row = 1, padx= 20,sticky='n')
        self.create_widgets()
        self.grid_rowconfigure(1, weight=1)
        self.configure(fg_color="transparent")

    def addlabel(self):
        """Add new label and entre to a list each time a buttom is click
        """
        frame_p = ctk.CTkFrame(self,fg_color="transparent")
        frame_p.grid(column = 0, row = 2+(len(self.lop)), sticky="w")
        self.parameter_n = ctk.CTkLabel(frame_p,
                                        text=f"Parameter {str(len(self.lop)+1)}",
                                        font=('comic',24))
        self.parameter_n.grid(column = 0, row = 1+(len(self.lop)), sticky='w')
        self.parameter_name = ctk.CTkLabel(frame_p,
                                           text=f"Name",
                                           font=('comic',18),
                                           anchor="e")
        self.parameter_name.grid(column = 0, row = 2+(len(self.lop)))
        self.parameter_name_entry = Entry(frame_p,
                                      "Default value",
                                      2,
                                      (2+(len(self.lop))),
                                      40)
        # self.parameter_name_entry = ctk.CTkEntry(frame_p,corner_radius=4)
        # self.parameter_name_entry.insert(0, "This is Temporary Text...")
        # self.parameter_name.grid(column = 2, row = 1+(len(self.lop)))
        self.parameter_v = ctk.CTkLabel(frame_p,
                                        text="Value",
                                        font=('comic',18),
                                        anchor="e")
        self.parameter_v.grid(column = 0, row = 3+(len(self.lop)))
        self.parameter_val = Entry(frame_p,
                                      "Default value",
                                      2,
                                      (3+(len(self.lop))),
                                      40)
        # self.parameter_val = ctk.CTkEntry(frame_p, corner_radius=4)
        # self.parameter_val.grid(column = 2, row = 2+(len(self.lop)))

        self.lop[self.parameter_name_entry] = self.parameter_val

    def movebuttom(self):
        """Move buttom position"""
        self.add_parameter_button.grid(column = 0, row = 3+(len(self.lop)))

    def create_widgets(self):
        """Create widgets for the class App
        """

        self.titre_variables = ctk.CTkLabel(self, text='Variables',
                                            font=('comic',36,'bold'),
                                            # text_color="#0087F2"
                                            )
        self.titre_variables.grid(column=0, row = 0, pady= 20)

        self.frame_para1 = ctk.CTkFrame(self,height = 100, width = 100, fg_color="transparent")
        self.frame_para1.grid(column = 0, row = 2, sticky='w')
        self.frame_para1.grid_rowconfigure(0, weight=1)
        self.frame_para1.grid_columnconfigure(0, weight=1)

        self.parameter_1 = ctk.CTkLabel(self.frame_para1,
                                        text=f"Parameter {str(len(self.lop)+1)}",
                                        font=('comic',24))
        self.parameter_1.grid(column=0, row = 0, sticky='w')
        self.parameter_1_name = ctk.CTkLabel(self.frame_para1,
                                             text="Name",
                                             font=('comic',18),
                                             anchor= "e")
        self.parameter_1_name.grid(column=0, row = 1)

        self.parameter_1_name_entry = Entry(self.frame_para1,
                                      "Default value",
                                      1,
                                      1,
                                      40)
        self.parameter_1_value = ctk.CTkLabel(self.frame_para1,
                                              text="Value",
                                              font=('comic',18),
                                              anchor= "e")
        self.parameter_1_value.grid(column=0, row = 2)
        self.parameter_1_value_entry = Entry(self.frame_para1,
                                      "Default value",
                                      1,
                                      2,
                                      40)
        
        Frame2.lop[self.parameter_1_name_entry] = self.parameter_1_value_entry
        self.add_parameter_button = ctk.CTkButton(self,
                                                  text="Add parameter +",
                                                  corner_radius=4,
                                                  font=('comic',18),
                                                  command=lambda: [self.addlabel(), self.movebuttom()])
        self.add_parameter_button.grid(column = 0, row = 6, sticky='w')

class App(ctk.CTk):
    """
    Class for the visual of the UI

    Attributes

    frame1 : Class Frame1 storing the path information

    frame2 : Class Frame2 storing the parameters information

    mode_button: customtkinter button, change theme

    stop_button : customtkinter buttom, stop the GUI

    launch_button : customtkinter buttom, excecute controller functions
    """
    mode=""
    def progbar(self):
        """Fonction adding the progression bar"""
        self.progressbar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progressbar.grid(column = 2, row = 1)

    # def tb_click(self, frame1, frame2):
    #     self.progbar()
    #     # self.prog_bar.start()
    #     self.queue = queue.Queue()
    #     ThreadedTask(self, self.queue, frame1, frame2).start()
    #     self.master.after(100, self.process_queue)

    def process_queue(self):
        """IN CONSTRUCTION
        Function checking the thread in process"""
        try:
            msg = self.queue.get_nowait()
            # Show result of the task if needed
            # self.prog_bar.stop()
        except queue.Empty:
            self.master.after(100, self.process_queue)

    def dark_light(self):
        """"""
        if self.mode == "":
            self.mode = "light"
        elif self.mode =="light":
            self.mode = "dark"
        else:
            self.mode = "light"

        ctk.set_appearance_mode(self.mode)

    def __init__(self):
        super().__init__()
        self.geometry("1200x1200")
        self.title("KaMi")
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.loadimage_mode = ctk.CTkImage(Image.open("button/dark_light.png"))

        self.kami = ctk.CTkLabel(self, text='KaMuI',
                                font=('comic',56,'bold'),
                                text_color="#0087F2")
        self.kami.grid(column=2, row=0)

        self.frame1 = Frame1(self)
        self.frame2 = Frame2(self)

        self.mode_button = ctk.CTkButton(self,
                                       text="",
                                        image=self.loadimage_mode,
                                        border_width=0,
                                        corner_radius=4,
                                        fg_color="transparent",
                                        width = 8,
                                        anchor='n',
                                        command=lambda: self.dark_light())
   
        self.mode_button.grid(column = 0, row = 0, sticky="nw")
        #Call the function in the controller file that will launch the model
        self.launch_button = ctk.CTkButton(self,
                                            text="Launch Simulation",
                                            command=lambda: [cg.tb_click(self, self.frame1, self.frame2)],
                                            width=120,
                                            # height=32,
                                            font=('comic',18,'bold'),
                                            border_width=0,
                                            corner_radius=8)

        self.launch_button.grid(column = 3, row = 2 )