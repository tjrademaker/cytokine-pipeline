#! /usr/bin/env python3
"""Project data in latent space of neural network"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle,sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from scripts.gui.plotting.plottingGUI import selectLevelsPage
from scripts.process.adapt_dataframes import set_standard_order

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('cytokine-pipeline-master')+1])+'/'

class WeightMatrixSelectionPage(tk.Frame):
    def __init__(self, master, lspB,nsp):
        tk.Frame.__init__(self, master)

        global latentSpaceBool,nextSwitchPage
        latentSpaceBool = lspB
        #Select levels page is nsp if we're starting at latent_space.py, Select Fit/action is nsp if starting at parameterization.py
        nextSwitchPage = nsp

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select trained weight matrix to project with:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        trainingDatasets = []
        print(path)
        for fileName in os.listdir(path+'output/trained-networks'):
            print(fileName)
            if '-' in fileName and '.pkl' in fileName and 'mlp' in fileName:
                datasetName = fileName.split('.')[0].split('-')[-1]
                if datasetName not in trainingDatasets:
                    trainingDatasets.append(datasetName)

        self.buttonWindow = tk.Frame(self)
        self.buttonWindow.pack(side=tk.BOTTOM, pady=10)
        tk.Button(self.buttonWindow, text="OK",
            command=lambda: collectInputs()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Back",
            command=lambda: master.switch_frame(master.homepage)
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Quit",
            command=lambda: quit()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)

        datasetVar = tk.StringVar(value=trainingDatasets[0])
        for i,dataset in enumerate(trainingDatasets):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            global datasetName
            datasetName = datasetVar.get()
            #Grab weight matrix
            mlp=pickle.load(open(path+'output/trained-networks/mlp-'+datasetName+'.pkl',"rb"))
            global weightMatrix
            weightMatrix = mlp
            #Grab training set min/max to normalize df to be projected
            global df_min,df_max
            df_min,df_max=pd.read_pickle(path+"output/trained-networks/min_max-"+datasetName+".pkl")
            master.switch_frame(WTorMutantDatasetSelectionPage)


class WTorMutantDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select projection type:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        #Mutant List
        mutantTypeList = ["Tumor","Activation","TCellNumber","Macrophages","CAR","TCellType","CD25Mutant","ITAMDeficient","DrugPerturbation","NewPeptide"]
        datasetTypes = ['WT']+mutantTypeList

        self.buttonWindow = tk.Frame(self)
        self.buttonWindow.pack(side=tk.BOTTOM,pady=10)
        tk.Button(self.buttonWindow, text="OK",
            command=lambda: collectInputs()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Back",
            command=lambda: master.switch_frame(WeightMatrixSelectionPage,latentSpaceBool,nextSwitchPage)
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Quit",
            command=lambda: quit()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)

        datasetVar = tk.StringVar(value=datasetTypes[0])
        for i,dataset in enumerate(datasetTypes):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            datasetType = datasetVar.get()
            if datasetType == 'WT':
                master.switch_frame(TrainingDatasetSelectionPage)
            else:
                #Load raw mutant data
                df_mutant = import_mutant_output(datasetType)
                #Subset features by what was used to train WT
                featureColumns = pd.read_pickle(path+"output/trained-networks/train-"+datasetName+".pkl").columns
                df_mutant = df_mutant.loc[:,featureColumns]
                #Normalize mutant
                df_mutant=(df_mutant - df_min)/(df_max - df_min)
                #Project mutant on latent space
                df_mutant_proj=pd.DataFrame(np.dot(df_mutant,weightMatrix.coefs_[0]),index=df_mutant.index,columns=["Node 1","Node 2"])
                projectionName = '-'.join([datasetName,datasetType])
                with open(path+'scripts/gui/plotting/projectionName.pkl','wb') as f:
                    pickle.dump(projectionName,f)
                df_mutant_proj = set_standard_order(df_mutant_proj.copy().reset_index(),returnSortedLevelValues=False)
                df_mutant_proj = pd.DataFrame(df_mutant_proj.iloc[:,-2:].values,index=pd.MultiIndex.from_frame(df_mutant_proj.iloc[:,:-2]),columns=['Node 1','Node 2'])
                with open(path+'output/projected-dataframes/projection-'+projectionName+'.pkl','wb') as f:
                    pickle.dump(df_mutant_proj,f)
                #switch to plotting
                if(latentSpaceBool):
                    proj_df = df_mutant_proj.iloc[::5,:]
                    master.switch_frame(nextSwitchPage,proj_df,WTorMutantDatasetSelectionPage)
                #switch to choosing fit type, then plotting
                else:
                    master.switch_frame(nextSwitchPage,df_mutant_proj,WTorMutantDatasetSelectionPage)


class TrainingDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select training dataset to project on:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        datasetRadioButtons = []
        trainingDatasets = []
        all_df_WT = pd.read_pickle(path+"output/all-WT.pkl")
        idx = pd.IndexSlice
        sortedWT = set_standard_order(all_df_WT.loc[idx[:,'100k',:,:,1,'IFNg','concentration'],:].reset_index())
        for dataName in pd.unique(sortedWT['Data']):
            trainingDatasets.append(dataName)

        self.buttonWindow = tk.Frame(self)
        self.buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(self.buttonWindow, text="OK",
            command=lambda: collectInputs()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Back",
            command=lambda: master.switch_frame(WTorMutantDatasetSelectionPage)
            ).pack(in_=self.buttonWindow,side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Quit",
            command=lambda: quit()
            ).pack(in_=self.buttonWindow,side=tk.LEFT)


        # Frame to contain the scrollable canvas and the scrollbars within the
        # main window.
        self.labelWindow1 = tk.Frame(self)
        self.labelWindow1.pack(side=tk.TOP,padx=10,fill=tk.X,expand=tk.NO)

        # Make canvas inside that frame
        self.w1 = tk.Canvas(self.labelWindow1, borderwidth=0, width=600,
            height=600)

        # Make scrollbar in side the self.labelWindow1 frame as well
        scr_v1 = tk.Scrollbar(self.labelWindow1, orient=tk.VERTICAL, command=self.w1.yview)
        scr_v1.pack(side=tk.RIGHT,fill=tk.Y)
        # Add and bind scrollbar to canvas
        self.w1.config(yscrollcommand=scr_v1.set)
        self.w1.pack(fill=tk.BOTH, expand=tk.NO)

        # Make another horizontal scrollbar
        scr_v2 = tk.Scrollbar(self.labelWindow1, orient=tk.HORIZONTAL, command=self.w1.xview)
        scr_v2.pack(side=tk.BOTTOM, fill=tk.X)
        self.w1.config(xscrollcommand=scr_v2.set)
        self.w1.pack(fill=tk.BOTH, expand=tk.NO)

        # Make a frame to contain the list of radio buttons inside the Canvas
        # This is to create all buttons at once so they can be scrolled
        self.labelWindow = tk.Frame(self.w1)
        self.labelWindow.pack(fill=tk.BOTH, expand=tk.NO)
        self.w1.create_window((0,0), window=self.labelWindow, anchor = tk.NW)

        # Bind the label frame's <Configure> to the canvas' size
        # See https://stackoverflow.com/questions/3085696/adding-a-scrollbar-to-a-group-of-widgets-in-tkinter
        self.labelWindow1.bind("<Configure>", self.onFrameConfigure)
        #labelWindow = tk.Frame(self)
        #labelWindow.pack(side=tk.TOP,padx=10,fill=tk.X,expand=True)

        #Make and add frame for widgets inside of canvas
        #canvas_frame = tk.Frame(w1)
        mainWindow = tk.Frame(self.w1)
        mainWindow.pack()
        self.w1.create_window((0,0),window=mainWindow, anchor = tk.NW)

        datasetVar = tk.StringVar(value=trainingDatasets[0])
        for i,dataset in enumerate(trainingDatasets):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        def collectInputs():
            datasetName2 = datasetVar.get()
            #Load files
            df_WT = pd.concat([all_df_WT.loc[datasetName2]],keys=[datasetName2],names=['Data'])
            df_WT = df_WT.unstack(['Feature','Cytokine']).loc[:,'value']
            #df_WT=pd.read_pickle(path+"output/trained-networks/train-"+datasetName2+".pkl")
            #Subset features by what was used to train WT
            featureColumns = pd.read_pickle(path+"output/trained-networks/train-"+datasetName+".pkl").columns
            df_WT = df_WT.loc[:,featureColumns]
            print(df_WT)
            #Normalize WT
            df_WT=(df_WT - df_min)/(df_max - df_min)
            #Project WT on latent space
            df_WT_proj=pd.DataFrame(np.dot(df_WT,weightMatrix.coefs_[0]),index=df_WT.index,columns=["Node 1","Node 2"])
            #switch to plotting
            projectionName = '-'.join([datasetName,datasetName2])
            with open(path+'scripts/gui/plotting/projectionName.pkl','wb') as f:
                pickle.dump(projectionName,f)
            df_WT_proj = set_standard_order(df_WT_proj.copy().reset_index(),returnSortedLevelValues=False)
            df_WT_proj = pd.DataFrame(df_WT_proj.iloc[:,-2:].values,index=pd.MultiIndex.from_frame(df_WT_proj.iloc[:,:-2]),columns=['Node 1','Node 2'])
            with open(path+'output/projected-dataframes/projection-'+projectionName+'.pkl','wb') as f:
                pickle.dump(df_WT_proj,f)
            if(latentSpaceBool):
                proj_df = df_WT_proj.iloc[::5,:]
                master.switch_frame(nextSwitchPage,proj_df,TrainingDatasetSelectionPage)
            #switch to choosing fit type, then plotting
            else:
                master.switch_frame(nextSwitchPage,df_WT_proj,TrainingDatasetSelectionPage)

    def onFrameConfigure(self, event):
        """ Reset the scroll region to encompass the entire inner frame,
        so no radio button labels are missing. """
        self.w1.configure(scrollregion=self.w1.bbox("all"))

    def resizeFrame(self, event):
        width = event.width
        self.labelWindow1.itemconfig(self)

def import_mutant_output(mutant, folder=path+"data/processed/"):
    """Import processed cytokine data from experiments that contains
        "mutant" conditions.

    Args:
        mutant (str): name of file with mutant data.
                Has to be one of the following: "Tumor", "Activation",
                    "TCellNumber", "Macrophages", "CAR", "TCellType",
                    "CD25Mutant", "ITAMDeficient", "Drug"
        folder (str): full path to folder containing .hdf files to parse

    Returns:
        df_full (dataframe): the dataframe with processed cytokine data
    """

    naive_level_values={
        "ActivationType": "Naive",
        "Antibody": "None",
        "APC": "B6",
        "APCType": "Splenocyte",
        "CARConstruct": "None",
        "CAR_Antigen": "None",
        "Genotype": "WT",
        "IFNgPulseConcentration": "None",
        "TCellType": "OT1",
        "TLR_Agonist": "None",
        "TumorCellNumber": "0k",
        "DrugAdditionTime": 36,
        "Drug": "Null",
        "ConditionType": "Control",
        "TCR": "OT1"
    }

    mutant_levels={
        "Tumor": ["APC", "APCType", "IFNgPulseConcentration"],
        "Activation": ["ActivationType"],
        "TCellNumber": [],
        "Macrophages": ["TLR_Agonist", "APCType"],
        "CAR":["CAR_Antigen", "Genotype", "CARConstruct"],
        "TCellType": ["TCellType", "TCR"],
        "CD25Mutant": ["Genotype"],
        "ITAMDeficient": ["Genotype"],
        "NewPeptide": [],
        "Drug": ["Drug", "DrugAdditionTime"]
    }

    essential_levels=["TCellNumber", "Peptide", "Concentration", "Time"]

    dfs_dict = {}
    for file in os.listdir(folder):
        if (mutant not in file) | (not file.endswith(".hdf")):
            continue
        df = pd.read_hdf(folder + file)

        # If level not in essential levels or mutant-required level,
        # keep naive level values and drop level
        for level in df.index.names:
            if level not in essential_levels + mutant_levels[mutant]:
                df = df[df.index.get_level_values(level) == naive_level_values[level]]
                df = df.droplevel(level, axis=0)
        dfs_dict[file[:-4]] = df
        print(file)
        print(df.index.names)

    # Concatenate all saved dataframes
    df_full = pd.concat(dfs_dict, names=["Data"])
    return df_full
