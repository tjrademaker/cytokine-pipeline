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
sys.path.insert(0, 'gui/plotting')
from plottingGUI import GUI_Start,selectLevelsPage
#from projectionGUI import WeightMatrixSelectionPage

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
        for fileName in os.listdir(path+'output'):
            if '-' in fileName and '.pkl' in fileName and 'mlp' in fileName:
                datasetName = fileName.split('.')[0].split('-')[-1]
                if datasetName not in trainingDatasets:
                    trainingDatasets.append(datasetName)

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
            mlp=pickle.load(open(path+'output/mlp-'+datasetName+'.pkl',"rb"))
            global weightMatrix
            weightMatrix = mlp
            master.switch_frame(WTorMutantDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class WTorMutantDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select projection type:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        #Mutant List
        mutantTypeList = ["Tumor","Activation","TCellNumber","Macrophages","CAR","TCellType","CD25Mutant","ITAMDeficient"]
        datasetTypes = ['WT']+mutantTypeList

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
                featureColumns = pd.read_pickle(path+"output/train-"+datasetName+".pkl").columns
                print(df_mutant)
                df_mutant = df_mutant.loc[:,featureColumns]
                #Project mutant on latent space
                df_mutant_proj=pd.DataFrame(np.dot(df_mutant,weightMatrix.coefs_[0]),index=df_mutant.index,columns=["Node 1","Node 2"])
                #switch to plotting
                if(latentSpaceBool):
                    proj_df = df_mutant_proj.iloc[::5,:]
                    master.switch_frame(nextSwitchPage,proj_df,WTorMutantDatasetSelectionPage)
                #switch to choosing fit type, then plotting
                else:
                    master.switch_frame(nextSwitchPage,df_mutant_proj,WTorMutantDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(WeightMatrixSelectionPage,latentSpaceBool,nextSwitchPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class TrainingDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select training dataset to project on:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        trainingDatasets = []
        for fileName in os.listdir(path+'output'):
            if '-' in fileName and '.pkl' in fileName:
                datasetName = fileName.split('.')[0].split('-')[-1]
                if datasetName not in trainingDatasets:
                    trainingDatasets.append(datasetName)

        datasetVar = tk.StringVar(value=trainingDatasets[0])
        for i,dataset in enumerate(trainingDatasets):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            datasetName = datasetVar.get()
            #Load files
            df_WT=pd.read_pickle(path+"output/train-"+datasetName+".pkl")
            #Subset features by what was used to train WT
            featureColumns = pd.read_pickle(path+"output/train-"+datasetName+".pkl").columns
            df_WT = df_WT.loc[:,featureColumns]
            #Project WT on latent space
            df_WT_proj=pd.DataFrame(np.dot(df_WT,weightMatrix.coefs_[0]),index=df_WT.index,columns=["Node 1","Node 2"])
            #switch to plotting
            if(latentSpaceBool):
                proj_df = df_WT_proj.iloc[::5,:]
                master.switch_frame(nextSwitchPage,proj_df,TrainingDatasetSelectionPage)
            #switch to choosing fit type, then plotting
            else:
                master.switch_frame(nextSwitchPage,df_WT_proj,TrainingDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(WTorMutantDatasetSelectionPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

def main():
    
    latentSpaceBool = True
    app = GUI_Start(WeightMatrixSelectionPage,latentSpaceBool,selectLevelsPage)
    app.mainloop()

def import_WT_output():
    """Import splines from wildtype naive OT-1 T cells by looping through all datasets
    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """

    folder="../data/processed/"

    naive_pairs={
            "ActivationType": "Naive",
            "Antibody": "None",
            "APC": "B6",
            "APCType": "Splenocyte",
            "CARConstruct":"None",
            "CAR_Antigen":"None",
            "Genotype": "WT",
            "IFNgPulseConcentration":"None",
            "TCellType": "OT1",
            "TLR_Agonist":"None",
            "TumorCellNumber":"0k"
            }

    for file in os.listdir(folder):

        if ".hdf" not in file:
            continue

        df=pd.read_hdf(folder + file)
        mask=[True] * len(df)

        for index_name in df.index.names:
            if index_name in naive_pairs.keys():
                mask=np.array(mask) & np.array([index == naive_pairs[index_name] for index in df.index.get_level_values(index_name)])
                df=df.droplevel([index_name])

        df=pd.concat([df[mask]],keys=[file[:-4]],names=["Data"]) #add experiment name as multiindex level 

        if "df_full" not in locals():
            df_full=df.copy()
        else:
            df_full=pd.concat((df_full,df))

    return df_full

def import_mutant_output(mutant):
    """Import processed cytokine data from an experiment that contains mutant data

    Args:
            mutant (str): name of file with mutant data.
                    Has to be one of the following "Tumor","Activation","TCellNumber","Macrophages","CAR","TCellType","CD25Mutant","ITAMDeficient"

    Returns:
        df_full (dataframe): the dataframe with processed cytokine data
    """
    
    naive_level_values={
                "ActivationType": "Naive",
                "Antibody": "None",
                "APC": "B6",
                "APCType": "Splenocyte",
                "CARConstruct":"None",
                "CAR_Antigen":"None",
                "Genotype": "WT",
                "IFNgPulseConcentration":"None",
                "TCellType": "OT1",
                "TLR_Agonist":"None",
            }
    
    mutant_levels={
                "Tumor": ["APC","APCType","IFNgPulseConcentration"],
                "Activation": ["ActivationType"],
                "TCellNumber": [],
                "Macrophages": ["TLR_Agonist"],
                "CAR":["CAR_Antigen","Genotype","CARConstruct"],
                "TCellType":["TCellType"],
                "CD25Mutant": ["Genotype"],
                "ITAMDeficient":["Genotype"],
            }
    
    essential_levels=["TCellNumber","Peptide","Concentration","Time"]

    folder="data/processed/"

    for file in os.listdir(folder):

        if (mutant not in file) | (".hdf" not in file):
            continue

        df=pd.read_hdf(folder + "/" + file)
        
        # If level not in essential levels or mutant-required level, keep naive level values and drop level
        for level in df.index.names:
            if level not in essential_levels+mutant_levels[mutant]:
                df=df[df.index.get_level_values(level)==naive_level_values[level]]
                df=df.droplevel(level,axis=0)
                
        df=pd.concat([df],keys=[file[:-4]],names=["Data"]) #add experiment name as multiindex level
        
        print(file)
        print(df.index.names)
        
        if "df_full" not in locals():
            df_full=df.copy()
        else:
            df_full=pd.concat([df_full,df],levels=["Data"]+mutant_levels[mutant]+essential_levels)

    return df_full

def plot_parameters(mutant):
    """Find the index level name associated to the provided mutant by returning the value associated to the key mutant in mutant_dict
        Args:
                mutant (str): string contains mutant
        Returns:
                level_name (str): the index level name of interest associated to the given mutant
    """

    level_dict={"20": "Data",
            "21": "TCellNumber",
            "22": "TCellNumber",
            "23": "TCellNumber",
            "Antagonism":"AntagonistToAgonistRatio",
            "CD25Mutant": "Genotype",
            "ITAMDeficient": "Genotype",
            "NaiveVsExpandedTCells": "ActivationType",
            "P14": "TCellType",
            "F5": "TCellType",
            "P14,F5": "TCellType",
            "TCellNumber": "TCellNumber",
            "Tumor": "APC Type",
            "WT": "Data"}

    peptides = ["N4","Q4","T4","V4","G4","E1"]#,"Y3","A2","A8","Q7"]
    concentrations=["1uM","100nM","10nM","1nM"]

    if mutant in ["20","23"]:
        peptides+=["A2","Y3"]#,"Q7","A8"]
    if mutant == "P14":
        peptides+=["A3V","AV","C4Y","G4Y","L6F","S4Y","gp33WT","mDBM"]
    if mutant == "F5":
        peptides+=["GAG","NP34","NP68"]
    if mutant == "P14,F5":
        peptides+=["A3V","AV","C4Y","G4Y","L6F","S4Y","gp33WT","mDBM","GAG","NP34","NP68"]
    if mutant == "Tumor":
        peptides = ["N4", "Q4", "T4", "Y3"]
        concentrations = ["1nM IFNg","1uM","100nM","1nM"]#,"100pM IFNg","10pM IFNg","1nM IFNg","500fM IFNg","250fM IFNg","125fM IFNg","0 IFNg"]

    return (level_dict[mutant],peptides,concentrations)

if __name__ == "__main__":
    main()
