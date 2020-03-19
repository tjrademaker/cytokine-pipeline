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


prop_cycle=plt.rcParams["axes.prop_cycle"]
colors=np.array([prop_cycle.by_key()["color"]]*2).flatten()
sizes=[1.5,1.2,0.9,0.6]
dashes = ["", (4, 1.5), (1, 1),(3, 1, 1.5, 1), (5, 1, 1, 1),(5, 1, 2, 1, 2, 1)]*3
markers = ["o","s","X", '^', '*', 'D', 'P','v']*3

peptides=["N4","Q4","T4","V4","G4","E1"][::-1]
concentrations=["1uM","100nM","10nM","1nM"]
tcellnumbers=["100k"]

class TrainingDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Training Dataset:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)
        
        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        trainingDatasets = []
        for fileName in os.listdir('output'):
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
            df_min,df_max=pickle.load(open("output/train-min_max-"+datasetName+".pkl","rb"))
            df_WT=pd.read_pickle("output/train-"+datasetName+".pkl")

            mlp=pickle.load(open("output/mlp-"+datasetName+".pkl","rb"))
            #Project WT on latent space
            df_WT_proj=pd.DataFrame(np.dot(df_WT,mlp.coefs_[0]),index=df_WT.index,columns=["Node 1","Node 2"])
            df_WT_proj.to_pickle("output/proj-WT-"+datasetName+".pkl")

            proj_df = project_in_latent_space(df_WT_proj,mutant="WT")
            
            master.switch_frame(selectLevelsPage,proj_df)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

def main():

    app = GUI_Start(TrainingDatasetSelectionPage)
    app.mainloop()
      
'''
- Loop over data
- 


'''
def import_mutant_output_new():
    """Import processed cytokine data and add level that concerns test

    Args:
            mutant (str): name of file with mutant data.
                    Has to be one of the following "Tumor","NaiveVsExpandedTCells","TCellNumber","Antagonism","CD25Mutant","ITAMDeficient"

    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """

    # TODO do something smart with "tests"
    tests=["CD25Mutant","ITAMDeficient","Tumor","TCellNumber","Activation","CAR","TCellType","Macrophages"]

    folder="output/dataframes/"

    for file in os.listdir(folder):

            if (mutant not in file) | (".hdf" not in file):
                    continue

            df=pd.read_hdf(folder + "/" + file)
            df=pd.concat([df],keys=[file[18:-4]],names=["Data"]) #add experiment name as multiindex level 

            if "df_full" not in locals():
                    df_full=df.copy()
            else:
                    df_full=pd.concat((df_full,df))

    return df_full


def import_mutant_output(mutant):
        """Import processed cytokine data from an experiment that contains mutant data

        Args:
                mutant (str): name of file with mutant data.
                        Has to be one of the following "Tumor","NaiveVsExpandedTCells","TCellNumber","Antagonism","CD25Mutant","ITAMDeficient"

        Returns:
                df_full (dataframe): the dataframe with processed cytokine data
        """

        folder="data/processed/"

        for file in os.listdir(folder):

                if (mutant not in file) | (".hdf" not in file):
                        continue

                df=pd.read_hdf(folder + "/" + file)
                df=pd.concat([df],keys=[file[18:-4]],names=["Data"]) #add experiment name as multiindex level 

                if "df_full" not in locals():
                        df_full=df.copy()
                else:
                        df_full=pd.concat((df_full,df))

        return df_full


def project_in_latent_space(df,**kwargs):

        print("/".join(kwargs.values()))

        level_name,peptides,concentrations=plot_parameters(kwargs["mutant"])
        df=df.iloc[::5,:]
        return df

        """
        h=sns.relplot(data=data,x="Node 1",y="Node 2",kind='line',sort=False,mew=0,ms=4,dashes=dashes,markers=markers,
                hue="Peptide",hue_order=peptides,#palette=colors[:len(peptides)],
                size="Concentration",size_order=concentrations,sizes=sizes,
                style="TCellNumber",style_order=tcellnumbers)

        [ax.set(xticks=[],yticks=[],xlabel="Node 1",ylabel="Node 2") for ax in h.axes.flat]
        """

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
