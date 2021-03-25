#! /usr/bin/env python3
"""Train and save a neural network"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
	import matplotlib
	matplotlib.use("TkAgg")
from sklearn import neural_network
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scripts.gui.plotting.plottingGUI import createLabelDict,checkUncheckAllButton,selectLevelsPage
from scripts.process.adapt_dataframes import set_standard_order

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('cytokine-pipeline-master')+1])+'/'

idx = pd.IndexSlice

#FOR NEURAL NETWORK DATA SELECTION
class InputDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        #Import data
        df = import_WT_output()
        dataset = df.stack().stack().to_frame('value')
        dataset.to_pickle(path+"output/all-WT.pkl")

        tk.Frame.__init__(self, master)

        global trueLabelDict
        trueLabelDict = {}

        #Sort by date/number/quality/quantity
        print("Sorting all labels and levels... it can take a while")
        dataset = set_standard_order(dataset.reset_index())
        sortedValues = set_standard_order(dataset.copy(),returnSortedLevelValues=True)

        dataset = pd.DataFrame(dataset['value'].values,index=pd.MultiIndex.from_frame(dataset.iloc[:,:-1]),columns=['value'])
        trueLabelDict = createLabelDict(dataset.copy(),sortedValues=sortedValues)

        titleWindow = tk.Frame(self)
        titleWindow.pack(side=tk.TOP,padx=10,fill='none',expand=True)
        titleLabel = tk.Label(titleWindow, text='Training set name:',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0)
        e1 = tk.Entry(titleWindow)
        e1.grid(row=0,column=1)
        e1.insert(0, 'default')

        timeWindow = tk.Frame(self)
        timeWindow.pack(side=tk.TOP,padx=10,fill='none',expand=True)
        timeLabel = tk.Label(timeWindow, text='Time range (START TIME-END TIME):',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0)
        e2 = tk.Entry(timeWindow)
        e2.grid(row=0,column=1)
        e2.insert(0, '1-72')

        # Buttons at the bottom, pack them first so they don't disappear
        # https://stackoverflow.com/questions/42074654/avoid-the-status-bar-footer-from-disappearing-in-a-gui-when-reducing-the-size
        self.buttonWindow = tk.Frame(self)
        self.buttonWindow.pack(side=tk.BOTTOM,pady=10)

        # Find the max number of level values
        maxNumLevelValues = 0
        for labelList in trueLabelDict.values():
            if len(labelList) > maxNumLevelValues:
                maxNumLevelValues = len(labelList)

        tk.Button(self.buttonWindow, text="OK",
                command=lambda: collectInputs(dataset)
            ).pack(in_=self.buttonWindow, side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Back",
                command=lambda: master.switch_frame(master.homepage)
            ).pack(in_=self.buttonWindow, side=tk.LEFT)
        tk.Button(self.buttonWindow, text="Quit",
                command=lambda: quit()
            ).pack(in_=self.buttonWindow, side=tk.LEFT)

        # Frame to contain the scrollable canvas and the scrollbars within the
        # main window.
        self.labelWindow1 = tk.Frame(self)
        self.labelWindow1.pack(side=tk.TOP,padx=10,fill=tk.X,expand=tk.NO)

        # Make canvas inside that frame
        self.w1 = tk.Canvas(self.labelWindow1, borderwidth=0, width=1200,
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

        l1 = tk.Label(self.labelWindow, text='Parameters:',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0,columnspan=len(trueLabelDict)*6)
        levelValueCheckButtonList = []
        overallCheckButtonVariableList = []
        checkAllButtonList = []
        uncheckAllButtonList = []
        i=0
        maxNumLevelValues = 0
        for levelName in trueLabelDict:
            j=0
            levelCheckButtonList = []
            levelCheckButtonVariableList = []
            levelLabel = tk.Label(self.labelWindow, text=levelName+':')
            levelLabel.grid(row=1,column = i*6,sticky=tk.N,columnspan=5)
            for levelValue in trueLabelDict[levelName]:
                includeLevelValueBool = tk.BooleanVar()
                cb = tk.Checkbutton(self.labelWindow, text=levelValue, variable=includeLevelValueBool)
                cb.grid(row=j+4,column=i*6+2,columnspan=2,sticky=tk.W)
                self.labelWindow.grid_columnconfigure(i*6+3,weight=1)
                cb.select()
                levelCheckButtonList.append(cb)
                levelCheckButtonVariableList.append(includeLevelValueBool)
                j+=1

            checkAllButton1 = checkUncheckAllButton(self.labelWindow,levelCheckButtonList, text='Check All')
            checkAllButton1.configure(command=checkAllButton1.checkAll)
            checkAllButton1.grid(row=2,column=i*6,sticky=tk.N,columnspan=3)
            checkAllButtonList.append(checkAllButton1)

            uncheckAllButton1 = checkUncheckAllButton(self.labelWindow,levelCheckButtonList, text='Uncheck All')
            uncheckAllButton1.configure(command=checkAllButton1.uncheckAll)
            uncheckAllButton1.grid(row=2,column=i*6+3,sticky=tk.N,columnspan=3)
            uncheckAllButtonList.append(checkAllButton1)

            levelValueCheckButtonList.append(levelCheckButtonList)
            overallCheckButtonVariableList.append(levelCheckButtonVariableList)
            if len(trueLabelDict[levelName]) > maxNumLevelValues:
                maxNumLevelValues = len(trueLabelDict[levelName])
            i+=1

        def collectInputs(dataset):
            includeLevelValueList = []
            #Decode boolean array of checkboxes to level names
            i = 0
            for levelName,checkButtonVariableList in zip(trueLabelDict,overallCheckButtonVariableList):
                tempLevelValueList = []
                for levelValue,checkButtonVariable in zip(trueLabelDict[levelName],checkButtonVariableList):
                    if checkButtonVariable.get():
                        tempLevelValueList.append(levelValue)
                #Add time level values in separately using time range entrybox
                if i == len(trueLabelDict.keys()) - 2:
                    timeRange = e2.get().split('-')
                    includeLevelValueList.append(list(range(int(timeRange[0]),int(timeRange[1]))))
                includeLevelValueList.append(tempLevelValueList)
                i+=1

            print(dataset)
            df = dataset.loc[tuple(includeLevelValueList),:].unstack(['Feature','Cytokine']).loc[:,'value']
            print(df)

            trainingSetName = e1.get()
            #Save min/max and normalize data
            pickle.dump([df.min(),df.max()],open(path+"output/trained-networks/min_max-"+trainingSetName+".pkl","wb"))
            df=(df - df.min())/(df.max()-df.min())
            df.to_pickle(path+"output/trained-networks/train-"+trainingSetName+".pkl")

            #Needed to adapt this to work with any selection of peptides; not sure if what I'm doing here is correct
            peptides=["N4","Q4","T4","V4","G4","E1"][::-1]
            full_peptide_dict={k:v for v,k in enumerate(peptides)}
            peptide_dict = {}
            for peptide in full_peptide_dict:
                if peptide in pd.unique(df.index.get_level_values("Peptide")):
                    peptide_dict[peptide] = full_peptide_dict[peptide]

            #Extract times and set classes
            y=df.index.get_level_values("Peptide").map(peptide_dict)

            mlp=neural_network.MLPClassifier(
                    activation="tanh",hidden_layer_sizes=(2,),max_iter=5000,
                    solver="adam",random_state=90,learning_rate="adaptive",alpha=0.01).fit(df,y)

            score=mlp.score(df,y); print("Training score %.1f"%(100*score));
            pickle.dump(mlp,open(path+"output/trained-networks/mlp-"+trainingSetName+".pkl","wb"))

            df_WT_proj=pd.DataFrame(np.dot(df,mlp.coefs_[0]),index=df.index,columns=["Node 1","Node 2"])
            df_WT_proj.to_pickle(path+"output/trained-networks/proj-WT-"+trainingSetName+".pkl")

            proj_df = df_WT_proj.iloc[::5,:]
            master.switch_frame(selectLevelsPage,proj_df,InputDatasetSelectionPage)
    def onFrameConfigure(self, event):
        """ Reset the scroll region to encompass the entire inner frame,
        so no radio button labels are missing. """
        self.w1.configure(scrollregion=self.w1.bbox("all"))

    def resizeFrame(self, event):
        width = event.width
        self.labelWindow1.itemconfig(self)


def import_WT_output(folder=path+"data/processed/"):
    """Import splines from wildtype naive OT-1 T cells by looping through all datasets

    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """
    naive_pairs={
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
        "ConditionType": "Null",
        "TCR": "OT1"
    }

    dfs_dict = {}
    for file in os.listdir(folder):
        if ".hdf" not in file:
            continue

        df=pd.read_hdf(folder + file)
        mask=[True] * len(df)

        for index_name in df.index.names:
            if index_name in naive_pairs.keys():
                mask=np.array(mask) & np.array([index == naive_pairs[index_name] for index in df.index.get_level_values(index_name)])
                df=df.droplevel([index_name])
        dfs_dict[file[:-4]] = df[mask]
        print(file)
        print(df[mask].index.names)
    # Concatenate all dfs
    df_full = pd.concat(dfs_dict, names=["Data"])
    return df_full

def plot_weights(mlp,cytokines,peptides,**kwargs):

    fig,ax=plt.subplots(1,2,figsize=(8,2))
    ax[0].plot(mlp.coefs_[0],marker="o",lw=2,ms=7)
    ax[1].plot(mlp.coefs_[1].T[::-1],marker="o",lw=2,ms=7)
    [a.legend(["Node 1","Node 2"]) for a in ax]

    ax[0].set(xlabel="Cytokine",xticks=np.arange(len(cytokines)),xticklabels=cytokines,ylabel="Weights")
    ax[1].set(xlabel="Peptide",xticks=np.arange(len(mlp.coefs_[1].T)),xticklabels=peptides)
    plt.savefig("%s/weights.pdf"%tuple(kwargs.values()),bbox_inches="tight")
