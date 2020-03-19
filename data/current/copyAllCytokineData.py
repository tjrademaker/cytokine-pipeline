#!/usr/bin/env python3 
import json,pickle,math,matplotlib,sys,os,string,subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pathToExperiment = '../../../../allExperiments/QualityQuantityDeconvolution/'
allExperiments = [name for name in os.listdir(pathToExperiment) if os.path.isdir(os.path.join(pathToExperiment,name))]
for experimentFolder in allExperiments:
    if 'outputData' in os.listdir(pathToExperiment+experimentFolder):
        allFiles = os.listdir(pathToExperiment+experimentFolder+'/outputData/pickleFiles')
        for fileName in allFiles:
            if '-modified' in fileName and 'cytokineConcentrationPickleFile' in fileName:
                subprocess.run(['cp',pathToExperiment+experimentFolder+'/outputData/pickleFiles/'+fileName,'.'])
