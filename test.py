#! /usr/bin/env python3
import pickle,sys,os
import pandas as pd

#df = pickle.load(open('data/final/cytokineConcentrationPickleFile-20200202-Macrophages_1-final.pkl','rb'))
df = pd.read_hdf('data/processed/DrugPerturbation.hdf')
print(df)
