""" Manually split an experiment called HighMI into one df for each
replicate.

Run this before any other option in the GUI, including adapt_dataframes.

Should make this part of adapt_dataframes at some point.

@author:frbourassa (suggested by tjrademaker and acharsj)
September 13, 2020
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle
import numpy as np
import pandas as pd

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('cytokine-pipeline-master')+1])+'/'
folder=path+"data/current/"

def main_split():
    for file in os.listdir(folder):
        if not file.endswith(".pkl"):
            continue
        # Import the df
        try:
            tmp = pd.read_pickle(folder+file)
            reps = tmp.index.get_level_values("Replicate").unique()
        except KeyError:
            continue
        except FileNotFoundError:
            continue
        else:
            print("Splitting", file, "in {} parts".format(len(reps)))
            for i in reps:
                # Slice for current replicate i, drop the index level
                df_i = tmp.xs(i, level="Replicate", axis=0, drop_level=True)
                suffix = "-modified.pkl"  # usual suffix
                fname = file[:-len(suffix)] + "-" + str(i) + suffix
                df_i.to_pickle(folder+fname)

            # Finally, rename the old file to something non-pickle (like .bak)
            os.rename(folder+file, folder+file[:-4]+".bak")


if __name__ == "__main__":
    main_split()
