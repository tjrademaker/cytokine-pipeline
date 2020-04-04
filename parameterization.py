#! /usr/bin/env python3
"""Fit curves in latent space"""

import warnings
warnings.filterwarnings("ignore")

import pickle,sys,os
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

from scipy.optimize import curve_fit
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, 'gui/plotting')
from plottingGUI import GUI_Start,selectLevelsPage
from latent_space import WeightMatrixSelectionPage

# Fit a straight line to the final time points of the (node1, node2) curve. 
# Add time points until R^2 falls under tol_r2, which means the curve is no longer straight. 
# Requires at least 5 data points for a good fit
def fit_vt_vm(node1, node2, tol_r2=0.99):
    n_points = 1
    r2 = 1
    while (r2 > tol_r2) and (n_points < len(node1)):
        n_points += 1
        slope,_,r2,_,_ = linregress(node1[-n_points:], node2[-n_points:])

    if n_points > 5:
        return slope
    else:
        return np.nan

# Compute ratio vt / vm experiment-wide
def compute_vt_vm(df):
    slopes = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=["slope"])
    # Fit a straight line to as as many final points as possible (at least 5)
    for idx in slopes.index:
        slopes.loc[idx] = fit_vt_vm(df.loc[idx,"Node 1"], df.loc[idx,"Node 2"])

    # Return mean or median dropping conditions for which no slope could be reliably calculated (n_points <= 5)
    median_slope = slopes.dropna().median().values[0]
    mean_slope = slopes.dropna().mean().values[0]
    print("\tMedian slope:\t",np.around(median_slope,2))
    print("\tMean slope:\t",np.around(mean_slope,2))
    return mean_slope


# Piecewise ballistic function
def ballisticConstantVelocity(times, v_0, theta, t_0, v_t):
    
    # Only keep unique values in times vector (which has double entries)
    times=np.unique(times)
    
    # Initialize some variables
    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    v_x = v_0 * np.cos(theta * 2 * np.pi / 360)
    v_y = v_0 * np.sin(theta * 2 * np.pi / 360)
    v_m = v_t / vt_vm_ratio
    
    # Phase 1
    prop = (times <= t_0)
    x[prop] = v_x * times[prop]
    y[prop] = v_y * times[prop]
    
    # Phase 2
    r0 = [v_x*t_0,v_y*t_0]  # Position at the end of the propulsion phase
    delta_t = times[~prop] - t_0
    
    x[~prop] = (v_x + v_m) * (1-np.exp(-2*delta_t))/2 - v_m * delta_t + r0[0]
    y[~prop] = (v_y + v_t) * (1-np.exp(-2*delta_t))/2 - v_t * delta_t + r0[1]
    
    return np.array([x, y]).flatten()

# Find the best fit for each time course in the DataFrame. 
def return_fit_params(df,fittingFunction,fittingFunctionBounds,fittingFunctionInitialGuess,fittingParameterLabels,time_scale=20):
    # Initialize a dataframe that will record parameters and parameter variances fitted to each curve. 
    fittingParameterVarianceLabels = []
    for param in fittingParameterLabels:
        fittingParameterVarianceLabels.append('var '+param)
    cols = fittingParameterLabels+fittingParameterVarianceLabels 
    
    df_params = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=cols)
    times=np.tile(df.index.get_level_values("Time").astype("int")/time_scale,[2])    
    
    # Fit each curve, then return the parameters
    for idx in df_params.index:
        # Each row contains one node, each column is one time point. Required by curve_fit
        popt, pcov = curve_fit(fittingFunction, xdata=times, ydata=df.loc[idx,:].values.T.flatten(), p0=fittingFunctionInitialGuess, bounds=fittingFunctionBounds)

        df_params.loc[idx, fittingParameterLabels] = popt
        df_params.loc[idx, fittingParameterVarianceLabels] = np.diag(pcov)
    
    return df_params

def combine_spline_df_with_fit(df,df_fit):
    """Combine splines and parameterized curves
    Args:
        df (pd.DataFrame): data sampled from splines
        df_fit (pd.DataFrame): data sampled from parameterized curves
    Returns:
        df_compare (pd.DataFrame): combined dataframe with additional index levels (feature, Processing type)
    """
    # Add Processing type index level
    df["Processing type"]="Splines"
    df_fit["Processing type"]="Fit"
    # Combine dataframes
    df_compare=pd.concat([df,df_fit])
    df_compare.set_index("Processing type",inplace=True,append=True)
    # Compute derivatives of df_compare data and return dataframe to same format
    placeholder=df_compare[["Node 1","Node 2"]].stack().unstack("Time")
    placeholder[0]=0
    placeholder=placeholder.sort_index(axis=1).astype("float").diff(axis=1).dropna(axis=1).unstack().stack("Time")
    placeholder.columns = pd.MultiIndex.from_product([ ['concentration'], placeholder.columns])
    placeholder=placeholder.swaplevel("Processing type","Time")
    # Add feature column level and combine integral and derivatives
    df_compare.columns = pd.MultiIndex.from_product([ ['integral'], df_compare.columns])
    df_compare=pd.concat([df_compare,placeholder],axis=1)
    df_compare.columns.names=["Feature","Node"]
    df_compare = df_compare.stack('Feature')
    df_compare.columns.name = ''

    return df_compare

def return_param_and_fitted_latentspace_dfs(df,fittingFunctionName):
    """Returns parameterized dataframes to put into plotting GUI
    Args:
        df (pd.DataFrame): data in nodespace sampled from splines
        fittingFunctionName (String): name of fitting function; must be a key value in all dictionaries below
    Returns:
        df_params (pd.DataFrame): contains all parameters from fitting function
        df_compare (pd.DataFrame): combined dataframe with additional index levels (feature, Processing type)
    """
    time_scale=20

    fittingFunctionDict = {'ballisticConstantVelocity':ballisticConstantVelocity}
    fittingFunctionBoundsDict = {'ballisticConstantVelocity':[(0, -135, 0, 0), (20, 90, 5, 20)]}
    fittingFunctionInitialGuessDict = {'ballisticConstantVelocity':[3, -90, 0., 3]}
    fittingFunctionParameterLabelsDict = {'ballisticConstantVelocity':["v_0", "theta", "t_0", "v_t"]}
    
    # Fit curves
    fittingFunction = fittingFunctionDict[fittingFunctionName]
    fittingFunctionBounds = fittingFunctionBoundsDict[fittingFunctionName]
    fittingFunctionInitialGuess = fittingFunctionInitialGuessDict[fittingFunctionName]
    fittingFunctionParameterLabels = fittingFunctionParameterLabelsDict[fittingFunctionName]
    
    datasetParamList = []
    datasetFitList = []
    datasets = list(pd.unique(df.index.get_level_values('Data')))
    for dataset in datasets:
        datasetDf = df.xs([dataset],level=['Data'])
        global vt_vm_ratio
        vt_vm_ratio = compute_vt_vm(datasetDf)
        dataset_df_params = return_fit_params(datasetDf,fittingFunction,fittingFunctionBounds,fittingFunctionInitialGuess,fittingFunctionParameterLabels,time_scale=time_scale)

        # Compute latent space coordinates from parameters fits
        dataset_df_fit=datasetDf.copy()
        for idx in dataset_df_params.index:
            fittingVars = dataset_df_params.loc[idx,fittingFunctionParameterLabels]
            times = dataset_df_fit.loc[idx,:].index.get_level_values("Time").astype("float")/time_scale
            dataset_df_fit.loc[idx,:] = fittingFunction(times, *fittingVars).reshape(2,-1).T
        
        datasetParamList.append(dataset_df_params)
        datasetFitList.append(dataset_df_fit)

    df_params = pd.concat(datasetParamList,keys=datasets,names=['Data'])
    df_fit = pd.concat(datasetFitList,keys=datasets,names=['Data'])

    # Compare fit vs splines
    df_compare = combine_spline_df_with_fit(df,df_fit)
    
    return df_params,df_compare

class FittingFunctionSelectionPage(tk.Frame):
    def __init__(self, master,df,backPage):
        tk.Frame.__init__(self, master)

        mainWindow = tk.Frame(self)
        l1 = tk.Label(mainWindow, text="Select function to fit latent space with:", font='Helvetica 18 bold').grid(row=0,column=0,sticky=tk.W)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        functionList = ['ballisticConstantVelocity','ballisticConstantForce']
        functionVar = tk.StringVar(value=functionList[0])
        rblist = []
        for i,function in enumerate(functionList):
            rb = tk.Radiobutton(mainWindow, text=function,padx = 20, variable=functionVar,value=function)
            rb.grid(row=i+1,column=0,sticky=tk.W)
            rblist.append(rb)
        
        def collectInputs():
            functionName = functionVar.get()
            global df_params_plot,df_compare_plot
            df_params_plot,df_compare_plot = return_param_and_fitted_latentspace_dfs(df,functionName)
            with open('temp-params.pkl','wb') as f:
                pickle.dump(df_params_plot,f)
            with open('temp-compare.pkl','wb') as f:
                pickle.dump(df_compare_plot,f)
            master.switch_frame(PlottingDataframeSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(backPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class PlottingDataframeSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        mainWindow = tk.Frame(self)
        l1 = tk.Label(mainWindow, text="Select type of parameterized dataframe to plot:", font='Helvetica 18 bold').grid(row=0,column=0,sticky=tk.W)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        dfTypeList = ['parameters','parameterized latent space']
        dfTypeVar = tk.StringVar(value=dfTypeList[0])
        rblist = []
        for i,dfType in enumerate(dfTypeList):
            rb = tk.Radiobutton(mainWindow, text=dfType,padx = 20, variable=dfTypeVar,value=dfType)
            rb.grid(row=i+1,column=0,sticky=tk.W)
            rblist.append(rb)
        
        def collectInputs():
            dfType = dfTypeVar.get()
            if dfType == 'parameters': 
                df_plot = df_params_plot.copy()
            else:
                df_plot = df_compare_plot.copy()
                df_plot = df_plot.swaplevel(i=-3,j=-2)
                df_plot = df_plot.swaplevel(i=-2,j=-1)
            master.switch_frame(selectLevelsPage,df_plot,PlottingDataframeSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(backPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)


def main():
    latentSpaceBool = False 
    app = GUI_Start(WeightMatrixSelectionPage,latentSpaceBool,FittingFunctionSelectionPage)
    app.mainloop()

if __name__ == "__main__":
    main()
