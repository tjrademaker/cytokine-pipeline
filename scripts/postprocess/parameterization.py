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
sys.path.insert(0, '../gui/plotting')
from plottingGUI import selectLevelsPage
from latent_space import WeightMatrixSelectionPage
sys.path.insert(0, '../process')
from adapt_dataframes import set_standard_order
            
splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('cytokine-pipeline-master')+1])+'/'

def fit_vt_vm(node1, node2, tol_r2=0.99):
    """Fit a straight line to the final time points of latent space. 
    Add time points until R^2 falls below tol_r2, when the curve is no longer straight.
    Requires at least 5 timepoints for a good fit, else returns NaN
    
    Args:
        node1 (1darray): timepoints of node 1
        node2 (1darray): timepoints of node 2
        tol_r2 (float): tolerance of the fit

    Returns:
        slope (float): slope of the fit (vt/vm ratio) or NaN when the curve could not be fit
    """

    def check_uniqueness(n_points=1):
        """Recursively find number of points to start fitting with more than one unique value in Node 1 or Node 2
        
        Args:
            n_points (int): default is 1

        Returns:
            n_points (int): number of points to start fit with
        """

        if ((len(np.unique(node1[-n_points:])) == 1) and (len(np.unique(node2[-n_points:])) == 1)):
            if len(node1) < n_points:
                return n_points
            else:
                return check_uniqueness(n_points+1)

        else:
            return n_points

    n_points = check_uniqueness(4)
    if len(node1) < n_points:
        return np.nan
    r2 = 1
    
    while (r2 > tol_r2) and (n_points < len(node1)):
        slope,_,r2,_,_ = linregress(node1[-n_points:], node2[-n_points:])
        n_points += 1
        
    if ((n_points >= 5) and (np.abs(slope) > 1)):
        return slope
    else:
        return np.nan

def compute_vt_vm(df,slope_type="median"):
    """ Compute vt/vm ratio.
    Fit all conditions separately and return the mean or median slope (whichever is better)

    Args:
        df (pd.DataFrame): dataframe with all conditions
        slope_type (string): determine measure to output slope from distribution. Currently implemented measures are mean or median. Default is median.

    Returns:
        slope (int): mean or median slope corresponding to vt/vm ratio

    """
    slopes = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=["slope"])
    # Fit a straight line to as as many final points as possible (at least 5)
    for idx in slopes.index:
        slopes.loc[idx] = fit_vt_vm(df.loc[idx,"Node 1"], df.loc[idx,"Node 2"])
    # Return mean or median dropping conditions for which no slope could be reliably calculated (n_points <= 5)
    median_slope = slopes.dropna().median().values[0]
    mean_slope = slopes.dropna().mean().values[0]
    print("\tMedian slope:\t",np.around(median_slope,2))
    print("\tMean slope:\t",np.around(mean_slope,2))

    if slope_type == "median":
        return median_slope
    elif slope_type == "mean":
        return mean_slope
    else:
        raise NotImplementedError("%s measure is not yet implemented"%slope_type)


def ballistic_v0(times, v0, t0, theta, vt, fit = True):
    """Piecewise ballisfic function that computes node 1 and node 2 from given parameters.
    Phase 1 is constant velocity regime.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories. Vector is doubled to accommodate fitting procedure
        v0 (float): constant speed
        t0 (float): time until phase 1
        theta (float): angle of constant speed (radians)
        vt (float): terminal velocity

    Returns: 
        curves (1darray): array with node1 and node 2 trajectory

    """

    # Only keep unique values in times vector (which has double entries) and remove "times" for regularization at 0
    times=np.unique(times)
    times=times[times>0]
    
    # Initialize some variables
    vm = vt / vt_vm_ratio
    
    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    vx = v0 * np.cos(theta-np.pi/2)
    vy = v0 * np.sin(theta-np.pi/2)
    
    # Phase 1
    prop = (times <= t0)
    x[prop] = vx * times[prop]
    y[prop] = vy * times[prop]
    
    # Phase 2
    r0 = [vx*t0,vy*t0]  # Position at the end of the propulsion phase
    delta_t = times[~prop] - t0
    
    x[~prop] = (vx + vm) * (1-np.exp(-2*delta_t))/2 - vm * delta_t + r0[0]
    y[~prop] = (vy + vt) * (1-np.exp(-2*delta_t))/2 - vt * delta_t + r0[1]
    
    if fit:
        curves = np.array([list(x)+[0,0,0,0], list(y)+list(np.sqrt(np.abs([v0,t0,theta,vt])))]).flatten()
        return curves
    else:
        return np.array([x,y]).flatten()


def ballistic_F(times, F,  t0, theta, vt, fit=True):

    """Piecewise ballisfic function that computes node 1 and node 2 from given parameters.
    Phase 1 is constant force regime.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories. Vector is doubled to accommodate fitting procedure
        F (float): constant speed
        t0 (float): time until phase 1
        theta (float): angle of constant speed
        vt (float): terminal velocity

    Returns: 
        curves (1darray): array with node1 and node 2 trajectory

    """
    
    def phase1_position(F,t):
        return F * (t - 1 + np.exp(-t))

    def phase1_velocity(F,t):
        return F * (1 - np.exp(-t))
    
    # Only keep unique values in the times vector (which has double entries) and remove "times" for regularization at 0
    times=np.unique(times)
    times=times[times>0]
    
    # Initialize some variables
    vm = vt / vt_vm_ratio
    
    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    Fx = F * np.cos(theta-np.pi/2)# - vm
    Fy = F * np.sin(theta-np.pi/2)# - vt
    
    # Phase 1
    prop = (times <= t0)
    x[prop] = phase1_position(Fx,times[prop])
    y[prop] = phase1_position(Fy,times[prop])
                
    # Position and velocity at end of phase 1
    r0 = phase1_position(np.array([Fx,Fy]),t0)
    v0 = phase1_velocity(np.array([Fx,Fy]),t0)
    
    # Phase 2
    delta_t = times[~prop] - t0
    
    x[~prop] = (v0[0] + vm) * (1-np.exp(-delta_t)) - vm * delta_t + r0[0]
    y[~prop] = (v0[1] + vt) * (1-np.exp(-delta_t)) - vt * delta_t + r0[1]
    
    if fit:
        curves = np.array([list(x)+[0,0,0,0], list(y)+list(np.sqrt(np.abs([F,t0,theta,vt])))]).flatten()
        return curves
    else:
        return np.array([x,y]).flatten()


# Find the best fit for each time course in the DataFrame. 
def return_fit_params(df,func,bounds,p0,param_labels,time_scale=20):
    # Initialize a dataframe that will record parameters and parameter variances fitted to each curve. 
    var_param_labels = ["var" + param for param in param_labels]
    cols = param_labels+var_param_labels 
    
    df_params = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=cols)
    xdata = df.iloc[:,0].index.get_level_values("Time").astype("float")/time_scale
    xdata=np.tile(list(xdata)+[0,0,0,0],[2])

    # Fit each curve, then return the parameters
    for idx in df_params.index:

        x,y=df.loc[idx,:].values.T
        ydata = np.array([list(x)+[0,0,0,0],list(y)+[0,0,0,0]]).flatten()

        # Each row contains one node, each column is one time point. Required by curve_fit
        popt, pcov = curve_fit(func, xdata=xdata, ydata=ydata, absolute_sigma=True, p0=p0, bounds= bounds)

        df_params.loc[idx, param_labels] = popt
        df_params.loc[idx, var_param_labels] = np.diag(pcov)
    
    return df_params


def combine_spline_df_with_fit(df,df_fit):
    """Combine splines and parameterized curves

    Args:
        df (pd.DataFrame): latent space data sampled from splines
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
    tmp=df_compare[["Node 1","Node 2"]].stack().unstack("Time")
    tmp[0]=0
    tmp=tmp.sort_index(axis=1).astype("float").diff(axis=1).dropna(axis=1).unstack().stack("Time")
    tmp.columns = pd.MultiIndex.from_product([ ['concentration'], tmp.columns])
    tmp=tmp.swaplevel("Processing type","Time")

    # Add feature as column level, combine integral and derivatives, stack dataframe and rename feature index level
    df_compare.columns = pd.MultiIndex.from_product([ ['integral'], df_compare.columns])
    df_compare=pd.concat([df_compare,tmp],axis=1).stack(0)
    df_compare.index.names=df_compare.index.names[:-1]+["Feature"]
    
    return df_compare


def return_param_and_fitted_latentspace_dfs(df,fittingFunctionName):
    """Returns parameterized dataframes to put into plotting GUI
    Args:
        df (pd.DataFrame): latent space data sampled from splines
        fittingFunctionName (str): name of fitting function; must be a key value in all dictionaries below
    Returns:
        df_params (pd.DataFrame): contains all parameters from fitting function
        df_compare (pd.DataFrame): combined dataframe with additional index levels (feature, Processing type)
    """
    time_scale=20

    func_dict = {'Constant velocity':ballistic_v0, 'Constant force': ballistic_F}
    bounds_dict = {'Constant velocity':[(0, 0, 0, 0), (5, 5, np.pi, 5)],
                   'Constant force': [(0, 0, 0, 0), (5, 5, np.pi, 5)]}
    p0_dict = {'Constant velocity':[1, 1, 1, 1],'Constant force':[1, 1, 1, 1]}
    param_labels_dict = {'Constant velocity':["v0", "t0", "theta", "vt"],
                         'Constant force':["F", "t0", "theta", "vt"]}
    
    # Fit curves
    func = func_dict[fittingFunctionName]
    bounds = bounds_dict[fittingFunctionName]
    p0 = p0_dict[fittingFunctionName]
    param_labels = param_labels_dict[fittingFunctionName]
    
    datasetParamList = []
    datasetFitList = []
    datasets = list(pd.unique(df.index.get_level_values('Data')))
    for dataset in datasets:
        datasetDf = df.xs([dataset],level=['Data'])
        global vt_vm_ratio
        vt_vm_ratio = compute_vt_vm(datasetDf)
        dataset_df_params = return_fit_params(datasetDf,func,bounds,p0,param_labels,time_scale=time_scale)

        # Compute latent space coordinates from parameters fits
        dataset_df_fit=datasetDf.copy()
        for idx in dataset_df_params.index:
            fittingVars = dataset_df_params.loc[idx,param_labels]
            times = dataset_df_fit.loc[idx,:].index.get_level_values("Time").astype("float")/time_scale
            dataset_df_fit.loc[idx,:] = func(times, *fittingVars,fit=False).reshape(2,-1).T
        
        datasetParamList.append(dataset_df_params)
        datasetFitList.append(dataset_df_fit)

    df_params = pd.concat(datasetParamList,keys=datasets,names=['Data'])
    df_fit = pd.concat(datasetFitList,keys=datasets,names=['Data'])

    # Compare fit vs splines
    df_compare = combine_spline_df_with_fit(df,df_fit)
    
    return df_params,df_compare



class FittingFunctionSelectionPage(tk.Frame):
    def __init__(self, master,df,bp):
        tk.Frame.__init__(self, master)
        global backPage
        global dfToParameterize
        dfToParameterize = df.copy()
        backPage = bp

        mainWindow = tk.Frame(self)
        l1 = tk.Label(mainWindow, text="Select function to fit latent space with:", font='Helvetica 18 bold').grid(row=0,column=0,sticky=tk.W)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        functionList = ['Constant velocity','Constant force']
        functionVar = tk.StringVar(value=functionList[0])
        rblist = []
        for i,function in enumerate(functionList):
            rb = tk.Radiobutton(mainWindow, text=function,padx = 20, variable=functionVar,value=function)
            rb.grid(row=i+1,column=0,sticky=tk.W)
            rblist.append(rb)
        
        def collectInputs():
            functionName = functionVar.get()
            global df_params_plot,df_compare_plot
            df_params_plot,df_compare_plot = return_param_and_fitted_latentspace_dfs(dfToParameterize,functionName)
            
            df_params_columns = list(df_params_plot.columns)
            df_compare_columns = list(df_compare_plot.columns)

            #Sort dataframes
            sorted_df_params_plot = set_standard_order(df_params_plot.reset_index())
            df_params_plot = pd.DataFrame(sorted_df_params_plot.loc[:,df_params_columns].values,index=pd.MultiIndex.from_frame(sorted_df_params_plot.iloc[:,:-1*len(df_params_columns)]))
            df_params_plot.columns = df_params_columns

            sorted_df_compare_plot = set_standard_order(df_compare_plot.reset_index())
            df_compare_plot = pd.DataFrame(sorted_df_compare_plot.loc[:,df_compare_columns].values,index=pd.MultiIndex.from_frame(sorted_df_compare_plot.iloc[:,:-1*len(df_compare_columns)]))
            df_compare_plot.columns = df_compare_columns
            
            print(df_compare_plot)
            print(df_params_plot)

            #Save dataframes:
            projectionName = pickle.load(open(path+'scripts/gui/plotting/projectionName.pkl','rb'))
            with open(path+'output/parameter-dataframes/params-'+projectionName+'-'+functionName+'.pkl','wb') as f:
                pickle.dump(df_params_plot,f)
            with open(path+'output/parameter-space-dataframes/paramSpace-'+projectionName+'-'+functionName+'.pkl','wb') as f:
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
                df_plot_reset = df_plot.reset_index()
                df_plot = pd.DataFrame(df_plot_reset.iloc[:,-3:].values,index=pd.MultiIndex.from_frame(df_plot_reset.iloc[:,:-3]),columns=list(df_plot_reset.columns)[-3:])
                df_plot = df_plot[['Node 1','Node 2','Time']] 
                print(df_plot)

            master.switch_frame(selectLevelsPage,df_plot,PlottingDataframeSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(FittingFunctionSelectionPage,dfToParameterize,backPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)
