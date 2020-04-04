"""Fit curves in latent space"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import linregress

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
            return check_uniqueness(n_points+1)
        else:
            return n_points

    n_points = check_uniqueness()
    r2 = 1
    
    while (r2 > tol_r2) and (n_points < len(node1)):
        slope,_,r2,_,_ = linregress(node1[-n_points:], node2[-n_points:])
        n_points += 1

    if ((n_points > 5) and (slope > 1)):
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


# Piecewise ballistic function
def ballistic(times, v0, t0, theta, vt, vm=0):
    """Piecewise ballisfic function that computes node 1 and node 2 from given parameters.
    Phase 1 is constant velocity regime.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories. Vector is doubled to accommodate fitting procedure
        v0 (float): constant speed
        t0 (float): time until phase 1
        theta (float): angle of constant speed
        vt (float): terminal velocity
        vm (float): optional parameter only set when calling ballistic to create parameter fits.

    Returns: 
        curves (1darray): array with node1 and node 2 trajectory

    """

    # Only keep unique values in times vector (which has double entries)
    times=np.unique(times)
    
    # Initialize some variables
    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    vx = v0 * np.cos(theta * 2 * np.pi / 360)
    vy = v0 * np.sin(theta * 2 * np.pi / 360)

    # Only compute vm from the global vt/vm ratio if it was not 
    if vm == 0:
        vm = vt / vt_vm_ratio
    
    # Phase 1
    prop = (times <= t0)
    x[prop] = vx * times[prop]
    y[prop] = vy * times[prop]
    
    # Phase 2
    r0 = [vx*t0,vy*t0]  # Position at the end of the propulsion phase
    delta_t = times[~prop] - t0
    
    x[~prop] = (vx + vm) * (1-np.exp(-2*delta_t))/2 - vm * delta_t + r0[0]
    y[~prop] = (vy + vt) * (1-np.exp(-2*delta_t))/2 - vt * delta_t + r0[1]

    curves = np.array([x, y]).flatten()
    
    return curves


def fit_all_curves(df):
    """Fit each time course in the dataframe
    
    Args:
        df (pd.DataFrame): dataframe with timecourses of all conditions

    Returns
        df_params (pd.DataFrame): dataframe with parameters
    """

    # Initialize a dataframe to record parameters fitted to each curve. 
    df_params = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=["v0", "theta", "t0", "vt", "vm"])
    times=np.tile(df.index.get_level_values("Time").astype("int")/time_scale,[2])    
    
    # Fit curves for all conditions
    for idx in df_params.index:
        peptide = idx[df_params.index.names.index("Peptide")]

        bounds = [(0, 0, -135, 0), (20, 5, 90, 20)]
        p0 = [3, 0, -90, 3]

        # Each row contains one node, each column is one time point. Required by curve_fit
        popt, _ = curve_fit(ballistic, xdata=times, ydata=df.loc[idx,:].values.T.flatten(), p0=p0, bounds=bounds)

        df_params.loc[idx, ["v0", "t0", "theta", "vt"]] = popt

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

    # Add feature as column level, combine integral and derivatives, stack dataframe and rename feature index level
    df_compare.columns = pd.MultiIndex.from_product([ ['integral'], df_compare.columns])
    df_compare=pd.concat([df_compare,placeholder],axis=1).stack(0)
    df_compare.index.names=df_compare.index.names[:-1]+["Feature"]
    
    return df_compare



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


# Variables to set with first GUI
slope_type="median"
time_scale=20
cytokines="IFNg+IL-2+IL-6+IL-17A+TNFa"

tcellnumbers=["100k","30k","10k","3k"]
peptides=["N4","Q4","T4","V4"]
concentrations=["1uM","100nM","10nM","1nM"]

experiment_type = "TCellNumber"


# Load data, min/maxes and weight matrix
df_WT=import_mutant_output(experiment_type)

df_min,df_max=pd.read_pickle("output/train-min-max.pkl")
df=(df_WT - df_min)/(df_max - df_min)

df=df.loc[:,("integral",cytokines.split("+"))]
mlp=pickle.load(open("output/mlp.pkl", "rb"))

# Project on latent space
df=pd.DataFrame(np.dot(df,mlp.coefs_[0]),index=df.index,columns=["Node 1","Node 2"])

# Initialize dataframe with fitted parameters. 
df_all_params=pd.DataFrame([],columns=list(df.index.names[:-1])+["v0","t0","theta","vt","vm"])

# Fit curves of all experiments
for exp in df.index.levels[0]:

    print(exp)
    # Get latent space projection
    df_tmp=df.loc[exp].copy()

    # Fit curves
    vt_vm_ratio = compute_vt_vm(df_tmp)
    df_params = fit_all_curves(df_tmp)
    df_params.loc[:,"vm"]=df_params.loc[:,"vt"]/vt_vm_ratio

    # Set experiments as level name and add to big dataframe df_all_params
    df_params["Data"]=exp
    df_all_params=pd.concat([df_all_params,df_params.reset_index()])    


df_all_params.set_index(df.index.names[:-1],inplace=True,append=False)
df_all_params["v0*t0"]=np.product(df_all_params[["v0","t0"]])

# Parameter space plotting GUI
if 0: 
    h=sns.relplot(data=df_all_params.reset_index(), x="t0",y="v0",
    	hue="TCellNumber", hue_order=tcellnumbers,
    	style="Peptide",style_order=peptides,
        size="Concentration",size_order=concentrations,
        col="Data")
    plt.show()

# Comparison splines vs fit plotting GUI
latent_space_feature="integral"
if 1: 

    # Compute latent space coordinates from parameters fits
    df_fit=df.copy()
    for idx in df_all_params.index:
        v0, t0, theta, vt, vm = df_all_params.loc[idx,["v0","t0","theta","vt","vm"]]
        times = df_fit.loc[idx,:].index.get_level_values("Time").astype("float")/time_scale
        df_fit.loc[idx,:]=ballistic(times, v0, t0, theta, vt).reshape(2,-1).T

    df_compare=combine_spline_df_with_fit(df,df_fit)
    df_compare=df_compare[df_compare.index.get_level_values("Feature") == latent_space_feature]

    sns.relplot(data=df_compare.reset_index(),kind="line",sort=False,x="Node 1",y="Node 2",
                hue="TCellNumber",hue_order=tcellnumbers,
                col="Peptide",col_order=peptides,
                row="Data",
                size="Concentration",size_order=concentrations,
                style="Processing type",dashes=["",(1,1)])
    plt.show()