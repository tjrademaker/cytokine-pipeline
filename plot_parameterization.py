"""Relate parameter fits to data features"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D


time_scale=20
filepath="output/analysis/integral/"
cytokines="IFNg+IL-2+IL-6+IL-17A+TNFa"

tcellnumbers=["100k","30k","10k","3k"]
peptides=["N4","Q4","T4","V4"]
concentrations=["1uM","100nM","10nM","1nM"]

colors=sns.color_palette('deep', 4)
markers=["o","X","s","P"]
sizes=[50,30,20,10]

# Ugly way to define color dictionary for each of the variables
color_dict={var:color for var,color in zip(tcellnumbers+peptides+concentrations,colors+colors+colors)}
marker_dict={var:marker for var,marker in zip(tcellnumbers+peptides+concentrations,markers+markers+markers)}
size_dict={var:size for var,size in zip(tcellnumbers+peptides+concentrations,sizes+sizes+sizes)}

peptide_dict={pep:i for pep,i in zip(peptides,range(len(peptides)))}


# Fit a straight line and return its slope
def custom_fit(group):

	def f(x,A,B):
		return A*x + B

	popt,_=curve_fit(f,xdata=group["t_0"],ydata=group["v_0"])
	return popt[0]


# Compute features v0*t0, v0/t0 and Dr/r
def extract_features(df):

	df.set_index(["Data","TCellNumber","Peptide","Concentration"],inplace=True)
	df["Dv"]=0
	df["v0/t0"]=df["v_0"]/df["t_0"]

	for idx in df.groupby(["Data","TCellNumber","Peptide"]).size().index:		
		idxy=tuple(list(idx)+["1uM"])

		for idz in df.loc[idx].index:
			idxz=tuple(list(idx)+[idz])
			df.loc[idxz,"Dv"]=(df.loc[idxy,"v_0"]-df.loc[idxz,"v_0"])#*np.sqrt(df.loc[idxz,"v_0"])

	return df


# Adjustment to ax.scatter to take markers m as a keyword
# Proposed in https://stackoverflow.com/questions/51810492/how-can-i-add-a-list-of-marker-styles-in-matplotlib
def mscatter(x,y,z, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,z,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return ax


# Find variable that correlates with T cell number, concentration and peptide
def plot_correlating_variables(df_all_params):
	# Plot slope A=v0/t0 vs T cell number
	df_full=df_all_params.groupby(["Data","TCellNumber"]).apply(custom_fit).reset_index()
	df_full["TCellNumber"] = [int(num[:-1]) for num in df_full["TCellNumber"]]
	df_full.rename(columns={0: "v0/t0"},inplace=True)

	h=sns.relplot(data=df_full,x="TCellNumber",y="v0/t0",hue="Data")
	h.ax.set(xscale="log",title="Variable for T cell number")

	# Plot v_0 vs peptide
	df_full=df_all_params[["Data","Peptide","Concentration","v_0"]]
	df_full.set_index(["Data","Peptide","Concentration"],inplace=True)
	df_full=df_full.loc[(slice(None),peptides,concentrations),:]
	df_full.v_0=df_full.v_0.astype("float")
	df_full=df_full.groupby(["Data","Peptide","Concentration"]).mean().reset_index()

	h=sns.relplot(data=df_full.reset_index(),x="Peptide",y="v_0",hue="Concentration",hue_order=concentrations,col="Data",col_wrap=3)

	# Plot Dv / v vs concentration
	df_full=extract_features(df_all_params)
	df_full=df_full.groupby(["Concentration","Peptide"]).mean().reset_index()
	index=df_full.Concentration.map({conc:i for conc,i in zip(concentrations,range(4))}).sort_values().index
	df_full=df_full.loc[index,:]

	h=sns.relplot(data=df_full.reset_index(),x="Concentration",y="Dv",hue="Peptide")
	h.ax.set(ylim=[0,0.5])#="log",title="Variable for concentration")

	return None


def project_features_in_2d_space(params):
	# Plot projections of features on 2d planes
	h=sns.relplot(data=params.reset_index(), x="Dv",y="v0/t0",
		hue="TCellNumber",
		style="Peptide",style_order=peptides,
		size="Concentration",size_order=concentrations,
		col="Data",col_wrap=3)
	[ax.set(ylim=[0,0.6]) for ax in h.axes.flat]
	plt.savefig("figures/fitting/2d-projection-1.pdf")

	h=sns.relplot(data=params.reset_index(), x="Dv",y="v0/t0",
		hue="TCellNumber",
		style="Peptide",style_order=peptides,
		size="Concentration",size_order=concentrations)
	[ax.set(ylim=[0,0.6]) for ax in h.axes.flat]
	plt.savefig("figures/fitting/2d-projection-11.pdf")

	h=sns.relplot(data=params.reset_index(), x="Dv",y="v_0",
		hue="Peptide",
		style="Concentration",style_order=concentrations,
		size="TCellNumber",size_order=tcellnumbers,
		col="Data",col_wrap=3)
	plt.savefig("figures/fitting/2d-projection-2.pdf")

	h=sns.relplot(data=params.reset_index(), x="Dv",y="v_0",
		hue="Peptide",
		style="Concentration",style_order=concentrations,
		size="TCellNumber",size_order=tcellnumbers)
	plt.savefig("figures/fitting/2d-projection-22.pdf")

	h=sns.relplot(data=params.reset_index(), x="v0/t0",y="Dv",
		hue="Concentration",
		style="TCellNumber",style_order=tcellnumbers,
		size="Peptide",size_order=peptides,
		col="Data",col_wrap=3)
	[ax.set(xlim=[0,0.6]) for ax in h.axes.flat]
	plt.savefig("figures/fitting/2d-projection-3.pdf")

	h=sns.relplot(data=params.reset_index(), x="v0/t0",y="Dv",
		hue="Concentration",
		style="TCellNumber",style_order=tcellnumbers,
		size="Peptide",size_order=peptides)
	[ax.set(xlim=[0,0.6]) for ax in h.axes.flat]
	plt.savefig("figures/fitting/2d-projection-33.pdf")

	return None


def plot_features_in_3d_space(params): 
	# Plot features in 3d space - 
	# Color by T cell number
	fig = plt.figure(figsize=(6,6))
	ax=Axes3D(fig)
	mscatter(params["Dv"], params["v0/t0"], params["v_0"], ax=ax,
		c=list(params.index.get_level_values("TCellNumber").map(color_dict)),
		m=list(params.index.get_level_values("Peptide").map(marker_dict)),
		s=list(params.index.get_level_values("Concentration").map(size_dict)),
		depthshade=False)
	ax.set(xlabel='Dv',ylabel='v0 / t0',zlabel='v0',ylim=[0,0.6],title="Color by T cell number")
	ax.view_init(azim=-90, elev=90)
	plt.savefig("figures/fitting/3d-projection-1.pdf")

	# Color by peptide
	fig = plt.figure(figsize=(6,6))
	ax=Axes3D(fig)
	mscatter(params["Dv"], params["v0/t0"], params["v_0"], ax=ax,
		c=list(params.index.get_level_values("Peptide").map(color_dict)),
		m=list(params.index.get_level_values("Concentration").map(marker_dict)),
		s=list(params.index.get_level_values("TCellNumber").map(size_dict)),
		depthshade=False)
	ax.set(xlabel='Dv',ylabel='v0 / t0',zlabel='v0',ylim=[0,0.6],title="Color by peptide")
	ax.view_init(azim=-90, elev=0)
	plt.savefig("figures/fitting/3d-projection-2.pdf")

	# Color by concentration
	fig = plt.figure(figsize=(6,6))
	ax=Axes3D(fig)
	mscatter(params["Dv"], params["v0/t0"], params["v_0"], ax=ax,
		c=list(params.index.get_level_values("Concentration").map(color_dict)),
		m=list(params.index.get_level_values("TCellNumber").map(marker_dict)),
		s=list(params.index.get_level_values("Peptide").map(size_dict)),
		depthshade=False)
	ax.set(xlabel='Dv',ylabel='v0 / t0',zlabel='v0',ylim=[0,0.6], title="Color by concentration")
	ax.view_init(azim=0, elev=-90)
	plt.savefig("figures/fitting/3d-projection-3.pdf")

	return None


if __name__ == "__main__":

	# Read data
	df_all_params=pd.read_pickle(filepath+"all_fit_params.pkl")
	plot_correlating_variables(df_all_params)

	# Extract features v0*t0, v0/t0, Dr/r from fitted parameters
	params = extract_features(df_all_params.reset_index())
	params=params.loc[(slice(None),tcellnumbers,peptides,concentrations),:]

	project_features_in_2d_space(params)
	plot_features_in_3d_space(params)
	plt.show()