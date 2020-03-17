"""Compute geometrical features from latent space coordinates"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

prop_cycle=plt.rcParams["axes.prop_cycle"]
colors=np.array([prop_cycle.by_key()["color"]]*2).flatten()
sizes=[50,30,20,10]
markers = ["o","s","X", '^', '*', 'D', 'P','v']*3


def main():

	#Set parameters
	features="integral" #integral+concentration+derivative"
	filepath="output/analysis/"+features

	df_WT_proj=pickle.load(open(filepath+"/dataframes/proj-WT.pkl","rb"))
	df_WT_geom=extract_features(df_WT_proj,mutant="WT")
	pickle.dump(df_WT_geom,open(filepath+"/dataframes/geom-WT.pkl","wb"))

	plot_geometric_features(df_WT_geom,
		path=filepath,
		mutant="WT",
		norm_level="Absolute",
		plot_style="averaged")

	#TODO: fix duplicate entries in "Antagonism","ITAMDeficient"
	#TODO: when P14 and F5 have more concentrations, add them here (think about markers)
	mutants=["NaiveVsExpandedTCells"]#["P14,F5"]#["20"]#"NaiveVsExpandedTCells"]#"20","21","22","23","CD25Mutant","NaiveVsExpandedTCells","TCellNumber","Tumor"]
	# test_timeseries=["TCellTypeComparison_OT1,P14,F5_Timeseries_2"]

	for mutant in mutants:
		#Load dataframes with projection in latent space
		df_mut_proj=pickle.load(open(filepath+"/dataframes/proj-%s.pkl"%mutant,"rb"))

		#Compute geometrical features and save dataframe
		df_mut_geom=extract_features(df_mut_proj,mutant=mutant)	
		pickle.dump(df_mut_geom,open(filepath+"/dataframes/geom-%s.pkl"%mutant,"wb"))

		for norm_level in ["Absolute","Relative"]:#TODO: include "Normalized"
			for plot_style in ["color","style"]:
				plot_geometric_features(df_mut_geom,
					path=filepath,
					mutant=mutant,
					norm_level=norm_level,
					plot_style=plot_style)


def custom_fit(group):

	def f(x,A,B):
		return A*x + B

	popt,_=curve_fit(f,xdata=group["Node 1"],ydata=group["Node 2"])
	return popt[0]


def extract_features(df,**kwargs): ###TODO - REMOVE KWARGS ONCE LEVELS ARE FIXED

	### TODO - REMOVE ONCE TUMOR/ANTAGONISM LEVELS ARE FIXED
	n_levels=len(df.index.levels[:-1])
	levels=[level for level in range(n_levels)]

	#Compute angle for each timeseries
	cutoff_time = 20
	slope=df[df.index.get_level_values("Time") <= cutoff_time].groupby(level=levels).apply(custom_fit)
	angle=360/(2*np.pi)*np.arctan(slope)

	#Compute v0 for each timeseries
	# velocity=df[df.index.get_level_values("Time") <= cutoff_time].groupby(level=levels).diff().mean()

    #Get value of node 1 in latent representation at max_time, which is determined by minimum length of the different WT timeseries
	max_time=df.unstack("Time").dropna(axis=1).columns[-1][-1]
	df=df[df.index.get_level_values("Time").astype("int") <= max_time]

    #Create geometric feature dataframe
	df_geom=pd.DataFrame([],index=angle.index,columns=[["Absolute"]*2+["Relative"]*2,["Node 1","Angle"]*2],dtype="float")
	for idx in angle.index:
		df_geom.loc[idx,("Absolute","Node 1")]=df.loc[tuple(list(idx)+[max_time]),"Node 1"]

	df_geom["Absolute","Angle"]=angle

	#Correct angles that were oriented 180 or more w.r.t. N4 1uM (same slope but opposite direction, generally null peptides)
	correction=df_geom["Absolute","Angle"] > df_geom["Absolute","Angle"].max(level="Peptide")["N4"]
	df_geom["Absolute","Angle"][correction] -= 180

	df_geom["Relative"]=df_geom["Absolute"].copy()

	#Subtract angle WT N4 1uM
	### TODO - CHANGE CONDITIONAL ONCE LEVELS ARE FIXED
	if kwargs["mutant"] == "Tumor":
		df_geom["Relative"] -= df_geom.loc[("Splenocyte","N4","1uM"),"Absolute"].values
	elif kwargs["mutant"] in ["P14","F5","P14,F5"]:
		df_geom["Relative"] -= df_geom.loc[("OT1","N4","1uM"),"Absolute"].values

	# elif kwargs["mutant"] == "Antagonism":
	# 	df_geom["Relative"] -= df_geom.loc[(slice(None),"None","None","N4","1uM"),"Absolute"].values

	else:	
		for idx in df.groupby(level=levels[:-2]).size().index:

			if type(idx) == str: idy=tuple([idx]+["N4"]+["1uM"])
			else: idy=tuple(list(idx)+["N4"]+["1uM"])

			rel_N4=df_geom.loc[idy,"Absolute"] #At N4 1uM,angle relative to the horizontal and node 1 relative to 0

			for idz in df_geom.loc[idx].index:

				if type(idx) == str: idz=tuple([idx]+list(idz))
				else: idz=tuple(list(idx)+list(idz))

				df_geom["Relative"].loc[idz] -= rel_N4

	# for data in df.groupby(level=levels[:-2]).size().index:
	# 	for peptide in df.loc[data,:].index.get_level_values("Peptide").unique():
	# 		print(data,peptide)

	# print(df.groupby(level=levels[:-2]).size().index)

	# df_geom["Normalized"]=df_geom["Relative"]

	return df_geom


def plot_geometric_features(df,**kwargs):

	print("-".join(kwargs.values()))

	level_name,peptides,concentrations=plot_parameters(kwargs["mutant"])

	df = df[kwargs["norm_level"]]
	data=df.reset_index()
	df_ave=df.groupby(level=[1,2]).agg({"Node 1":["mean","std"],"Angle":["mean","std"]})
	data_ave=df_ave.loc[:,(slice(None),"mean")].droplevel(level=1,axis=1).reset_index()

	if kwargs["plot_style"] == "color":
		sns.relplot(data=data,x="Node 1",y="Angle",
			style=level_name,hue="Peptide",hue_order=peptides,palette=colors[:len(data["Peptide"].unique())],#col=level_name,
			size="Concentration",size_order=concentrations,sizes=sizes)

	elif kwargs["plot_style"] == "style":
		sns.relplot(data=data,x="Node 1",y="Angle",
			hue=level_name,style="Peptide",style_order=peptides,markers=markers,
			size="Concentration",size_order=concentrations,sizes=sizes,#col=level_name
			)

	elif (kwargs["plot_style"] == "averaged") & (kwargs["mutant"] == "WT"):

		g=sns.relplot(data=data_ave,x="Node 1",y="Angle",
			hue="Peptide",hue_order=peptides,
			style="Concentration",style_order=concentrations,markers=markers,s=50)
		#Add error bars
		ax=g.fig.axes[0]
		for peptide,color in zip(peptides[:-2],colors):
			for conc,size in zip(concentrations,sizes):
				x,xerr,y,yerr=df_ave.loc[peptide,conc].values
				ax.errorbar(x,y,xerr=xerr,yerr=yerr,color=color,elinewidth=0.5)

	plt.savefig("%s/geometrical-features/%s/%s-%s.pdf"%tuple(kwargs.values()),bbox_inches="tight")


def plot_parameters(mutant):
	"""Find the index level name associated to the provided mutant by returning the value associated to the key mutant in mutant_dict

	Args:
		mutant (str)
	Returns:
		level_name (str): the index level name associated to the given mutant
		peptides (str): peptides in this datasert
		concentrations (str): concentrations in this dataset
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

	peptides = ["N4","Q4","T4","V4","G4","E1"]
	concentrations=["1uM","100nM","10nM","1nM"]

	if mutant in ["20","23"]:
		peptides+=["A2","Y3"]
	elif mutant == "P14":
		peptides+=["A3V","AV","C4Y","G4Y","L6F","S4Y","gp33WT","mDBM"]
	elif mutant == "F5":
		peptides+=["GAG","NP34","NP68"]
	if mutant == "P14,F5":
		peptides+=["A3V","AV","C4Y","G4Y","L6F","S4Y","gp33WT","mDBM","GAG","NP34","NP68"]
	elif mutant == "Tumor":
		peptides = ["N4", "Q4", "T4", "Y3"]
		concentrations = ["1nM IFNg","1uM","100nM","1nM"]#,"100pM IFNg","10pM IFNg","1nM IFNg","500fM IFNg","250fM IFNg","125fM IFNg","0 IFNg"]

	return (level_dict[mutant],peptides,concentrations)


if __name__ == "__main__":
	main()
