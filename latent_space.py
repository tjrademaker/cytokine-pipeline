"""Project data in latent space of neural network"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

prop_cycle=plt.rcParams["axes.prop_cycle"]
colors=np.array([prop_cycle.by_key()["color"]]*2).flatten()
sizes=[1.5,1.2,0.9,0.6]
dashes = ["", (4, 1.5), (1, 1),(3, 1, 1.5, 1), (5, 1, 1, 1),(5, 1, 2, 1, 2, 1)]*3
markers = ["o","s","X", '^', '*', 'D', 'P','v']*3

peptides=["N4","Q4","T4","V4","G4","E1"][::-1]
concentrations=["1uM","100nM","10nM","1nM"]
tcellnumbers=["100k"]

def main():

	#Set parameters
	features="integral"
	cytokines="IFNg+IL-2+IL-6+IL-17A+TNFa"

	#Load files
	df_min,df_max=pickle.load(open("output/train-min-max.pkl","rb"))
	df_WT=pd.read_pickle("output/train.pkl")

	mlp=pickle.load(open("output/mlp.pkl","rb"))
	#Project WT on latent space
	df_WT_proj=pd.DataFrame(np.dot(df_WT,mlp.coefs_[0]),index=df_WT.index,columns=["Node 1","Node 2"])
	df_WT_proj.to_pickle("output/proj-WT.pkl")

	project_in_latent_space(df_WT_proj,mutant="WT")

	plt.show()


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
	data=df.reset_index()
	data=data.iloc[::5,:]

	h=sns.relplot(data=data,x="Node 1",y="Node 2",kind='line',sort=False,mew=0,ms=4,dashes=dashes,markers=markers,
		hue="Peptide",hue_order=peptides,#palette=colors[:len(peptides)],
		size="Concentration",size_order=concentrations,sizes=sizes,
		style="TCellNumber",style_order=tcellnumbers)

	[ax.set(xticks=[],yticks=[],xlabel="Node 1",ylabel="Node 2") for ax in h.axes.flat]


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
