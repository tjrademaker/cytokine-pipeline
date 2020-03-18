"""Train and save a neural network"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn import neural_network
import matplotlib.pyplot as plt


def main():

	#Set parameters
	features="integral"
	cytokines="IFNg+IL-2+IL-6+IL-17A+TNFa"
	times=np.arange(20,73)
	train_timeseries=[
						"PeptideComparison_OT1_Timeseries_18",
						"PeptideComparison_OT1_Timeseries_19",
						"PeptideComparison_OT1_Timeseries_20",
						"PeptideTumorComparison_OT1_Timeseries_1",
						"Activation_Timeseries_1",
						"TCellNumber_OT1_Timeseries_7"
						]

	peptides=["N4","Q4","T4","V4","G4","E1"][::-1]
	concentrations=["1uM","100nM","10nM","1nM"]
	tcellnumbers=["100k"]

	peptide_dict={k:v for v,k in enumerate(peptides)}

	#Import data
	df=import_WT_output()
	df.to_pickle("../output/all-WT.pkl")
	print(df)

	# TODO: Add GUI instead of manually selecting levels

	df=df.loc[(train_timeseries,tcellnumbers,peptides,concentrations,times),(features.split("+"),cytokines.split("+"))]

	# df=df[~ (((df.index.get_level_values("Peptide") == "V4")  & (df.index.get_level_values("Concentration") == "1nM")) 
	# 	| ((df.index.get_level_values("Peptide") == "T4")  & (df.index.get_level_values("Concentration") == "1nM"))
	# 	| ((df.index.get_level_values("Peptide") == "Q4")  & (df.index.get_level_values("Concentration") == "1nM")))]

	#Save min/max and normalize data
	pickle.dump([df.min(),df.max()],open("output/train-min-max.pkl","wb"))
	df=(df - df.min())/(df.max()-df.min())
	df.to_pickle("output/train.pkl")

	#Extract times and set classes
	y=df.index.get_level_values("Peptide").map(peptide_dict)

	mlp=neural_network.MLPClassifier(
		activation="tanh",hidden_layer_sizes=(2,),max_iter=5000,
		solver="adam",random_state=90,learning_rate="adaptive",alpha=0.01).fit(df,y)

	score=mlp.score(df,y); print("Training score %.1f"%(100*score));
	pickle.dump(mlp,open("output/mlp.pkl","wb"))

	# plot_weights(mlp,cytokines.split("+"),peptides,filepath=filepath)

def import_WT_output():
	"""Import splines from wildtype naive OT-1 T cells by looping through all datasets

	Returns:
		df_full (dataframe): the dataframe with processed cytokine data
	"""

	folder="../data/processed/"

	naive_pairs={
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
					"TumorCellNumber":"0k"
				}

	for file in os.listdir(folder):

		if ".hdf" not in file:
			continue

		df=pd.read_hdf(folder + file)
		mask=[True] * len(df)
		
		for index_name in df.index.names:
			if index_name in naive_pairs.keys():
				mask=np.array(mask) & np.array([index == naive_pairs[index_name] for index in df.index.get_level_values(index_name)])
				df=df.droplevel([index_name])

		df=pd.concat([df[mask]],keys=[file[:-4]],names=["Data"]) #add experiment name as multiindex level 

		if "df_full" not in locals():
			df_full=df.copy()
		else:
			df_full=pd.concat((df_full,df))

	return df_full


def plot_weights(mlp,cytokines,peptides,**kwargs):

	fig,ax=plt.subplots(1,2,figsize=(8,2))
	ax[0].plot(mlp.coefs_[0],marker="o",lw=2,ms=7)
	ax[1].plot(mlp.coefs_[1].T[::-1],marker="o",lw=2,ms=7)
	[a.legend(["Node 1","Node 2"]) for a in ax]

	ax[0].set(xlabel="Cytokine",xticks=np.arange(len(cytokines)),xticklabels=cytokines,ylabel="Weights")
	ax[1].set(xlabel="Peptide",xticks=np.arange(len(mlp.coefs_[1].T)),xticklabels=peptides)
	plt.savefig("%s/weights.pdf"%tuple(kwargs.values()),bbox_inches="tight")

if __name__ == "__main__":
	main()
