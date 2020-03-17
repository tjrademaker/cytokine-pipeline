# Cytokine processing pipeline
Pipeline to process cytokine data, extract integral features, train a neural network, and parameterize the latent space

To process the data, run the files in process/ in the following order
1. adapt_dataframes.py
2. process_raw_data.py
3. neural_network.py

Post processing files are
- latent_space.py
- geometrical_features.py
- parameterization.py
- plot_parameterization.py
- reconstruction.py

In notebooks/ there are notebooks to play with the parameterization of latent space (parameterization.ipynb & plot_parameterization.ipynb)
