# NHGPS :)

This repository contains the code for the paper **Variational Bayesian Inference for Nonlinear Hawkes Process with Gaussian Process Self Effects**. The paper will be linked once it is published.

This is still **Work in progress**. It only contains the code for the univariate version of te model.

## Source Code
- The source code can be found in the folder src.
- The python module hawkes_object.py is used to generate data from the model.
- The module variational_inference.py contains the code for the VI algorithm.
- To run the VI algorithm you need to install [JAX](http://handlebarsjs.com/https://github.com/google/jax). For larger datasets you may need access to GPU.

## Experiments
Examples of how to perform inference can be found in the Experiments folder. It contains scripts to perform inference and perform some further analysis from the inferred model (calculate the log likelihood, calculate the predictive density and estimate the underlying rate functions).

## Notebooks
- To learn how to generate data from the model see generate_data_from_the_model.ipynb.
- The notebook estimate_s_g.ipynb includes the code to generate figure 3 in the paper.
- The notebook single_neuron_data.ipynb contains the code to generate figures 5 and 6 in the paper.
