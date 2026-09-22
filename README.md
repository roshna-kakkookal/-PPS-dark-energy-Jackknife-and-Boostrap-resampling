# PPS-dark-energy-Jackknife-and-Boostrap-resampling
This is code for the analysis of dark-energy models using the PantheonPlus and SH0ES (PPS) dataset, with frequentist methods such as GLS, along with two resampling techniques, Jackknife and Bootstrap, and Bayesian methods (emcee, PolyChord). It also includes code for plotting corner plots to compare results from all the methods.

## Using this code
You can use this code provided that you cite the paper in your publications:
Roshna K, Nikhil Fernandes, P. Praveen, and V. Sreenath,  
[Constraining dark energy models using Jackknife and Bootstrap resampling.](https://arxiv.org/abs/2601.16197)

## About this code

1. This code consists of two Python files, namely `pantheonplussh0es_analysis.py` (the main analysis code) and `getdist_plotter.py` (used for plotting ).

2. The file `pantheonplussh0es_analysis.py` performs the analysis of dark-energy models using the PantheonPlus and SH0ES (PPS) dataset. It implements the generalized least-squares (GLS) method, Jackknife and Bootstrap resampling techniques, and Bayesian methods using `emcee` and `PolyChord`.

3. The code can be used to analyse dark-energy models, such as flat &Lambda;CDM, ΛCDM, flat wCDM, wCDM, and flat w₀wₐCDM. The model to be analyzed can be selected in the main program.

4. The `getdist_plotter.py` file is used to generate corner plots and compare the parameter constraints obtained from GLS, Jackknife, Bootstrap, and Bayesian methods.

5. The Jackknife and Bootstrap routines are used to investigate the effect of resampling on the parameter estimates and their uncertainties. The Bootstrap analysis uses 1000 resampled datasets, while the Jackknife analysis is performed by systematically resampling the dataset.

6. The code requires the PantheonPlus and SH0ES data files, together with their covariance matrix, as input. The relevant data files should be placed in the appropriate location before running the analysis.
