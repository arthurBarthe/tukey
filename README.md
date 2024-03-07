This repository contains the code used in the paper 'Tukey g-and-h neural network regression for non-Gaussian data'.
The src/losses.py contains the TukeyGandHLoss class which can be used to train a neural network on a Tukey g-and-h 
loss function.
The inversion of the Tukey g-and-h transform is performed within the class Tuckey_g_h_inverse. It relies on
a binary search. This class also implements the derivatives of the inverse Tukey g-and-h transform with
respect to its parameters, necessary to the evaluation of the likelihood.

The simulation studies from the paper can be found in the Jupyter notebook docs/simulation_study.ipynb.
The application to global crop yield prediction can be found in docs/crop_yield_application.ipynb.

NB: I need to clean these (and delete some uncessary notebooks) before making the repo public.
