#! /usr/bin/env python
'''
Module related to emulators, with functionality to train and call emulators for a given analysis run

The main functionalities are:
 - fit_emulators() performs PCA, fits an emulator to each PC, and writes the emulator to file
 - inverse_transform() transforms from PCA space to physical space

A configuration class EmulationConfig provides simple access to emulation settings 

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

import os
import yaml

import numpy as np
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.decomposition as sklearn_decomposition
import sklearn.gaussian_process as sklearn_gaussian_process

import data_IO
import common_base

####################################################################################################################
def fit_emulators(config):
    '''
    Do PCA, fit emulators, and write to file.

    Note that the emulators map design points to PCs; the output will need to be inverted from PCA space to physical space.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    '''

    # Check if emulator already exists
    emulator_path = os.path.join(config.output_dir, 'emulator.pkl')
    if os.path.exists(emulator_path):
        if config.retrain_emulator:
            os.remove(emulator_path)
            print(f'Removed {emulator_path}')
        else:
            print(f'Emulator already exists: {emulator_path} (to force retrain, set retrain_emulator: True')

    # Initialize observables
    observables = data_IO.read_data(os.path.join(config.output_dir, f'{config.analysis_name}_{config.parameterization}'), 
                                    filename='observables.h5')

    # Concatenate observables into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    print(f'Doing PCA...')
    observable_list = observables['Prediction'].keys()
    for i,observable_label in enumerate(observable_list):
        values = observables['Prediction'][observable_label]['y'].T
        if i==0:
            Y = values
        else:
            Y = np.concatenate([Y,values], axis=1)
    print(f'  Total shape of data (observable_bins, n_design_points): {Y.shape}')

    # Use sklearn to:
    #  - Center and scale each feature (and later invert)
    #  - Perform PCA to reduce to self.n_pc features
    # Note: The below code assumes Y is a 2D array of format (n_samples, n_features) and returns a
    #       2D array Z of format (n_samples, n_components) -- (see docs for StandardScaler and PCA)
    # TODO: do we want whiten=True?
    scaler = sklearn_preprocessing.StandardScaler()
    pca = sklearn_decomposition.PCA(svd_solver='full', whiten=True) # Include all PCs here, so we can access them later
    Z = pca.fit_transform(scaler.fit_transform(Y))[:,:config.n_pc]    # Select PCs here
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f'  Variance explained by first {config.n_pc} components: {np.sum(explained_variance_ratio[:config.n_pc])}')

    # Get design
    design = observables['Design']

    # Define GP kernel (covariance function)
    # TODO: abstract parameters / choices to config file
    min = np.array(config.analysis_config['parameters'][config.parameterization]['min'])
    max = np.array(config.analysis_config['parameters'][config.parameterization]['max'])
    length_scale = max - min
    length_scale_bounds = (np.outer(length_scale, (0.1, 10)))
    kernel_matern = sklearn_gaussian_process.kernels.Matern(length_scale=length_scale,
                                                            length_scale_bounds=length_scale_bounds,
                                                            nu=1.5,
                                                            )            
    noise0 = 0.5**2
    noisemin = 0.01**2
    noisemax = 1**2
    kernel_noise = sklearn_gaussian_process.kernels.WhiteKernel(noise_level=noise0,
                                                                noise_level_bounds=(noisemin, noisemax)
                                                               )
    kernel = (kernel_matern + kernel_noise)

    # Fit a GP (optimize the kernel hyperparameters) to map each design point to each of its PCs
    # Note that Z=(n_samples, n_components), so each PC corresponds to a row (i.e. a column of Z.T)
    alpha = 1.e-10
    print()
    print(f'Fitting GPs...')
    print(f'  The design has {design.shape[1]} parameters')
    gps = [sklearn_gaussian_process.GaussianProcessRegressor(kernel=kernel, 
                                                             alpha=alpha,
                                                             n_restarts_optimizer=config.n_restarts,
                                                             copy_X_train=False).fit(design, z) for z in Z.T]

    # Write the emulators to file

####################################################################################################################
class EmulationConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='', output_dir='', **kwargs):

        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = config_file
        self.output_dir = output_dir

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.force_retrain = config['force_retrain']
        self.n_pc = config['n_pc']
        self.n_restarts = config['n_restarts']
        self.mean_function = config['mean_function']
        self.constant = config['constant']
        self.linear_weights = config['linear_weights']
        self.covariance_function = config['covariance_function']
        self.matern_nu = config['matern_nu']
        self.variance = config['variance']
        self.noise = config['noise']