#! /usr/bin/env python
'''
Module related to emulators, with functionality to train and call emulators for a given analysis run

The main functionalities are:
 - fit_emulators() performs PCA, fits an emulator to each PC, and writes the emulator to file
 - predict() construct mean, std of emulator for a given set of parameter values

A configuration class EmulationConfig provides simple access to emulation settings

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

import os
import yaml

import numpy as np
import pickle
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.decomposition as sklearn_decomposition
import sklearn.gaussian_process as sklearn_gaussian_process

from bayesian_inference import data_IO
from bayesian_inference import common_base

####################################################################################################################
def fit_emulators(config):
    '''
    Do PCA, fit emulators, and write to file.
.
    The first config.n_pc principal components (PCs) are emulated by independent Gaussian processes (GPs)
    The emulators map design points to PCs; the output will need to be inverted from PCA space to physical space.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    '''

    # Check if emulator already exists
    if os.path.exists(config.emulation_outputfile):
        if config.force_retrain:
            os.remove(config.emulation_outputfile)
            print(f'Removed {config.emulation_outputfile}')
        else:
            print(f'Emulators already exist: {config.emulation_outputfile} (to force retrain, set force_retrain: True)')
            return

    # Initialize predictions into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    # A consistent order of observables is enforced internally in data_IO
    print(f'Doing PCA...')
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5')

    # Use sklearn to:
    #  - Center and scale each feature (and later invert)
    #  - Perform SVD-based PCA to reduce to config.n_pc features
    #
    # The input Y is a 2D array of format (n_samples, n_features).
    #
    # The output of pca.fit_transform() is a 2D array of format (n_samples, n_components),
    #   which is equivalent to:
    #     Y_pca = Y.dot(pca.components_.T), where:
    #       pca.components_ are the principal axes, sorted by decreasing explained variance -- shape (n_components, n_features)
    #
    # We can invert this back to the original feature space by: pca.inverse_transform(Y_pca),
    #   which is equivalent to:
    #     Y_reconstructed = Y_pca.dot(pca.components_)
    #
    # See docs for StandardScaler and PCA for further details.
    # This post explains exactly what fit_transform,inverse_transform do: https://stackoverflow.com/a/36567821
    #
    # TODO: do we want whiten=True? (NOTE: beware that inverse_transform also undoes whitening)
    scaler = sklearn_preprocessing.StandardScaler()
    pca = sklearn_decomposition.PCA(svd_solver='full', whiten=False) # Include all PCs here, so we can access them later
    Y_pca = pca.fit_transform(scaler.fit_transform(Y))
    Y_pca_truncated = Y_pca[:,:config.n_pc]    # Select PCs here
    Y_reconstructed_truncated = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f'  Variance explained by first {config.n_pc} components: {np.sum(explained_variance_ratio[:config.n_pc])}')

    # Get design
    design = data_IO.design_array_from_h5(config.output_dir, filename='observables.h5')

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
    emulators = [sklearn_gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                             alpha=alpha,
                                                             n_restarts_optimizer=config.n_restarts,
                                                             copy_X_train=False).fit(design, y) for y in Y_pca_truncated.T]

    # Print hyperparameters
    print()
    print('Kernel hyperparameters:')
    [print(f'  {emulator.kernel_}') for emulator in emulators]
    print()

    # Write all info we want to file
    output_dict = {}
    output_dict['PCA'] = {}
    output_dict['PCA']['Y'] = Y
    output_dict['PCA']['Y_pca'] = Y_pca
    output_dict['PCA']['Y_pca_truncated'] = Y_pca_truncated
    output_dict['PCA']['Y_reconstructed_truncated'] = Y_reconstructed_truncated
    output_dict['PCA']['pca'] = pca
    output_dict['emulators'] = emulators
    with open(config.emulation_outputfile, 'wb') as f:
	    pickle.dump(output_dict, f)

####################################################################################################################
def predict(parameters, results, config, validation_set=False):
    '''
    Construct dictionary of emulator predictions for each observable

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_design_points, n_parameters)
    :param str results: dictionary that stores emulator

    :return dict emulator_predictions: dictionary of emulator predictions, with format emulator_predictions[observable_label]
    '''

    # The emulators are stored as a list (one for each PC)
    emulators = results['emulators']

    # Get predictions (in PC space) from each emulator and concatenate them into a numpy array with shape (n_design_points, n_PCs)
    emulator_mean = np.zeros((parameters.shape[0], config.n_pc))
    emulator_std = np.zeros((parameters.shape[0], config.n_pc))
    for i,emulator in enumerate(emulators):
        y_mean, y_std = emulator.predict(parameters, return_std=True)
        #y_mean, y_cov = emulator.predict(parameters, return_cov=True)
        emulator_mean[:,i] = y_mean
        emulator_std[:,i] = y_std

    # Reconstruct the physical space from the PCs
    pca = results['PCA']['pca']
    emulator_mean_reconstructed = emulator_mean.dot(pca.components_[:config.n_pc,:])

    # TODO: propagate and return emulator_std

    # Construct dict of observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
    emulator_predictions = data_IO.prediction_dict_from_matrix(emulator_mean_reconstructed, observables, validation_set=validation_set)

    return emulator_predictions

####################################################################################################################
class EmulationConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='', **kwargs):

        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = config_file

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']
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

        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        self.emulation_outputfile = os.path.join(self.output_dir, 'emulation.pkl')
