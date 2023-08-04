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

from __future__ import annotations

import os
import logging
import yaml
from typing import Any

import numpy as np
import pickle
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.decomposition as sklearn_decomposition
import sklearn.gaussian_process as sklearn_gaussian_process

from bayesian_inference import data_IO
from bayesian_inference import common_base

logger = logging.getLogger(__name__)


####################################################################################################################
def fit_emulators(config: EmulationConfig) -> None:
    '''
    Do PCA, fit emulators, and write to file.

    The first config.n_pc principal components (PCs) are emulated by independent Gaussian processes (GPs)
    The emulators map design points to PCs; the output will need to be inverted from PCA space to physical space.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    '''

    # Check if emulator already exists
    if os.path.exists(config.emulation_outputfile):
        if config.force_retrain:
            os.remove(config.emulation_outputfile)
            logger.info(f'Removed {config.emulation_outputfile}')
        else:
            logger.info(f'Emulators already exist: {config.emulation_outputfile} (to force retrain, set force_retrain: True)')
            return

    # Initialize predictions into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    # A consistent order of observables is enforced internally in data_IO
    # NOTE: One sample corresponds to one design point, while one feature is one bin of one observable
    logger.info(f'Doing PCA...')
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
    # Then, we still need to make sure to undo the preprocessing (centering and scaling) by:
    #     Y_reconstructed_unscaled = scaler.inverse_transform(Y_reconstructed)
    #
    # See docs for StandardScaler and PCA for further details.
    # This post explains exactly what fit_transform,inverse_transform do: https://stackoverflow.com/a/36567821
    #
    # TODO: do we want whiten=True? (NOTE: beware that inverse_transform also undoes whitening)
    scaler = sklearn_preprocessing.StandardScaler()
    pca = sklearn_decomposition.PCA(svd_solver='full', whiten=False) # Include all PCs here, so we can access them later
    # Scale data and perform PCA
    Y_pca = pca.fit_transform(scaler.fit_transform(Y))
    Y_pca_truncated = Y_pca[:,:config.n_pc]    # Select PCs here
    # Invert PCA and undo the scaling
    Y_reconstructed_truncated = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])
    Y_reconstructed_truncated_unscaled = scaler.inverse_transform(Y_reconstructed_truncated)
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(f'  Variance explained by first {config.n_pc} components: {np.sum(explained_variance_ratio[:config.n_pc])}')

    # Get design
    design = data_IO.design_array_from_h5(config.output_dir, filename='observables.h5')

    # Define GP kernel (covariance function)
    min = np.array(config.analysis_config['parametrization'][config.parameterization]['min'])
    max = np.array(config.analysis_config['parametrization'][config.parameterization]['max'])
    length_scale = max - min
    length_scale_bounds = (np.outer(length_scale, tuple(config.length_scale_bounds)))
    kernel_matern = sklearn_gaussian_process.kernels.Matern(length_scale=length_scale,
                                                            length_scale_bounds=length_scale_bounds,
                                                            nu=config.matern_nu,
                                                            )

    # Potential addition of noise kernel
    kernel = kernel_matern
    if config.noise is not None:
        kernel_noise = sklearn_gaussian_process.kernels.WhiteKernel(
            noise_level=config.noise["args"]["noise_level"],
            noise_level_bounds=config.noise["args"]["noise_level_bounds"],
        )
        kernel = (kernel_matern + kernel_noise)

    # Fit a GP (optimize the kernel hyperparameters) to map each design point to each of its PCs
    # Note that Z=(n_samples, n_components), so each PC corresponds to a row (i.e. a column of Z.T)
    logger.info("")
    logger.info(f'Fitting GPs...')
    logger.info(f'  The design has {design.shape[1]} parameters')
    emulators = [sklearn_gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                             alpha=config.alpha,
                                                             n_restarts_optimizer=config.n_restarts,
                                                             copy_X_train=False).fit(design, y) for y in Y_pca_truncated.T]

    # Print hyperparameters
    logger.info("")
    logger.info('Kernel hyperparameters:')
    [logger.info(f'  {emulator.kernel_}') for emulator in emulators]  # type: ignore[func-returns-value]
    logger.info("")

    # Write all info we want to file
    output_dict: dict[str, Any] = {}
    output_dict['PCA'] = {}
    output_dict['PCA']['Y'] = Y
    output_dict['PCA']['Y_pca'] = Y_pca
    output_dict['PCA']['Y_pca_truncated'] = Y_pca_truncated
    output_dict['PCA']['Y_reconstructed_truncated'] = Y_reconstructed_truncated
    output_dict['PCA']['Y_reconstructed_truncated_unscaled'] = Y_reconstructed_truncated_unscaled
    output_dict['PCA']['pca'] = pca
    output_dict['PCA']['scaler'] = scaler
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
    emulator_central_value = np.zeros((parameters.shape[0], config.n_pc))
    emulator_std = np.zeros((parameters.shape[0], config.n_pc))
    for i,emulator in enumerate(emulators):
        y_central_value, y_std = emulator.predict(parameters, return_std=True)
        #y_central_value, y_cov = emulator.predict(parameters, return_cov=True)
        emulator_central_value[:,i] = y_central_value
        emulator_std[:,i] = y_std

    # Reconstruct the physical space from the PCs, and invert preprocessing
    pca = results['PCA']['pca']
    scaler = results['PCA']['scaler']
    emulator_central_value_reconstructed_scaled = emulator_central_value.dot(pca.components_[:config.n_pc,:])
    emulator_central_value_reconstructed = scaler.inverse_transform(emulator_central_value_reconstructed_scaled)

    # TODO: propagate and return emulator_std

    # Construct dict of observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
    emulator_predictions = data_IO.prediction_dict_from_matrix(emulator_central_value_reconstructed, observables, validation_set=validation_set)

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

        # Observable inputs
        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']

        ########################
        # Emulator configuration
        ########################
        emulator_configuration = self.analysis_config["parameters"]["emulator"]
        self.force_retrain = emulator_configuration['force_retrain']
        self.n_pc = emulator_configuration['n_pc']
        self.mean_function = emulator_configuration['mean_function']
        self.constant = emulator_configuration['constant']
        self.linear_weights = emulator_configuration['linear_weights']
        self.covariance_function = emulator_configuration['covariance_function']
        self.matern_nu = emulator_configuration['matern_nu']
        self.variance = emulator_configuration['variance']
        self.length_scale_bounds = emulator_configuration["length_scale_bounds"]

        # Noise
        self.noise = emulator_configuration['noise']
        # Validation for noise configuration: Either None (null in yaml) or the noise configuration
        if self.noise is not None:
            # Check we have the appropriate keys
            assert [k in self.noise.keys() for k in ["type", "args"]], "Noise configuration must have keys 'type' and 'args'"
            if self.noise["type"] == "white":
                # Validate arguments
                # We don't want to do too much since we'll just be reinventing the wheel, but a bit can be helpful.
                assert set(self.noise["args"]) == set(["noise_level", "noise_level_bounds"]), "Must provide arguments 'noise_level' and 'noise_level_bounds' for white noise kernel"
            else:
                raise ValueError("Unsupported noise kernel")

        # GPR
        self.n_restarts = emulator_configuration["GPR"]['n_restarts']
        self.alpha = emulator_configuration["GPR"]["alpha"]

        # Output options
        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        self.emulation_outputfile = os.path.join(self.output_dir, 'emulation.pkl')
