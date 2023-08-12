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
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
import pickle
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.decomposition as sklearn_decomposition
import sklearn.gaussian_process as sklearn_gaussian_process

from bayesian_inference import data_IO
from bayesian_inference import common_base

logger = logging.getLogger(__name__)


####################################################################################################################
def fit_emulators(emulation_config: EmulationConfig) -> None:
    """ Do PCA, fit emulators, and write to file.

    :param EmulationConfig config: Configuration for the emulators, including all groups.
    """
    emulator_groups_output = {}
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        emulator_groups_output[emulation_group_name] = fit_emulator_group(emulation_group_config)
    for emulation_group_config, emulation_group_output in zip(emulation_config.emulation_groups_config.values(), emulator_groups_output.values()):
        write_emulators(config=emulation_group_config, output_dict=emulation_group_output)


####################################################################################################################
def fit_emulator_group(config: EmulationGroupConfig) -> dict[str, Any]:
    '''
    Do PCA, fit emulators, and write to file for an individual emulation group.

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
            return {}

    # Initialize predictions into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    # A consistent order of observables is enforced internally in data_IO
    # NOTE: One sample corresponds to one design point, while one feature is one bin of one observable
    logger.info(f'Doing PCA...')
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', observable_filter=config.observable_filter)

    # Use sklearn to:
    #  - Center and scale each feature (and later invert)
    #  - Perform PCA to reduce to config.n_pc features.
    #      This amounts to finding the matrix S that diagonalizes the covariance matrix C = Y.T*Y = S*D^2*S.T
    #      Or equivalently the right singular vectors S.T in the SVD decomposition of Y: Y = U*D*S.T
    #      Given S, we can transform from feature space to PCA space with: Y_PCA = Y*S
    #               and from PCA space to feature space with: Y = Y_PCA*S.T
    #
    # The input Y is a 2D array of format (n_samples, n_features).
    #
    # The output of pca.fit_transform() is a 2D array of format (n_samples, n_components),
    #   which is equivalent to:
    #     Y_pca = Y.dot(pca.components_.T), where:
    #       pca.components_ are the principal axes, sorted by decreasing explained variance -- shape (n_components, n_features)
    #     In the notation above, pca.components_ = S.T, i.e.
    #       the rows of pca.components_ are the sorted eigenvectors of the covariance matrix of the scaled features.
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
    min = np.array(config.analysis_config['parameterization'][config.parameterization]['min'])
    max = np.array(config.analysis_config['parameterization'][config.parameterization]['max'])
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

    return output_dict


####################################################################################################################
def read_emulators(config: EmulationGroupConfig) -> dict[str, Any]:
    # Validation
    filename = Path(config.emulation_outputfile)

    with filename.open("rb") as f:
        results = pickle.load(f)
    return results

####################################################################################################################
def write_emulators(config: EmulationGroupConfig, output_dict: dict[str, Any]) -> None:
    """Write emulators stored in a result from `fit_emulator_group` to file."""
    # Validation
    filename = Path(config.emulation_outputfile)

    with filename.open('wb') as f:
	    pickle.dump(output_dict, f)

####################################################################################################################
def nd_block_diag(arrs):
    """See: https://stackoverflow.com/q/62384509"""
    shapes = np.array([i.shape for i in arrs])

    out = np.zeros(np.append(np.amax(shapes[:,:-2],axis=0), [shapes[:,-2].sum(), shapes[:,-1].sum()]))
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes[:,-2:]):
        out[..., r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc

    return out

####################################################################################################################
def predict(parameters: npt.NDArray[np.float64],
            emulation_config: EmulationConfig,
            validation_set: bool = False,
            merge_predictions_over_groups: bool = True,
            emulation_group_results: dict[str, dict[str, Any]] | None = None) -> dict[str, npt.NDArray[np.float64]]:
    """
    Construct dictionary of emulator predictions for each observable

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
    :param EmulationConfig emulation_config: configuration object for the overall emulator (including all groups)
    :param bool validation_set: whether to use the validation set (True) or the training set (False)
    :param bool merge_predictions_over_groups: whether to merge predictions over emulation groups (True)
                                               or return a dictionary of predictions for each group (False). Default: True
    :param dict emulator_group_results: dictionary containing results from each emulation group. If None, read from file.
    :return dict emulator_predictions: dictionary containing matrices of central values and covariance
    """
    if emulation_group_results is None:
        emulation_group_results = {}
    predict_output = {}
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        emulation_group_result = emulation_group_results.get(emulation_group_name, read_emulators(emulation_group_config))
        predict_output[emulation_group_name] = predict_emulation_group(
            parameters=parameters,
            results=emulation_group_result,
            config=emulation_group_config,
        )

    # Allow the option to return immediately to allow the study of performance per emulation group
    if not merge_predictions_over_groups:
        return predict_output

    # Now, we want to merge predictions over groups
    # First, we need to figure out how observables map to the emulator groups
    all_observables = data_IO.read_dict_from_h5(emulation_config.output_dir, 'observables.h5')
    emulation_group_prediction_observable_dict = {}
    # TODO: This isn't terribly efficient. Can we store this mapping somehow?
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        group_predict_output = predict_output[emulation_group_name]
        emulation_group_prediction_observable_dict[emulation_group_name] = data_IO.observable_dict_from_matrix(
            Y=group_predict_output["central_value"],
            observables=all_observables,
            cov=group_predict_output.get("cov", np.array([])),
            validation_set=validation_set,
            observable_filter=emulation_group_config.observable_filter,
        )
        logger.info("hold up")

    # Does it have "central_value"? "cov"?
    available_value_types = set([
        value_type
        for group in emulation_group_prediction_observable_dict.values()
        for value_type in group
    ])
    logger.warning(f"{predict_output=}")
    logger.warning(f"{available_value_types=}")
    logger.warning(f"{emulation_group_prediction_observable_dict=}")

    # We don't care about the observable groups anymore, and there should be no overlap, so we'll just merge them together.
    # Validation
    flat_predictions_dict = {
        value_type: {
            k: v
            for group in emulation_group_prediction_observable_dict.values()
            for k, v in group[value_type].items()
        }
        for value_type in available_value_types
    }
    if len(set(flat_predictions_dict["central_value"])) != len(flat_predictions_dict["central_value"]):
        raise ValueError("Duplicate observable keys found when merging emulator groups!")
    merged_output: dict[str, dict[str, npt.NDArray[np.float64]]] = {
        # NOTE: We're compiling from emulation_group_prediction_observable_dict, but this contains the same info
        #       on whether there are "cov" matrices
        k: {} for k in available_value_types
    }
    # Reorder to match observables
    for value_type in available_value_types:
        for observable_key in data_IO.sorted_observable_list_from_dict(all_observables):
            value = flat_predictions_dict[value_type].get(observable_key)
            if value is not None:
                merged_output[value_type][observable_key] = value

    # {
    #     "central_value": np.array.shape: (n_design, n_features),
    #     "cov": np.array.shape: (n_design, n_features),
    # }
    # TODO: What about covariance between the groups? The cov here only accounts for the covariance within each group.
    result = {
        "central_value": data_IO.observable_matrix_from_dict(Y_dict=merged_output, values_to_return="central_value")
    }
    if "cov" in available_value_types:
        result["cov"] = nd_block_diag(list(merged_output["cov"].values()))
        #import scipy.linalg
        #mats = np.array().reshape(5, 3, 2, 2)
        #result = [scipy.linalg.block_diag(*bmats) for bmats in mats]
    return result


    # Next, we will merge the predictions of the groups together, careful to follow the order of the observables

    # Below is debug code for converting without following the order. Keeping around for now in case we need it.
    ## Note that we don't care about the keys of predict_output (which are just the emulation group names),
    ## but rather the keys stored in the predict_single results, which are "central_values" and "cov".
    ## This first line just lets us avoid hard coding those names.
    #merged_output: dict[str, list[npt.NDArray[np.float64]]] = {
    #    k: [] for k in predict_output.values()
    #}
    ## Now, we extract the prediction results from inside of each emulation group to merge the final result together.
    ## NOTE: In doing so, we will append one emulation group after another. This should be consistent with our
    ##       convention of ordering results.
    #for k in merged_output:
    #    for emulation_group_output in predict_output.values():
    #        merged_output[k].append(emulation_group_output[k])
    #return {
    #    k: np.concatenate(v, axis=1)
    #    for k, v in merged_output.items()
    #}


####################################################################################################################
def predict_emulation_group(parameters, results, config):
    '''
    Construct dictionary of emulator predictions for each observable in an emulation group.

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
    :param str results: dictionary that stores emulator

    :return dict emulator_predictions: dictionary containing matrices of central values and covariance

    Note: One can easily construct a dict of predictions with format emulator_predictions[observable_label]
          from the returned matrix as follows (useful for plotting / troubleshooting):
              observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
              emulator_predictions = data_IO.observable_dict_from_matrix(emulator_central_value_reconstructed,
                                                                         observables,
                                                                         cov=emulator_cov_reconstructed,
                                                                         validation_set=validation_set)
    '''

    # The emulators are stored as a list (one for each PC)
    emulators = results['emulators']

    # Get predictions (in PC space) from each emulator and concatenate them into a numpy array with shape (n_samples, n_PCs)
    # Note: we just get the std rather than cov, since we are interested in the predictive uncertainty
    #       of a given point, not the correlation between different sample points.
    n_samples = parameters.shape[0]
    emulator_central_value = np.zeros((n_samples, config.n_pc))
    emulator_variance = np.zeros((n_samples, config.n_pc))
    for i,emulator in enumerate(emulators):
        y_central_value, y_std = emulator.predict(parameters, return_std=True) # Alternately: return_cov=True
        emulator_central_value[:,i] = y_central_value
        emulator_variance[:,i] = y_std**2
    # Construct (diagonal) covariance matrix from the variances, for use in uncertainty propagation
    emulator_cov = np.apply_along_axis(np.diagflat, 1, emulator_variance)
    assert emulator_cov.shape == (n_samples, config.n_pc, config.n_pc)

    # Reconstruct the physical space from the PCs, and invert preprocessing.
    # Note we use array broadcasting to calculate over all samples.
    pca = results['PCA']['pca']
    scaler = results['PCA']['scaler']
    emulator_central_value_reconstructed_scaled = emulator_central_value.dot(pca.components_[:config.n_pc,:])
    emulator_central_value_reconstructed = scaler.inverse_transform(emulator_central_value_reconstructed_scaled)

    # Propagate uncertainty through the linear transformation back to feature space.
    # Note that for a vector f = Ax, the covariance matrix of f is C_f = A C_x A^T.
    #   (see https://en.wikipedia.org/wiki/Propagation_of_uncertainty)
    #   (Note also that even if C_x is diagonal, C_f will not be)
    # In our case, we have Y[i].T = S*Y_PCA[i].T for each point i in parameter space, where
    #    Y[i].T is a column vector of features -- shape (n_features,)
    #    Y_PCA[i].T is a column vector of corresponding PCs -- shape (n_pc,)
    #    S is the transfer matrix described above -- shape (n_features, n_pc)
    # So C_Y[i] = S * C_Y_PCA[i] * S^T.
    # Note: should be equivalent to: https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145
    # TODO: one can make this faster with broadcasting/einsum
    n_features = pca.components_.shape[1]
    S = pca.components_.T[:,:config.n_pc]
    emulator_cov_reconstructed_scaled = np.zeros((n_samples, n_features, n_features))
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] = S.dot(emulator_cov[i_sample].dot(S.T))
    assert emulator_cov_reconstructed_scaled.shape == (n_samples, n_features, n_features)

    # Propagate uncertainty: inverse preprocessing
    # We only need to undo the unit variance scaling, since the shift does not affect the covariance matrix.
    # We can do this by computing a outer product (i.e. product of each pairwise scaling),
    #   and multiple each element of the covariance matrix by this.
    scale_factors = scaler.scale_
    emulator_cov_reconstructed = emulator_cov_reconstructed_scaled*np.outer(scale_factors, scale_factors)

    # TODO: include predictive variance due to truncated PCs (also propagated as above)
    #np.sum(pca.explained_variance_[config.n_pc:])

    # Return the stacked matrices:
    #   Central values: (n_samples, n_features)
    #   Covariances: (n_samples, n_features, n_features)
    emulator_predictions = {}
    emulator_predictions['central_value'] = emulator_central_value_reconstructed
    emulator_predictions['cov'] = emulator_cov_reconstructed

    return emulator_predictions

####################################################################################################################
class EmulationGroupConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='', emulation_group_name: str | None = None):

        self.analysis_name = analysis_name
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
        if emulation_group_name is None:
            emulator_configuration = self.analysis_config["parameters"]["emulators"]
        else:
            emulator_configuration = self.analysis_config["parameters"]["emulators"][emulation_group_name]
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

        # Observable list
        # None implies a convention of accepting all available data
        self.observable_filter = None
        observable_list = emulator_configuration.get("observable_list", [])
        observable_exclude_list = emulator_configuration.get("observable_exclude_list", [])
        if observable_list or observable_exclude_list:
            self.observable_filter = data_IO.ObservableFilter(
                include_list=observable_list,
                exclude_list=observable_exclude_list,
            )

        # Output options
        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        emulation_outputfile_name = 'emulation.pkl'
        if emulation_group_name is not None:
            emulation_outputfile_name = f'emulation_group_{emulation_group_name}.pkl'
        self.emulation_outputfile = os.path.join(self.output_dir, emulation_outputfile_name)

@attrs.define
class EmulationConfig(common_base.CommonBase):
    analysis_name: str
    parameterization: str
    config_file: Path = attrs.field(converter=Path)
    analysis_config: dict[str, Any] = attrs.field(factory=dict)
    emulation_groups_config: dict[str, EmulationGroupConfig] = attrs.field(factory=dict)
    config: dict[str, Any] = attrs.field(init=False)
    output_dir: Path = attrs.field(init=False)

    def __attrs_post_init__(self):
        with self.config_file.open() as stream:
            self.config = yaml.safe_load(stream)
        self.output_dir = os.path.join(self.config['output_dir'], f'{self.analysis_name}_{self.parameterization}')

    @classmethod
    def from_config_file(cls, analysis_name: str, parameterization: str, config_file: Path, analysis_config: dict[str, Any]):
        c = cls(
            analysis_name=analysis_name,
            parameterization=parameterization,
            config_file=config_file,
            analysis_config=analysis_config,
        )
        # Initialize the config for each emulation group
        c.emulation_groups_config = {
            k: EmulationGroupConfig(
                analysis_name=c.analysis_name,
                parameterization=c.parameterization,
                analysis_config=c.analysis_config,
                config_file=c.config_file,
                emulation_group_name=k,
            )
            for k in c.analysis_config["parameters"]["emulators"]
        }
        return c

    def read_all_emulator_groups(self) -> dict[str, dict[str, npt.NDarray[np.float64]]]:
        """ Read all emulator groups.

        Just a convenience function.
        """
        emulation_results = {}
        for emulation_group_name, emulation_group_config in self.emulation_groups_config.items():
            emulation_results[emulation_group_name] = read_emulators(emulation_group_config)
        return emulation_results