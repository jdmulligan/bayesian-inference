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

import logging
import os
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
        # NOTE: If it returns early because an emulator already exists, then we don't want to overwrite it!
        if emulator_groups_output[emulation_group_name]:
            write_emulators(config=emulation_group_config, output_dict=emulator_groups_output[emulation_group_name])
    # NOTE: We store everything in a dict so we can later return these if we decide it's helpful. However,
    #       it doesn't appear to be at the moment (August 2023), so we leave as is.

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
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename=config.observables_filename, observable_filter=config.observable_filter)

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
    # TODO: Do we want whiten the PCs, i.e. to scale the variances of each PC to 1?
    #       I don't see a compelling reason to do this...We are fitting separate GPs to each PC, 
    #       so standardizing the variance of each PC is not important.
    #       (NOTE: whitening can be done with whiten=True -- beware that inverse_transform also undoes whitening)
    scaler = sklearn_preprocessing.StandardScaler()
    # This adopts the sklearn convention, but then sets a max cap of 30 PCs (arbitrarily chosen) to
    # reduce computation time.
    max_n_components = config.max_n_components_to_calculate
    if max_n_components is not None:
        logger.info(f"Running with max n_pc={max_n_components}")
    pca = sklearn_decomposition.PCA(n_components=max_n_components, svd_solver='full', whiten=False) # Include all PCs here, so we can access them later
    # Scale data and perform PCA
    Y_pca = pca.fit_transform(scaler.fit_transform(Y))
    Y_pca_truncated = Y_pca[:,:config.n_pc]    # Select PCs here
    # Invert PCA and undo the scaling
    Y_reconstructed_truncated = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])
    Y_reconstructed_truncated_unscaled = scaler.inverse_transform(Y_reconstructed_truncated)
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(f'  Variance explained by first {config.n_pc} components: {np.sum(explained_variance_ratio[:config.n_pc])}')

    # Get design
    design = data_IO.design_array_from_h5(config.output_dir, filename=config.observables_filename)

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
    # Note that Y_PCA=(n_samples, n_components), so each PC corresponds to a row (i.e. a column of Y_PCA.T)
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
def nd_block_diag(arrays):
    """ Add 2D matrices into a block diagonal matrix in n-dimensions.

    See: https://stackoverflow.com/q/62384509

    :param arrays list[np.array]: List of arrays to block diagonalize.
    """
    shapes = np.array([i.shape for i in arrays])

    out = np.zeros(np.append(np.amax(shapes[:,:-2],axis=0), [shapes[:,-2].sum(), shapes[:,-1].sum()]))
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes[:,-2:]):
        out[..., r:r + rr, c:c + cc] = arrays[i]
        r += rr
        c += cc

    return out


####################################################################################################################
@attrs.define
class SortEmulationGroupObservables:
    """ Class to track and convert between emulation group matrices to match sorted observables.

    emulation_group_to_observable_matrix: Mapping from emulation group matrix to the matrix of observables. Format:
        {observable_name: (emulator_group_name, slice in output_matrix, slice in emulator_group_matrix)}
    shape: Shape of matrix output. Format: (n_design_points, n_features). Note that we may only be predicting
        one design point at a time, so we pick out the number of design points for the output based on the provided
        group outputs (which implicitly contains the required number of design points).
    available_value_types: Available value types in the group matrices. These will be extracted when the mapping is learned.
    """
    emulation_group_to_observable_matrix: dict[str, tuple[str, slice, slice]]
    shape: tuple[int, int]
    _available_value_types: set[str] | None = attrs.field(init=False, default=None)

    @classmethod
    def learn_mapping(cls, emulation_config: EmulationConfig) -> SortEmulationGroupObservables:
        """ Construct this object by learning the mapping from the emulation group prediction matrices to the sorted and merged matrices.

        :param EmulationConfig emulation_config: Configuration for the emulator(s).
        :return: Constructed object.
        """
        # NOTE: This could be configurable (eg. for validation). However, we don't seem to immediately
        #       need this functionality, so we'll omit it for now.
        prediction_key = "Prediction"

        # Now we need the mapping from emulator groups to observables with the right indices.
        # First, we need to start with all available observables (beyond just what's in any given group)
        # to learn the entire mapping
        # NOTE: It doesn't matter what observables file we use here since it's just to find all of the observables which are used.
        all_observables = data_IO.read_dict_from_h5(emulation_config.output_dir, 'observables.h5')
        current_position = 0
        observable_slices = {}
        for observable_key in data_IO.sorted_observable_list_from_dict(all_observables[prediction_key]):
            n_bins = all_observables[prediction_key][observable_key]['y'].shape[0]
            observable_slices[observable_key] = slice(current_position, current_position + n_bins)
            current_position += n_bins

        # Now, take advantage of the ordering in the emulator groups. (ie. the ordering in the group
        # matrix is consistent with the order of the observable names).
        observable_emulation_group_map = {}
        for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
            emulation_group_observable_keys = data_IO.sorted_observable_list_from_dict(all_observables[prediction_key], observable_filter=emulation_group_config.observable_filter)
            current_group_bin = 0
            for observable_key in emulation_group_observable_keys:
                observable_slice = observable_slices[observable_key]
                observable_emulation_group_map[observable_key] = (
                    emulation_group_name,
                    observable_slice,
                    slice(current_group_bin, current_group_bin + (observable_slice.stop - observable_slice.start))
                )
                current_group_bin += (observable_slice.stop - observable_slice.start)
                logger.debug(f"{observable_key=}, {observable_emulation_group_map[observable_key]=}, {current_group_bin=}")
        logger.debug(f"Sorted order: {observable_slices=}")

        # And then finally put them in the proper sorted observable order
        observable_emulation_group_map = {
            k: observable_emulation_group_map[k]
            for k in observable_slices
        }

        # We want the shape to allow us to preallocate the array:
        # Default shape: (n_design_points, n_features)
        last_observable = list(observable_slices)[-1]
        shape = (all_observables[prediction_key][observable_key]['y'].shape[1], observable_slices[last_observable].stop)
        logger.debug(f"{shape=} (note: for all design points)")

        return cls(
            emulation_group_to_observable_matrix=observable_emulation_group_map,
            shape=shape,
        )

    def convert(self, group_matrices: dict[str, dict[str, npt.NDArray[np.float64]]]) -> dict[str, npt.NDArray[np.float64]]:
        """ Convert a matrix to match the sorted observables.

        :param group_matrices: Matrixes to convert by emulation group. eg:
            {"group_1": {"central_value": np.array, "cov": [...]}, "group_2": np.array}
        :return: Converted matrix for each available value type.
        """
        if self._available_value_types is None:
            self._available_value_types = set([
                value_type
                for group in group_matrices.values()
                for value_type in group
            ])

        output = {}
        # Requires special handling since we're adding matrices (ie. 3d rather than 2d)
        if "cov" in self._available_value_types:
            # Setup
            value_type = "cov"

            # We have to sort them according to the mapping that we've derived.
            # However, it's not quite as trivial to just insert them (as we do for the central values),
            # so we'll use the output matrix slice as the key to sort by below.
            inputs_for_block_diag = {}
            for observable_name, (emulation_group_name, slice_in_output_matrix, slice_in_emulation_group_matrix) in self.emulation_group_to_observable_matrix.items():
                emulation_group_matrix = group_matrices[emulation_group_name]
                # NOTE: The slice_in_output_matrix.start should provide unique integers to sort by
                #       (basically, we just use the starting position instead of inserting it directly).
                inputs_for_block_diag[slice_in_output_matrix.start] = emulation_group_matrix[value_type][:, slice_in_emulation_group_matrix, slice_in_emulation_group_matrix]

            # And then merge them together in a block diagonal, sorting to put them in the right order
            output[value_type] = nd_block_diag(
                # sort based on the start value of the slice in the output matrix.
                [
                    # NOTE: We don't want to pass the key, but we need it for sorting, so we then
                    #       have to explicitly select the actual matrices (ie. the v of the k, v pair)
                    #       to pass along.
                    m[1]
                    for m in sorted(
                        inputs_for_block_diag.items(), key=lambda x: x[0]
                    )
                ]
            )

        # Handle the other values (as of 14 August 2023, it's just "central_value")
        for value_type in self._available_value_types:
            # Skip over "cov" since we handled it explicitly above.
            if value_type == "cov":
                continue

            # Since the number of design points that we want to predict varies, we can't define the output
            # until we can extract it from one group output. So we wait to initialize the output matrix until
            # we have the first group output.
            output[value_type] = None
            for observable_name, (emulation_group_name, slice_in_output_matrix, slice_in_emulation_group_matrix) in self.emulation_group_to_observable_matrix.items():
                emulation_group_matrix = group_matrices[emulation_group_name]
                if output[value_type] is None:
                    output[value_type] = np.zeros((emulation_group_matrix[value_type].shape[0], *self.shape[1:]))
                output[value_type][:, slice_in_output_matrix] = emulation_group_matrix[value_type][:, slice_in_emulation_group_matrix]

        return output


####################################################################################################################
def predict(parameters: npt.NDArray[np.float64],
            emulation_config: EmulationConfig,
            merge_predictions_over_groups: bool = True,
            emulation_group_results: dict[str, dict[str, Any]] | None = None) -> dict[str, npt.NDArray[np.float64]]:
    """
    Construct dictionary of emulator predictions for each observable

    :param ndarray[float] parameters: list of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
    :param EmulationConfig emulation_config: configuration object for the overall emulator (including all groups)
    :param bool merge_predictions_over_groups: whether to merge predictions over emulation groups (True)
                                               or return a dictionary of predictions for each group (False). Default: True
    :param dict emulator_group_results: dictionary containing results from each emulation group. If None, read from file.
    :return dict emulator_predictions: dictionary containing matrices of central values and covariance
    """
    if emulation_group_results is None:
        emulation_group_results = {}
    predict_output = {}
    for emulation_group_name, emulation_group_config in emulation_config.emulation_groups_config.items():
        emulation_group_result = emulation_group_results.get(emulation_group_name)
        # Only load the emulator group directly from file if needed. If called frequently
        # (eg. in the MCMC), it's probably better to load it once and pass it in.
        # NOTE: I know that get() can provide a second argument as the default, but a quick check showed that
        #       `read_emulators` was executing far more than expected (maybe trying to determine some default value?).
        #       However, separating it out like this seems to avoid the issue, but better to just avoid the issue.
        if emulation_group_result is None:
            emulation_group_result = read_emulators(emulation_group_config)

        predict_output[emulation_group_name] = predict_emulation_group(
            parameters=parameters,
            results=emulation_group_result,
            config=emulation_group_config,
        )

    # Allow the option to return immediately to allow the study of performance per emulation group
    if not merge_predictions_over_groups:
        return predict_output

    # Now, we want to merge predictions over groups
    return emulation_config.sort_observables_in_matrix.convert(group_matrices=predict_output)


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

    # Include predictive variance due to truncated PCs.
    # We can do this by decomposing the original covariance in feature space:
    #   C_Y = S D^2 S^T 
    #       = S_{<=n_pc} D^2_{<=n_pc} S_{<=n_pc}^T + S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T
    # In general, we want to estimate the covariance as a function of theta.
    # We can do this for the first term by estimating it with the emulator covariance constructed above,
    #   as a function of theta.
    # We can't do this with the second term, since we didn't emulate it -- so we estimate it,
    #   treating it as independent of theta, and add it to the emulator covariance:
    #     Sigma_unexplained = 1/n_samples * S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T,
    #   where we include the 1/n_samples factor to account for the fact that we are estimating the covariance from a set of samples.
    # See eqs 21-22 of https://arxiv.org/pdf/2102.11337.pdf
    # TODO: double check this (and compare to https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145)
    S_unexplained = pca.components_.T[:,config.n_pc:]
    D_unexplained = np.diag(pca.explained_variance_[config.n_pc:])
    emulator_cov_unexplained = S_unexplained.dot(D_unexplained.dot(S_unexplained.T)) / n_samples
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] += emulator_cov_unexplained

    # Propagate uncertainty: inverse preprocessing
    # We only need to undo the unit variance scaling, since the shift does not affect the covariance matrix.
    # We can do this by computing an outer product (i.e. product of each pairwise scaling),
    #   and multiplying each element of the covariance matrix by this.
    scale_factors = scaler.scale_
    emulator_cov_reconstructed = emulator_cov_reconstructed_scaled*np.outer(scale_factors, scale_factors)

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
        self.observables_filename = config["observables_filename"]

        ########################
        # Emulator configuration
        ########################
        if emulation_group_name is None:
            emulator_configuration = self.analysis_config["parameters"]["emulators"]
        else:
            emulator_configuration = self.analysis_config["parameters"]["emulators"][emulation_group_name]
        self.force_retrain = emulator_configuration['force_retrain']
        self.n_pc = emulator_configuration['n_pc']
        self.max_n_components_to_calculate = emulator_configuration.get("max_n_components_to_calculate", None)
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
    # Optional objects that may provide useful additional functionality
    _observable_filter: data_IO.ObservableFilter | None = attrs.field(init=False, default=None)
    _sort_observables_in_matrix: SortEmulationGroupObservables | None = attrs.field(init=False, default=None)

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

    def read_all_emulator_groups(self) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
        """ Read all emulator groups.

        Just a convenience function.
        """
        emulation_results = {}
        for emulation_group_name, emulation_group_config in self.emulation_groups_config.items():
            emulation_results[emulation_group_name] = read_emulators(emulation_group_config)
        return emulation_results

    @property
    def observable_filter(self) -> data_IO.ObservableFilter:
        if self._observable_filter is None:
            if not self.emulation_groups_config:
                raise ValueError("Need to specify emulation groups to provide an observable filter")
            # Accumulate the include and exclude lists from all emulation groups
            include_list: list[str] = []
            exclude_list: list[str] = self.config.get("global_observable_exclude_list", [])
            for emulation_group_config in self.emulation_groups_config.values():
                include_list.extend(emulation_group_config.observable_filter.include_list)  # type: ignore[union-attr]
                exclude_list.extend(emulation_group_config.observable_filter.exclude_list)  # type: ignore[union-attr]

            self._observable_filter = data_IO.ObservableFilter(
                include_list=include_list,
                exclude_list=exclude_list,
            )
        return self._observable_filter

    @property
    def sort_observables_in_matrix(self) -> SortEmulationGroupObservables:
        if self._sort_observables_in_matrix is None:
            if not self.emulation_groups_config:
                raise ValueError("Need to specify emulation groups to provide an sorting for observable group observables")
            # Accumulate the include and exclude lists from all emulation groups
            self._sort_observables_in_matrix = SortEmulationGroupObservables.learn_mapping(self)
        return self._sort_observables_in_matrix