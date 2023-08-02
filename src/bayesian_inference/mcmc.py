#! /usr/bin/env python
'''
Module related to MCMC, with functionality to compute posterior for a given analysis run

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() compute credible interval for a given posterior

A configuration class MCMCConfig provides simple access to emulation settings

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''

import os
import yaml
import logging
import os
import pickle

import emcee
import numpy as np
from scipy.linalg import lapack
from multiprocessing import Pool

from bayesian_inference import common_base
from bayesian_inference import data_IO
from bayesian_inference import emulation

logger = logging.getLogger(__name__)


####################################################################################################################
def run_mcmc(config):
    '''
    Run MCMC to compute posterior

    Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
    sampler (emcee) <http://dfm.io/emcee>`.
    '''

    # Get parameter names and min/max
    names = config.analysis_config['parameters'][config.parameterization]['names']
    min = config.analysis_config['parameters'][config.parameterization]['min']
    max = config.analysis_config['parameters'][config.parameterization]['max']
    ndim = len(names)

    # Load emulators
    with open(config.emulation_outputfile, 'rb') as f:
        emulators = pickle.load(f)

    # Load experimental data into dict: data[observable_label]['xmin'/'xmax'/'y'/'y_err']
    output_dir = config.output_dir
    filename = 'observables.h5'
    experimental_results = data_IO.data_array_from_h5(output_dir, filename, observable_table_dir=None)

    # TODO: If needed we can create a h5py dataset for compression/chunking
    #dset = f.create_dataset('chain', dtype='f8',
    #                        shape=(config.n_walkers, 0, ndim),
    #                        chunks=(config.n_walkers, 1, ndim),
    #                        maxshape=(config.n_walkers, None, ndim),
    #                        compression='lzf')

    # Construct sampler (we create a dummy daughter class from emcee.EnsembleSampler, to add some logging info)
    # Note: we pass the emulators and experimental data as args to the log_posterior function
    # TODO: Set pool parameter for parallelization
    #with Pool() as pool:
    #sampler = LoggingEnsembleSampler(config.n_walkers, self.ndim, _log_posterior, pool=pool)
    sampler = LoggingEnsembleSampler(config.n_walkers, ndim, _log_posterior, args=[min, max, config, emulators, experimental_results])

    # Generate random starting positions for each walker
    random_pos = np.random.uniform(min, max, (config.n_walkers, ndim))

    # Run first half of burn-in
    logging.info('Starting initial burn-in...')
    nburn0 = config.n_burn_steps // 2
    sampler.run_mcmc(random_pos, nburn0, n_logging_steps=config.n_logging_steps)

    # Reposition walkers to the most likely points in the chain, then run the second half of burn-in.
    # This significantly accelerates burn-in and helps prevent stuck walkers.
    logging.info('Resampling walker positions...')
    X0 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-config.n_walkers:]]
    sampler.reset()
    X0 = sampler.run_mcmc(X0, config.n_burn_steps - nburn0, n_logging_steps=config.n_logging_steps, storechain=False)[0]
    sampler.reset()
    logging.info('Burn-in complete.')

    # Production samples
    logging.info('Starting production...')
    sampler.run_mcmc(X0, config.n_sampling_steps, n_logging_steps=config.n_logging_steps)

    # Write to file
    logging.info('Writing chain to file...')
    output_dict = {}
    output_dict['chain'] = sampler.chain
    filename = os.path.join(config.output_dir, 'mcmc_chain.h5')
    data_IO.write_dict_to_h5(output_dict, filename, verbose=True)
    #dset.resize(dset.shape[1] + config.n_sampling_steps, 1)
    #dset[:, -config.n_sampling_steps:, :] = sampler.chain
    logging.info('Done.')

####################################################################################################################
def credible_interval(samples, confidence=0.9):
    """
    Compute the highest-posterior density (HPD) credible interval (default 90%) for an array of samples.
    """
    # number of intervals to compute
    nci = int((1 - confidence)*samples.size)

    # find highest posterior density (HPD) credible interval
    # i.e. the one with minimum width
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])   # interval lows
    cih = np.sort(samples[argp[-nci:]])  # interval highs
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]

#---------------------------------------------------------------
def _log_posterior(X, min, max, config, emulators, experimental_results):
    """
    Function to evaluate the log-posterior for a given set of input parameters.

    This function is called by https://emcee.readthedocs.io/en/stable/user/sampler/

    :X: input ndarray of parameter space values
    :min: list of minimum boundaries for each emulator parameter
    :max: list of maximum boundaries for each emulator parameter
    :config: configuration object
    :emulators: dict of emulators
    :experimental_results: dict of experimental results
    """

    # Convert to 2darray of shape (n_samples, n_parameters)
    X = np.array(X, copy=False, ndmin=2)

    # Initialize log-posterior array, which we will populate and return
    log_posterior = np.zeros(X.shape[0])

    # Check if any samples are outside the parameter bounds, and set log-posterior to -inf for those
    inside = np.all((X > min) & (X < max), axis=1)
    log_posterior[~inside] = -np.inf

    # Evaluate log-posterior for samples inside parameter bounds
    n_samples = np.count_nonzero(inside)
    if n_samples > 0:

        # Compute emulator prediction
        # returns dictionary of emulator predictions, with format emulator_predictions[observable_label]
        emulator_predictions = emulation.predict(X[inside], emulators, config)

        # Count the number of bins that we have, and construct arrays to store the difference between
        # emulator prediction and experimental data, and the covariance matrix
        sorted_observable_list = data_IO.sorted_observable_list_from_dict(experimental_results)
        n_data_points = 0
        for observable_label in sorted_observable_list:
            n_data_points += len(experimental_results[observable_label]['y'])

        # Loop through sorted list of observables and compute:
        #  - Difference between emulator prediction and experimental data for each bin
        #  - Covariance matrix
        dY = np.empty((n_samples, n_data_points))
        covariance_matrix = np.empty((n_samples, n_data_points, n_data_points))
        i_data_point = 0
        for observable_label in sorted_observable_list:

            #-------------------------
            # Get experimental data
            # TODO: include covariance matrix
            data_y = experimental_results[observable_label]['y']
            data_y_err = experimental_results[observable_label]['y_err']

            #-------------------------
            # Get emulator prediction
            # TODO: include covariance matrix
            emulator_prediction = emulator_predictions[observable_label]

            # Check that emulator prediction has same length as experimental data
            assert data_y.shape[0] == emulator_prediction.shape[1]

            #-------------------------
            # Compute difference (using broadcasting to subtract each data point from each emulator prediction)
            dY[:,i_data_point:i_data_point+data_y.shape[0]] = emulator_prediction - data_y

            #-------------------------
            # Fill covariance array
            # TODO: We want to add data covariance and emulator covariance.
            #       Currently we only include uncorrelated data uncertainty, and no emulator covariance.
            covariance_matrix[:,i_data_point:i_data_point+data_y.shape[0],i_data_point:i_data_point+data_y.shape[0]] = np.diag(data_y_err)

            # Increment data point counter
            i_data_point += data_y.shape[0]

        # Check that we have iterated through the appropriate number of data points
        assert i_data_point == n_data_points

        # Compute log likelihood at each point in the sample
        log_posterior[inside] += list(map(_loglikelihood, dY, covariance_matrix))

    return log_posterior

#---------------------------------------------------------------
def _loglikelihood(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()

####################################################################################################################
class LoggingEnsembleSampler(emcee.EnsembleSampler):
    '''
    Add some logging to the emcee.EnsembleSampler class.
    Inherit from: https://emcee.readthedocs.io/en/stable/user/sampler/
    '''

    #---------------------------------------------------------------
    def run_mcmc(self, X0, n_sampling_steps, n_logging_steps=100, **kwargs):
        """
        Run MCMC with logging every 'logging_steps' steps (default: log every 100 steps).
        """
        logging.info(f'running {self.nwalkers} walkers for {n_sampling_steps} steps')
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                logging.info(f'step {n}: acceptance fraction: mean {af.mean()}, std {af.std()}, min {af.min()}, max {af.max()}')

        return result

####################################################################################################################
class MCMCConfig(common_base.CommonBase):

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

        emulator_configuration = config["emulator_parameters"]
        self.n_pc = emulator_configuration['n_pc']

        mcmc_configuration = config["mcmc_parameters"]
        self.n_walkers = mcmc_configuration['n_walkers']
        self.n_burn_steps = mcmc_configuration['n_burn_steps']
        self.n_sampling_steps = mcmc_configuration['n_sampling_steps']
        self.n_logging_steps = mcmc_configuration['n_logging_steps']

        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        self.emulation_outputfile = os.path.join(self.output_dir, 'emulation.pkl')
