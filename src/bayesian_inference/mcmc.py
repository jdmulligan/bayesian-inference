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

import emcee
import numpy as np
from scipy.linalg import lapack
from multiprocessing import Pool

from bayesian_inference import common_base
from bayesian_inference import data_IO
from bayesian_inference import emulation

####################################################################################################################
def run_mcmc(config):
    '''
    Run MCMC to compute posterior

    Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
    sampler (emcee) <http://dfm.io/emcee>`_.
    '''

    # Get parameter names and min/max
    names = config.analysis_config['parameters'][config.parameterization]['names']
    min = config.analysis_config['parameters'][config.parameterization]['min']
    max = config.analysis_config['parameters'][config.parameterization]['max']
    ndim = len(names)

    # TODO: If needed we can create a h5py dataset for compression/chunking
    #dset = f.create_dataset('chain', dtype='f8',
    #                        shape=(config.n_walkers, 0, ndim),
    #                        chunks=(config.n_walkers, 1, ndim),
    #                        maxshape=(config.n_walkers, None, ndim),
    #                        compression='lzf')

    # TODO: Set pool parameter for parallelization
    #with Pool() as pool:
    #sampler = LoggingEnsembleSampler(config.n_walkers, self.ndim, _log_posterior, pool=pool)
    sampler = LoggingEnsembleSampler(config.n_walkers, ndim, _log_posterior, args=[min, max])

    # Run first half of burn-in starting from random positions.
    logging.info('Starting initial burn-in...')
    nburn0 = config.n_burn_steps // 2
    sampler.run_mcmc(_random_pos(config.n_walkers), nburn0, n_logging_steps=config.n_logging_steps)

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
    Compute the highest-posterior density (HPD) credible interval (default 90%)
    for an array of samples.
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
def _log_posterior(self, X, min, max, extra_std_prior_scale=0.05, model_sys_error=False):
    """
    Evaluate the posterior at `X`.

    `extra_std_prior_scale` is the scale parameter for the prior
    distribution on the model sys error parameter:

        prior ~ sigma^2 * exp(-sigma/scale)

    This model sys error parameter is not by default implemented.
    """
    X = np.array(X, copy=False, ndmin=2)

    lp = np.zeros(X.shape[0])

    inside = np.all((X > min) & (X < max), axis=1)
    lp[~inside] = -np.inf

    nsamples = np.count_nonzero(inside)

    if nsamples > 0:
        if model_sys_error:
            extra_std = X[inside, -1]
        else:
            extra_std = 0.0

        pred = _predict(X[inside], return_cov=True, extra_std=extra_std)

        # allocate difference (model - expt) and covariance arrays
        nobs = self._expt_y.size
        dY = np.empty((nsamples, nobs))
        cov = np.empty((nsamples, nobs, nobs))

        model_Y, model_cov = pred

        # copy predictive mean and covariance into allocated arrays
        for obs1, subobs1, slc1 in self._slices:
            dY[:, slc1] = model_Y[obs1][subobs1]
            for obs2, subobs2, slc2 in self._slices:
                cov[:, slc1, slc2] = model_cov[(obs1, subobs1), (obs2, subobs2)]

        # subtract expt data from model data
        dY -= self._expt_y

        # add expt cov to model cov
        cov += self._expt_cov

        # compute log likelihood at each point
        lp[inside] += list(map(_mvn_loglike, dY, cov))

        # add prior for extra_std (model sys error)
        if model_sys_error:
            lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale

    return lp

#---------------------------------------------------------------
def _predict(self, X, **kwargs):
    """
    Call each system emulator to predict model output at X.

    """

    emulator_predictions = emulation.predict(X, results, config, validation_set=validation_set)

    return {
        sys: emulator.Emulator.from_cache(sys, self.workdir).predict(
            X[:, ],#[n] + self._common_indices],
            **kwargs
        )
        for n, sys in enumerate(self.systems)
    }

#---------------------------------------------------------------
def _mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

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

#---------------------------------------------------------------
def _random_pos(self, n=1):
    """
    Generate `n` random positions in parameter space.
    """
    return np.random.uniform(self.min, self.max, (n, self.ndim))

####################################################################################################################
class LoggingEnsembleSampler(emcee.EnsembleSampler):

    #---------------------------------------------------------------
    def run_mcmc(self, X0, n_sampling_steps, n_logging_steps=100, **kwargs):
        """
        Run MCMC with logging every 'logging_steps' steps (default: log every 100 steps).
        """
        logging.info(f'running {self.k} walkers for {n_sampling_steps} steps')
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
        self.n_walkers = config['n_walkers']
        self.n_burn_steps = config['n_burn_steps']
        self.n_sampling_steps = config['n_sampling_steps']
        self.n_logging_steps = config['n_logging_steps']

        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        self.emulation_outputfile = os.path.join(self.output_dir, 'emulation.pkl')
