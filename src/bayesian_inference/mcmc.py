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
import multiprocessing
import numpy as np

from bayesian_inference import common_base
from bayesian_inference import data_IO
from bayesian_inference import emulation
from bayesian_inference import log_posterior

logger = logging.getLogger(__name__)


####################################################################################################################
def run_mcmc(config, closure_index=-1):
    '''
    Run MCMC to compute posterior

    Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
    sampler (emcee) <http://dfm.io/emcee>`.

    :param MCMCConfig config: Instance of MCMCConfig
    :param int closure_index: Index of validation design point to use for MCMC closure. Off by default.
                              If non-negative index is specified, will construct pseudodata from the design point
                              and use that for the closure test.
    '''

    # Get parameter names and min/max
    names = config.analysis_config['parameterization'][config.parameterization]['names']
    min = config.analysis_config['parameterization'][config.parameterization]['min']
    max = config.analysis_config['parameterization'][config.parameterization]['max']
    ndim = len(names)

    # Load emulators
    emulation_config = emulation.EmulationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    emulation_results = emulation_config.read_all_emulator_groups()

    # Pre-compute the predictive variance due to PC truncation, since it is independent of theta.
    emulator_cov_unexplained = emulation.compute_emulator_cov_unexplained(emulation_config, emulation_results)

    # Load experimental data into arrays: experimental_results['y'/'y_err'] (n_features,)
    # In the case of a closure test, we use the pseudodata from the validation design point
    experimental_results = data_IO.data_array_from_h5(config.output_dir, 'observables.h5', pseudodata_index=closure_index, observable_filter=emulation_config.observable_filter)

    # TODO: By default the chain will be stored in memory as a numpy array
    #       If needed we can create a h5py dataset for compression/chunking

    # We can use multiprocessing in emcee to parallelize the independent walkers
    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, the some of the caching mechanisms
    #       (eg. used in learning the emulator group mapping doesn't work)
    # NOTE: We use `get_context` here to avoid having to globally specify the context. Plus, it then should be fine
    #       to repeated call this function. (`set_context` can only be called once - otherwise, it's a runtime error).
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(initializer=log_posterior.initialize_pool_variables, initargs=[min, max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained]) as pool:

        # Construct sampler (we create a dummy daughter class from emcee.EnsembleSampler, to add some logging info)
        # Note: we pass the emulators and experimental data as args to the log_posterior function
        logger.info('Initializing sampler...')
        sampler = LoggingEnsembleSampler(config.n_walkers, ndim, log_posterior.log_posterior,
                                        #args=[min, max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained],
                                        pool=pool)

        # Generate random starting positions for each walker
        random_pos = np.random.uniform(min, max, (config.n_walkers, ndim))

        # Run first half of burn-in
        logger.info(f'Parallelizing over {pool._processes} processes...')
        logger.info('Starting initial burn-in...')
        nburn0 = config.n_burn_steps // 2
        sampler.run_mcmc(random_pos, nburn0, n_logging_steps=config.n_logging_steps)

        # Reposition walkers to the most likely points in the chain, then run the second half of burn-in.
        # This significantly accelerates burn-in and helps prevent stuck walkers.
        logger.info('Resampling walker positions...')
        X0 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-config.n_walkers:]]
        sampler.reset()
        X0 = sampler.run_mcmc(X0, config.n_burn_steps - nburn0, n_logging_steps=config.n_logging_steps)[0]
        sampler.reset()
        logger.info('Burn-in complete.')

        # Production samples
        logger.info('Starting production...')
        sampler.run_mcmc(X0, config.n_sampling_steps, n_logging_steps=config.n_logging_steps)

        # Write to file
        logger.info('Writing chain to file...')
        output_dict = {}
        output_dict['chain'] = sampler.get_chain()
        output_dict['acceptance_fraction'] = sampler.acceptance_fraction
        output_dict['log_prob'] = sampler.get_log_prob()
        try:
            output_dict['autocorrelation_time'] = sampler.get_autocorr_time()
        except Exception as e:
            output_dict['autocorrelation_time'] = None
            logger.info(f"Could not compute autocorrelation time: {str(e)}")
        # If closure test, save the design point parameters and experimental pseudodata
        if closure_index >= 0:
            design_point =  data_IO.design_array_from_h5(config.output_dir, filename='observables.h5', validation_set=True)[closure_index]
            output_dict['design_point'] = design_point
            output_dict['experimental_pseudodata'] = experimental_results
        data_IO.write_dict_to_h5(output_dict, config.mcmc_output_dir, 'mcmc.h5', verbose=True)

        # Save the sampler to file as well, in case we want to access it later
        #   e.g. sampler.get_chain(discard=n_burn_steps, thin=thin, flat=True)
        # Note that currently we use sampler.reset() to discard the burn-in and reposition
        #   the walkers (and free memory), but it prevents us from plotting the burn-in samples.
        with open(config.sampler_outputfile, 'wb') as f:
            pickle.dump(sampler, f)

        logger.info('Done.')

####################################################################################################################
def credible_interval(samples, confidence=0.9, interval_type='quantile'):
    '''
    Compute the credible interval for an array of samples.

    TODO: one could also call the versions in pymc3 or arviz

    :param 1darray samples: Array of samples
    :param float confidence: Confidence level (default 0.9)
    :param str type: Type of credible interval to compute. Options are:
                        'hpd' - highest-posterior density
                        'quantile' - quantile interval
    '''

    if interval_type == 'hpd':
        # number of intervals to compute
        nci = int((1 - confidence)*samples.size)
        # find highest posterior density (HPD) credible interval i.e. the one with minimum width
        argp = np.argpartition(samples, [nci, samples.size - nci])
        cil = np.sort(samples[argp[:nci]])   # interval lows
        cih = np.sort(samples[argp[-nci:]])  # interval highs
        ihpd = np.argmin(cih - cil)
        ci = cil[ihpd], cih[ihpd]

    elif interval_type == 'quantile':
        cred_range = [(1-confidence)/2, 1-(1-confidence)/2]
        ci = np.quantile(samples, cred_range)

    return ci

####################################################################################################################
def map_parameters(posterior, method='quantile'):
    '''
    Compute the MAP parameters

    :param 1darray posterior: Array of samples
    :param str method: Method used to compute MAP. Options are:
                        'quantile' - take a narrow quantile interval and compute mean of parameters in that interval
    :return 1darray map_parameters: Array of MAP parameters
    '''

    if method == 'quantile':
        central_quantile = 0.01
        lower_bounds = np.quantile(posterior, 0.5-central_quantile/2, axis=0)
        upper_bounds = np.quantile(posterior, 0.5+central_quantile/2, axis=0)
        mask = (posterior >= lower_bounds) & (posterior <= upper_bounds)
        map_parameters = np.array([posterior[mask[:,i],i].mean() for i in range(posterior.shape[1])])

    return map_parameters

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
        logger.info(f'  running {self.nwalkers} walkers for {n_sampling_steps} steps')
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                logger.info(f'  step {n}: acceptance fraction: mean {af.mean()}, std {af.std()}, min {af.min()}, max {af.max()}')

        return result

####################################################################################################################
class MCMCConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='',
                       closure_index=-1, **kwargs):

        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = config_file

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']
        self.observables_filename = config["observables_filename"]

        mcmc_configuration = analysis_config["parameters"]["mcmc"]
        self.n_walkers = mcmc_configuration['n_walkers']
        self.n_burn_steps = mcmc_configuration['n_burn_steps']
        self.n_sampling_steps = mcmc_configuration['n_sampling_steps']
        self.n_logging_steps = mcmc_configuration['n_logging_steps']

        self.output_dir = os.path.join(config['output_dir'], f'{analysis_name}_{parameterization}')
        self.emulation_outputfile = os.path.join(self.output_dir, 'emulation.pkl')
        self.mcmc_outputfilename = 'mcmc.h5'
        if closure_index < 0:
            self.mcmc_output_dir = self.output_dir
        else:
            self.mcmc_output_dir = os.path.join(self.output_dir, f'closure/results/{closure_index}')
        self.mcmc_outputfile = os.path.join(self.mcmc_output_dir, 'mcmc.h5')
        self.sampler_outputfile = os.path.join(self.mcmc_output_dir, 'mcmc_sampler.pkl')

        # Update formatting of parameter names for plotting
        unformatted_names = self.analysis_config['parameterization'][self.parameterization]['names']
        self.analysis_config['parameterization'][self.parameterization]['names'] = [rf'{s}' for s in unformatted_names]
