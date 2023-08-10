#! /usr/bin/env python
'''
Module related to generate plots for MCMC

authors: J.Mulligan, R.Ehlers
'''

import logging
import os

import numpy as np
import pandas as pd
import emcee
import pymc

from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian_inference import data_IO

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    '''
    Generate plots for MCMC, using data written to mcmc.h5 file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param MCMCConfig config: we take an instance of MCMCConfig as an argument to keep track of config info.
    '''

    # Check if mcmc.h5 file exists
    if not os.path.exists(config.mcmc_outputfile):
        logger.info(f'MCMC output does not exist: {config.mcmc_outputfile}')
        return

    # Get results from file
    results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)

    # Plot output dir
    plot_dir = os.path.join(config.output_dir, 'plot_mcmc')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Check that results match config file
    chain = results['chain']
    n_sampling_steps, n_walkers, n_dim = chain.shape
    logger.info(f'Plotting MCMC results for chain with n_walkers={n_walkers}, n_sampling_steps={n_sampling_steps}, n_dim={n_dim}')
    logger.info(f'Chain is of size: {os.path.getsize(config.mcmc_outputfile)/(1024*1024):.1f} MB')
    assert chain.shape[0] == config.n_sampling_steps
    assert chain.shape[1] == config.n_walkers
    assert chain.shape[2] == len(config.analysis_config['parametrization'][config.parameterization]['names'])

    # MCMC plots
    _plot_acceptance_fraction(results['acceptance_fraction'], plot_dir, config)
    _plot_log_posterior(results['log_prob'], plot_dir, config)
    _plot_autocorrelation_time(results, plot_dir, config)
    _plot_posterior_pairplot(chain, plot_dir, config)

#---------------------------------------------------------------
def _plot_acceptance_fraction(acceptance_fraction, plot_dir, config):
    '''
    Plot histogram of acceptance_fraction for each walker.

    Typically we want to check that the acceptance fraction is not too low (e.g. < 0.1) 
    and that it is fairly consistent across walkers, in order to ensure walkers are not getting stuck.

    :param 1darray acceptance_fraction: fraction of steps accepted for each walker -- shape (n_walkers,)
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(config.n_walkers), acceptance_fraction, 
             marker='o', color=sns.xkcd_rgb['denim blue'])
    plt.ylim(0,1)
    plt.xlabel('Walker Index')
    plt.ylabel('Acceptance Fraction')
    outputfile = os.path.join(plot_dir, 'acceptance_fraction.pdf')
    plt.savefig(outputfile)
    plt.close()

#---------------------------------------------------------------
def _plot_log_posterior(log_posterior, plot_dir, config):
    '''
    Plot unnormalized log probability (i.e. likelihood times prior) for each walker
    as a function of step number.

    Typically we want to check that:
        - The distribution has converged to a fluctuating state, independent of step number (i.e. sufficient burn-in)
        - The walkers are not getting stuck in some part of phase space
        - There is not a multi-model posterior (which MCMC can struggle with)

    :param 1darray log_prob: log probability at each step for each walker -- shape (n_steps, n_walkers)
    '''

    n_steps = log_posterior.shape[0]
    n_walkers = log_posterior.shape[1]

    # Plot heatmap: log posterior for each walker as a function of step number
    plt.figure(figsize=(10, 6))
    sns.heatmap(log_posterior, cmap='viridis')
    plt.xlabel('Walker')
    plt.ylabel('Step Number')
    plt.title('Log Posterior (unnormalized)')
    outputfile = os.path.join(plot_dir, 'log_posterior_2D.pdf')
    plt.savefig(outputfile)
    plt.close()

    # Plot mean and stdev of each walker as a function of step number
    mean_log_posterior = log_posterior.mean(axis=1)    
    std_log_posterior = log_posterior.std(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_log_posterior, label='mean over walkers')
    plt.fill_between(range(n_steps), mean_log_posterior - std_log_posterior, 
                                     mean_log_posterior + std_log_posterior, 
                                     alpha=0.3, label='std over walkers')
    plt.xlabel('Step Number')
    plt.ylabel('Log Posterior (unnormalized)')
    plt.legend()
    outputfile = os.path.join(plot_dir, 'log_posterior_1D_steps.pdf')
    plt.savefig(outputfile)
    plt.close()

    # Plot mean and stdev of each step number as a function of walkers
    mean_log_posterior = log_posterior.mean(axis=0)    
    std_log_posterior = log_posterior.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_log_posterior, label='mean over steps')
    plt.fill_between(range(n_walkers), mean_log_posterior - std_log_posterior, 
                                       mean_log_posterior + std_log_posterior,
                                       alpha=0.3, label='std over steps')
    plt.xlabel('Walker')
    plt.ylabel('Log Posterior (unnormalized)')
    plt.legend()
    outputfile = os.path.join(plot_dir, 'log_posterior_1D_walkers.pdf')
    plt.savefig(outputfile)
    plt.close()
    
#---------------------------------------------------------------
def _plot_autocorrelation_time(results, plot_dir, config):
    '''
    Plot autocorrelation time

    The autocorrelation time is crucial because ideally we would want to draw independent 
    samples from the posterior, but in practice MCMC draws correlated samples -- thereby 
    affecting sampling uncertainty.

    For a given walker, the autocorrelation function is defined as
       C(dt) = Cov(f(t),f(t+dt)) / Var(f(t))
    where f(t) is typically either the value of a given parameter or the log posterior.

    From the autocorrelation function, the autocorrelation time is defined as the
    integral of the autocorrelation function:
       tau = 1 + 2 * sum_{dt=1}^N C(dt)
    where N is the number of steps in the chain (although often N is taken smaller).

    Typically we want to compute the autocorrelation time tau for each parameter (and/or the log posterior),
    which gives an estimate of how many independent samples are in the chain: N_independent = N_steps / tau.
    We often then want to set a thinning factor equal to or larger than the autocorrelation time.

    In order to have a reliable estimate of tau, we typically want a chain of N > 50*tau.

    For further info see: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr

    :param 1darray autocorrelation_time: estimate of autocorrelation time for each parameter -- shape (n_dim,)
    '''
    
    # For each walker, use emcee to compute the autocorrelation time for each parameter
    chain = results['chain']
    _, n_walkers, n_dim = chain.shape
    autocorrelation_time_parameters = np.zeros((n_walkers, n_dim))
    for i in range(n_walkers):
        try:
            autocorrelation_time_parameters[i] = emcee.autocorr.integrated_time(chain[:,i,:]) 
        except emcee.autocorr.AutocorrError as e:
            logger.info(f"Autocorrelation time could not be computed for walker {i}: {e}")

    # Compute the mean and stdev over all walkers
    mean_autocorrelation_time_parameters = autocorrelation_time_parameters.mean(axis=0)    
    std_autocorrelation_time_parameters = autocorrelation_time_parameters.std(axis=0)

    # Also compute the autocorrelation time for the log posterior
    log_posterior = results['log_prob']
    autocorrelation_time_posterior = np.zeros((n_walkers, 1))
    for i in range(n_walkers):
        try:
            autocorrelation_time_posterior[i] = emcee.autocorr.integrated_time(log_posterior[:,i]) 
        except emcee.autocorr.AutocorrError as e:
            logger.info(f"Autocorrelation time could not be computed for log_posterior: {e}")
    mean_autocorrelation_time_posterior = autocorrelation_time_posterior.mean(axis=0)    
    std_autocorrelation_time_posterior = autocorrelation_time_posterior.std(axis=0)

    # Concatenate the autocorrelation time for the parameters and the log posterior
    mean_autocorrelation_time = np.concatenate((mean_autocorrelation_time_parameters, 
                                                mean_autocorrelation_time_posterior))
    std_autocorrelation_time = np.concatenate((std_autocorrelation_time_parameters, 
                                               std_autocorrelation_time_posterior))

    # Bar plot
    plt.figure(figsize=(10, 6))
    parameter_names = config.analysis_config['parametrization'][config.parameterization]['names']
    labels = parameter_names + ['log_posterior']
    plt.bar(labels, mean_autocorrelation_time, yerr=std_autocorrelation_time)
    plt.ylabel('Autocorrelation time')
    plt.title('Autocorrelation time (mean,stdev over walkers)')
    outputfile = os.path.join(plot_dir, 'autocorrelation_time.pdf')
    plt.savefig(outputfile)
    plt.close()

    # Compare to autocorrelation estimate from the sampler (if present)
    if 'autocorrelation_time' in results.keys():
        autocorrelation_time = results['autocorrelation_time']
        if autocorrelation_time is None:
            logger.info('No autocorrelation time data found.')
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(parameter_names, results['autocorrelation_time'])
        plt.ylabel('Autocorrelation time')
        outputfile = os.path.join(plot_dir, 'autocorrelation_time_sampler.pdf')
        plt.savefig(outputfile)
        plt.close()

#---------------------------------------------------------------
def _plot_posterior_pairplot(chain, plot_dir, config, holdout_test = False, holdout_point = None):
    '''
    Plot posterior pairplot
    Optionally, we can also display the holdout point (if holdout_test = True)

    :param 3darray chain: positions of walkers at each step -- shape (n_steps, n_walkers, n_dim)
    :param bool holdout_test (optional): whether to display holdout point
    :param 1darray holdout_point (optional): point to display
    '''

    # Flatten chain to shape (n_steps*n_walkers, n_dim)
    samples = chain.reshape((chain.shape[0]*chain.shape[1], chain.shape[2]))

    # Construct dataframe of samples
    names = [rf'{s}' for s in config.analysis_config['parametrization'][config.parameterization]['names']]
    df = pd.DataFrame(samples, columns=names)

    # Plot posterior pairplot
    g = sns.pairplot(df, diag_kind='kde', 
                     plot_kws={'alpha':0.1, 's':1, 'color':sns.xkcd_rgb['light blue']}, 
                     diag_kws={'color':'blue', 'fill':True})
    
    # Rasterize the scatter points but not the diagonal, to keep file size small
    for i, row_axes in enumerate(g.axes):
        for j, ax in enumerate(row_axes):
            if i != j:
                for artist in ax.get_children():
                    if isinstance(artist, PathCollection):
                        artist.set_rasterized(True)

    # If holdout test, draw the holdout point
    # (and we will return whether it is contained in the credible region)
    if holdout_test:
        theta_closure = True
        for i, row_axes in enumerate(g.axes):
            for j, ax in enumerate(row_axes):
                if i == j: # Along diagonal, draw the highest posterior density interval (HPDI)
                    
                    credible_interval = pymc.stats.hpd(np.array(samples[:,i]), config.confidence)
                    ymax = ax.get_ylim()[1]
                    ax.fill_between(credible_interval, [ymax,ymax], color=sns.xkcd_rgb['almost black'], alpha=0.1)
                        
                    # Store whether truth value is contained within credible region
                    theta_truth = holdout_point[i]
                    if (theta_truth > credible_interval[1]) or (theta_truth < credible_interval[0]):
                        theta_closure = False
                        
                if i != j: # Off diagonal, draw the holdout point
                    ax.scatter(holdout_point[j], holdout_point[i], color=sns.xkcd_rgb['almost black'])

    plt.savefig(f'{plot_dir}/posterior_pairplot.pdf')
    plt.close('all')

    if holdout_test:
        return theta_closure