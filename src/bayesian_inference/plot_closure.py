#! /usr/bin/env python
'''
Module to plot closure test results

The basic idea of the closure tests is: 
    - For each validation point, compute “true qhat” and get “pseudodata” of 
        observables (taking experimental uncertainties from actual data). 
    - Using originally trained emulator, run the MCMC and compute qhat posterior — then compare to “true qhat”.

authors: J.Mulligan, R.Ehlers
'''

import logging
import os
from functools import partial

import numpy as np
import scipy

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian_inference import data_IO
from bayesian_inference import plot_qhat
from bayesian_inference import mcmc

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    '''
    Generate closure tests plots, using data written to mcmc.h5 file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param MCMCConfig config: we take an instance of MCMCConfig as an argument to keep track of config info.
    '''

    # For each closure point, plot qhat posterior and compare to true qhat
    # We will make one plot for fixed E, and one plot for fixed T
    n_design_points = config.analysis_config['validation_indices'][1] - config.analysis_config['validation_indices'][0]
    cred_level = 0.9
    E = 100
    T = 0.3
    n_x = 50    # number of T or E points to plot
    n_theta_samples = 200

    # Construct a dict of info to save in order to make summary plots.
    # This bookkeeping can probably be cleaned up.
    closure_summary = {}
    closure_summary[f'E{E}'] = {}
    closure_summary[f'E{E}']['qhat_closure_array'] = np.zeros((n_design_points, n_x))
    closure_summary[f'E{E}']['qhat_mean'] = np.zeros((n_design_points, n_x))
    closure_summary[f'T{T}'] = {}
    closure_summary[f'T{T}']['qhat_closure_array'] = np.zeros((n_design_points, n_x))
    closure_summary[f'T{T}']['qhat_mean'] = np.zeros((n_design_points, n_x))

    parameter_names = [rf'{s}' for s in config.analysis_config['parametrization'][config.parameterization]['names']]
    for parameter in parameter_names:
        closure_summary[parameter] = {}
        closure_summary[parameter]['theta_truth'] = np.zeros((n_design_points))
        closure_summary[parameter]['theta_closure_array'] = np.zeros((n_design_points))
        closure_summary[parameter]['qhat_mean'] = np.zeros((n_design_points))

    # Loop over all closure points
    for design_point_index in range(n_design_points):

        # Check if mcmc.h5 file exists
        result_dir = os.path.join(config.output_dir, f'closure/results/{design_point_index}')
        mcmc_outputfile = os.path.join(result_dir, 'mcmc.h5')
        if not os.path.exists(mcmc_outputfile):
            logger.info(f'MCMC output does not exist: {mcmc_outputfile}')
            return

        # Get results from file
        results = data_IO.read_dict_from_h5(result_dir, 'mcmc.h5', verbose=True)

        # Get posterior samples
        n_walkers, n_steps, n_params = results['chain'].shape
        posterior = results['chain'].reshape((n_walkers*n_steps, n_params))

        # Get target design point, so we can compute "true" qhat
        target_design_point = results['design_point'].reshape((1, n_params))

        # Plot qhat vs. T,E and return boolean array of whether target qhat is within credible interval
        # Then save relevant info to make summary plots over all closure points
        qhat_plot_dir = os.path.join(config.output_dir, f'closure/results/{design_point_index}')
        qhat_closure_dict = plot_qhat.plot_qhat(posterior, qhat_plot_dir, config, E=E, cred_level=cred_level, 
                                                n_samples=1000, n_x=n_x, target_design_point=target_design_point)
        closure_summary[f'E{E}']['qhat_closure_array'][design_point_index] = qhat_closure_dict['qhat_closure_array']
        closure_summary[f'E{E}']['qhat_mean'][design_point_index] = qhat_closure_dict['qhat_mean']
        closure_summary[f'E{E}']['x_array'] = qhat_closure_dict['x_array']
        closure_summary[f'E{E}']['cred_level'] = qhat_closure_dict['cred_level']

        qhat_closure_dict = plot_qhat.plot_qhat(posterior, qhat_plot_dir, config, T=T, cred_level=cred_level, 
                                                n_samples=1000, n_x=n_x, target_design_point=target_design_point)
        closure_summary[f'T{T}']['qhat_closure_array'][design_point_index] = qhat_closure_dict['qhat_closure_array']
        closure_summary[f'T{T}']['qhat_mean'][design_point_index] = qhat_closure_dict['qhat_mean']
        closure_summary[f'T{T}']['x_array'] = qhat_closure_dict['x_array']
        closure_summary[f'T{T}']['cred_level'] = qhat_closure_dict['cred_level']

        # Compute the credible interval for the design parameter, and check whether target is within it
        for i,parameter in enumerate(parameter_names):
            results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
            chain = results['chain']
            posterior = chain.reshape((chain.shape[0]*chain.shape[1], chain.shape[2]))
            idx = np.random.choice(posterior.shape[0], size=n_theta_samples, replace=False)
            posterior_samples = posterior[idx,:]
            credible_interval = mcmc.credible_interval(posterior_samples[:,i], confidence=cred_level)
            theta_truth = target_design_point[0][i]
            closure_summary[parameter]['theta_truth'][design_point_index] = theta_truth
            closure_summary[parameter]['theta_closure_array'][design_point_index] = (theta_truth > credible_interval[0]) and (theta_truth < credible_interval[1])
            closure_summary[parameter]['qhat_mean'][design_point_index] = np.mean(plot_qhat.qhat(target_design_point, config, T=T, E=E))

    # Create summary plots over all closure points
    plot_dir = os.path.join(config.output_dir, 'closure/summary_plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot as a function of E,T
    for key,qhat_closure_dict in closure_summary.items():
        if key == f'E{E}' or key == f'T{T}':
            _plot_closure_summary_qhat(key, qhat_closure_dict, plot_dir)

    # Plot as a function of design parameters
    for i,parameter in enumerate(parameter_names):
        _plot_closure_summary_theta(closure_summary[parameter], parameter, i, cred_level, E, T, config, plot_dir)

#---------------------------------------------------------------
def _plot_closure_summary_qhat(key, qhat_closure_dict, plot_dir):
    '''
    Plot summary of closure tests as a function of E or T.

    We will construct a 2D histogram of <qhat> vs. T or E, with z-axis the fraction of closure tests that passed.
    This allows us to see both the aggregate closure statistics, as well as to look differentially
    in qhat to see if our closure is successful across the space.
    
    :param str key: E{E} or T{T}, specifying which variable is fixed (plot as a function of the other one)
    :param dict qhat_closure_dict: dict containing qhat closure results
    '''
    if 'E' in key:
        E = float(key[1:])
        xlabel = 'T (GeV)'
        ylabel = rf'$\left< \hat{{q}}/T^3 \right>_{{E={E}\;\rm{{GeV}}}}$'
    elif 'T' in key:
        T = float(key[1:])
        xlabel = 'E (GeV)'
        ylabel = rf'$\left< \hat{{q}}/T^3 \right>_{{T={T}\;\rm{{GeV}}}}$'

    # Get relevant info from qhat_closure_dict
    #   - qhat_closure_array: boolean array of whether target qhat is contained in credible region -- shape (n_design_points, n_x)
    #   - qhat_mean: mean qhat for each design point -- shape (n_design_points, n_x)
    #   - x_array: array of T or E values -- shape (n_x,)
    qhat_closure_array = qhat_closure_dict['qhat_closure_array']
    qhat_mean = qhat_closure_dict['qhat_mean']
    x_array = qhat_closure_dict['x_array']
    cred_level = qhat_closure_dict['cred_level']

    # We want to histogram the fraction of successes (z) as a function of E/T,<qhat> (x,y)
    # Here we want x,y,z with equal sizes, so we flatten y,z and repeat x_array
    x = np.tile(x_array, qhat_mean.shape[0])
    y = qhat_mean.flatten()
    z = qhat_closure_array.flatten()

    xbins = np.linspace(x_array[0], x_array[-1], num=8)

    _plot_closure_2D_histogram(x, y, z, xbins, cred_level, xlabel, ylabel, key, plot_dir)

#---------------------------------------------------------------
def _plot_closure_summary_theta(parameter_closure_dict, parameter, i, cred_level, E, T, config, plot_dir):
    '''
    Plot summary of closure tests as a function of design parameter theta[i].

    We will construct a 2D histogram of <qhat> vs. theta[i], with z-axis the fraction of closure tests that passed.
    This allows us to see both the aggregate closure statistics, as well as to look differentially
    in qhat to see if our closure is successful across the space.
    
    :param dict parameter_closure_dict: dict containing theta closure results
    :param str parameter: design parameter label
    '''

    # To do this, we need to compute the credible intervals for each parameter from the samples,
    # as in the mcmc pairplot

    # Get relevant info from parameter_closure_dict
    #   - parameter_closure_dict: boolean array of whether target qhat is contained in credible region -- shape (n_design_points,)
    #   - qhat_mean: mean qhat for each design point -- shape (n_design_points,)
    #   - x_array: array of parameter values -- shape (n_design_points,)
    theta_closure_array = parameter_closure_dict['theta_closure_array']
    qhat_mean = parameter_closure_dict['qhat_mean']
    x_array = parameter_closure_dict['theta_truth']

    # We want to histogram the fraction of successes (z) as a function of theta[i],<qhat> (x,y)
    # Here we want x,y,z with equal sizes, so we flatten y,z and repeat x_array
    x = x_array
    y = qhat_mean
    z = theta_closure_array

    parameter_min = config.analysis_config['parametrization'][config.parameterization]['min'][i]
    parameter_max = config.analysis_config['parametrization'][config.parameterization]['max'][i]
    xbins = np.linspace(parameter_min, parameter_max, num=8)

    xlabel = parameter
    ylabel = rf'$\left< \hat{{q}}/T^3 \right>_{{E={E},T={T}\;\rm{{GeV}}}}$'
    _plot_closure_2D_histogram(x, y, z, xbins, cred_level, xlabel, ylabel, f'theta{i}', plot_dir)

#---------------------------------------------------------------
def _plot_closure_2D_histogram(x, y, z, xbins, cred_level, xlabel, ylabel, suffix, plot_dir):
    '''
    Construct a 2D histogram of <qhat> vs. X, with z-axis the fraction of closure tests that passed.
    Here, X can be anything -- we use E/T or theta[i].

    :param 1darray x: array of x values
    :param 1darray y: array of y values
    :param 1darray z: array of z values
    :param 1darray xbins: array of x-axis bins (we will combine multiple x points (e.g. E,T values) per bin for visualization)
    :param float cred_level: credible level
    '''

    # Define y-axis bins
    qhat_bins =  np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12])
    qhat_bins_center = (qhat_bins[:-1] + qhat_bins[1:]) / 2.0
    
    # Generate and plot histogram
    H, xedges, yedges, _ = scipy.stats.binned_statistic_2d(x, y, z, statistic=np.mean,
                                                           bins=[xbins, qhat_bins])
    H = np.ma.masked_invalid(H) # mask where there is no data
    XX, YY = np.meshgrid(xedges, yedges)
    fig = plt.figure(figsize = (11,9))
    ax1=plt.subplot(111)
    plot1 = ax1.pcolormesh(XX, YY, H.T)
    fig.colorbar(plot1, ax=ax1)
    
    # Generate histogram of binomial uncertainty, and print success rate in each bin
    statistic = partial(efficiency_uncertainty, nbins=xbins.shape[0])
    Herr, xedges, yedges, _ = scipy.stats.binned_statistic_2d(x, y, z,
                                                              statistic=statistic,
                                                              bins=[xbins, qhat_bins])
    xbins_center = (xbins[:-1] + xbins[1:]) / 2.0
    for i in range(len(xbins)-1):
        for j in range(len(qhat_bins)-1):
            zval = H[i][j]
            zerr = Herr[i][j]
            if np.isnan(zval) or np.isnan(zerr):
                continue
            ax1.text(xbins_center[i], qhat_bins_center[j], rf'{zval:0.2f}$\pm${zerr:0.2f}',
                     size=8, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    
    # Print aggregated closure success rate
    mean = np.mean(z)
    unc = efficiency_uncertainty(z, 1) # Here, we take just one point per curve
    plt.gca().text(0.95, 0.95, rf'mean: {mean:0.2f}$\pm${unc:0.2f}', ha='right', va='top', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(facecolor='white', alpha=1.0, boxstyle="round,pad=0.3"))

    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(f'Fraction of closure tests contained in {100*cred_level}% CR', size=14)
    plt.savefig(f'{plot_dir}/Closure_Summary2D_{suffix}.pdf')
    plt.close('all')
    
#---------------------------------------------------------------
def efficiency_uncertainty(success_array, nbins=0, type='bayesian'):
    '''
    Compute bayesian uncertainty on efficiency from an array of True/False values

    :param 1darray success_array: array of True/False values
    :param 1darray bins: bins that we will use to bin success_array for plotting
    '''
    length = success_array.shape[0]
    sum = np.sum(success_array)
    mean = 1.*sum/length
    
    # We have multiple E,T points per bin, which would underestimate the uncertainty
    # since neighboring points are highly correlated -- so we average all points in a bin
    real_length = length / nbins
    
    # Bayesian uncertainty: http://phys.kent.edu/~smargeti/STAR/D0/Ullrich-Errors.pdf
    if type == 'bayesian':
        k = mean*real_length
        n = real_length
        variance = (k+1)*(k+2)/((n+2)*(n+3)) - (k+1)*(k+1)/((n+2)*(n+2))
        uncertainty = np.sqrt(variance)
    # Binomial uncertainty
    elif type == 'binomial':
        variance = real_length*mean*(1-mean)
        sigma = np.sqrt(variance)
        uncertainty = sigma/real_length

    return uncertainty