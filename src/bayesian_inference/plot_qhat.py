#! /usr/bin/env python
'''
Module related to generate qhat plots

authors: J.Mulligan, R.Ehlers
'''

import logging
import os

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian_inference import data_IO
from bayesian_inference import mcmc

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    '''
    Generate qhat plots, using data written to mcmc.h5 file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param MCMCConfig config: we take an instance of MCMCConfig as an argument to keep track of config info.
    '''

    # Check if mcmc.h5 file exists
    if not os.path.exists(config.mcmc_outputfile):
        logger.info(f'MCMC output does not exist: {config.mcmc_outputfile}')
        return

    # Get results from file
    results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
    n_walkers, n_steps, n_params = results['chain'].shape
    posterior = results['chain'].reshape((n_walkers*n_steps, n_params))

    # Plot output dir
    plot_dir = os.path.join(config.output_dir, 'plot_qhat')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # qhat plots
    plot_qhat(posterior, plot_dir, config, E=100, cred_level=0.9, n_samples=1000)
    plot_qhat(posterior, plot_dir, config, T=0.3, cred_level=0.9, n_samples=1000)

#---------------------------------------------------------------[]
def plot_qhat(posterior, plot_dir, config, E=0, T=0, cred_level=0., n_samples=5000, n_x=50, 
              plot_prior=True, target_design_point=np.array([])):
    '''
    Plot qhat credible interval from posterior samples,
    as a function of either E or T (with the other held fixed)

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    :param int n_x: number of T or E points to plot
    :param 1darray target_design_point: if closure test, design point corresponding to "truth" qhat value
    '''

    # Sample posterior parameters without replacement
    if posterior.shape[0] < n_samples:
        n_samples = posterior.shape[0]
        logger.warning(f'Not enough posterior samples to plot {n_samples} samples, using {n_samples} instead')
    idx = np.random.choice(posterior.shape[0], size=n_samples, replace=False)
    posterior_samples = posterior[idx,:]

    # Compute qhat for each sample, as a function of T or E
    #   qhat_posteriors will be a 2d array of shape (x_array.size, n_samples)
    if E:
        xlabel = 'T (GeV)'
        suffix = f'E{E}'
        label = f'E = {E} GeV'
        x_array = np.linspace(0.16, 0.5, n_x)
        qhat_posteriors = np.array([qhat(posterior_samples, config, T=T, E=E) for T in x_array])
    elif T:
        xlabel = 'E (GeV)'
        suffix = f'T{T}'
        label = f'T = {T} GeV'
        x_array = np.linspace(5, 200, n_x)
        qhat_posteriors = np.array([qhat(posterior_samples, config, T=T, E=E) for E in x_array])

    # Get mean qhat values for each T or E
    qhat_mean = np.mean(qhat_posteriors, axis=1)
    plt.plot(x_array, qhat_mean, sns.xkcd_rgb['denim blue'],
             linewidth=2., linestyle='--', label='Mean')
    plt.xlabel(xlabel)
    plt.ylabel(r'$\hat{q}/T^3$')
    ymin = 0
    ymax = 2*max(qhat_mean)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])

    # Get credible interval for each T or E
    h = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up =  [i[1] for i in h]
    plt.fill_between(x_array, credible_low, credible_up, color=sns.xkcd_rgb['light blue'],
                     label=f'Posterior {int(cred_level*100)}% Credible Interval')

    # Plot prior as well, for comparison
    # TODO: one could also plot some type of "information gain" metric, e.g. KL divergence
    if plot_prior:

        # Generate samples
        prior_samples = _generate_prior_samples(config, n_samples=n_samples)

        # Compute qhat for each sample, as a function of T or E
        if E:
            qhat_priors = np.array([qhat(prior_samples, config, T=T, E=E) for T in x_array])
        elif T:
            qhat_priors = np.array([qhat(prior_samples, config, T=T, E=E) for E in x_array])

        # Get credible interval for each T or E
        h_prior = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_priors]
        credible_low_prior = [i[0] for i in h_prior]
        credible_up_prior =  [i[1] for i in h_prior]
        plt.fill_between(x_array, credible_low_prior, credible_up_prior, color=sns.xkcd_rgb['light blue'],
                         alpha=0.3, label=f'Prior {int(cred_level*100)}% Credible Interval')

    # If closure test: Plot truth qhat value
    # We will return a dict of info needed for plotting closure plots, including a
    #   boolean array (as a fcn of T or E) of whether the truth value is contained within credible region
    if target_design_point.any():
        if E:
            qhat_truth = [qhat(target_design_point, config, T=T, E=E) for T in x_array]
        elif T:
            qhat_truth = [qhat(target_design_point, config, T=T, E=E) for E in x_array]
        plt.plot(x_array, qhat_truth, sns.xkcd_rgb['pale red'],
                linewidth=2., label='Target')

        qhat_closure = {}
        qhat_closure['qhat_closure_array'] = np.array([((qhat_truth[i] < credible_up[i]) and (qhat_truth[i] > credible_low[i])) for i,_ in enumerate(x_array)]).squeeze()
        qhat_closure['qhat_mean'] = qhat_mean
        qhat_closure['x_array'] = x_array
        qhat_closure['cred_level'] = cred_level

    plt.legend(title=f'{label}, {config.parameterization}', title_fontsize=12,
               loc='upper right', fontsize=12)

    plt.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close('all')

    if target_design_point.any():
        return qhat_closure

#---------------------------------------------------------------
def qhat(posterior_samples, config, T=0, E=0) -> float:
    '''
    Evaluate qhat/T^3 from posterior samples of parameters,
    for fixed E and T

    See: https://github.com/raymondEhlers/STAT/blob/1b0df83a9fd479f8110fd326ae26c0ce002a1109/run_analysis_base.py

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 1darray: qhat/T^3 -- shape (n_samples,)
    '''

    if config.parameterization == "exponential":

        alpha_s_fix = posterior_samples[:,0]
        active_flavor = 3
        C_a = 3.0  # Extracted from JetScapeConstants

        # From GeneralQhatFunction
        debye_mass_square = alpha_s_fix * 4 * np.pi * np.power(T, 2.0) * (6.0 + active_flavor) / 6.0
        scale_net = 2 * E * T
        if scale_net < 1.0:
            scale_net = 1.0

        # alpha_s should be taken as 2*E*T, per Abhijit
        # See: https://jetscapeworkspace.slack.com/archives/C025X5NE9SN/p1648404101376299
        square_lambda_QCD_HTL = np.exp( -12.0 * np.pi/( (33 - 2 * active_flavor) * scale_net) )
        running_alpha_s = 12.0 * np.pi/( (33.0 - 2.0 * active_flavor) * np.log(scale_net/square_lambda_QCD_HTL) )
        if scale_net < 1.0:
            running_alpha_s = scale_net
        answer = (C_a * 50.4864 / np.pi) * running_alpha_s * alpha_s_fix * np.abs(np.log(scale_net / debye_mass_square))

        return answer * 0.19732698   # 1/GeV to fm

#---------------------------------------------------------------
def _generate_prior_samples(config, n_samples=100):
    '''
    Generate samples of prior parameters

    The prior is uniform in the parameter space -- except for c1,c2,c3 it is the log that is uniform.

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 2darray: samples -- shape (n_samples,n_params)
    '''
    names = config.analysis_config['parameterization'][config.parameterization]['names']
    parameter_min = config.analysis_config['parameterization'][config.parameterization]['min'].copy()
    parameter_max = config.analysis_config['parameterization'][config.parameterization]['max'].copy()

    # Transform c1,c2,c3 to log
    n_params = len(names)
    for i,name in enumerate(names):
        if 'c_' in name:
            parameter_min[i] = np.log(parameter_min[i])
            parameter_max[i] = np.log(parameter_max[i])

    # Generate uniform samples
    samples = np.random.uniform(parameter_min, parameter_max, (n_samples, n_params))

    # Transform log(c1,c2,c3) back to c1,c2,c3
    for i,name in enumerate(names):
        if 'c_' in name:
            samples[:,i] = np.exp(samples[:,i])

    return samples