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

logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config):
    '''
    Generate qhat plots and closure tests, using data written to mcmc.h5 file in analysis step.
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
    _plot_qhat(posterior, plot_dir, config, E=100, cred_level=0.9, n_samples=1000)
    _plot_qhat(posterior, plot_dir, config, T=0.3, cred_level=0.9, n_samples=1000)

#---------------------------------------------------------------
def _plot_qhat(posterior, plot_dir, config, E=0, T=0, cred_level=0., n_samples=5000):
    '''
    Plot qhat credible interval from posterior samples, 
    as a function of either E or T (with the other held fixed)

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    '''

    # Sample posterior parameters without replacement
    idx = np.random.choice(posterior.shape[0], size=n_samples, replace=False)
    posterior_samples = posterior[idx,:]

    # Compute qhat for each sample, as a function of T or E
    #   qhat_posteriors will be a 2d array of shape (x_array.size, n_samples)
    if E:
        xlabel = 'T (GeV)'
        suffix = f'E{E}'
        label = f'E = {E} GeV'
        x_array = np.linspace(0.16, 0.5)
        qhat_posteriors = np.array([_qhat(posterior_samples, config, T=T, E=E) for T in x_array])           
    elif T:
        xlabel = 'E (GeV)'
        suffix = f'T{T}'
        label = f'T = {T} GeV'
        x_array = np.linspace(5, 200)
        qhat_posteriors = np.array([_qhat(posterior_samples, config, T=T, E=E) for E in x_array])
    
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
    # TODO: For simplicity I use quantiles, although it may be slightly better to use 
    #       the highest posterior density (HPD) interval, e.g. in pymc3 or arviz
    cred_range = [(1-cred_level)/2, 1-(1-cred_level)/2]
    h = [np.quantile(qhat_values, cred_range) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up =  [i[1] for i in h]
    plt.fill_between(x_array, credible_low, credible_up, color=sns.xkcd_rgb['light blue'],
                     label=f'{int(cred_level*100)}% Credible Interval')

    plt.legend(title=f'{label}, {config.parameterization}', title_fontsize=12,
               loc='upper right', fontsize=12)

    plt.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close('all')

#---------------------------------------------------------------
def _qhat(posterior_samples, config, T=0, E=0) -> float:
    '''
    Evaluate qhat/T^3 from posterior samples of parameters,
    as a function of either E or T (with the other held fixed)

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