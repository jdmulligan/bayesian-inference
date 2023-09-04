"""Plot across analyses

authors: J. Mulligan, R. Ehlers
"""

from __future__ import annotations
from typing import Any

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bayesian_inference import data_IO, mcmc
from bayesian_inference import plot_qhat

logger = logging.getLogger(__name__)


def plot(analyses: dict[str, Any], config_file: str, output_dir: str) -> None:
    """Plot across selected analyses

    :param dict[str, MCMCConfig] configs: dictionary of MCMCConfig objects, with keys corresponding to analysis names

    :return None: we save plots to disk
    """
    # Setup
    configs = {}
    for analysis_name, analysis_config in analyses.items():
        for parameterization in analysis_config['parameterizations']:
            configs[f"{analysis_name}_{parameterization}"] = mcmc.MCMCConfig(
                analysis_name=analysis_name,
                parameterization=parameterization,
                analysis_config=analysis_config,
                config_file=config_file
            )

    # Validation and setup
    results = {}
    posteriors = {}
    for analysis_name, config in configs.items():
        # Check if mcmc.h5 file exists
        if not os.path.exists(config.mcmc_outputfile):
            logger.info(f'MCMC output does not exist: {config.mcmc_outputfile}')
            return

        # Get results from file
        results[analysis_name] = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
        n_walkers, n_steps, n_params = results[analysis_name]['chain'].shape
        posteriors[analysis_name] = results[analysis_name]['chain'].reshape((n_walkers*n_steps, n_params))

    # Plot output dir
    plot_dir = os.path.join(output_dir, 'plot_analyses')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_qhat_across_analyses(
        results=results,
        posteriors=posteriors,
        configs=configs,
        plot_dir=plot_dir,
        E=100,
        cred_level=0.9, n_samples=5000,
    )


#---------------------------------------------------------------[]
def plot_qhat_across_analyses(
    results,
    posteriors,
    plot_dir,
    configs, E=0, T=0, cred_level=0., n_samples=5000, n_x=50,
    plot_prior=True, plot_mean=True, plot_map=False, target_design_point=np.array([])):
    '''
    Plot qhat credible interval from posterior samples,
    as a function of either E or T (with the other held fixed)

    Pretty much copied from plot_qhat, hastily modified due to QM.

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    :param int n_x: number of T or E points to plot
    :param 1darray target_design_point: if closure test, design point corresponding to "truth" qhat value
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    for (analysis_name, result), posterior, config in zip(results.items(), posteriors.values(), configs.values()):
        # Sample posterior parameters without replacement
        if posterior.shape[0] < n_samples:
            n_samples = posterior.shape[0]
            logger.warning(f'Not enough posterior samples to plot {n_samples} samples, using {n_samples} instead')
        idx = np.random.choice(posterior.shape[0], size=n_samples, replace=False)
        posterior_samples = posterior[idx,:]

        # Compute qhat for each sample (as well as MAP value), as a function of T or E
        #   qhat_posteriors will be a 2d array of shape (x_array.size, n_samples)
        if E:
            xlabel = 'T (GeV)'
            suffix = f'E{E}'
            label = f'E = {E} GeV'
            x_array = np.linspace(0.16, 0.5, n_x)
            qhat_posteriors = np.array([plot_qhat.qhat(posterior_samples, config, T=T, E=E) for T in x_array])
        elif T:
            xlabel = 'E (GeV)'
            suffix = f'T{T}'
            label = f'T = {T} GeV'
            x_array = np.linspace(5, 200, n_x)
            qhat_posteriors = np.array([plot_qhat.qhat(posterior_samples, config, T=T, E=E) for E in x_array])

        # Plot mean qhat values for each T or E
        if plot_mean:
            qhat_mean = np.mean(qhat_posteriors, axis=1)
            ax.plot(x_array, qhat_mean, #sns.xkcd_rgb['denim blue'],
                    linewidth=2., linestyle='--', label='Mean')

        # Plot the MAP value as well for each T or E
        if plot_map:
            if E:
                qhat_map = np.array([plot_qhat.qhat(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for T in x_array])
            elif T:
                qhat_map = np.array([plot_qhat.qhat(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for E in x_array])
            ax.plot(x_array, qhat_map, #sns.xkcd_rgb['medium green'],
                    linewidth=2., linestyle='--', label='MAP')

        # Get credible interval for each T or E
        h = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_posteriors]
        credible_low = [i[0] for i in h]
        credible_up =  [i[1] for i in h]
        ax.fill_between(x_array, credible_low, credible_up, #color=sns.xkcd_rgb['light blue'],
                        label=f'Posterior {int(cred_level*100)}% Credible Interval')

        # Plot prior as well, for comparison
        # TODO: one could also plot some type of "information gain" metric, e.g. KL divergence
        if plot_prior:

            # Generate samples
            prior_samples = plot_qhat._generate_prior_samples(config, n_samples=n_samples)

            # Compute qhat for each sample, as a function of T or E
            if E:
                qhat_priors = np.array([plot_qhat.qhat(prior_samples, config, T=T, E=E) for T in x_array])
            elif T:
                qhat_priors = np.array([plot_qhat.qhat(prior_samples, config, T=T, E=E) for E in x_array])

            # Get credible interval for each T or E
            h_prior = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_priors]
            credible_low_prior = [i[0] for i in h_prior]
            credible_up_prior =  [i[1] for i in h_prior]
            ax.fill_between(x_array, credible_low_prior, credible_up_prior, color=sns.xkcd_rgb['light blue'],
                            alpha=0.3, label=f'Prior {int(cred_level*100)}% Credible Interval')

        # If closure test: Plot truth qhat value
        # We will return a dict of info needed for plotting closure plots, including a
        #   boolean array (as a fcn of T or E) of whether the truth value is contained within credible region
        if target_design_point.any():
            if E:
                qhat_truth = [plot_qhat.qhat(target_design_point, config, T=T, E=E) for T in x_array]
            elif T:
                qhat_truth = [plot_qhat.qhat(target_design_point, config, T=T, E=E) for E in x_array]
            ax.plot(x_array, qhat_truth, sns.xkcd_rgb['pale red'],
                    linewidth=2., label='Target')

            qhat_closure = {}
            qhat_closure['qhat_closure_array'] = np.array([((qhat_truth[i] < credible_up[i]) and (qhat_truth[i] > credible_low[i])) for i,_ in enumerate(x_array)]).squeeze()
            qhat_closure['qhat_mean'] = qhat_mean
            qhat_closure['x_array'] = x_array
            qhat_closure['cred_level'] = cred_level

    # Plot formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\hat{q}/T^3$')
    ymin = 0
    if plot_mean:
        ymax = 2*max(qhat_mean)
    elif plot_map:
        ymax = 2*max(qhat_map)
    ax.set_ylim([ymin, ymax])
    ax.legend(title=f'{label}, {config.parameterization}', title_fontsize=12,
            loc='upper right', fontsize=12)

    fig.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close('all')

    if target_design_point.any():
        return qhat_closure
