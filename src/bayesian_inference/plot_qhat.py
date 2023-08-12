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
from bayesian_inference import emulation
from bayesian_inference import mcmc
from bayesian_inference import plot_utils

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

    # Observable sensitivity plots
    _plot_observable_sensitivity(posterior, plot_dir, config, delta=0.1, n_samples=1000)

#---------------------------------------------------------------[]
def plot_qhat(posterior, plot_dir, config, E=0, T=0, cred_level=0., n_samples=5000, n_x=50, 
              plot_prior=True, plot_mean=True, plot_map=False, target_design_point=np.array([])):
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

    # Compute qhat for each sample (as well as MAP value), as a function of T or E
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

    # Plot mean qhat values for each T or E
    if plot_mean:
        qhat_mean = np.mean(qhat_posteriors, axis=1)
        plt.plot(x_array, qhat_mean, sns.xkcd_rgb['denim blue'],
                linewidth=2., linestyle='--', label='Mean')

    # Plot the MAP value as well for each T or E
    if plot_map:
        if E:
            qhat_map = np.array([qhat(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for T in x_array])
        elif T:
            qhat_map = np.array([qhat(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for E in x_array])
        plt.plot(x_array, qhat_map, sns.xkcd_rgb['medium green'],
                linewidth=2., linestyle='--', label='MAP')

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

    # Plot formatting
    plt.xlabel(xlabel)
    plt.ylabel(r'$\hat{q}/T^3$')
    ymin = 0
    if plot_mean:
        ymax = 2*max(qhat_mean)
    elif plot_map: 
        ymax = 2*max(qhat_map)
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])
    plt.legend(title=f'{label}, {config.parameterization}', title_fontsize=12,
               loc='upper right', fontsize=12)

    plt.savefig(f'{plot_dir}/qhat_{suffix}.pdf')
    plt.close('all')

    if target_design_point.any():
        return qhat_closure

#---------------------------------------------------------------
def _plot_observable_sensitivity(posterior, plot_dir, config, delta=0.1, n_samples=1000):
    '''
    Plot local sensitivity index (at the MAP value) for each parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    Note: this is just a normalized partial derivative, dO_j/dx_i * (x_i/O_j)

    Based on: 
      - https://arxiv.org/abs/2011.01430
      - https://link.springer.com/article/10.1007/BF00547132
    '''

    # Find the MAP value
    map_parameters = mcmc.map_parameters(posterior)

    # Plot sensitivity index for each parameter
    for i_parameter in range(posterior.shape[1]):
        _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter, 
                                                      plot_dir, config, delta=delta)
    
    # TODO: Plot sensitivity for qhat:
    #   S(qhat, O_j, delta) = 1/delta * [O_j(qhat_map') - O_j(qhat_map)] / O_j(qhat)
    # In the current qhat formulation, qhat = qhat(x_0=alpha_s_fix) only depends on x_0=alpha_s_fix.
    # So this information is already captured in the x_0 sensitivity plot above.
    # If we want to explicitly compute S(qhat), we need to evaluate the emulator at qhat_map'=(1+delta)*qhat_map.
    # In principle one should find the x_0 corresponding to (1+delta)*qhat_map.
    # For simplicity we can just evaluate x_0'=x_0(1+delta) and then redefine delta=qhat(x_0')-qhat(x_0) -- but
    #   this is excatly the same as the S(x_0) plot above, up the redefinition of delta.
    # It may nevertheless be nice to add since a plot of S(qhat) will likely be more salient to viewers.

#---------------------------------------------------------------
def _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter, plot_dir, config, delta=0.1):
    '''
    Plot local sensitivity index (at the MAP value) for a single parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    TODO: We probably want to add higher level summary plot, e.g. take highest 5 observables for each parameter,
            or average bins over a given observable.
          Could also construct a 2D plot where color shows the sensitivity
    '''

    # Define the two parameter points we would like to evaluate
    x = map_parameters.copy()
    x_prime = map_parameters.copy()
    x_prime[i_parameter] = (1+delta)*x_prime[i_parameter]
    x = np.expand_dims(x, axis=0)
    x_prime = np.expand_dims(x_prime, axis=0)
    
    # Get emulator predictions at the two points
    emulation_config = emulation.EmulationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    emulation_results = emulation_config.read_all_emulator_groups()
    emulator_predictions_x = emulation.predict(x, emulation_config, emulation_group_results=emulation_results)
    emulator_predictions_x_prime = emulation.predict(x_prime, emulation_config, emulation_group_results=emulation_results)

    # Convert to dict: emulator_predictions[observable_label]
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
     # FIXME: The observable list doesn't match up in order with the emulator results
    emulator_predictions_x_dict = data_IO.observable_dict_from_matrix(emulator_predictions_x['central_value'], observables)
    emulator_predictions_x_prime_dict = data_IO.observable_dict_from_matrix(emulator_predictions_x_prime['central_value'], observables)

    # Construct dict of sensitivity index, in same format as emulator_predictions['central_value']
    sensitivity_index_dict = emulator_predictions_x_prime_dict['central_value'].copy()
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)
    for observable_label in sorted_observable_list:
        x = emulator_predictions_x_dict['central_value'][observable_label]
        x_prime = emulator_predictions_x_prime_dict['central_value'][observable_label]
        sensitivity_index_dict[observable_label] = 1/delta * (x_prime - x) / x
    
    # Plot
    plot_list = [sensitivity_index_dict]
    columns = [0]
    labels = [rf'Sensitivity index at MAP, $\delta={delta}$']
    colors = [sns.xkcd_rgb['dark sky blue']]
    param = config.analysis_config['parameterization'][config.parameterization]['names'][i_parameter][1:-1].replace('{', '{{').replace('}', '}}')
    ylabel = rf'$S({param}, \mathcal{{O}}, \delta)$'
    #ylabel = rf'$S({param}, \mathcal{{O}}, \delta) = \frac{{1}}{{\delta}} \frac{{\mathcal{{O}}([1+\delta] {param})-\mathcal{{O}}({param})}}{{\mathcal{{O}}({param})}}$'
    filename = f'sensitivity_index_{i_parameter}.pdf'
    plot_utils.plot_observable_panels(plot_list, labels, colors, columns, config, plot_dir, filename, 
                                      linewidth=1, ymin=-5, ymax=5, ylabel=ylabel, plot_exp_data=False, bar_plot=True)

#---------------------------------------------------------------
def qhat(posterior_samples, config, T=0, E=0) -> float:
    '''
    Evaluate qhat/T^3 from posterior samples of parameters,
    for fixed E and T

    See: https://github.com/raymondEhlers/STAT/blob/1b0df83a9fd479f8110fd326ae26c0ce002a1109/run_analysis_base.py

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 1darray: qhat/T^3 -- shape (n_samples,)
    '''

    if posterior_samples.ndim == 1:
        posterior_samples = np.expand_dims(posterior_samples, axis=0)

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