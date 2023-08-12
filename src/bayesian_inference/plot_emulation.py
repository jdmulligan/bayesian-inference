#! /usr/bin/env python
'''
Module related to generate plots for PCA/emulators

authors: J.Mulligan, R.Ehlers
'''

import os
import itertools
import logging
import pickle
import yaml

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian_inference import data_IO
from bayesian_inference import emulation
from bayesian_inference import plot_utils

logger = logging.getLogger(__name__)


####################################################################################################################
def plot(config):
    '''
    Generate plots for PCA/emulators, using data written to file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    '''
    emulation_results = {}
    for emulation_group_name, emulation_group_config in config.emulation_groups_config.items():
        # Check if emulator already exists
        if not os.path.exists(emulation_group_config.emulation_outputfile):
            logger.info(f'Emulator output does not exist: {emulation_group_config.emulation_outputfile}')
            continue
        emulation_results[emulation_group_name] = emulation.read_emulators(emulation_group_config)

        # Plot output dir
        plot_dir = os.path.join(emulation_group_config.output_dir, f'plot_emulation_group_{emulation_group_name}')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # PCA plots
        results = emulation_results[emulation_group_name]
        _plot_pca_reconstruction_error(results, plot_dir, emulation_group_config)
        _plot_pca_reconstruction_observables(results, emulation_group_config, plot_dir)
        # TODO: These deserve to be a global plot. Also nice to have for the group
        _plot_pca_explained_variance(results, plot_dir, emulation_group_config)
        _plot_pca_reconstruction_error_by_feature(results, plot_dir, emulation_group_config)

        # Emulator plots
        # TODO: validation_set doesn't do anything here yet because predict() doesn't use the validation_set argument!
        _plot_emulator_observables(results, emulation_group_config, plot_dir, validation_set=False)
        _plot_emulator_observables(results, emulation_group_config, plot_dir, validation_set=True)

        _plot_emulator_residuals(results, emulation_group_config, plot_dir, validation_set=False)
        _plot_emulator_residuals(results, emulation_group_config, plot_dir, validation_set=True)

#---------------------------------------------------------------
def _plot_pca_explained_variance(results, plot_dir, config):
    '''
    Plot fraction of explained variance as a function of number of principal components
    '''

    pca = results['PCA']['pca']
    n_pc_max = 30
    n_pc_selected = config.n_pc

    x = range(n_pc_max)
    y = [np.sum(pca.explained_variance_ratio_[:n_pc]) for n_pc in x]

    plt.title('PCA: explained variance', fontsize=14)
    plt.xlabel('number of principal components', fontsize=16)
    plt.ylabel('fraction explained variance', fontsize=16)
    plt.grid(True)
    plt.plot(x, y, linewidth=2, linestyle='-', alpha=1., color=sns.xkcd_rgb['dark sky blue'])
    plt.plot([], [], ' ', label=f"n_pc = {n_pc_selected}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'PCA_explained_variance.pdf'))
    plt.close()

#---------------------------------------------------------------
def _plot_pca_reconstruction_error(results, plot_dir, config):
    '''
    Compute reconstruction error -- inverse transform and then compute residuals
    https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
    '''

    pca = results['PCA']['pca']
    Y = results['PCA']['Y']
    Y_pca = results['PCA']['Y_pca']
    scaler = results['PCA']['scaler']

    n_pc_max = 30
    n_pc_selected = config.n_pc

    x = range(n_pc_max)
    y = [np.sum((Y - scaler.inverse_transform(Y_pca[:,:n_pc].dot(pca.components_[:n_pc,:])))**2, axis=1).mean() for n_pc in x]

    # Alternately can call:
    # Y_reconstructed = pca.inverse_transform(Y_pca)

    plt.title('PCA: reconstruction error', fontsize=14)
    plt.xlabel('number of principal components', fontsize=16)
    plt.ylabel('reconstruction error', fontsize=16)
    plt.grid(True)
    plt.plot(x, y, linewidth=2, linestyle='-', alpha=1., color=sns.xkcd_rgb['dark sky blue'])
    plt.plot([], [], ' ', label=f"n_pc = {n_pc_selected}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'PCA_reconstruction_error.pdf'))
    plt.close()


def _plot_pca_reconstruction_error_by_feature(results, plot_dir, config):
    """ Plot reconstruction (ie. truncated PCA) - nominal vs feature as a function of n_pc.

    Here, features are the point by point observables. We'll plot a separate figure per PC.

    Expect to see big bumps at a few points corresponding to those which are poorly reconstructed in the truncated PCA.
    """
    # Select all design points
    selected_design_point = slice(None, None, sum)
    # or just one
    #selected_design_point = 0
    n_pc_max = 30
    n_pc_per_figure = 5

    pca = results['PCA']['pca']
    Y = results['PCA']['Y']
    Y_pca = results['PCA']['Y_pca']
    scaler = results['PCA']['scaler']

    colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['light blue'], sns.xkcd_rgb['pale red'], sns.xkcd_rgb['medium green']]

    for n_chunk in range(1, n_pc_max, n_pc_per_figure):
        n_pc_range = list(range(n_chunk, n_chunk + n_pc_per_figure))
        # Split into groups of n_pcs for readability
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'PCA: reconstruction error n_pc={"-".join([str(n_pc_range[0]), str(n_pc_range[-1])])}', fontsize=14)
        ax.set_xlabel('PCA feature', fontsize=16)
        ax.set_ylabel('reconstruction error', fontsize=16)
        ax.grid(True)

        for i, n_pc in enumerate(n_pc_range):
            # Need to invert PCA and undo the scaling
            feature_diff_at_fixed_n_pc = Y - scaler.inverse_transform(Y_pca[:, :n_pc].dot(pca.components_[:n_pc,:]))
            x = np.arange(0, feature_diff_at_fixed_n_pc.shape[1])

            if isinstance(selected_design_point, slice) and selected_design_point.step is sum:
                # Normalize per n_pc
                y = np.sum(np.abs(feature_diff_at_fixed_n_pc), axis=0) / feature_diff_at_fixed_n_pc.shape[0]
            else:
                y = np.abs(feature_diff_at_fixed_n_pc[selected_design_point, :])

            ax.plot(
                x,
                y,
                linewidth=2,
                linestyle='-',
                alpha=1.,
                color=colors[n_pc % n_pc_per_figure],
                label=f"n_pc = {n_pc}",
                zorder=3 + i,
            )

        ax.legend(frameon=False, loc="upper right", fontsize=14)
        fig.tight_layout()
        selected_design_point_str = ""
        if not isinstance(selected_design_point, slice):
            selected_design_point_str = str(selected_design_point)
        elif selected_design_point.step is sum:
            selected_design_point_str = "s_all"
        else:
            start, stop = selected_design_point.start, selected_design_point.stop
            if start is None:
                start = 0
            if stop is None:
                stop = feature_diff_at_fixed_n_pc.shape[-1]
            selected_design_point_str = f"s_{start}_{stop}"

        _path = os.path.join(plot_dir, f'PCA_reconstruction_error__design_point_{selected_design_point_str}__n_pc_{"_".join([str(n_pc_range[0]), str(n_pc_range[-1])])}.pdf')
        fig.savefig(_path)
        plt.close(fig)


#---------------------------------------------------------------
def _plot_pca_reconstruction_observables(results, config, plot_dir):
    '''
    Plot observables before and after PCA -- for fixed n_pc
    '''

    # Get PCA results -- 2D arrays: (design_point_index, observable_bins)
    Y = results['PCA']['Y']
    Y_reconstructed_truncated = results['PCA']['Y_reconstructed_truncated_unscaled']
    # Translate matrix of stacked observables to a dict of matrices per observable
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    Y_dict = data_IO.observable_dict_from_matrix(Y, observables, config=config, validation_set=False, observable_filter=config.observable_filter)
    Y_dict_truncated_reconstructed = data_IO.observable_dict_from_matrix(Y_reconstructed_truncated, observables, validation_set=False, observable_filter=config.observable_filter)

    # Pass in a list of dicts to plot, each of which has structure Y[observable_label][design_point_index]
    plot_list = [Y_dict['central_value'], Y_dict_truncated_reconstructed['central_value']]
    labels = [r'JETSCAPE (before PCA)', r'JETSCAPE (after PCA)']
    colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue']] # sns.xkcd_rgb['light blue'], sns.xkcd_rgb['pale red'], sns.xkcd_rgb['medium green']

    design_point_index = 0
    filename = f'PCA_observables__design_point{design_point_index}'
    plot_utils.plot_observable_panels(plot_list, labels, colors, [design_point_index], config, plot_dir, filename, observable_filter=config.observable_filter)

#-------------------------------------------------------------------------------------------
def _plot_emulator_observables(results, config, plot_dir, validation_set=False):
    '''
    Plot observables predicted by emulator, and compare to JETSCAPE calculations
    '''

    # Get observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)

    # Get design points
    design = data_IO.design_array_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set)

    # Get JETSCAPE predictions
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set, observable_filter=config.observable_filter)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.observable_dict_from_matrix(Y, observables, config=config, validation_set=validation_set, observable_filter=config.observable_filter)

    # Get emulator predictions
    emulator_predictions = emulation.predict_emulation_group(parameters=design, results=results, config=config)
    emulator_predictions_dict = data_IO.observable_dict_from_matrix(emulator_predictions['central_value'],
                                                                    observables,
                                                                    validation_set=validation_set,
                                                                    observable_filter=config.observable_filter)

    # Plot
    design_point_index = 0
    if validation_set:
        plot_list = [Y_dict['central_value'], emulator_predictions_dict['central_value']]
        labels = [r'JETSCAPE (before PCA)', r'Emulator']
        colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['light blue']]
        filename = f'emulator_observables_validation_design_point{design_point_index}'
    else:
        # Get PCA results -- 2D arrays: (design_point_index, observable_bins)
        Y_reconstructed_truncated = results['PCA']['Y_reconstructed_truncated_unscaled']
        # Translate matrix of stacked observables to a dict of matrices per observable
        Y_dict_truncated_reconstructed = data_IO.observable_dict_from_matrix(Y_reconstructed_truncated, observables, validation_set=validation_set, observable_filter=config.observable_filter)

        plot_list = [Y_dict['central_value'], Y_dict_truncated_reconstructed['central_value'], emulator_predictions_dict['central_value']]
        labels = [r'JETSCAPE', r'JETSCAPE (reconstructed)', r'Emulator']
        colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['light blue']]
        filename = f'emulator_observables_training__design_point{design_point_index}'

    plot_utils.plot_observable_panels(plot_list, labels, colors, [design_point_index], config, plot_dir, filename, observable_filter=config.observable_filter)

#-------------------------------------------------------------------------------------------
def _plot_emulator_residuals(results, config, plot_dir, validation_set=False):
    '''
    Plot residuals between emulator and JETSCAPE calculations
    '''

    # Get observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)

    # Get design points
    design = data_IO.design_array_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set)

    # Get JETSCAPE predictions
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set, observable_filter=config.observable_filter)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.observable_dict_from_matrix(Y, observables, config=config, validation_set=validation_set, observable_filter=config.observable_filter)

    # Get emulator predictions
    emulator_predictions = emulation.predict_emulation_group(parameters=design, results=results, config=config)
    emulator_predictions_dict = data_IO.observable_dict_from_matrix(emulator_predictions['central_value'],
                                                                    observables,
                                                                    cov=emulator_predictions['cov'],
                                                                    validation_set=validation_set,
                                                                    observable_filter=config.observable_filter)

    # Construct scatter plot of |RAA_true - RAA_emulator| over all design points
    # We will also project the residuals into a histogram as a function of RAA_true

    # TODO: note that all observables are not RAA...may want to plot this more differentially
    #       depending on the observable considered

    # Construct a figure with two plots
    plt.figure(1, figsize=(10, 6))
    ax_scatter = plt.axes([0.1, 0.13, 0.6, 0.8]) # [left, bottom, width, height]
    ax_residual = plt.axes([0.81, 0.13, 0.15, 0.8])

    markers = ['o', 's', 'D']
    colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['pale red']]

    # Loop through observables
    RAA_true = np.array([])
    RAA_emulator = np.array([])
    std_emulator = np.array([])
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables, observable_filter=config.observable_filter)
    for observable_label in sorted_observable_list:
        sqrts, system, observable_type, observable, subobserable, centrality = data_IO.observable_label_to_keys(observable_label)

        RAA_true = np.concatenate((RAA_true, np.ravel(Y_dict['central_value'][observable_label])))
        RAA_emulator = np.concatenate((RAA_emulator, np.ravel(emulator_predictions_dict['central_value'][observable_label])))
        std_emulator = np.concatenate((std_emulator, np.ravel(emulator_predictions_dict['std'][observable_label])))

    residual = RAA_true - RAA_emulator
    normalized_residual = np.divide(residual, std_emulator)

    # Draw scatter plot
    ax_scatter.scatter(RAA_true, RAA_emulator, s=5, marker=markers[0],
                        color=colors[0], alpha=0.7, label=r'$\rm{{{}}}$'.format(''), linewidth=0,
                        rasterized=True)
    ax_scatter.set_ylim([0, 1.19])
    ax_scatter.set_xlim([0, 1.19])
    ax_scatter.set_xlabel(r'$R_{\rm{AA}}^{\rm{true}}$', fontsize=20)
    ax_scatter.set_ylabel(r'$R_{\rm{AA}}^{\rm{emulator}}$', fontsize=20)
    ax_scatter.legend(title='', title_fontsize=16,
                        loc='upper left', fontsize=14, markerscale=5)
    plt.setp(ax_scatter.get_xticklabels(), fontsize=14)
    plt.setp(ax_scatter.get_yticklabels(), fontsize=14)

    # Alternately, plot stdev as a function of RAA_true
    #ax_scatter.scatter(true_raa_i, emulator_raa_stdev_i, s=5,
    #                    color=color, alpha=0.7, label=r'$\rm{{{}}}$'.format(system_label), linewidth=0)
    #ax_scatter.set_xlabel(r'$R_{\rm{AA}}^{\rm{true}}$', fontsize=18)
    #ax_scatter.set_ylabel(r'$\sigma_{\rm{emulator}}$', fontsize=18)

    # Draw line with slope 1
    ax_scatter.plot([0,1], [0,1], sns.xkcd_rgb['almost black'], alpha=0.3,
                    linewidth=3, linestyle='--')

    # Print mean value of emulator uncertainty
    stdev_mean_relative = np.divide(std_emulator, RAA_emulator)
    stdev_mean = np.mean(stdev_mean_relative)
    text = rf'$\left< \sigma_{{\rm{{emulator}}}} \right> = {100*stdev_mean:.1f}\%$'
    ax_scatter.text(0.6, 0.15, text, fontsize=16)

    # Draw residuals
    max = 3
    bins = np.linspace(-max, max, 30)
    x = (bins[1:] + bins[:-1])/2
    h = ax_residual.hist(normalized_residual, color=colors[0], histtype='step',
                        orientation='horizontal', linewidth=3, alpha=0.8, density=True, bins=bins)
    ax_residual.scatter(h[0], x, color=colors[0], s=10, marker=markers[0])
    ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right) / \sigma_{\rm{emulator}}$', fontsize=20)
    #ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right)$', fontsize=20)
    plt.setp(ax_residual.get_xticklabels(), fontsize=14)
    plt.setp(ax_residual.get_yticklabels(), fontsize=14)

    # Print out indices of points that deviate significantly
    # if np.abs(normalized_residual) > 3*stdev:
    #     logger.info('Index {} has poor  emulator validation...'.format(j))

    if validation_set:
        filename = 'emulator_residuals_validation'
    else:
        filename = 'emulator_residuals_training'

    plt.savefig(os.path.join(plot_dir, f'{filename}.pdf'))
    plt.close('all')