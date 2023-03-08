#! /usr/bin/env python
'''
Module related to generate plots for PCA/emulators

authors: J.Mulligan, R.Ehlers
'''

import os
import yaml
import pickle

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

import data_IO
import emulation
import plot_utils

####################################################################################################################
def plot(config):
    '''
    Generate plots for PCA/emulators, using data written to file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    '''

    # Check if emulator already exists
    if not os.path.exists(config.emulation_outputfile):
        print(f'Emulator output does not exist: {config.emulation_outputfile}')
        return

    # Get results from file
    with open(config.emulation_outputfile, 'rb') as f:
	    results = pickle.load(f)

    # Plot output dir
    plot_dir = os.path.join(config.output_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
            
    # PCA plots
    _plot_pca_explained_variance(results, plot_dir)
    _plot_pca_reconstruction_error(results, plot_dir)
    _plot_pca_reconstruction_observables(results, config, plot_dir)

    # Emulator plots
    _plot_emulator_observables(results, config, plot_dir, validation_set=False)
    _plot_emulator_observables(results, config, plot_dir, validation_set=True)

    _plot_emulator_residuals(results, config, plot_dir, validation_set=False)
    _plot_emulator_residuals(results, config, plot_dir, validation_set=True)

#---------------------------------------------------------------
def _plot_pca_explained_variance(results, plot_dir):
    '''
    Plot fraction of explained variance as a function of number of principal components
    '''

    pca = results['PCA']['pca'] 
    n_pc_max = 30 # results['PCA']['Y_pca'].shape[1]

    x = range(n_pc_max)
    y = [np.sum(pca.explained_variance_ratio_[:n_pc]) for n_pc in x]

    plt.title('PCA: explained variance', fontsize=14)
    plt.xlabel('number of principal components', fontsize=16)
    plt.ylabel('fraction explained variance', fontsize=16)
    plt.grid(True)
    #plt.xscale('log')
    plt.plot(x, y, linewidth=2, linestyle='-', alpha=1., color=sns.xkcd_rgb['dark sky blue'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'PCA_explained_variance.pdf'))
    plt.close()
 
#---------------------------------------------------------------
def _plot_pca_reconstruction_error(results, plot_dir):
    '''
    Compute reconstruction error -- inverse transform and then compute residuals
    https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
    '''

    pca = results['PCA']['pca'] 
    Y = results['PCA']['Y']
    Y_pca = results['PCA']['Y_pca']

    n_pc_max = 30 # Y_pca.shape[1]

    x = range(n_pc_max)
    y = [np.sum((Y - Y_pca[:,:n_pc].dot(pca.components_[:n_pc,:]))**2, axis=1).mean() for n_pc in x]

    # Alternately can call:
    # Y_reconstructed = pca.inverse_transform(Y_pca)

    plt.title('PCA: reconstruction error', fontsize=14)
    plt.xlabel('number of principal components', fontsize=16)
    plt.ylabel('reconstruction error', fontsize=16)
    plt.grid(True)
    plt.plot(x, y, linewidth=2, linestyle='-', alpha=1., color=sns.xkcd_rgb['dark sky blue'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'PCA_reconstruction_error.pdf'))
    plt.close()

#---------------------------------------------------------------
def _plot_pca_reconstruction_observables(results, config, plot_dir):
    '''
    Plot observables before and after PCA -- for fixed n_pc
    '''

    # Get PCA results -- 2D arrays: (design_point_index, observable_bins)
    Y = results['PCA']['Y']
    Y_reconstructed_truncated = results['PCA']['Y_reconstructed_truncated']
    # Translate matrix of stacked observables to a dict of matrices per observable
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    Y_dict = data_IO.prediction_dict_from_matrix(Y, observables, config, validation_set=False)
    Y_dict_truncated_reconstructed = data_IO.prediction_dict_from_matrix(Y_reconstructed_truncated, observables, validation_set=False)

    # Pass in a list of dicts to plot, each of which has structure Y[observable_label][design_point_index]
    plot_list = [Y_dict, Y_dict_truncated_reconstructed]
    labels = [r'JETSCAPE (before PCA)', r'JETSCAPE (after PCA)']
    colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue']] # sns.xkcd_rgb['light blue'], sns.xkcd_rgb['pale red'] 
    design_point_index = 0
    filename = f'PCA_observables__design_point{design_point_index}'
    plot_utils.plot_observable_panels(plot_list, labels, colors, design_point_index, config, plot_dir, filename)

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
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.prediction_dict_from_matrix(Y, observables, config, validation_set=validation_set)

    # Get emulator predictions
    emulator_predictions = emulation.predict(design, results, config, validation_set=validation_set)

    # Plot
    design_point_index = 0
    if validation_set:
        plot_list = [Y_dict, emulator_predictions]
        labels = [r'JETSCAPE (before PCA)', r'Emulator']
        colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['light blue']] 
        filename = f'emulator_observables_validation_design_point{design_point_index}'
    else:
        # Get PCA results -- 2D arrays: (design_point_index, observable_bins)
        Y_reconstructed_truncated = results['PCA']['Y_reconstructed_truncated']
        # Translate matrix of stacked observables to a dict of matrices per observable
        Y_dict_truncated_reconstructed = data_IO.prediction_dict_from_matrix(Y_reconstructed_truncated, observables, validation_set=validation_set)

        plot_list = [Y_dict, Y_dict_truncated_reconstructed, emulator_predictions]
        labels = [r'JETSCAPE', r'JETSCAPE (reconstructed)', r'Emulator']
        colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['light blue']]
        filename = f'emulator_observables_training__design_point{design_point_index}'
    plot_utils.plot_observable_panels(plot_list, labels, colors, design_point_index, config, plot_dir, filename)

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
    Y = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=validation_set)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.prediction_dict_from_matrix(Y, observables, config, validation_set=validation_set)

    # Get emulator predictions
    emulator_predictions = emulation.predict(design, results, config, validation_set=validation_set)

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
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)
    for i_observable,observable_label in enumerate(sorted_observable_list):
        sqrts, system, observable_type, observable, subobserable, centrality = data_IO.observable_label_to_keys(observable_label)

        RAA_true = np.concatenate((RAA_true, np.ravel(Y_dict[observable_label])))
        RAA_emulator = np.concatenate((RAA_emulator, np.ravel(emulator_predictions[observable_label])))
        #std_emulator = std_emulator.concatenate(emulator_predictions[observable_label])

    residual = RAA_true - RAA_emulator
    #normalized_residual = np.divide(RAA_true-RAA_emulator, std_emulator)

    # Draw scatter plot
    ax_scatter.scatter(RAA_true, RAA_emulator, s=5, marker=markers[0],
                        color=colors[0], alpha=0.7, label=r'$\rm{{{}}}$'.format(''), linewidth=0)
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
    #stdev_mean_relative = np.divide(emulator_raa_stdev_i, true_raa_i)
    #stdev_mean = np.mean(stdev_mean_relative)
    #text = r'$\left< \sigma_{{\rm{{emulator}}}}^{{\rm{{{}}}}} \right> = {:0.1f}\%$'.format(system_label, 100*stdev_mean)
    #ax_scatter.text(0.4, 0.17-0.09*i, text, fontsize=16)
    
    # Draw residuals
    max = 3
    bins = np.linspace(-max, max, 30)
    x = (bins[1:] + bins[:-1])/2
    h = ax_residual.hist(residual, color=colors[0], histtype='step',
                        orientation='horizontal', linewidth=3, alpha=0.8, density=True, bins=bins)
    ax_residual.scatter(h[0], x, color=colors[0], s=10, marker=markers[0])
    #ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right) / \sigma_{\rm{emulator}}$', fontsize=20)
    ax_residual.set_ylabel(r'$\left(R_{\rm{AA}}^{\rm{true}} - R_{\rm{AA}}^{\rm{emulator}}\right)$', fontsize=20)
    plt.setp(ax_residual.get_xticklabels(), fontsize=14)
    plt.setp(ax_residual.get_yticklabels(), fontsize=14)
                            
    # Print out indices of points that deviate significantly
    # if np.abs(normalized_residual) > 3*stdev:
    #     print('Index {} has poor  emulator validation...'.format(j))
            
    if validation_set:
        filename = 'emulator_residuals_validation'
    else:
        filename = 'emulator_residuals_training'

    plt.savefig(os.path.join(plot_dir, f'{filename}.pdf'))
    plt.close('all')                  