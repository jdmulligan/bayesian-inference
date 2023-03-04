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

    # Y, Y_truncated_reconstructed are 2D arrays: (design_point_index, observable_bins)
    pca = results['PCA']['pca'] 
    Y = results['PCA']['Y']
    Y_pca = results['PCA']['Y_pca']
    Y_pca_truncated = Y_pca[:,:config.n_pc]
    Y_truncated_reconstructed = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])

    # We will use the JETSCAPE-analysis config files for plotting metadata
    plot_config_dir = config.observable_config_dir

    # Get data (Note: this is where the bin values are stored)
    data = data_IO.data_array_from_h5(config.output_dir, filename='observables.h5')

    # Get sorted list of observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)

    # Translate matrix of stacked observables to a dict of matrices per observable 
    Y_dict = data_IO.prediction_dict_from_matrix(Y, observables)
    Y_dict_truncated_reconstructed = data_IO.prediction_dict_from_matrix(Y_truncated_reconstructed, observables)

    #-----
    # Loop through observables and plot
    # TODO: probably want to put the main pieces of this functionality in plot_base.py or something

    for observable_label in sorted_observable_list:
        sqrts, system, observable_type, observable, subobserable, centrality = data_IO.observable_label_to_keys(observable_label)
        
        # Get JETSCAPE-analysis config block for that observable
        plot_config_file = os.path.join(plot_config_dir, f'STAT_{sqrts}.yaml')
        with open(plot_config_file, 'r') as stream:
            plot_config = yaml.safe_load(stream)
            plot_block = plot_config[observable_type][observable]
        xtitle = _latex_from_tlatex(plot_block['xtitle'])
        ytitle = _latex_from_tlatex(plot_block['ytitle_AA'])

        color_data = sns.xkcd_rgb['almost black']
        color_prediction = sns.xkcd_rgb['dark sky blue']
        color_prediction_reconstructed = sns.xkcd_rgb['denim blue']  # sns.xkcd_rgb['light blue'], sns.xkcd_rgb['pale red']
        
        linewidth = 2
        alpha = 0.7

        # Get bins
        xmin = data[observable_label]['xmin']
        xmax = data[observable_label]['xmax']
        x = (xmin + xmax) / 2
        xerr = (xmax - x)

        # Get experimental data
        data_y = data[observable_label]['y']
        data_y_err = data[observable_label]['y_err'] 

        # To start, let's just grab a single design point
        design_point_index = 0

        # Get prediction
        prediction_y = Y_dict[observable_label][design_point_index]

        # Get PCA-reconstructed prediction
        prediction_y_truncated_reconstruated = Y_dict_truncated_reconstructed[observable_label][design_point_index]

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(rf'{xtitle}', fontsize=20)
        ax.set_ylabel(rf'{ytitle}', fontsize=20)
        ax.set_ylim([0., 2.])
        ax.tick_params(labelsize=15)
        ax.set_xlim(xmin[0], xmax[-1])

        # Draw data and predictions
        ax.errorbar(x, data_y, xerr=xerr, yerr=data_y_err,
                      color=color_data, marker='s', markersize=8, linestyle='', label='Experimental data')
        
        ax.plot(x, prediction_y, label=r'JETSCAPE',
                color=color_prediction, linewidth=linewidth, alpha=alpha)
        
        ax.plot(x, prediction_y_truncated_reconstruated, label=r'JETSCAPE (reconstructed)',
                color=color_prediction_reconstructed, linewidth=linewidth, alpha=alpha)

        # Draw dashed line at RAA=1
        ax.plot([xmin[0], xmax[-1]], [1, 1],
                sns.xkcd_rgb['almost black'], alpha=alpha, linewidth=linewidth, linestyle='dotted')

        # Draw legend
        ax.legend(loc='best', title=observable_label, title_fontsize=14, fontsize=14, frameon=False)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'PCA_{observable_label}.pdf'))
        plt.close()

#-------------------------------------------------------------------------------------------
def _latex_from_tlatex(s):
    '''
    Convert from tlatex to latex

    :param str s: TLatex string
    :return str s: latex string
    ''' 
    s = f'${s}$'
    s = s.replace('#it','')
    s = s.replace(' ','\;')
    s = s.replace('} {','},\;{')
    s = s.replace('#','\\')
    s = s.replace('SD',',\;SD')
    s = s.replace(', {\\beta} = 0', '')
    s = s.replace('{\Delta R}','')
    s = s.replace('Standard_WTA','\mathrm{Standard-WTA}')
    s = s.replace('{\\lambda}_{{\\alpha}},\;{\\alpha} = ','\lambda_')
    return s