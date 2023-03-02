#! /usr/bin/env python
'''
Module related to generate plots for PCA/emulators

authors: J.Mulligan, R.Ehlers
'''

import os

import numpy as np
import pickle

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

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

    pca = results['PCA']['pca'] 
    Y = results['PCA']['Y']
    Y_pca = results['PCA']['Y_pca']
    Y_pca_truncated = Y_pca[:,:config.n_pc]
    Y_truncated_reconstructed = Y_pca_truncated.dot(pca.components_[:config.n_pc,:])

    # Y, Y_truncated_reconstructed are 2D arrays: (design_point_index, observable_bins)
    # For each design point, we want to parse the observable bins back to the observable they came from

    # To start, let's just grab a single design point
    observable_bins = Y[0]
    print(f'There are {observable_bins.shape[0]} bins')

    # TODO: should move dict<-->ndarray of Predictions to data_IO class, and ensure order of observables is fixed

    # Loop through JETSCAPE-analysis config to construct jet observables
    # Filter on observables in this analysis
    # For each observable:
    #   - plot prediction
    #   - plot PCA-reconstructed prediction
    #   - plot data
    #   - include labels
    # TODO: probably want to put the main pieces of this functionality in plot_base.py or something