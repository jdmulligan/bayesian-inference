#! /usr/bin/env python
'''
Module related to generate plots for MCMC

authors: J.Mulligan, R.Ehlers
'''

import os

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

from bayesian_inference import data_IO

####################################################################################################################
def plot(config):
    '''
    Generate plots for MCMC
    '''

    # Plot output dir
    plot_dir = os.path.join(config.output_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)