#! /usr/bin/env python
'''
Module with plotting utilities that can be shared across multiple other plotting modules

authors: J.Mulligan, R.Ehlers
'''

import os
import yaml

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('paper', rc={'font.size':18,'axes.titlesize':18,'axes.labelsize':18})

import data_IO             

#---------------------------------------------------------------
def plot_observable_panels(plot_list, labels, colors, design_point_index, config, plot_dir, filename):
    '''
    Plot observables before and after PCA -- for fixed n_pc
    '''
    # Loop through observables and plot 
    # Get sorted list of observables
    observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)

    # Get data (Note: this is where the bin values are stored)
    data = data_IO.data_array_from_h5(config.output_dir, filename='observables.h5', 
                                      observable_table_dir=config.observable_table_dir)

    # Group observables into subplots, with shapes specified in config
    plot_panel_shapes = config.analysis_config['plot_panel_shapes']
    n_panels = sum(x[0]*x[1] for x in plot_panel_shapes)
    assert len(sorted_observable_list) < n_panels, f'You specified {n_panels} panels, but have {len(sorted_observable_list)} observables'
    i_plot = 0
    i_subplot = 0
    fig, axs = None, None

    # We will use the JETSCAPE-analysis config files for plotting metadata
    plot_config_dir = config.observable_config_dir

    for i_observable,observable_label in enumerate(sorted_observable_list):
        sqrts, system, observable_type, observable, subobserable, centrality = data_IO.observable_label_to_keys(observable_label)
        
        # Get JETSCAPE-analysis config block for that observable
        plot_config_file = os.path.join(plot_config_dir, f'STAT_{sqrts}.yaml')
        with open(plot_config_file, 'r') as stream:
            plot_config = yaml.safe_load(stream)
        plot_block = plot_config[observable_type][observable]
        xtitle = latex_from_tlatex(plot_block['xtitle'])
        ytitle = latex_from_tlatex(plot_block['ytitle_AA'])

        color_data = sns.xkcd_rgb['almost black']
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

        # Plot -- create new plot and/or fill appropriate subplot
        plot_shape = plot_panel_shapes[i_plot]
        fontsize = 14./plot_shape[0]
        markersize = 8./plot_shape[0]
        if i_subplot == 0:
            fig, axs = plt.subplots(plot_shape[0], plot_shape[1], constrained_layout=True)
            for ax in axs.flat:
                ax.tick_params(labelsize=fontsize)
            row = 0
            col = 0
        else:
            col = i_subplot // plot_shape[0]
            row = i_subplot % plot_shape[0]

        axs[row,col].set_xlabel(rf'{xtitle}', fontsize=fontsize)
        axs[row,col].set_ylabel(rf'{ytitle}', fontsize=fontsize)
        axs[row,col].set_ylim([0., 2.])
        axs[row,col].set_xlim(xmin[0], xmax[-1])

        # Draw predictions
        for i_prediction,_ in enumerate(plot_list):
            axs[row,col].plot(x, plot_list[i_prediction][observable_label][design_point_index], 
                              label=labels[i_prediction], color=colors[i_prediction], 
                              linewidth=linewidth, alpha=alpha)
        
        # Draw data
        axs[row,col].errorbar(x, data_y, xerr=xerr, yerr=data_y_err,
                              color=color_data, marker='s', markersize=markersize, linestyle='', label='Experimental data')

        # Draw dashed line at RAA=1
        axs[row,col].plot([xmin[0], xmax[-1]], [1, 1],
                          sns.xkcd_rgb['almost black'], alpha=alpha, linewidth=linewidth, linestyle='dotted')

        # Draw legend
        axs[row,col].legend(loc='upper right', title=observable_label, 
                            title_fontsize=fontsize, fontsize=fontsize, frameon=True)

        # Increment subplot, and save if done with plot
        i_subplot += 1
        if i_subplot == plot_shape[0]*plot_shape[1] or i_observable == len(sorted_observable_list)-1:
            i_plot += 1
            i_subplot = 0
            
            plt.savefig(os.path.join(plot_dir, f'{filename}__{i_plot}.pdf'))
            plt.close()   

#-------------------------------------------------------------------------------------------
def latex_from_tlatex(s):
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