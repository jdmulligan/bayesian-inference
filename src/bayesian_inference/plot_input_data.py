""" Plot input data and predictions for Bayesian inference.

authors: J.Mulligan, R.Ehlers
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bayesian_inference import data_IO, emulation


logger = logging.getLogger(__name__)

####################################################################################################################
def plot(config: emulation.EmulationConfig):
    '''
    Generate plots for input experimental data and predictions, using data written to file in the data import.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
        Although these plots are before the emulation, we need emulation info (eg. what groups?), so it's extremely
        useful to have the config object.
    '''
    # Plot output dir
    plot_dir = Path(config.output_dir) / 'plot_input_data'
    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_pairplot_correlations(config=config, plot_dir=plot_dir)


####################################################################################################################
def _plot_pairplot_correlations(config: emulation.EmulationConfig, plot_dir: Path, use_experimental_data: bool = False) -> None:
    """

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    :param Path plot_dir: Directory in which to save the plots.
    :param bool use_experimental_data: If true, use experimental data. Otherwise, use predictions. Default: False.
        The experimental data isn't especially useful, but was helpful for initial conceptual development.
    """

    if use_experimental_data:
        observables = data_IO.data_array_from_h5(config.output_dir, filename='observables.h5', observable_filter=config.observable_filter)
        # Focus on central values
        observables = observables["y"]
        # In the case of data, this is trivially one "design point"
    else:
        observables = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=False, observable_filter=config.observable_filter)

    # We want a shape of (n_design_points, n_features)
    df = pd.DataFrame(observables[:, :10])
    # Add design point as a column so we can use it for hue
    df["design_point"] = np.arange(df.shape[0])

    # Plot
    g = sns.pairplot(
        df,
        #hue="design_point",
        #diag_kind='hist',
        #plot_kws={'alpha':0.7, 's':3, 'color':'blue'},
        #diag_kws={'color':'blue', 'fill':True, 'bins':20}
    )
    #plt.tight_layout()
    plt.savefig(plot_dir / "pairplot_correlations.pdf")
    # Cleanup
    plt.close('all')
