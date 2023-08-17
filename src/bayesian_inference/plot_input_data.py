""" Plot input data and predictions for Bayesian inference.

authors: J.Mulligan, R.Ehlers
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

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

    _plot_pairplot_correlations(config=config, plot_dir=plot_dir, annotate_design_points=False)
    _plot_pairplot_correlations(config=config, plot_dir=plot_dir, annotate_design_points=True)


####################################################################################################################
def _plot_pairplot_correlations(
    config: emulation.EmulationConfig,
    plot_dir: Path,
    annotate_design_points: bool,
    use_experimental_data: bool = False,
) -> None:
    """

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    :param Path plot_dir: Directory in which to save the plots.
    :param bool annotate_design_points: If true, annotate the data points with their design point index.
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
    #df = pd.DataFrame(observables[:, :3])
    df = pd.DataFrame(observables)
    # Add design point as a column so we can use it (eg. with hue)
    all_observables_dict = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    design_points = np.array(all_observables_dict["Design_indices"])
    df["design_point"] = design_points

    current_index = 0
    n_features_per_group = 5
    for i_group in range(observables.shape[1] // n_features_per_group):
        # Select multiple slices of columns: the values of interest + design_point column at -1
        # See: https://stackoverflow.com/a/39393929
        #current_df = df.iloc[:, np.r_[current_index:current_index + n_features_per_group, -1]]
        current_df = df.iloc[:, np.r_[current_index:current_index + int(n_features_per_group / 2), -current_index - int(n_features_per_group/2) : -current_index-1, -1]]
        logger.info(f"Pair plotting columns: {current_df.columns=}")

        # TEMP
        if i_group > 2:
            break

        # Plot
        #g = sns.pairplot(
        #    df,
        #    #hue="design_point",
        #    #diag_kind='hist',
        #    #plot_kws={'alpha':0.7, 's':3, 'color':'blue'},
        #    #diag_kws={'color':'blue', 'fill':True, 'bins':20}
        #)
        # Manually creating the same type of pair plot, but with more control
        variables = list(current_df.columns)
        # Need to drop the design_point column, as we just want it for labeling
        variables.remove("design_point")
        # And finally plot
        #g = sns.PairGrid(current_df, vars=variables)
        g = PairGridWithRegression(current_df, vars=variables)
        #g.map_lower(sns.scatterplot)
        # NOTE: Can ignore outliers via `robust=True`, although need to install statsmodel
        regression_results = g.map_lower(simple_regplot)
        #g.map_lower(sns.regplot)
        g.map_diag(sns.histplot)

        import IPython; IPython.embed()

        # Annotate data points with labels
        if annotate_design_points:
            count = 0
            for i, axis_row in enumerate(variables):
                for j, axis_col in enumerate(variables):
                    if i < j:  # Skip the upper triangle + diagonal
                        current_ax = g.axes[j, i]
                        current_ax.text(0.1, 0.9, s=f"count={count}", fontsize=8, color='blue', transform=current_ax.transAxes)
                        count += 1
                        for (design_point, x, y) in zip(current_df["design_point"], current_df[axis_row], current_df[axis_col]):
                            current_ax.annotate(design_point, (x, y), fontsize=8, color='red')

        #plt.tight_layout()
        filename = "pairplot_correlations"
        if annotate_design_points:
            filename += "_annotated"
        plt.savefig(plot_dir / f"{filename}__group_{i_group}.pdf")
        # Cleanup
        plt.close('all')

        current_index += n_features_per_group

class PairGridWithRegression(sns.PairGrid):
    """ PairGrid where we can return the regression results.

    Sadly, this isn't possible with seaborn, so we have to work it out ourselves.
    Made minimal edits of seaborn.PairGrid from:
    https://github.com/mwaskom/seaborn/blob/aebf7d8dba58b41090cefbfea4074682edf62349/seaborn/axisgrid.py#L1172-L1670
    commit: aebf7d8dba58b41090cefbfea4074682edf62349 . All that was added was the simplest way to return values.
    It's not robust, and almost certainly won't work in all cases, but I think it will fail clearly.
    """
    def map(self, func, **kwargs):
        """Plot with the same function in every subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        row_indices, col_indices = np.indices(self.axes.shape)
        indices = zip(row_indices.flat, col_indices.flat)
        results = self._map_bivariate(func, indices, **kwargs)

        return results

    def map_lower(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        indices = zip(*np.tril_indices_from(self.axes, -1))
        logger.warning(f"map_lower: {func=}")
        results = self._map_bivariate(func, indices, **kwargs)
        return results

    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        indices = zip(*np.triu_indices_from(self.axes, 1))
        results = self._map_bivariate(func, indices, **kwargs)
        return results

    def map_offdiag(self, func, **kwargs):
        """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        results = {}
        if self.square_grid:
            results = self.map_lower(func, **kwargs)
            if not self._corner:
                results.update(self.map_upper(func, **kwargs))
        else:
            indices = []
            for i, (y_var) in enumerate(self.y_vars):
                for j, (x_var) in enumerate(self.x_vars):
                    if x_var != y_var:
                        indices.append((i, j))
            results = self._map_bivariate(func, indices, **kwargs)
        return results

    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        # This is a hack to handle the fact that new distribution plots don't add
        # their artists onto the axes. This is probably superior in general, but
        # we'll need a better way to handle it in the axisgrid functions.
        from seaborn.distributions import histplot, kdeplot
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True

        kws = kwargs.copy()  # Use copy as we insert other kwargs
        results = {}
        for i, j in indices:
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:  # i.e. we are in corner mode
                continue
            results[(i, j)] = self._plot_bivariate(x_var, y_var, ax, func, **kws)
            logger.info(f"{i=}, {j=}, {results[(i, j)]=}, {func=}")
        self._add_axis_labels()

        if "hue" in inspect.signature(func).parameters:
            self.hue_names = list(self._legend_data)

        return results

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot on the specified axes."""
        if "hue" not in inspect.signature(func).parameters:
            logger.info("here")
            results = self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
            return results

        kwargs = kwargs.copy()
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = ax
        else:
            plt.sca(ax)

        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]

        if self._hue_var is not None and self._hue_var not in axes_vars:
            axes_vars.append(self._hue_var)

        data = self.data[axes_vars]
        if self._dropna:
            data = data.dropna()

        x = data[x_var]
        y = data[y_var]
        if self._hue_var is None:
            hue = None
        else:
            hue = data.get(self._hue_var)

        if "hue" not in kwargs:
            kwargs.update({
                "hue": hue, "hue_order": self._hue_order, "palette": self._orig_palette,
            })
        logger.warning(f"{func=}")
        result = func(x=x, y=y, **kwargs)

        self._update_legend_data(ax)

        return result

    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot while iterating over hue subsets."""
        kwargs = kwargs.copy()
        if str(func.__module__).startswith("seaborn"):
            kwargs["ax"] = ax
        else:
            plt.sca(ax)

        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]

        # NOTE RJE: This only supports one hue group. Note that hue_vals is **not**
        #           what we want to check here (it's some series)
        if len(self._hue_order) > 1:
            msg = f"Only one hue group is supported. Provided: {self._hue_order}"
            raise NotImplementedError(msg)

        hue_grouped = self.data.groupby(self.hue_vals)
        results = None
        for k, label_k in enumerate(self._hue_order):

            kws = kwargs.copy()

            # Attempt to get data for this level, allowing for empty
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=axes_vars,
                                      dtype=float)

            if self._dropna:
                data_k = data_k[axes_vars].dropna()

            x = data_k[x_var]
            y = data_k[y_var]

            for kw, val_list in self.hue_kws.items():
                kws[kw] = val_list[k]
            kws.setdefault("color", self.palette[k])
            if self._hue_var is not None:
                kws["label"] = label_k

            if str(func.__module__).startswith("seaborn"):
                results = func(x=x, y=y, **kws)
            else:
                results = func(x, y, **kws)

        self._update_legend_data(ax)

        return results


def simple_regplot(
    x, y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None,
    **kwargs,
):
    """ Draw a regression line with error interval.

    From: https://stackoverflow.com/a/59756979 . This can be used as an approximately drop-in replacement for
    sns.regplot, but the difference is that it returns the fit results. This works nicely for simpler plots,
    but unfortunately requires the PairGridWithRegression class to actually return those values.
    """
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit).fit()

    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # draw the fit line and error interval
    ci_kws = {} if ci_kws is None else ci_kws
    ax.fill_between(
        eval_x[:, 1],
        pred.predicted_mean - n_std * pred.se_mean,
        pred.predicted_mean + n_std * pred.se_mean,
        alpha=0.5,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws)

    # draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    ax.scatter(x, y, c=h[0].get_color(), **scatter_kws)

    return fit_results
