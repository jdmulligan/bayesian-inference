""" Plot input data and predictions for Bayesian inference.

authors: J.Mulligan, R.Ehlers
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
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

    # Identify outliers via large statistical uncertainties
    _identify_large_statistical_uncertainty_points(config=config, validation_set=False)

    # First, plot the pair correlations for each observables
    _plot_pairplot_correlations(
        config=config,
        plot_dir=plot_dir,
        observable_grouping=ObservableGrouping(observable_by_observable=True),
        annotate_design_points=False,
    )
    # And an annotated version
    _plot_pairplot_correlations(
        config=config,
        plot_dir=plot_dir,
        observable_grouping=ObservableGrouping(observable_by_observable=True),
        annotate_design_points=True,
    )
    #
    _plot_pairplot_correlations(
        config=config,
        plot_dir=plot_dir,
        outliers_config=OutliersConfig(),
    )


####################################################################################################################
def _identify_large_statistical_uncertainty_points(config: emulation.EmulationConfig, validation_set: bool) -> None:
    # Setup
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    # Retrieve all observables, and check each for large statistical uncertainty points
    all_observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    for observable_key in data_IO.sorted_observable_list_from_dict(
        all_observables[prediction_key],
    ):
        outliers = _large_statistical_uncertainty_points(
            values=all_observables[prediction_key][observable_key]["y"],
            y_err=all_observables[prediction_key][observable_key]["y_err"],
        )
        logger.info(f"{observable_key=}, {outliers=}")

        # Check whether we have any design points where we have two points in a row.
        # In that case, extrapolation may be more problematic.
        outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
        for i_feature, i_design_point in zip(*outliers):
            outlier_features_per_design_point[i_design_point].update([i_feature])

        for k, v in outlier_features_per_design_point.items():
            if len(v) > 1 and np.all(np.diff(list(v)) == 1):
                logger.warning(f"Observable {observable_key}, Design point {k} has two consecutive outliers: {v}")

####################################################################################################################
def _large_statistical_uncertainty_points(values: npt.NDArray[np.float64], y_err: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """

    Best to do this observable-by-observable because the relative uncertainty will vary for each one.
    """
    relative_error = y_err / values
    rms = np.sqrt(np.mean(relative_error**2, axis=-1))
    #import IPython; IPython.embed()
    # NOTE: shape is (n_features, n_design_points).
    #       Remember that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(relative_error > 2 * rms[:, np.newaxis])
    return outliers  # type: ignore[return-value]

@attrs.frozen
class OutliersConfig:
    outliers_n_RMS_away_from_fit: float = 2.

@attrs.frozen
class ObservableGrouping:
    observable_by_observable: bool = False
    fixed_size: int | None = None

####################################################################################################################
def _plot_pairplot_correlations(
    config: emulation.EmulationConfig,
    plot_dir: Path,
    observable_grouping: ObservableGrouping | None = None,
    outliers_config: OutliersConfig | None = None,
    annotate_design_points: bool = False,
    use_experimental_data: bool = False,
) -> None:
    """ Plot pair correlations.

    Note that there are many configuration options, and they may not all be compatible with each other.

    :param EmulationConfig config: we take an instance of EmulationConfig as an argument to keep track of config info.
    :param Path plot_dir: Directory in which to save the plots.
    :param bool observable_by_observable: If true, plot each observable separately. Otherwise, plot for fixed size of observables.
    :param bool annotate_design_points: If true, annotate the data points with their design point index.
    :param bool calculate_RMS_distance: If true, calculate the RMS distance of each point from the fit line, and
        identify outliers as those more than `outliers_n_RMS_away_from_fit` away from the fit.
    :param float outliers_n_RMS_away_from_fit: Number of RMS away from the fit to identify outliers. Default: 2.
    :param bool use_experimental_data: If true, use experimental data. Otherwise, use predictions. Default: False.
        The experimental data isn't especially useful, but was helpful for initial conceptual development.
    """
    # Validation
    if observable_grouping is None:
        observable_grouping = ObservableGrouping(fixed_size=5)

    # Setup
    if use_experimental_data:
        observables = data_IO.data_array_from_h5(config.output_dir, filename='observables.h5', observable_filter=config.observable_filter)
        # Focus on central values
        observables = observables["y"]
        # In the case of data, this is trivially one "design point"
    else:
        observables = data_IO.predictions_matrix_from_h5(config.output_dir, filename='observables.h5', validation_set=False, observable_filter=config.observable_filter)

    # Determine output name
    filename = "pairplot_correlations"
    if observable_grouping is not None:
        if observable_grouping.observable_by_observable:
            filename += "__observable_by_observable"
        elif observable_grouping.fixed_size is not None:
            filename += f"__observable_group_by_{observable_grouping.fixed_size}"
    if annotate_design_points:
        filename += "__annotated"
    if outliers_config is not None:
        filename += "__outliers"

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

        #import IPython; IPython.embed()

        # Determine outliers by calculating the RMS distance from a linear fit
        if outliers_config:
            for i_col, x_column in enumerate(variables):
                for i_row, y_column in enumerate(variables):
                    if i_col < i_row:  # Skip the upper triangle + diagonal
                        current_ax = g.axes[i_row, i_col]

                        fit_result = regression_results[(i_row, i_col)]
                        #import IPython; IPython.embed()
                        logger.info(f"{fit_result=}, {fit_result.params=}")
                        # NOTE: The slope_key is the apparently taken from one of the columns of the df.
                        #       It's easier to just search for the right one here.
                        slope_key = [key for key in fit_result.params.keys() if key != "const"][0]
                        distances = _distance_from_line(
                            x=current_df[x_column],
                            y=current_df[y_column],
                            m=fit_result.params[slope_key],
                            b=fit_result.params["const"],
                        )
                        rms = np.sqrt(np.mean(distances**2))
                        logger.info(f"RMS distance: {rms:.2f}")

                        # Identify outliers by distance > outliers_n_RMS_away_from_fit * RMS
                        outlier_indices = np.where(distances > outliers_config.outliers_n_RMS_away_from_fit * rms)[0]

                        # Draw RMS on plot for reference
                        # NOTE: This isn't super precise, so don't be surprised if it doesn't perfectly match the outliers right along the line
                        # NOTE: Number of points is arbitrarily chosen - just want it to be dense
                        _x = np.linspace(np.min(current_df[x_column]), np.max(current_df[x_column]), 100)
                        # I'm sure that there's a way to do this directly from statsmodels, but I find their docs to be difficult to read.
                        # Since this is a simple case, we'll just do it by hand
                        linear_fit = fit_result.params[slope_key] * _x + fit_result.params["const"]
                        current_ax.plot(_x, linear_fit + 2 * rms, color='red', linestyle="dashed", linewidth=1.5)
                        current_ax.plot(_x, linear_fit - 2 * rms, color='red', linestyle="dashed", linewidth=1.5)

                        for (design_point, x, y) in zip(
                            current_df["design_point"][outlier_indices],
                            current_df[x_column][outlier_indices],
                            current_df[y_column][outlier_indices],
                        ):
                            logger.info(f"Outlier at {design_point=}")
                            current_ax.annotate(f"outlier! {design_point}", (x, y), fontsize=8, color=sns.xkcd_rgb['dark sky blue'])

        # Annotate data points with labels
        if annotate_design_points:
            count = 0
            for i_col, x_column in enumerate(variables):
                for i_row, y_column in enumerate(variables):
                    if i_col < i_row:  # Skip the upper triangle + diagonal
                        current_ax = g.axes[i_row, i_col]
                        current_ax.text(0.1, 0.9, s=f"count={count}", fontsize=8, color='blue', transform=current_ax.transAxes)
                        count += 1
                        for (design_point, x, y) in zip(current_df["design_point"], current_df[x_column], current_df[y_column]):
                            current_ax.annotate(design_point, (x, y), fontsize=8, color='red')

        #plt.tight_layout()
        plt.savefig(plot_dir / f"{filename}__group_{i_group}.pdf")
        # Cleanup
        plt.close('all')

        current_index += n_features_per_group

def _distance_from_line(x: npt.NDArray[np.number], y: npt.NDArray[np.number], m: float, b: float) -> np.ndarray:
    """ Calculate the distance of each point from a line.

    :param np.ndarray x: x values of points.
    :param np.ndarray y: y values of points.
    :param float m: slope of line.
    :param float b: y-intercept of line.
    :returns: distance of each point from the line.
    :rtype: np.ndarray
    """
    return np.abs(m * x - y + b) / np.sqrt(m**2 + 1)


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
