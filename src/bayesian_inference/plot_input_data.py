""" Plot input data and predictions for Bayesian inference.

authors: J.Mulligan, R.Ehlers
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Iterable

import attrs
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from bayesian_inference import data_IO, emulation, preprocess_input_data


logger = logging.getLogger(__name__)


def chunk_observables_in_dataframe(
    df: pd.DataFrame,
    chunk_size: int,
    base_label: str,
    base_title: str,
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    current_index = 0
    #for _ in range(observables.shape[1] // chunk_size):
    for _ in range((len(df.columns) - 1) // chunk_size):
        # Select multiple slices of columns: the values of interest + design_point column at -1
        # See: https://stackoverflow.com/a/39393929
        # Continuous
        current_df = df.iloc[:, np.r_[current_index:current_index + chunk_size, -1]]
        # Correlate early and late together
        #current_df = df.iloc[:, np.r_[current_index:current_index + int(self.fixed_size / 2), -current_index - int(self.fixed_size/2) : -current_index-1, -1]]
        label = f"{current_index}_{current_index + chunk_size}"
        if base_label:
            label = f"{base_label}_{label}"
        title = f"{current_index} - {current_index + chunk_size}"
        if base_title:
            title = f"{base_title} {title}"
        yield label, title, current_df

        current_index += chunk_size


@attrs.frozen
class ObservableGrouping:
    observable_by_observable: bool = False
    emulator_groups: bool = False
    fixed_size: int | None = None

    @property
    def label(self) -> str:
        label = ""
        if self.observable_by_observable:
            label += "observable_by_observable"
        elif self.emulator_groups:
            label += "emulator_groups"
        elif self.fixed_size is not None:
            label += f"observable_group_by_{self.fixed_size}"
        else:
            raise ValueError(f"Invalid ObservableGrouping settings: {self}")
        return label

    def gen(self, config: emulation.EmulationConfig, observables_filename: str, validation_set: bool) -> Iterable[tuple[str, str, pd.DataFrame]]:
        """ Generate a sequence of DataFrames, each of which contains a subset of the observables.

        :param np.ndarray observables: Predictions to be grouped.
        :param EmulationConfig config: The emulation config.
        :returns: A sequence of (str, str, DataFrames), each of which contains a label (for filenames), a title (for the figure), and a subset of the observables.
        """
        # Setup
        # Data
        all_observables_dict = data_IO.read_dict_from_h5(config.output_dir, observables_filename)
        #df = pd.DataFrame(observables[:, :3])
        #df = pd.DataFrame(observables)
        # Add design point as a column so we can use it (eg. with hue)
        design_points = np.array(all_observables_dict["Design_indices"])

        if self.observable_by_observable:
            observables = data_IO.predictions_matrix_from_h5(
                config.output_dir,
                filename=observables_filename,
                validation_set=validation_set,
                observable_filter=config.observable_filter
            )
            observables_dict = data_IO.observable_dict_from_matrix(
                observables,
                all_observables_dict,
                observable_filter=config.observable_filter
            )

            for observable_key, observable in observables_dict["central_value"].items():
                df = pd.DataFrame(observable)
                df["design_point"] = design_points
                yield f"observable_{observable_key}", observable_key, df
        elif self.emulator_groups:
            for emulation_group_name, emulation_group_config in config.emulation_groups_config.items():
                observables = data_IO.predictions_matrix_from_h5(
                    config.output_dir,
                    filename=observables_filename,
                    validation_set=validation_set,
                    observable_filter=emulation_group_config.observable_filter,
                )
                df = pd.DataFrame(observables)
                df["design_point"] = design_points

                # The max chunk size is selected somewhat arbitrarily to keep memory usage within reason
                max_chunk_size = 30
                if len(df.columns) > max_chunk_size:
                    yield from chunk_observables_in_dataframe(
                        df=df,
                        chunk_size=max_chunk_size,
                        base_label=f"{emulation_group_name}",
                        base_title=f"Group {emulation_group_name}",
                    )
                else:
                    yield f"{emulation_group_name}", f"Group {emulation_group_name}", df

        elif self.fixed_size is not None:
            observables = data_IO.predictions_matrix_from_h5(
                config.output_dir,
                filename=observables_filename,
                validation_set=validation_set,
                observable_filter=config.observable_filter
            )

            df = pd.DataFrame(observables)
            df["design_point"] = design_points

            yield from chunk_observables_in_dataframe(
                df=df,
                chunk_size=self.fixed_size,
                base_label=f"",
                base_title=f"Fixed size: {self.fixed_size}",
            )
        else:
            raise ValueError(f"Invalid ObservableGrouping settings: {self}")


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

    # Compare smoothed predictions for all design points
    # Start with individual so we can look in detail
    #_plot_predictions_for_all_design_points(
    #    config=config,
    #    plot_dir=plot_dir,
    #    select_which_to_plot=["standard"],
    #    grid_size=(3, 3),
    #    validation_set=False,
    #)
    #_plot_predictions_for_all_design_points(
    #    config=config,
    #    plot_dir=plot_dir,
    #    select_which_to_plot=["preprocessed"],
    #    grid_size=(3, 3),
    #    validation_set=False,
    #)
    ## And then combined for convenient comparison
    #_plot_predictions_for_all_design_points(
    #    config=config,
    #    plot_dir=plot_dir,
    #    select_which_to_plot=["standard", "preprocessed"],
    #    grid_size=(3, 3),
    #    validation_set=False,
    #)

    #for observables_filename in ["observables.h5", "observables_preprocessed.h5"]:
    for observables_filename in ["observables_preprocessed.h5"]:
        ## First, plot the pair correlations for each observables
        #_plot_pairplot_correlations(
        #    config=config,
        #    plot_dir=plot_dir,
        #    observable_grouping=ObservableGrouping(observable_by_observable=True),
        #    annotate_design_points=False,
        #    observables_filename=observables_filename,
        #)
        # Observable-by-observable, labeling and printing problematic design points
        identified_outliers = _plot_pairplot_correlations(
            config=config,
            plot_dir=plot_dir,
            observable_grouping=ObservableGrouping(observable_by_observable=True),
            outliers_config=preprocess_input_data.OutliersConfig(n_RMS=4.),
            observables_filename=observables_filename,
        )
        logger.info(f"{identified_outliers=}")
        summarized_design_points = set()
        for outlier_design_points in identified_outliers.values():
            summarized_design_points.update(outlier_design_points)
        logger.info(f"Summary of outlier design points ({len(summarized_design_points)=}): {summarized_design_points}")
        # Annotate all design points observable-by-observable
        _plot_pairplot_correlations(
            config=config,
            plot_dir=plot_dir,
            observable_grouping=ObservableGrouping(observable_by_observable=True),
            annotate_design_points=True,
            observables_filename=observables_filename,
        )
        # Group by emulator groups
        #_plot_pairplot_correlations(
        #    config=config,
        #    plot_dir=plot_dir,
        #    observable_grouping=ObservableGrouping(emulator_groups=True),
        #    observables_filename=observables_filename,
        #)


####################################################################################################################
def _plot_predictions_for_all_design_points(
    config: emulation.EmulationConfig,
    plot_dir: Path,
    select_which_to_plot: list[str],
    grid_size: tuple[int, int] | None = None,
    validation_set: bool = False,
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """ Plot comparison of predictions for all design points. """
    # Validation
    if grid_size is None:
        grid_size = (4, 4)
    if legend_kwargs is None:
        legend_kwargs = {}

    # Setup
    logger.info(f"Plotting standard vs preprocessed predictions for {select_which_to_plot=}")
    fontsize = 14. / grid_size[0]
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    # Get data (Note: this is where the bin values are stored)
    # NOTE: It doesn't matter which filename we use here - it's the same for both
    data = data_IO.data_dict_from_h5(config.output_dir, filename='observables.h5')

    # Grab the observables to compare
    all_observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    all_observables_preprocessed = data_IO.read_dict_from_h5(config.output_dir, 'observables_preprocessed.h5')
    colors = [sns.xkcd_rgb['dark sky blue'], sns.xkcd_rgb['medium green']]

    sorted_observable_keys_iter = iter(list(data_IO.sorted_observable_list_from_dict( all_observables[prediction_key],)))
    counter = 0
    while True:
        fig, axes = plt.subplots(grid_size[0], grid_size[1], constrained_layout=True)
        # This feels really not pythonic, but I guess the zip captures the StopIteration,
        # so we can't just rely on that to break the loop.
        observable_key_ax_pairs = list(zip(sorted_observable_keys_iter, axes.flat))
        # Out of observables - time to stop
        if len(observable_key_ax_pairs) == 0:
            break
        for observable_key, ax in observable_key_ax_pairs:
            # Use data to get the bin centers
            xmin = data[observable_key]['xmin']
            xmax = data[observable_key]['xmax']
            x = (xmin + xmax) / 2
            # Plot each design point separately
            observable = all_observables[prediction_key][observable_key]["y"]
            observable_preprocessed = all_observables_preprocessed[prediction_key][observable_key]["y"]
            for i_design_point in range(observable.shape[1]):
                for label, color, obs in zip(["standard", "preprocessed"], colors, [observable, observable_preprocessed]):
                    if label not in select_which_to_plot:
                        continue
                    ax.plot(
                        x,
                        obs[:, i_design_point],
                        linewidth=2,
                        alpha=0.2,
                        color=color,
                        label=label if i_design_point == 0 else None
                    )

            # This should practically cover reasonable observable ranges, but prevent giant outliers
            # from obscuring the details that we're interested in
            ax.set_ylim([-0.5, 2])
            ax.legend(
                loc='upper right',
                title=observable_key,
                title_fontsize=fontsize,
                fontsize=fontsize,
                frameon=False,
                **legend_kwargs
            )

        # Write figure to file and move to next one
        name = "compare_predictions_all_design_points"
        if validation_set:
            name += "__validation"
        select_which_to_plot_str = "_".join(select_which_to_plot)
        name += f"__{select_which_to_plot_str}__{counter}.pdf"
        _path = plot_dir / name
        fig.savefig(_path)
        plt.close(fig)

        counter += 1


####################################################################################################################
def _plot_pairplot_correlations(
    config: emulation.EmulationConfig,
    plot_dir: Path,
    observable_grouping: ObservableGrouping | None = None,
    outliers_config: preprocess_input_data.OutliersConfig | None = None,
    annotate_design_points: bool = False,
    use_experimental_data: bool = False,
    observables_filename: str = "observables.h5",
) -> dict[str, set]:
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
        observables = data_IO.data_array_from_h5(config.output_dir, filename=observables_filename, observable_filter=config.observable_filter)
        # Focus on central values
        observables = observables["y"]
        # In the case of data, this is trivially one "design point"
    else:
        observables = data_IO.predictions_matrix_from_h5(config.output_dir, filename=observables_filename, validation_set=False, observable_filter=config.observable_filter)

    # Determine output name
    observables_filename_label = observables_filename.split(".")[0]
    filename = f"{observables_filename_label}_pairplot_correlations"
    if observable_grouping is not None:
        filename += f"__{observable_grouping.label}"
    if annotate_design_points:
        filename += "__annotated"
    if outliers_config is not None:
        filename += "__outliers"

    # We want a shape of (n_design_points, n_features)
    df_generator = observable_grouping.gen(config=config, observables_filename=observables_filename, validation_set=False)

    # k -> v is label -> design_points
    identified_outliers: dict[str, set[int]] = {}
    for i_group, (label, title, current_df) in enumerate(df_generator):
        logger.debug(f"Pair plotting columns: {current_df.columns=}")

        # Useful early stopping for debugging...
        #if i_group > 3:
        #    break

        # Here, we'll practically want to create an sns.pairplot, but we can't configure it directly
        # for what we need, so we'll have to do it by hand.
        # Setup
        variables = list(current_df.columns)
        # Need to drop the design_point column from the variables list, as we just want it for labeling
        variables.remove("design_point")
        # And finally plot
        # With the standard sns call, we can't access the regression
        #g = sns.PairGrid(current_df, vars=variables)
        # This new class allows us to access the regression results
        g = PairGridWithRegression(current_df, vars=variables)
        regression_results = None
        if outliers_config:
            # NOTE: Can ignore outliers via `robust=True`, although need to install statsmodel
            regression_results = g.map_lower(simple_regplot)
        else:
            g.map_lower(sns.scatterplot)

        g.map_diag(sns.histplot)

        # Determine outliers by calculating the RMS distance from a linear fit
        if outliers_config:
            assert regression_results is not None
            identified_outliers[label] = set()
            for i_col, x_column in enumerate(variables):
                for i_row, y_column in enumerate(variables):
                    if i_col < i_row:  # Skip the upper triangle + diagonal
                        current_ax = g.axes[i_row, i_col]

                        fit_result = regression_results[(i_row, i_col)]
                        logger.debug(f"{fit_result=}, {fit_result.params=}")
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
                        logger.debug(f"RMS distance: {rms:.2f}")

                        # Identify outliers by distance > outliers_n_RMS_away_from_fit * RMS
                        outlier_indices = np.where(distances > outliers_config.n_RMS * rms)[0]

                        # Draw RMS on plot for reference
                        # NOTE: This isn't super precise, so don't be surprised if it doesn't perfectly match the outliers right along the line
                        # NOTE: Number of points is arbitrarily chosen - just want it to be dense
                        _x = np.linspace(np.min(current_df[x_column]), np.max(current_df[x_column]), 100)
                        # I'm sure that there's a way to do this directly from statsmodels, but I find their docs to be difficult to read.
                        # Since this is a simple case, we'll just do it by hand
                        linear_fit = fit_result.params[slope_key] * _x + fit_result.params["const"]
                        current_ax.plot(_x, linear_fit + outliers_config.n_RMS * rms, color='red', linestyle="dashed", linewidth=1.5)
                        current_ax.plot(_x, linear_fit - outliers_config.n_RMS * rms, color='red', linestyle="dashed", linewidth=1.5)

                        for (design_point, x, y) in zip(
                            current_df["design_point"][outlier_indices],
                            current_df[x_column][outlier_indices],
                            current_df[y_column][outlier_indices],
                        ):
                            logger.debug(f"Outlier at {design_point=}")
                            current_ax.annotate(f"_{design_point}", (x, y), fontsize=8, color=sns.xkcd_rgb['dark sky blue'])
                            identified_outliers[label].add(design_point)
            # Don't bother keeping if it's fully empty (just simplifies the output)
            if not identified_outliers[label]:
                del identified_outliers[label]

        # Annotate data points with design point labels to identify them
        if annotate_design_points:
            #count = 0
            for i_col, x_column in enumerate(variables):
                for i_row, y_column in enumerate(variables):
                    if i_col < i_row:  # Skip the upper triangle + diagonal
                        current_ax = g.axes[i_row, i_col]
                        #current_ax.text(0.1, 0.9, s=f"count={count}", fontsize=8, color='blue', transform=current_ax.transAxes)
                        #count += 1
                        for (design_point, x, y) in zip(
                            current_df["design_point"],
                            current_df[x_column],
                            current_df[y_column]
                        ):
                            current_ax.annotate(design_point, (x, y), fontsize=8, color='red')

        # Add title
        g.fig.suptitle(title, fontsize=26)

        #plt.tight_layout()
        name = f"{filename}__{label}"
        logger.info(f"Plotting {name=}")
        plt.savefig(plot_dir / f"{name}.pdf")
        # Cleanup
        plt.close('all')

    return identified_outliers


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
        self._add_axis_labels()

        if "hue" in inspect.signature(func).parameters:
            self.hue_names = list(self._legend_data)

        return results

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot on the specified axes."""
        if "hue" not in inspect.signature(func).parameters:
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
