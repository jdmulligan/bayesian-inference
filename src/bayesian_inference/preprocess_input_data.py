""" Preprocess the input data (eg. outliers removal, etc)

authors: J.Mulligan, R.Ehlers
"""

import logging
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

from bayesian_inference import data_IO, emulation

logger = logging.getLogger(__name__)

@attrs.frozen
class OutliersConfig:
    """Configuration for identifying outliers.

    :param float n_RMS: Number of RMS away from the value to identify as an outlier. Default: 2.
    """
    n_RMS: float = 2.


def preprocess(
    config: emulation.EmulationConfig,
) -> dict[str, Any]:
    # First, remove outliers
    observables = smooth_statistical_outliers_in_predictions(
        config=config,
        simple_interpolation=True,
        outliers_config=OutliersConfig(n_RMS=2.),
    )

    # TODO: Need to make the observables.h5 file an input option
    return observables


def smooth_statistical_outliers_in_predictions(
    config: emulation.EmulationConfig,
    simple_interpolation: bool,
    outliers_config: OutliersConfig,
) -> dict[str, Any]:
    # Setup
    all_observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    new_observables = {}
    new_observables["Prediction"] = _smooth_statistical_outliers_in_predictions(
        config=config,
        all_observables=all_observables,
        validation_set=False,
        simple_interpolation=simple_interpolation,
        outliers_config=outliers_config,
    )
    new_observables["Prediction_validation"] = _smooth_statistical_outliers_in_predictions(
        config=config,
        all_observables=all_observables,
        validation_set=True,
        simple_interpolation=simple_interpolation,
        outliers_config=outliers_config,
    )

    # And then fill in the rest of the observable quantities.
    # This way, we can use it as a drop-in replacement.
    for k in all_observables:
        if k not in new_observables:
            new_observables[k] = all_observables[k]

    return new_observables

def _smooth_statistical_outliers_in_predictions(
    all_observables: dict[str, dict[str, dict[str, Any]]],
    config: emulation.EmulationConfig,
    validation_set: bool,
    simple_interpolation: bool,
    outliers_config: OutliersConfig | None = None,
) -> dict[str, Any]:
    # Validation
    if outliers_config is None:
        outliers_config = OutliersConfig()
    # Setup
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    # Retrieve all observables, and check each for large statistical uncertainty points
    new_observables: dict[str, dict[str, dict[str, Any]]] = {prediction_key: {}}
    for observable_key in data_IO.sorted_observable_list_from_dict(
        all_observables[prediction_key],
    ):
        outliers = _find_large_statistical_uncertainty_points(
            values=all_observables[prediction_key][observable_key]["y"],
            y_err=all_observables[prediction_key][observable_key]["y_err"],
            outliers_config=outliers_config,
        )
        logger.info(f"{observable_key=}, {outliers=}")

        # Check whether we have any design points where we have two points in a row.
        # In that case, extrapolation may be more problematic.
        outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
        for i_feature, i_design_point in zip(*outliers):
            outlier_features_per_design_point[i_design_point].update([i_feature])

        outlier_features_to_interpolate = {}
        for k, v in outlier_features_per_design_point.items():
            if len(v) > 1 and np.all(np.diff(list(v)) == 1):
                logger.warning(f"Observable {observable_key}, Design point {k} has two consecutive outliers: {v}")
                # TEMP: Don't propagate these points
            else:
                outlier_features_to_interpolate[k] = list(v)

        # Interpolate to find the value and error at the outlier point(s)
        new_observables[prediction_key][observable_key] = {}
        for key_type in ["y", "y_err"]:
            new_observables[prediction_key][observable_key][key_type] = np.array(
                all_observables[prediction_key][observable_key][key_type], copy=True,
            )
            observable_bin_centers = (
                all_observables["Data"][observable_key]["xmin"] + (
                    all_observables["Data"][observable_key]["xmax"] -
                    all_observables["Data"][observable_key]["xmin"]
                ) / 2.
            )
            if len(observable_bin_centers) == 1:
                # Skip - we can't interpolate one point.
                logger.debug(f"Skipping observable {observable_key} because it has only one point.")
                continue

            for design_point, points_to_interpolate in outlier_features_to_interpolate.items():
                mask = np.ones_like(observable_bin_centers, dtype=bool)
                # We don't want to data points that we points to interpolate to the interpolation function.
                # Otherwise, it will negatively impact the interpolation.
                mask[points_to_interpolate] = False
                # NOTE: ROOT::Interpolator uses a Cubic Spline, so this might be a reasonable future approach
                #       However, I think it's slower, so we'll start with this simple approach.
                if simple_interpolation:
                    interpolated_values = np.interp(
                        observable_bin_centers[points_to_interpolate],
                        observable_bin_centers[mask],
                        new_observables[prediction_key][observable_key][key_type][:, design_point][mask],
                    )
                else:
                    import scipy.interpolate
                    cs = scipy.interpolate.CubicSpline(
                        observable_bin_centers[mask],
                        new_observables[prediction_key][observable_key][key_type][:, design_point][mask],
                    )
                    interpolated_values = cs(observable_bin_centers[points_to_interpolate])

                new_observables[prediction_key][observable_key][key_type][points_to_interpolate, design_point] = interpolated_values

    return new_observables

def _find_large_statistical_uncertainty_points(
    values: npt.NDArray[np.float64],
    y_err: npt.NDArray[np.float64],
    outliers_config: OutliersConfig,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Find problematic points based on large statistical uncertainty points.

    Best to do this observable-by-observable because the relative uncertainty will vary for each one.
    """
    relative_error = y_err / values
    rms = np.sqrt(np.mean(relative_error**2, axis=-1))
    # NOTE: shape is (n_features, n_design_points).
    #       Remember that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(relative_error > outliers_config.n_RMS * rms[:, np.newaxis])
    return outliers  # type: ignore[return-value]