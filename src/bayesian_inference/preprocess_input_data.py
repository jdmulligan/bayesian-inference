""" Preprocess the input data (eg. outliers removal, etc)

authors: J.Mulligan, R.Ehlers
"""

import logging
from typing import Any
from itertools import groupby

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

    return observables


def smooth_statistical_outliers_in_predictions(
    config: emulation.EmulationConfig,
    simple_interpolation: bool,
    outliers_config: OutliersConfig,
) -> dict[str, Any]:
    """ Steer smoothing of statistical outliers in predictions. """
    # Setup
    all_observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5')
    new_observables = {}
    # Adds the outputs under the "Prediction" key
    new_observables.update(
        _smooth_statistical_outliers_in_predictions(
            config=config,
            all_observables=all_observables,
            validation_set=False,
            simple_interpolation=simple_interpolation,
            outliers_config=outliers_config,
        )
    )
    # Adds the outputs under the "Prediction_validation" key
    new_observables.update(
        _smooth_statistical_outliers_in_predictions(
            config=config,
            all_observables=all_observables,
            validation_set=True,
            simple_interpolation=simple_interpolation,
            outliers_config=outliers_config,
        )
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
        logger.error("=====================================")
        logger.error(f"{observable_key=}, {outliers=}")

        # Check whether we have any design points where we have two points in a row.
        # In that case, extrapolation may be more problematic.
        outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
        for i_feature, design_point in zip(*outliers):
            if observable_key == "2760__PbPb__hadron__pt_ch_atlas____5-10" and design_point == 119:
                logger.warning(f"{outlier_features_per_design_point[design_point]=}, adding {i_feature=}")

            outlier_features_per_design_point[design_point].update([i_feature])
        # We expect these to be sorted (for finding distances between them), but sets are unordered, so we need to sort
        for design_point in outlier_features_per_design_point:
            outlier_features_per_design_point[design_point] = sorted(outlier_features_per_design_point[design_point])  # type: ignore[assignment]

        # If there are multiple points in a row, we skip extrapolation since it's not clear that we
        # can reliably extrapolate.
        # TODO: If we have to skip, we should probably consider excluding the observable for this design point.
        outlier_features_to_interpolate_per_design_point: dict[int, list[int]] = {}
        # TODO: Make this configurable
        # TODO: Default to 1 to be conservative!
        max_points_in_row_to_interpolate = 2
        for k, v in outlier_features_per_design_point.items():
            logger.warning("------------------------")
            logger.warning(f"{k=}, {v=}")
            distance_between_outliers = np.diff(list(v))
            indices_of_outliers_that_are_one_apart = set()
            accumulated_indices_to_remove = set()

            for distance, lower_feature_index, upper_feature_index in zip(distance_between_outliers, list(v)[:-1], list(v)[1:]):
                # We're really only worried about points which are right next to each other
                if distance == 1:
                    indices_of_outliers_that_are_one_apart.update([lower_feature_index, upper_feature_index])
                else:
                    # Only remove if it passes our threshold.
                    # NOTE: We want strictly greater than because we add two points per distance being greater than 1.
                    #       eg. one distance(s) of 1 -> two points
                    #           two distance(s) of 1 -> three points (due to set)
                    #           three distance(s) of 1 -> four points (due to set)
                    if len(indices_of_outliers_that_are_one_apart) > max_points_in_row_to_interpolate:
                        accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)
                    else:
                        # For debugging, keep track of when we skip removing points (ie. keep them for interpolation) because they're below our max threshold of consecutive points
                        # NOTE: No point in warning if empty, since that case is trivial
                        if len(indices_of_outliers_that_are_one_apart) > 0:
                            msg = (
                                f"Will continue with interpolating consecutive indices {indices_of_outliers_that_are_one_apart}"
                                f" because the their number is within the allowable range (n_consecutive<={max_points_in_row_to_interpolate})."
                            )
                            logger.warning(msg)
                    # Reset
                    indices_of_outliers_that_are_one_apart = set()

                # Update for next loop
                #previous_distance = distance
            outlier_features_to_interpolate_per_design_point[k] = sorted(list(set(v) - accumulated_indices_to_remove))
            logger.info(f"features kept for interpolation: {outlier_features_to_interpolate_per_design_point[k]}")

            # Now we want to look for multiple points in a row.
            #for group_value, group_members in groupby(np.diff(list(v))):
            #    if group_value == 1 and len(group_members):
            #        indices_of_outliers_that_are_one_apart.remove(list(group_members))

            #outlier_features_to_keep_for_interpolation = list(v)
            #distance_between_outliers = np.diff(list(v))
            ## Loop over possible outliers, and then if they're too close, remove both edges from the distance
            #previous_distance = 1e3
            #same_distance_in_a_row = 0
            #for distance, lower_feature_index, upper_feature_index in zip(distance_between_outliers, list(v)[:-1], list(v)[1:]):
            #    # First, derive some quantities
            #    if distance == previous_distance:
            #        same_distance_in_a_row += 1
            #    else:
            #        same_distance_in_a_row = 0
            #    # We're really only worried about points which are right next to each other
            #    if distance == 1:
            #        if max_points_in_row_to_interpolate > 1 and same_distance_in_a_row >= max_points_in_row_to_interpolate:
            #            logger.warning(f"Skipping {k} because we have {same_distance_in_a_row} points in a row.")
            #            outlier_features_to_interpolate[k] = []
            #            break

            #        outlier_features_to_keep_for_interpolation.remove(lower_feature_index)
            #        outlier_features_to_keep_for_interpolation.remove(upper_feature_index)

            #    # Update for the next time around
            #    previous_distance = distance

            #if len(v) > 1:
            #    outlier_features_to_keep_for_interpolation = []
            #    distance_between_outliers = np.diff(list(v))
            #    run_lengths = groupby(distance_between_outliers)
            #    current_index = 0
            #    just_skipped = False
            #    for group_value, group_members in run_lengths:
            #        logger.info(f"{current_index=}, {group_value=}, {list(group_members)=}")
            #        # Reset
            #        just_skipped = False
            #        if group_value == 1:
            #            # Skip
            #            current_index += (len(list(group_members)) + 1)
            #            just_skipped = True
            #        else:
            #            outlier_features_to_keep_for_interpolation.append(list(v)[current_index])
            #        current_index += 1
            #    # Get the last point
            #    if not just_skipped:
            #        outlier_features_to_keep_for_interpolation.append(list(v)[-1])
            #    else:
            #        logger.warning(f"Skipping last point for {k}")

            #    logger.info(f"{v=}, {distance_between_outliers=}, {outlier_features_to_keep_for_interpolation=}")
            #    outlier_features_to_interpolate[k] = outlier_features_to_keep_for_interpolation
            #else:
            #    logger.info("Just keeping")
            #    outlier_features_to_interpolate[k] = list(v)

            #import IPython; IPython.embed()
            #if len(v) > 1 and np.any(np.diff(list(v)) == 1):
            #    logger.warning(f"Observable {observable_key}, Design point {k} has multiple consecutive outliers: {v}")
            #    # TODO: If they're not sequential, we should still pick them out.
            #else:
            #    outlier_features_to_interpolate[k] = list(v)

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

            for design_point, points_to_interpolate in outlier_features_to_interpolate_per_design_point.items():
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