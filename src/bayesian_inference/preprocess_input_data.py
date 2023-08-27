""" Preprocess the input data (eg. outliers removal, etc)

authors: J.Mulligan, R.Ehlers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import yaml

from bayesian_inference import common_base, data_IO

logger = logging.getLogger(__name__)

@attrs.frozen
class OutliersConfig:
    """Configuration for identifying outliers.

    :param float n_RMS: Number of RMS away from the value to identify as an outlier. Default: 2.
    """
    n_RMS: float = 2.


def preprocess(
    preprocessing_config: PreprocessingConfig,
) -> dict[str, Any]:
    # First, smooth predictions
    observables = smooth_statistical_outliers_in_predictions(
        preprocessing_config=preprocessing_config,
    )

    return observables


def smooth_statistical_outliers_in_predictions(
    preprocessing_config: PreprocessingConfig,
) -> dict[str, Any]:
    """ Steer smoothing of statistical outliers in predictions. """
    logger.info("Smoothing outliers in predictions...")

    # Setup for observables
    all_observables = data_IO.read_dict_from_h5(preprocessing_config.output_dir, 'observables.h5')
    new_observables = {}
    # Adds the outputs under the "Prediction" key
    new_observables.update(
        _smooth_statistical_outliers_in_predictions(
            all_observables=all_observables,
            validation_set=False,
            preprocessing_config=preprocessing_config,
        )
    )
    # Adds the outputs under the "Prediction_validation" key
    new_observables.update(
        _smooth_statistical_outliers_in_predictions(
            all_observables=all_observables,
            validation_set=True,
            preprocessing_config=preprocessing_config,
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
    validation_set: bool,
    preprocessing_config: PreprocessingConfig,
) -> dict[str, Any]:
    """Smooth out statistical outliers in predictions.

    Args:
        all_observables: Dictionary of observables from read_dict_from_h5.
        validation_set: Whether to use the validation set or not.
        preprocessing_config: Configuration for preprocessing.
    """
    # Setup
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    # These will contain our interpolated predictions
    new_observables: dict[str, dict[str, dict[str, Any]]] = {prediction_key: {}}
    # Outliers which we are unable to remove. We keep track of this to help guide if we need to exclude a design point
    # Format: {observable_key: {design_point: set[feature_index]}
    outliers_we_are_unable_to_remove: dict[str, dict[int, set[int]]] = {}
    for observable_key in data_IO.sorted_observable_list_from_dict(
        all_observables[prediction_key],
    ):
        # First, find the outliers based on large statistical uncertainty points
        outliers = _find_large_statistical_uncertainty_points(
            values=all_observables[prediction_key][observable_key]["y"],
            y_err=all_observables[prediction_key][observable_key]["y_err"],
            outliers_config=preprocessing_config.smoothing_outliers_config,
        )

        # Next, we want to do quality checks.
        # If there are multiple problematic points in a row, we want to skip interpolation since
        # it's not clear that we can reliably interpolate.
        # First, we need to put the features into a more useful order:
        # outliers: zip(feature_index, design_point) -> dict: (design_point, feature_index)
        # NOTE: The `design_point` here is the index in the design point array of the design points
        #       that we've using for this analysis. To actually use them (ie. in print outs), we'll
        #       need to apply them to the actual design point array.
        outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
        for i_feature, design_point in zip(*outliers):
            outlier_features_per_design_point[design_point].update([i_feature])
        # These features must be sorted to finding distances between them, but sets are unordered,
        # so we need to explicitly sort them
        for design_point in outlier_features_per_design_point:
            outlier_features_per_design_point[design_point] = sorted(outlier_features_per_design_point[design_point])  # type: ignore[assignment]

        # Since the feature values of one design point shouldn't impact another, we'll want to
        # check one design point at a time.
        # NOTE: If we have to skip, we record the design point so we can consider excluding it due
        #       to that observable.
        outlier_features_to_interpolate_per_design_point: dict[int, list[int]] = {}
        for k, v in outlier_features_per_design_point.items():
            #logger.debug("------------------------")
            #logger.debug(f"{k=}, {v=}")
            # Calculate the distance between the outlier indices
            distance_between_outliers = np.diff(list(v))
            # And we'll keep track of which ones pass our quality requirements (not too many in a row).
            indices_of_outliers_that_are_one_apart = set()
            accumulated_indices_to_remove = set()

            for distance, lower_feature_index, upper_feature_index in zip(distance_between_outliers, list(v)[:-1], list(v)[1:]):
                # We're only worried about points which are right next to each other
                if distance == 1:
                    indices_of_outliers_that_are_one_apart.update([lower_feature_index, upper_feature_index])
                else:
                    # In this case, we now have points that aren't right next to each other.
                    # Here, we need to figure out what we're going to do with the points that we've found
                    # that **are** right next to each other. Namely, we'll want to remove them from the list
                    # to be interpolated, but if there are more points than our threshold.
                    # NOTE: We want strictly greater than because we add two points per distance being greater than 1.
                    #       eg. one distance(s) of 1 -> two points
                    #           two distance(s) of 1 -> three points (due to set)
                    #           three distance(s) of 1 -> four points (due to set)
                    if len(indices_of_outliers_that_are_one_apart) > preprocessing_config.smoothing_max_n_feature_outliers_to_interpolate:
                        # Since we are looking at the distances, we want to remove the points that make up that distance.
                        accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)
                    else:
                        # For debugging, keep track of when we find points that are right next to each other but
                        # where we skip removing them (ie. keep them for interpolation) because they're below our
                        # max threshold of consecutive points
                        # NOTE: There's no point in warning if empty, since that case is trivial
                        if len(indices_of_outliers_that_are_one_apart) > 0:
                            msg = (
                                f"Will continue with interpolating consecutive indices {indices_of_outliers_that_are_one_apart}"
                                f" because the their number is within the allowable range (n_consecutive<={preprocessing_config.smoothing_max_n_feature_outliers_to_interpolate})."
                            )
                            logger.info(msg)
                    # Reset for the next point
                    indices_of_outliers_that_are_one_apart = set()

            # Now that we've determine which points we want to remove from our interpolation (accumulated_indices_to_remove),
            # let's actually remove them from our list.
            # NOTE: We sort again because sets are not ordered.
            outlier_features_to_interpolate_per_design_point[k] = sorted(list(set(v) - accumulated_indices_to_remove))
            #logger.debug(f"features kept for interpolation: {outlier_features_to_interpolate_per_design_point[k]}")

            # And we'll keep track of what we can't interpolate
            if accumulated_indices_to_remove:
                if observable_key not in outliers_we_are_unable_to_remove:
                    outliers_we_are_unable_to_remove[observable_key] = {}
                outliers_we_are_unable_to_remove[observable_key][k] = accumulated_indices_to_remove

        # Finally, interpolate at the selected outlier point features to find the value and error
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
                logger.debug(f"Skipping observable \"{observable_key}\" because it has only one point.")
                continue

            for design_point, points_to_interpolate in outlier_features_to_interpolate_per_design_point.items():
                # We want to train the interpolation only on all good points, so we make them out.
                # Otherwise, it will negatively impact the interpolation.
                mask = np.ones_like(observable_bin_centers, dtype=bool)
                mask[points_to_interpolate] = False

                # Validation
                if len(observable_bin_centers[mask]) == 1:
                    # Skip - we can't interpolate one point.
                    msg = f"Skipping observable \"{observable_key}\", {design_point=} because it has only one point to anchor the interpolation."
                    logger.info(msg)
                    # And add to the list since we can't make it work.
                    if observable_key not in outliers_we_are_unable_to_remove:
                        outliers_we_are_unable_to_remove[observable_key] = {}
                    if k not in outliers_we_are_unable_to_remove[observable_key]:
                        outliers_we_are_unable_to_remove[observable_key][k] = set()
                    outliers_we_are_unable_to_remove[observable_key][k].update(points_to_interpolate)
                    continue

                # NOTE: ROOT::Interpolator uses a Cubic Spline, so this might be a reasonable future approach
                #       However, I think it's slower, so we'll start with this simple approach.
                # TODO: We entirely ignore the interpolation error here. Some approaches for trying to account for it:
                #       - Attempt to combine the interpolation error with the statistical error
                #       - Randomly remove a few percent of the points which are used for estimating the interpolation,
                #         and then see if there are significant changes in the interpolated parameters
                #       - Could vary some parameters (perhaps following the above) and perform the whole
                #         Bayesian analysis, again looking for how much the determined parameters change.
                if preprocessing_config.smoothing_interpolation_method == "simple":
                    interpolated_values = np.interp(
                        observable_bin_centers[points_to_interpolate],
                        observable_bin_centers[mask],
                        new_observables[prediction_key][observable_key][key_type][:, design_point][mask],
                    )
                elif preprocessing_config.smoothing_interpolation_method == "cubic_spline":
                    cs = scipy.interpolate.CubicSpline(
                        observable_bin_centers[mask],
                        new_observables[prediction_key][observable_key][key_type][:, design_point][mask],
                    )
                    interpolated_values = cs(observable_bin_centers[points_to_interpolate])

                new_observables[prediction_key][observable_key][key_type][points_to_interpolate, design_point] = interpolated_values

    # Reformat the outliers_we_are_unable_to_remove to be more useful and readable
    #logger.info(
    #    f"Observables which we are unable to remove outliers from: {outliers_we_are_unable_to_remove}"
    #)
    # NOTE: The typing is wrong because I based the type annotations on the "Predictions" key only,
    #       since it's more useful here.
    # NOTE: We need to map the i_design_point to the actual design point indices for them to be useful!
    design_point_array: npt.NDArray[np.intp] = all_observables["Design_indices" + ("_validation" if validation_set else "")]  # type: ignore[assignment]
    design_points_we_may_want_to_remove: dict[int, dict[str, set[int]]] = {}
    for observable_key, _v in outliers_we_are_unable_to_remove.items():
        for i_design_point, i_feature in _v.items():
            actual_design_point = design_point_array[i_design_point]
            if actual_design_point not in design_points_we_may_want_to_remove:
                design_points_we_may_want_to_remove[actual_design_point] = {}
            if observable_key not in design_points_we_may_want_to_remove[actual_design_point]:
                design_points_we_may_want_to_remove[actual_design_point][observable_key] = set()
            design_points_we_may_want_to_remove[actual_design_point][observable_key].update(i_feature)
    logger.warning(
        f"Design points which we may want to remove: {sorted(list(design_points_we_may_want_to_remove.keys()))}, length: {len(design_points_we_may_want_to_remove)}"
    )
    logger.info(
        f"In further detail: {design_points_we_may_want_to_remove}"
    )

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


@attrs.define
class PreprocessingConfig(common_base.CommonBase):
    analysis_name: str
    parameterization: str
    config_file: Path = attrs.field(converter=Path)
    analysis_config: dict[str, Any] = attrs.field(factory=dict)
    config: dict[str, Any] = attrs.field(init=False)
    output_dir: Path = attrs.field(init=False)

    def __attrs_post_init__(self):
        with self.config_file.open() as stream:
            self.config = yaml.safe_load(stream)

        # Retrieve parameters from the config
        # Smoothing parameters
        smoothing_parameters = self.analysis_config['parameters']['preprocessing']['smoothing']
        self.smoothing_outliers_config = OutliersConfig(n_RMS=smoothing_parameters["outlier_n_RMS"])
        self.smoothing_interpolation_method = smoothing_parameters["interpolation_method"]
        # Validation
        if self.smoothing_interpolation_method not in ["simple", "cubic_spline"]:
            msg = f"Unrecognized interpolation method {self.smoothing_interpolation_method}."
            raise ValueError(msg)
        self.smoothing_max_n_feature_outliers_to_interpolate = smoothing_parameters["max_n_feature_outliers_to_interpolate"]

        # I/O
        self.output_dir = Path(self.config['output_dir']) / f'{self.analysis_name}_{self.parameterization}'
