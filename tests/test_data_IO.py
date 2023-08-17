"""Tests for the data_IO module.

authors: J.Mulligan, R.Ehlers
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_inference import data_IO

logger = logging.getLogger(__name__)

_data_dir = Path(__file__).parent / "test_data"

def test_observable_matrix_round_trip(caplog: Any) -> None:
    """Integration test for observable matrix round trip."""
    # Setup
    caplog.set_level(logging.DEBUG)

    # Get observables
    observables = data_IO.read_dict_from_h5(str(_data_dir), 'observables.h5', verbose=False)
    # Get JETSCAPE predictions
    Y = data_IO.predictions_matrix_from_h5(str(_data_dir), filename='observables.h5', validation_set=False)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.observable_dict_from_matrix(Y, observables, validation_set=False)

    Y_round_trip = data_IO.observable_matrix_from_dict(Y_dict)
    np.testing.assert_allclose(Y, Y_round_trip)

@pytest.mark.parametrize(
    "design_points_to_exclude",
    [[17, 43, 203], []],
    ids=["exclude", "no_exclude"],
)
@pytest.mark.parametrize(
    "parameterization",
    ["test1", "test2"],
    ids=["continuous", "discontinuous"],
)
def test_exclude_design_points(caplog: Any, design_points_to_exclude: list[int], parameterization: str) -> None:
    """ Test excluding design point indices.

    We require a bit of care here to ensure that we don't confuse indices vs design points.
    Here, continuous vs discontinuous refers to whether the stored designed points are continuous
    (ie all 0-229) or discontinuous (ie 0-229 with some indices missing).
    """
    # Setup
    caplog.set_level(logging.DEBUG)
    excluded_values = {
        i: list(range(i * 6, i * 6 + 6))
        for i in design_points_to_exclude
    }

    read_design_point_parameters = np.loadtxt(_data_dir / "tables" / "Design" / f"Design__{parameterization}.dat", ndmin=2)

    # This check is designed to simulate if some design points are missing in the output.
    # NOTE: Remember that this is **separate** from the points that we'll exclude based on excluded_values
    # NOTE: This depends on how we prepared the file. I just chose removing 2 arbitrarily
    n_points_missing = 0 if "test1" in parameterization else 2
    assert read_design_point_parameters.shape == (230 - n_points_missing, 6)

    # NOTE: This extracts the design points and tries to use them as indices, but this isn't so trivial
    #       because we may be missing some design points. Thus, they aren't indices.
    design_points = data_IO._read_design_points_from_design_dat(_data_dir / "tables", parameterization)
    training_indices, training_design_points, validation_indices, validation_design_points = data_IO._split_training_validation_indices(
        design_points=design_points,
        validation_indices=list(range(200, 230)),
        design_points_to_exclude=design_points_to_exclude
    )

    # Determine the design point parameters for the training and validation sets
    design_points_parameters = read_design_point_parameters[training_indices]
    design_points_parameters_validation = read_design_point_parameters[validation_indices]

    # Check shape
    excluded_values_in_main_points = [i for i in design_points_to_exclude if i < 200]
    excluded_values_in_validation_points = [i for i in design_points_to_exclude if i >= 200]
    assert design_points_parameters.shape == (200 - len(excluded_values_in_main_points) - n_points_missing, 6)
    assert design_points_parameters_validation.shape == (30 - len(excluded_values_in_validation_points), 6)

    # Check that excluded values are not present
    for excluded_point, values in excluded_values.items():
        assert excluded_point not in training_design_points
        assert excluded_point not in validation_design_points

        assert values not in design_points_parameters
        assert values not in design_points_parameters_validation
