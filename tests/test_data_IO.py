"""Tests for the data_IO module.

authors: J.Mulligan, R.Ehlers
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from bayesian_inference import data_IO

logger = logging.getLogger(__name__)

_data_dir = Path(__file__).parent / "test_data"

def test_observable_matrix_round_trip():
    """Integration test for observable matrix round trip."""
    # Get observables
    observables = data_IO.read_dict_from_h5(str(_data_dir), 'observables.h5', verbose=False)
    # Get JETSCAPE predictions
    Y = data_IO.predictions_matrix_from_h5(str(_data_dir), filename='observables.h5', validation_set=False)
    # Translate matrix of stacked observables to a dict of matrices per observable
    Y_dict = data_IO.observable_dict_from_matrix(Y, observables, validation_set=False)

    Y_round_trip = data_IO.observable_matrix_from_dict(Y_dict)
    np.testing.assert_allclose(Y, Y_round_trip)
