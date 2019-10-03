# -*- coding: utf-8 -*-

import os
import numpy as np

from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh as TVBReducedWongWangExcIOInhI

from tvb_nest.tests.rww_dfun import compute_rww_dfun

TESTS_PATH = os.path.dirname(os.path.realpath(__file__))


def reduced_wong_wang_exc_io_inh_i(D, N, abs_err, tvb_model):

    exc = 1.1 - 1.2*np.random.uniform(size=(D, N))
    inh = 1.1 - 1.2*np.random.uniform(size=(D, N))

    state = np.array([exc, inh])
    coupling = exc[np.newaxis]

    computed_output = compute_rww_dfun(exc, inh, exc)
    computed_dstate = np.array([computed_output[0], computed_output[1]])

    numpy_dstate = tvb_model._numpy_dfun(state, coupling)
    max_difference = np.max(np.abs(numpy_dstate - computed_dstate))
    assert computed_dstate.shape == (2, 100, 100)
    assert max_difference < abs_err

    for ii in range(N):
        this_state = (state[:, :, ii])[:, :, np.newaxis]
        this_coupling = (coupling[:, :, ii])[:, :, np.newaxis]

        numba_dstate = (tvb_model.dfun(this_state, this_coupling)).squeeze()

        max_difference = np.max(np.abs(numba_dstate - computed_dstate[:, :, ii]))
        assert max_difference < abs_err

        max_difference = np.max(np.abs(numba_dstate - numpy_dstate[:, :, ii]))
        assert max_difference < abs_err

def test_reduced_wong_wang_exc_io_inh_i_internal():
    reduced_wong_wang_exc_io_inh_i(100, 100, 2*1e-1, ReducedWongWangExcIOInhI())

def test_reduced_wong_wang_exc_io_inh_i_external():
    reduced_wong_wang_exc_io_inh_i(100, 100, 2*1e-1, TVBReducedWongWangExcIOInhI())















