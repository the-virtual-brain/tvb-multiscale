# -*- coding: utf-8 -*-

import os
from oct2py import octave
import numpy as np
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh as TVBReducedWongWangExcIOInhI

TESTS_PATH = os.path.dirname(os.path.realpath(__file__))
octave.addpath(os.path.join(TESTS_PATH, "DMF2014"))


def reduced_wong_wang_exc_io_inh_i(N, abs_err, tvb_model, lamda):
    exc = 1.0 * np.random.uniform(size=(N, 1))
    inh = 1.0 * np.random.uniform(size=(N, 1))

    state = np.array([exc, inh])
    coupling = exc[np.newaxis]

    matlab_output = octave.matlab_rww_dfun(exc, inh, exc, lamda)
    matlab_dstate = np.array([matlab_output[0, 0], matlab_output[0, 1]]).squeeze()

    numpy_dstate = tvb_model._numpy_dfun(state, coupling).squeeze()
    max_difference = np.max(np.abs(numpy_dstate - matlab_dstate))
    assert matlab_dstate.shape == (2, N)
    assert max_difference < abs_err

    numba_dstate = tvb_model.dfun(state, coupling).squeeze()
    max_difference = np.max(np.abs(numpy_dstate - numba_dstate))
    assert max_difference < abs_err
    max_difference = np.max(np.abs(numba_dstate - matlab_dstate))
    assert max_difference < abs_err


def test_reduced_wong_wang_exc_io_inh_i_internal():
    reduced_wong_wang_exc_io_inh_i(100, 1e-12, ReducedWongWangExcIOInhI(), 1.0)


def test_reduced_wong_wang_exc_io_inh_i_external():
    reduced_wong_wang_exc_io_inh_i(100, 1e-12, TVBReducedWongWangExcIOInhI(), 0.0)
    octave.exit()


if __name__ == "__main__":
    test_reduced_wong_wang_exc_io_inh_i_internal()
    test_reduced_wong_wang_exc_io_inh_i_external()


