# -*- coding: utf-8 -*-

import os
from oct2py import octave
import numpy as np
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh as TVBReducedWongWangExcIOInhI


TESTS_PATH = os.path.dirname(os.path.realpath(__file__))
octave.addpath(os.path.join(TESTS_PATH, "DMF2014"))


def reduced_wong_wang_exc_io_inh_i(D, N, abs_err, tvb_model):

    exc = 1.1 - 1.2*np.random.uniform(size=(D, N))
    inh = 1.1 - 1.2*np.random.uniform(size=(D, N))

    state = np.array([exc, inh])
    coupling = exc[np.newaxis]

    matlab_output = octave.matlab_rww_dfun(exc, inh, exc)
    matlab_dstate = np.array([matlab_output[0, 0], matlab_output[0, 1]])

    numpy_dstate = tvb_model._numpy_dfun(state, coupling)
    max_difference = np.max(np.abs(numpy_dstate - matlab_dstate))
    assert matlab_dstate.shape == (2, 100, 100)
    assert max_difference < abs_err

    for ii in range(N):
        this_state = (state[:, :, ii])[:, :, np.newaxis]
        this_coupling = (coupling[:, :, ii])[:, :, np.newaxis]

        numba_dstate = (tvb_model.dfun(this_state, this_coupling)).squeeze()

        max_difference = np.max(np.abs(numba_dstate - matlab_dstate[:, :, ii]))
        assert max_difference < abs_err

        max_difference = np.max(np.abs(numba_dstate - numpy_dstate[:, :, ii]))
        assert max_difference < abs_err


def test_reduced_wong_wang_exc_io_inh_i_internal():
    reduced_wong_wang_exc_io_inh_i(100, 100, 1e-12, ReducedWongWangExcIOInhI())


def test_reduced_wong_wang_exc_io_inh_i_external():
    reduced_wong_wang_exc_io_inh_i(100, 100, 1e-12, TVBReducedWongWangExcIOInhI())
    octave.exit()











