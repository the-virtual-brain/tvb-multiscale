# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear

from tvb_multiscale.core.config import CONFIGURED
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import \
    serialize_tvb_cosimulator, dump_serial_tvb_cosimulator, load_serial_tvb_cosimulator


def test_simulator_serialization(test_models=[Linear, WilsonCowan, LinearReducedWongWangExcIO,
                                              ReducedWongWangExcIO, ReducedWongWangExcIOInhI]):
    for test_model in test_models:
        simulator_builder = CoSimulatorBuilder()
        simulator_builder.connectivity = CONFIGURED.DEFAULT_CONNECTIVITY_ZIP
        simulator_builder.model = test_model
        simulator = simulator_builder.build()
        serial_sim = serialize_tvb_cosimulator(simulator)
        filepath = os.path.join(CONFIGURED.out.FOLDER_RES, serial_sim["model"]+".pkl")
        dump_serial_tvb_cosimulator(filepath)
        serial_sim2 = load_serial_tvb_cosimulator()
        for key, val in serial_sim.items():
            assert np.all(serial_sim2[key] == val)







