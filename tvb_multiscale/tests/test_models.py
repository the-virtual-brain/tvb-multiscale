# -*- coding: utf-8 -*-
import os

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import numpy as np

from tvb_multiscale.examples.simulate_tvb_only import main_example

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely



class TestModel(object):

    simulation_length = 110.0
    transient = 10.0
    model = None

    model_params = {}

    results_path = ""

    def __init__(self, model, model_params={}):
        self.model = model
        self.model_params = model_params
        self.results_path = os.path.join(os.getcwd(), "outputs", self.model.__name__)

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(tvb_sim_model=self.model,
                            simulation_length=self.simulation_length, transient=self.transient,
                            **self.model_params)


# -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

model_params_wc = {
        "r_e": np.array([0.0]),
        "r_i": np.array([0.0]),
        "k_e": np.array([1.0]),
        "k_i": np.array([1.0]),
        "tau_e": np.array([10.0]),
        "tau_i": np.array([10.0]),
        "c_ee": np.array([10.0]),
        "c_ei": np.array([6.0]),
        "c_ie": np.array([10.0]),
        "c_ii": np.array([1.0]),
        "alpha_e": np.array([1.2]),
        "alpha_i": np.array([2.0]),
        "a_e": np.array([1.0]),
        "a_i": np.array([1.0]),
        "b_e": np.array([0.0]),
        "b_i": np.array([0.0]),
        "c_e": np.array([1.0]),
        "c_i": np.array([1.0]),
        "theta_e": np.array([2.0]),
        "theta_i": np.array([3.5]),
        "P": np.array([0.5]),
        "Q": np.array([0.0])
    }

model_params_redww_exc_io = {"G": np.array([20.0, ])}

model_params_redww_exc_io_inn_i = {"G": np.array([20.0, ]), "lamda": np.array([0.5, ])}

# ----------------------------------------SpikingWongWangExcIOInhI/MultiscaleWongWangExcIOInhI----------------------

model_params_sp = {
        "N_E": np.array([16, ]),
        "N_I": np.array([4, ]),
        "w_IE": np.array([1.0, ]),
        "lamda": np.array([0.5, ]),
        "G": np.array([200.0, ])
    }


if __name__ == "__main__":

    for model, model_params in zip([WilsonCowan,
                                    ReducedWongWangExcIO,
                                    ReducedWongWangExcIOInhI,
                                    SpikingWongWangExcIOInhI],
                                   [model_params_wc,
                                    model_params_redww_exc_io,
                                    model_params_redww_exc_io_inn_i,
                                    model_params_sp]
                                   ):
        test_model = TestModel(model, model_params)
        print(test_model.run())
        del test_model
