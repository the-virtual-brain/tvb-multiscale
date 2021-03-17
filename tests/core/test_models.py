# -*- coding: utf-8 -*-
import os
import shutil

import gc
from time import sleep
from tvb.basic.profile import TvbProfile

from tvb_multiscale.core.config import Config
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import numpy as np

from examples.simulate_tvb_only import main_example, results_path_fun

from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear

# from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


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

model_params_redww_exc_io = {"G": np.array([2.0, ])}

model_params_redww_exc_io_inn_i = {"G": np.array([2.0, ]), "lamda": np.array([0.5, ])}

# ----------------------------------------SpikingWongWangExcIOInhI/MultiscaleWongWangExcIOInhI----------------------

model_params_sp = {
    "N_E": np.array([16, ]),
    "N_I": np.array([4, ]),
    "w_IE": np.array([1.0, ]),
    "lamda": np.array([0.5, ]),
    "G": np.array([200.0, ])
}


class TestModel(object):
    simulation_length = 36.0
    transient = 6.0
    model = None
    model_params = {}
    plot_write = False

    @property
    def results_path(self):
        return results_path_fun(self.model)

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(tvb_sim_model=self.model,
                            simulation_length=self.simulation_length, transient=self.transient,
                            plot_write=self.plot_write, **self.model_params)


class TestLinear(TestModel):
    model = Linear

    model_params = {}

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestWilsonCowan(TestModel):
    model = WilsonCowan

    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
    model_params = model_params_wc

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestLinearReducedWongWangExcIO(TestModel):
    model = LinearReducedWongWangExcIO
    model_params = model_params_redww_exc_io

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestReducedWongWangExcIO(TestModel):
    model = ReducedWongWangExcIO
    model_params = model_params_redww_exc_io

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestReducedWongWangExcIOInhI(TestModel):
    model = ReducedWongWangExcIOInhI
    model_params = model_params_redww_exc_io_inn_i

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


# class TestSpikingWongWangExcIOInhI(TestModel):
#     model = SpikingWongWangExcIOInhI
#     model_params = model_params_sp
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.run()


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def loop_all(models_to_test=[TestLinear, TestWilsonCowan,
                             TestLinearReducedWongWangExcIO, TestReducedWongWangExcIO, TestReducedWongWangExcIOInhI]):
    import time
    import numpy as np
    from collections import OrderedDict
    success = OrderedDict()
    for test_model_class in models_to_test:
        test_model = test_model_class()
        print("\n******************************************************")
        print("******************************************************")
        print(test_model_class.__name__)
        for test in dir(test_model):
            if test[:4] == "test":
                print("******************************************************")
                print(test)
                print("******************************************************")
                try:
                    tic = time.time()
                    getattr(test_model, test)()
                    print("\nSuccess in time %f sec!" % (time.time() - tic))
                    success[test_model_class.__name__] = True
                except Exception as e:
                    success[test_model_class.__name__] = e
                    print("\nError after time %f sec!" % (time.time() - tic))
                del test_model
                gc.collect()
                print("******************************************************\n")
                sleep(5)
        print("\n******************************************************")
        print("******************************************************")
    if not np.all([result is True for result in list(success.values())]):
        raise Exception("%s\nmodels' tests failed! Details:\n%s" % (str(os.getcwd()), str(success)))
    else:
        print(success)
    print("******************************************************\n")


if __name__ == "__main__":
    loop_all()
