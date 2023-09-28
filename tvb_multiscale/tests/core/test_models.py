# -*- coding: utf-8 -*-
import os
import shutil
import gc
from time import sleep, time
from collections import OrderedDict
import warnings
import numpy as np

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


def run_test(test_model_class, success={}):
    test_model = test_model_class()
    print("******************************************************")
    print("******************************************************")
    try:
        tic = time()
        test_model.test()
        print("\nSuccess in %g sec!" % (time() - tic))
        success[test_model_class.__name__] = True
    except Exception as e:
        raise e
        # success[test_model_class.__name__] = str(e)
        # print("\nError in %g sec!" % (time() - tic))
        # warnings.warn(e)
    print("******************************************************\n")
    del test_model
    gc.collect()
    sleep(1)
    return success


def loop_all(models_to_test=[]):
    success = OrderedDict()
    for test_model_class in models_to_test:
        print("\n******************************************************")
        print("******************************************************")
        print(test_model_class.__name__)
        success = run_test(test_model_class, success)
        print("\n******************************************************")
        print("******************************************************")
    for model, result in success.items():
        if result is True:
            print("\n%s SUCCESS!" % model)
        else:
            warnings.warn("\n%s ERROR!:\n%s" % (model, result))
    if not np.all([result is True for result in list(success.values())]):
        raise Exception("%s\nmodels' tests failed!" % str(os.getcwd()))
    print("******************************************************\n")


models_to_test_TVB = [TestLinear,                       # 0
                      TestWilsonCowan,                  # 1
                      TestLinearReducedWongWangExcIO,   # 2
                      TestReducedWongWangExcIO,         # 3
                      TestReducedWongWangExcIOInhI]     # 4


def test_models(models_to_test=models_to_test_TVB, iM=0):
    if iM >= 0:
        print(run_test(models_to_test[iM]))
    else:
        loop_all(models_to_test)


if __name__ == "__main__":
    import sys

    iM = -1
    if len(sys.argv) > 1:
        iM = int(sys.argv[1])

    if iM >= 0:
        print("\n\nTesting model %d" % iM)
        test_models(iM=iM)
    else:
        test_models(iM=-1)
