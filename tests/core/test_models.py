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

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI

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


class TestModel(object):
    use_numba = True
    simulation_length = 55.0
    transient = 5.0
    model = None
    model_params = {}
    results_path = ""
    plot_write = True

    def __init__(self, model, model_params={}):
        self.model = model
        self.model_params = model_params

    @property
    def results_path(self):
        return results_path_fun(self.model)

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(tvb_sim_model=self.model,
                            simulation_length=self.simulation_length, transient=self.transient,
                            use_numba=self.use_numba, plot_write=self.plot_write,
                            **self.model_params)


class TestWilsonCowan(TestModel):
    model = WilsonCowan

    def __init__(self):
        super(TestWilsonCowan, self).__init__(WilsonCowan, model_params_wc)


class TestReducedWongWangExcIO(TestModel):
    model = ReducedWongWangExcIO

    def __init__(self):
        super(TestReducedWongWangExcIO, self).__init__(ReducedWongWangExcIO, model_params_redww_exc_io)


class TestReducedWongWangExcIOInhI(TestModel):
    model = ReducedWongWangExcIOInhI

    def __init__(self):
        super(TestReducedWongWangExcIOInhI, self).__init__(ReducedWongWangExcIOInhI, model_params_redww_exc_io_inn_i)


class TestSpikingWongWangExcIOInhI(TestModel):
    model = SpikingWongWangExcIOInhI

    def __init__(self):
        super(TestSpikingWongWangExcIOInhI, self).__init__(SpikingWongWangExcIOInhI, model_params_sp)


# @pytest.mark.skip(reason="These tests are taking too much time") # only TestSpikingWongWangExcIOInhI takes time
def test_models(models_to_test=[TestWilsonCowan,
                                TestReducedWongWangExcIO,
                                TestReducedWongWangExcIOInhI], # , TestSpikingWongWangExcIOInhI
                use_numba=True):
    import time
    import numpy as np
    from collections import OrderedDict
    success = OrderedDict()
    for test_model_class in models_to_test:
        test_model = test_model_class()
        test_model.use_numba = use_numba
        print("\n******************************************************")
        print("******************************************************")
        print(test_model_class.__name__)
        print("******************************************************")
        try:
            tic = time.time()
            test_model.run()
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
    return success

def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    print("************Testing with use_numba=False!*************\n")
    test_models(use_numba=False)
    print("************Testing with use_numba=True!**************\n")
    test_models(use_numba=True)
