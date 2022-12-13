#import unittest

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

from tests.core.test_models import test_models
from tests.core.test_spikeNet_models import TestSpikeNetModel

from examples.tvb_netpyne.example import default_example
from examples.tvb_netpyne.models.wilson_cowan import wilson_cowan_example
from examples.tvb_netpyne.models.red_wong_wang import excio_inhi_example

class TestDefault(TestSpikeNetModel):
#    multisynapse = False
    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)

class TestDefaultRATE(TestDefault):
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

class TestWilsonCowan(TestSpikeNetModel):
#    multisynapse = False
    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)

class TestWilsonCowanRATE(TestDefault):
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

models_to_test_netpyne = [
    TestDefaultRATE,
    TestWilsonCowanRATE
]
#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    test_models(models_to_test_netpyne)

#    unittest.main()
