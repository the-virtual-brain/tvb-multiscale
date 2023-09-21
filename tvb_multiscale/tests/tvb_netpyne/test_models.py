from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

from tvb_multiscale.tests.core.test_models import test_models
from tvb_multiscale.tests.core.test_spikeNet_models import TestSpikeNetModel

from examples.tvb_netpyne.example import default_example
from examples.tvb_netpyne.models.wilson_cowan import wilson_cowan_example
from examples.tvb_netpyne.models.red_wong_wang import excio_inhi_example


class TestDefault(TestSpikeNetModel):

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)


class TestDefaultRATE(TestDefault):
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


# class TestDefaultSPIKES(TestDefault):
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()


class TestWilsonCowan(TestSpikeNetModel):

    def run_fun(self):
        wilson_cowan_example(model=self.tvb_to_spikeNet_mode,
                             spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                             exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                             simulation_length=self.simulation_length, transient=self.transient,
                             plot_write=self.plot_write)


class TestWilsonCowanRATE(TestWilsonCowan):
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


# class TestWilsonCowanSPIKES(TestWilsonCowan):
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()


class TestRedWongWang(TestSpikeNetModel):
    def run_fun(self):
        excio_inhi_example(model=self.tvb_to_spikeNet_mode,
                           spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                           exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                           simulation_length=self.simulation_length, transient=self.transient,
                           plot_write=self.plot_write)


class TestRedWongWangRATE(TestRedWongWang):
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


models_to_test_netpyne = [
    TestDefaultRATE,
    # TestDefaultSPIKES, # Not implemented yet
    TestWilsonCowanRATE,
    # TestWilsonCowanSPIKES, # Not implemented yet
    TestRedWongWangRATE
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        iM = int(sys.argv[1])
        print("\n\nTesting model %d" % iM)
        test_models(models_to_test_netpyne, iM=iM)
    else:
        test_models(models_to_test_netpyne, iM=-1)
