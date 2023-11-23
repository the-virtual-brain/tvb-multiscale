# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

import matplotlib as mpl

mpl.use('Agg')

from examples.tvb_nest.example import default_example
from examples.tvb_nest.models.wilson_cowan import wilson_cowan_example
from examples.tvb_nest.models.red_wong_wang import \
    red_wong_wang_excio_example, red_wong_wang_excio_inhi_example_2013, red_wong_wang_excio_inhi_example_2014
from examples.tvb_nest.models.basal_ganglia_izhiikevich import basal_ganglia_izhikevich_example

from tvb_multiscale.tests.core.test_models import test_models
from tvb_multiscale.tests.core.test_spikeNet_models import TestSpikeNetModel


# TODO: Solve problem with memory garbage to run all tests!!!


class TestDefault(TestSpikeNetModel):

    multisynapse = False

    tvb_to_spikeNet_model = "RATE"
    tvb_to_spikeNet_transformer_model = None
    spikeNet_to_tvb_transformer_model = None

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_model, multisynapse=self.multisynapse,
                        tvb_to_spikeNet_transformer_model=self.tvb_to_spikeNet_transformer_model,
                        spikeNet_to_tvb_model=self.spikeNet_to_tvb_transformer_model,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)


class TestDefaultRATE(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        super(TestDefaultRATE, self).run()


class TestDefaultSPIKES_TO_RATE(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.spikeNet_to_tvb_transformer_model = "SPIKES_TO_RATE"
        super(TestDefaultSPIKES_TO_RATE, self).run()


class TestDefaultSPIKES_TO_HIST(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.spikeNet_to_tvb_transformer_model = "SPIKES_TO_HIST"
        super(TestDefaultSPIKES_TO_HIST, self).run()


class TestDefaultSPIKES(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_model = "SPIKES"
        self.tvb_to_spikeNet_transformer_model = "SPIKES"
        self.run()


class TestDefaultSPIKES_SINGLE_INTERACTION(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_model = "SPIKES"
        self.tvb_to_spikeNet_transformer_model = "SPIKES_SINGLE_INTERACTION"
        self.run()


class TestDefaultSPIKES_MULTIPLE_INTERACTION(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_model = "SPIKES"
        self.tvb_to_spikeNet_transformer_model = "SPIKES_MULTIPLE_INTERACTION"
        self.run()


class TestDefaultMultisynapse(TestSpikeNetModel):

    multisynapse = True

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_model, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)


class TestDefaultMultisynapseRATE(TestDefaultMultisynapse):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestDefaultMultisynapseSPIKES(TestDefaultMultisynapse):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestWilsonCowan(TestSpikeNetModel):

    multisynapse = False

    def run_fun(self):
        wilson_cowan_example(model=self.tvb_to_spikeNet_model, multisynapse=self.multisynapse,
                             spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                             exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                             simulation_length=self.simulation_length, transient=self.transient,
                             plot_write=self.plot_write)


class TestWilsonCowanRATE(TestWilsonCowan):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestWilsonCowanSPIKES(TestWilsonCowan):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestWilsonCowanMultisynapse(TestWilsonCowan):

    multisynapse = True


class TestWilsonCowanMultisynapseRATE(TestWilsonCowanMultisynapse):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestWilsonCowanMultisynapseSPIKES(TestWilsonCowanMultisynapse):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestReducedWongWangExcIO(TestSpikeNetModel):

    def run_fun(self):
        red_wong_wang_excio_example(model=self.tvb_to_spikeNet_mode,
                                    spiking_proxy_inds=self.spiking_proxy_inds,
                                    population_order=self.population_order,
                                    exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                    simulation_length=self.simulation_length, transient=self.transient,
                                    plot_write=self.plot_write)


class TestReducedWongWangExcIORATE(TestReducedWongWangExcIO):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestReducedWongWangExcIOSPIKES(TestReducedWongWangExcIO):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestReducedWongWangExcIOCURRENT(TestReducedWongWangExcIO):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


class TestReducedWongWangExcIOInhI2013(TestSpikeNetModel):

    def run_fun(self):
        red_wong_wang_excio_inhi_example_2013(model=self.tvb_to_spikeNet_mode,
                                              spiking_proxy_inds=self.spiking_proxy_inds,
                                              population_order=self.population_order,
                                              exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                              simulation_length=self.simulation_length, transient=self.transient,
                                              plot_write=self.plot_write)


class TestReducedWongWangExcIOInhI2013RATE(TestReducedWongWangExcIOInhI2013):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestReducedWongWangExcIOInhI2013SPIKES(TestReducedWongWangExcIOInhI2013):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestReducedWongWangExcIOInhI2013CURRENT(TestReducedWongWangExcIOInhI2013):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


class TestReducedWongWangExcIOInhI2014(TestSpikeNetModel):

    def run_fun(self):
        red_wong_wang_excio_inhi_example_2014(model=self.tvb_to_spikeNet_mode,
                                              spiking_proxy_inds=self.spiking_proxy_inds,
                                              population_order=self.population_order,
                                              exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                              simulation_length=self.simulation_length, transient=self.transient,
                                              plot_write=self.plot_write)


class TestReducedWongWangExcIOInhI2014RATE(TestReducedWongWangExcIOInhI2014):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestReducedWongWangExcIOInhI2014SPIKES(TestReducedWongWangExcIOInhI2014):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestReducedWongWangExcIOInhI2014CURRENT(TestReducedWongWangExcIOInhI2014):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


class TestBasalGangliaIzhikevich(TestSpikeNetModel):

    def run_fun(self):
        basal_ganglia_izhikevich_example(model=self.tvb_to_spikeNet_mode,
                                         population_order=self.population_order,
                                         exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                         simulation_length=self.simulation_length, transient=self.transient,
                                         plot_write=self.plot_write)


class TestBasalGangliaIzhikevichRATE(TestBasalGangliaIzhikevich):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()


class TestBasalGangliaIzhikevichSPIKES(TestBasalGangliaIzhikevich):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestBasalGangliaIzhikevichCURRENT(TestBasalGangliaIzhikevich):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


models_to_test_NEST = [
                       TestDefaultRATE,  # 0
                       TestDefaultMultisynapseRATE,  # 1
                       TestWilsonCowanRATE,  # 2
                       TestWilsonCowanMultisynapseRATE,  # 3

                       TestDefaultSPIKES,  # 4
                       TestDefaultMultisynapseSPIKES,  # 5
                       TestWilsonCowanSPIKES,  # 6
                       TestWilsonCowanMultisynapseSPIKES,  # 7

                       TestDefaultSPIKES_SINGLE_INTERACTION,    # 8
                       TestDefaultSPIKES_MULTIPLE_INTERACTION,  # 9
                       TestDefaultSPIKES_TO_RATE,               # 10
                       TestDefaultSPIKES_TO_HIST,               # 11

                       TestBasalGangliaIzhikevichRATE,          # 12
                       TestBasalGangliaIzhikevichSPIKES,        # 13
                       TestBasalGangliaIzhikevichCURRENT,        # 14

                       TestReducedWongWangExcIORATE,
                            TestReducedWongWangExcIOSPIKES,
                                 TestReducedWongWangExcIOCURRENT,

                       TestReducedWongWangExcIOInhI2013RATE,
                           TestReducedWongWangExcIOInhI2013SPIKES,
                               TestReducedWongWangExcIOInhI2013CURRENT,

                       TestReducedWongWangExcIOInhI2014RATE,
                           TestReducedWongWangExcIOInhI2014SPIKES,
                               TestReducedWongWangExcIOInhI2014CURRENT

                       ]


if __name__ == "__main__":
    import sys

    iM = -1
    if len(sys.argv) > 1:
        iM = int(sys.argv[1])
    if iM >= 0:
        print("\n\nTesting model %d" % iM)
    test_models(models_to_test_NEST, iM=iM)
