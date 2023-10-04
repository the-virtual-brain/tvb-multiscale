# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

import matplotlib as mpl

mpl.use('Agg')

from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

from tvb_multiscale.tvb_nest.nest_models.models.default import DefaultExcIOBuilder, DefaultExcIOMultisynapseBuilder
from tvb_multiscale.tvb_nest.nest_models.models.wilson_cowan import WilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder
# from tvb_multiscale.tvb_nest.nest_models.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
# from tvb_multiscale.tvb_nest.nest_models.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.interfaces.models.default import \
    DefaultTVBNESTInterfaceBuilder, DefaultMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import \
    WilsonCowanTVBNESTInterfaceBuilder, WilsonCowanMultisynapseTVBNESTInterfaceBuilder
# from tvb_multiscale.tvb_nest.interfaces.models.red_wong_wang import \
#     RedWongWangExcIOTVBNESTInterfaceBuilder, RedWongWangExcIOInhITVBNESTInterfaceBuilder
# from tvb_multiscale.tvb_nest.interfaces.models.basal_ganglia_izhikevich import \
#     BasalGangliaIzhikevichTVBNESTInterfaceBuilder

from examples.tvb_nest.example import default_example
from examples.tvb_nest.models.wilson_cowan import wilson_cowan_example
from examples.tvb_nest.models.red_wong_wang import \
    red_wong_wang_excio_example, red_wong_wang_excio_inhi_example_2013, red_wong_wang_excio_inhi_example_2014
from examples.tvb_nest.models.basal_ganglia_izhiikevich import basal_ganglia_izhikevich_example

from tvb_multiscale.tests.core.test_models import test_models
from tvb_multiscale.tests.core.test_spikeNet_models import TestSpikeNetModel


# TODO: Solve problem with memory garbage to run all tests!!!


class TestDefault(TestSpikeNetModel):
    # model = Linear()
    # model_params = {}
    # spikeNet_model_builder = DefaultExcIOBuilder()
    # tvb_spikeNet_model_builder = DefaultTVBNESTInterfaceBuilder()
    multisynapse = False

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, population_order=self.population_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)


class TestDefaultRATE(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "RATE"
        super(TestDefaultRATE, self).run()


class TestDefaultSPIKES(TestDefault):

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestDefaultMultisynapse(TestSpikeNetModel):
    # model = Linear()
    # model_params = {}
    # spikeNet_model_builder = DefaultExcIOMultisynapseBuilder()
    # tvb_spikeNet_model_builder = DefaultMultisynapseTVBNESTInterfaceBuilder()
    multisynapse = True

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
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
    # model = WilsonCowan()
    # model_params = {}
    # spikeNet_model_builder = WilsonCowanBuilder()
    # tvb_spikeNet_model_builder = WilsonCowanTVBNESTInterfaceBuilder()
    multisynapse = False

    def run_fun(self):
        wilson_cowan_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
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
    # spikeNet_model_builder = WilsonCowanMultisynapseBuilder()
    # tvb_spikeNet_model_builder = WilsonCowanMultisynapseTVBNESTInterfaceBuilder()
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
#
#
# class TestReducedWongWangExcIO(TestSpikeNetModel):
#
#     # model = ReducedWongWangExcIO()
#     # spikeNet_model_builder = WWDeco2013Builder()
#     # tvb_spikeNet_model_builder = RedWongWangExcIOTVBNESTInterfaceBuilder()
#
#     def run_fun(self):
#         red_wong_wang_excio_example(model=self.tvb_to_spikeNet_mode,
#                                     spiking_proxy_inds=self.spiking_proxy_inds,
#                                     population_order=self.population_order,
#                                     exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
#                                     simulation_length=self.simulation_length, transient=self.transient,
#                                     plot_write=self.plot_write)
#
#
# class TestReducedWongWangExcIORATE(TestReducedWongWangExcIO):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "RATE"
#         self.run()
#
#
# class TestReducedWongWangExcIOSPIKES(TestReducedWongWangExcIO):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()
#
#
# class TestReducedWongWangExcIOCURRENT(TestReducedWongWangExcIO):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "CURRENT"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2013(TestSpikeNetModel):
#
#     # model = ReducedWongWangExcIOInhI()
#     # spikeNet_model_builder = WWDeco2014Builder()
#     # tvb_spikeNet_model_builder = RedWongWangExcIOInhITVBNESTInterfaceBuilder()
#
#     def run_fun(self):
#         red_wong_wang_excio_inhi_example_2013(model=self.tvb_to_spikeNet_mode,
#                                               spiking_proxy_inds=self.spiking_proxy_inds,
#                                               population_order=self.population_order,
#                                               exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
#                                               simulation_length=self.simulation_length, transient=self.transient,
#                                               plot_write=self.plot_write)
#
#
# class TestReducedWongWangExcIOInhI2013RATE(TestReducedWongWangExcIOInhI2013):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "RATE"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2013SPIKES(TestReducedWongWangExcIOInhI2013):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2013CURRENT(TestReducedWongWangExcIOInhI2013):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "CURRENT"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2014(TestSpikeNetModel):
#
#     # model = ReducedWongWangExcIOInhI()
#     # spikeNet_model_builder = WWDeco2014Builder()
#     # tvb_spikeNet_model_builder = RedWongWangExcIOInhITVBNESTInterfaceBuilder()
#
#     def run_fun(self):
#         red_wong_wang_excio_inhi_example_2014(model=self.tvb_to_spikeNet_mode,
#                                               spiking_proxy_inds=self.spiking_proxy_inds,
#                                               population_order=self.population_order,
#                                               exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
#                                               simulation_length=self.simulation_length, transient=self.transient,
#                                               plot_write=self.plot_write)
#
#
# class TestReducedWongWangExcIOInhI2014RATE(TestReducedWongWangExcIOInhI2014):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "RATE"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2014SPIKES(TestReducedWongWangExcIOInhI2014):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()
#
#
# class TestReducedWongWangExcIOInhI2014CURRENT(TestReducedWongWangExcIOInhI2014):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "CURRENT"
#         self.run()
#
#
# class TestBasalGangliaIzhikevich(TestSpikeNetModel):
#     # model = LinearReducedWongWangExcIO()
#     # spikeNet_model_builder = BasalGangliaIzhikevichBuilder()
#     # tvb_spikeNet_model_builder = BasalGangliaIzhikevichTVBNESTInterfaceBuilder()
#     spiking_proxy_inds = np.arange(10).tolist()
#
#     def run_fun(self):
#         basal_ganglia_izhikevich_example(model=self.tvb_to_spikeNet_mode,
#                                          spiking_proxy_inds=self.spiking_proxy_inds,
#                                          population_order=self.population_order,
#                                          exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
#                                          simulation_length=self.simulation_length, transient=self.transient,
#                                          plot_write=self.plot_write)
#
#
# class TestBasalGangliaIzhikevichRATE(TestBasalGangliaIzhikevich):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "RATE"
#         self.run()
#
#
# class TestBasalGangliaIzhikevichSPIKES(TestBasalGangliaIzhikevich):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "SPIKES"
#         self.run()
#
#
# class TestBasalGangliaIzhikevichCURRENT(TestBasalGangliaIzhikevich):
#
#     # @pytest.mark.skip(reason="These tests are taking too much time")
#     def test(self):
#         self.tvb_to_spikeNet_mode = "CURRENT"
#         self.run()

# TODO: Solve error with models 4 and 5!!!


models_to_test_NEST = [TestDefaultRATE,  # 0
                       TestDefaultMultisynapseRATE,  # 1
                       TestWilsonCowanRATE,  # 2
                       TestWilsonCowanMultisynapseRATE,  # 3


                       TestDefaultSPIKES,  # 4
                       TestDefaultMultisynapseSPIKES,  # 5
                       TestWilsonCowanSPIKES,  # 6
                       TestWilsonCowanMultisynapseSPIKES,  # 7
                       #
                       # TestReducedWongWangExcIORATE,
                       #      TestReducedWongWangExcIOSPIKES,
                       #          TestReducedWongWangExcIOCURRENT,
                       #
                       # TestReducedWongWangExcIOInhI2013RATE,
                       #     TestReducedWongWangExcIOInhI2013SPIKES,
                       #         TestReducedWongWangExcIOInhI2013CURRENT,
                       #
                       # TestReducedWongWangExcIOInhI2014RATE,
                       #     TestReducedWongWangExcIOInhI2014SPIKES,
                       #         TestReducedWongWangExcIOInhI2014CURRENT,
                       #
                       #  TestBasalGangliaIzhikevichRATE,
                       #     TestBasalGangliaIzhikevichSPIKES,
                       #         TestBasalGangliaIzhikevichCURRENT
                       ]


if __name__ == "__main__":
    import sys

    iM = -1
    if len(sys.argv) > 1:
        iM = int(sys.argv[1])

    if iM >= 0:
        print("\n\nTesting model %d" % iM)
        test_models(models_to_test_NEST, iM=iM)
    else:
        test_models(models_to_test_NEST, iM=-1)
