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
from tvb_multiscale.tvb_nest.nest_models.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_multiscale.tvb_nest.nest_models.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.interfaces.models.default import \
    DefaultTVBNESTInterfaceBuilder, DefaultMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import \
    WilsonCowanTVBNESTInterfaceBuilder, WilsonCowanMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.red_wong_wang import \
    RedWongWangExcIOTVBNESTInterfaceBuilder, RedWongWangExcIOInhITVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBNESTInterfaceBuilder

from examples.tvb_nest.example import default_example
from examples.tvb_nest.models.wilson_cowan import wilson_cowan_example
from examples.tvb_nest.models.red_wong_wang import red_wong_wang_excio_example, red_wong_wang_excio_inhi_example
from examples.tvb_nest.models.basal_ganglia_izhiikevich import basal_ganglia_izhikevich_example

from tests.core.test_models import loop_all
from tests.core.test_spikeNet_models import TestSpikeNetModel


class TestDefault(TestSpikeNetModel):
    model = Linear
    model_params = {}
    spikeNet_model_builder = DefaultExcIOBuilder
    tvb_spikeNet_model_builder = DefaultTVBNESTInterfaceBuilder
    multisynapse = False

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, populations_order=self.populations_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestDefaultMutisynapse(TestSpikeNetModel):
    model = Linear
    model_params = {}
    spikeNet_model_builder = DefaultExcIOMultisynapseBuilder
    tvb_spikeNet_model_builder = DefaultMultisynapseTVBNESTInterfaceBuilder
    multisynapse = True

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                        spiking_proxy_inds=self.spiking_proxy_inds, populations_order=self.populations_order,
                        exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                        simulation_length=self.simulation_length, transient=self.transient,
                        plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestWilsonCowan(TestSpikeNetModel):
    model = WilsonCowan
    model_params = {}
    spikeNet_model_builder = WilsonCowanBuilder
    tvb_spikeNet_model_builder = WilsonCowanTVBNESTInterfaceBuilder
    multisynapse = False

    def run_fun(self):
        wilson_cowan_example(model=self.tvb_to_spikeNet_mode, multisynapse=self.multisynapse,
                             spiking_proxy_inds=self.spiking_proxy_inds, populations_order=self.populations_order,
                             exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                             simulation_length=self.simulation_length, transient=self.transient,
                             plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestWilsonCowanMultisynapse(TestWilsonCowan):

    spikeNet_model_builder = WilsonCowanMultisynapseBuilder
    tvb_spikeNet_model_builder = WilsonCowanMultisynapseTVBNESTInterfaceBuilder
    multisynapse = True

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()


class TestReducedWongWangExcIO(TestSpikeNetModel):

    model = ReducedWongWangExcIO
    spikeNet_model_builder = WWDeco2013Builder
    tvb_spikeNet_model_builder = RedWongWangExcIOTVBNESTInterfaceBuilder

    def run_fun(self):
        red_wong_wang_excio_example(model=self.tvb_to_spikeNet_mode,
                                    spiking_proxy_inds=self.spiking_proxy_inds, populations_order=self.populations_order,
                                    exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                    simulation_length=self.simulation_length, transient=self.transient,
                                    plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_current(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


class TestReducedWongWangExcIOInhI(TestSpikeNetModel):

    model = ReducedWongWangExcIOInhI
    spikeNet_model_builder = WWDeco2014Builder
    tvb_spikeNet_model_builder = RedWongWangExcIOInhITVBNESTInterfaceBuilder

    def run_fun(self):
        red_wong_wang_excio_inhi_example(model=self.tvb_to_spikeNet_mode,
                                         spiking_proxy_inds=self.spiking_proxy_inds, populations_order=self.populations_order,
                                         exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                         simulation_length=self.simulation_length, transient=self.transient,
                                         plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_current(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


class TestBasalGangliaIzhikevich(TestSpikeNetModel):

    model = LinearReducedWongWangExcIO
    spikeNet_model_builder = BasalGangliaIzhikevichBuilder
    spiking_proxy_inds = np.arange(10).tolist()
    tvb_spikeNet_model_builder = BasalGangliaIzhikevichTVBNESTInterfaceBuilder

    def run_fun(self):
        basal_ganglia_izhikevich_example(model=self.tvb_to_spikeNet_mode,
                                         populations_order=self.populations_order,
                                         exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                                         simulation_length=self.simulation_length, transient=self.transient,
                                         plot_write=self.plot_write)

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.tvb_to_spikeNet_mode = "RATE"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_spikes(self):
        self.tvb_to_spikeNet_mode = "SPIKES"
        self.run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_current(self):
        self.tvb_to_spikeNet_mode = "CURRENT"
        self.run()


if __name__ == "__main__":
    loop_all(models_to_test=[
                             TestDefault, TestDefaultMutisynapse,
                             TestWilsonCowan, TestWilsonCowanMultisynapse,
                             TestBasalGangliaIzhikevich,
                             TestReducedWongWangExcIO, TestReducedWongWangExcIOInhI
    ])
