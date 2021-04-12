# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan

from tvb_multiscale.tvb_annarchy.annarchy_models.models.default import DefaultExcIOBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models.default import DefaultTVBANNarchyInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models.wilson_cowan import WilsonCowanTVBANNarchyInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder

from examples.tvb_annarchy.example import default_example
from examples.tvb_annarchy.models.wilson_cowan import wilson_cowan_example
from examples.tvb_annarchy.models.basal_ganglia_izhikevich import basal_ganglia_izhikevich_example

from tests.core.test_models import loop_all
from tests.core.test_spikeNet_models import TestSpikeNetModel


class TestDefault(TestSpikeNetModel):
    model = Linear
    model_params = {}
    spikeNet_model_builder = DefaultExcIOBuilder
    tvb_spikeNet_model_builder = DefaultTVBANNarchyInterfaceBuilder
    multisynapse = False

    def run_fun(self):
        default_example(model=self.tvb_to_spikeNet_mode,
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
    tvb_spikeNet_model_builder = WilsonCowanTVBANNarchyInterfaceBuilder
    multisynapse = False

    def run_fun(self):
        wilson_cowan_example(model=self.tvb_to_spikeNet_mode,
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


class TestBasalGangliaIzhikevich(TestSpikeNetModel):

    model = LinearReducedWongWangExcIO
    spikeNet_model_builder = BasalGangliaIzhikevichBuilder
    spiking_proxy_inds = np.arange(10).tolist()
    tvb_spikeNet_model_builder = BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder

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


def test_models(models_to_test=[
                             TestDefault,
                             TestWilsonCowan,
                             TestBasalGangliaIzhikevich
                               ]):
    loop_all(models_to_test)


if __name__ == "__main__":
    test_models()
