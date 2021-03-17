# -*- coding: utf-8 -*-
import os
import shutil

from tvb.basic.profile import TvbProfile
from tvb_multiscale.core.config import Config
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from examples.tvb_nest.example import main_example, results_path_fun
from tvb_multiscale.tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_multiscale.tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww_basal_ganglia_izhikevich import \
    RedWWexcIOBuilder as IzhikevichRedWWexcIOBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder as InterfaceWilsonCowanMultisynapseBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww import RedWWexcIOBuilder, RedWWexcIOinhIBuilder

from tests.core.test_models import model_params_wc, model_params_redww_exc_io, model_params_redww_exc_io_inn_i

from tests.core.test_models import loop_all, TestModel

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


class TestModelNEST(TestModel):
    nest_nodes_ids = []
    nest_model_builder = None
    interface_model_builder = None
    nest_populations_order = 10
    tvb_to_nest_mode = "rate"
    nest_to_tvb = True
    exclusive_nodes = True
    delays_flag = True

    @property
    def results_path(self):
        return results_path_fun(self.nest_model_builder, self.interface_model_builder,
                                self.tvb_to_nest_mode, self.nest_to_tvb)

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(self.model, self.nest_model_builder, self.interface_model_builder,
                            self.nest_nodes_ids, populations_order=self.nest_populations_order,
                            tvb_to_nest_mode=self.tvb_to_nest_mode, nest_to_tvb=self.nest_to_tvb,
                            exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                            simulation_length=self.simulation_length, transient=self.transient,
                            plot_write=self.plot_write,
                            **self.model_params)


class TestWilsonCowan(TestModelNEST):
    model = WilsonCowan
    model_params = model_params_wc
    nest_nodes_ids = [33, 34]
    nest_model_builder = WilsonCowanBuilder
    interface_model_builder = InterfaceWilsonCowanBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestWilsonCowanMultisynapse(TestWilsonCowan):
    nest_model_builder = WilsonCowanMultisynapseBuilder
    interface_model_builder = InterfaceWilsonCowanMultisynapseBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestReducedWongWangExcIO(TestModelNEST):
    model = ReducedWongWangExcIO
    model_params = model_params_redww_exc_io
    nest_nodes_ids = [33, 34]
    nest_model_builder = WWDeco2013Builder
    interface_model_builder = RedWWexcIOBuilder

    def run(self, interfaces=["param", "current", "rate"]):
        for tvb_to_nest_mode in interfaces:
            self.tvb_to_nest_mode = tvb_to_nest_mode
            super(TestReducedWongWangExcIO, self).run()

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.run(interfaces=["rate"])

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_current(self):
        self.run(interfaces=["current"])

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_param(self):
        self.run(interfaces=["param"])


class TestReducedWongWangExcIOinhI(TestReducedWongWangExcIO):
    model = ReducedWongWangExcIOInhI
    model_params = model_params_redww_exc_io_inn_i
    nest_model_builder = WWDeco2014Builder
    interface_model_builder = RedWWexcIOinhIBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_rate(self):
        self.run(interfaces=["rate"])

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_current(self):
        self.run(interfaces=["current"])

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test_param(self):
        self.run(interfaces=["param"])


class TestIzhikevichRedWWexcIO(TestModel):
    model = ReducedWongWangExcIO
    nest_nodes_ids = list(range(10))
    nest_model_builder = BasalGangliaIzhikevichBuilder
    interface_model_builder = IzhikevichRedWWexcIOBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    loop_all(models_to_test=[TestWilsonCowan, TestWilsonCowanMultisynapse,
                             TestIzhikevichRedWWexcIO,
                             TestReducedWongWangExcIOinhI, TestReducedWongWangExcIO])
