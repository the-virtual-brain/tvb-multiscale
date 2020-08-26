# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb_multiscale.examples.tvb_nest.example import main_example, results_path_fun
from tvb_multiscale.tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_multiscale.tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder, WilsonCowanMultisynapseBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww_basal_ganglia_izhikevich import \
    RedWWexcIOBuilder as IzhikevichRedWWexcIOBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder as InterfaceWilsonCowanMultisynapseBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww import RedWWexcIOBuilder, RedWWexcIOinhIBuilder

from tvb_multiscale.tests.core.test_models import model_params_wc, model_params_redww_exc_io, model_params_redww_exc_io_inn_i

from tvb_multiscale.tests.core.test_models import test_models as test_models_base

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


class TestModel(object):

    nest_nodes_ids = []
    nest_model_builder = None
    interface_model_builder = None
    nest_populations_order = 10
    tvb_to_nest_mode = "rate"
    nest_to_tvb = True
    exclusive_nodes = True
    delays_flag = True
    simulation_length = 55.0
    transient = 5.0
    plot_write = True

    def __init__(self, model, nest_nodes_ids, nest_model_builder, interface_model_builder, model_params={}):
        self.model = model
        self.nest_nodes_ids = nest_nodes_ids
        self.nest_model_builder = nest_model_builder
        self.interface_model_builder = interface_model_builder
        self.model_params = model_params

    @property
    def results_path(self):
        return results_path_fun(self.nest_model_builder, self.interface_model_builder,
                                self.tvb_to_nest_mode, self.nest_to_tvb)

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(self.model, self.nest_model_builder, self.interface_model_builder,
                            self.nest_nodes_ids, nest_populations_order=self.nest_populations_order,
                            tvb_to_nest_mode=self.tvb_to_nest_mode, nest_to_tvb=self.nest_to_tvb,
                            exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                            simulation_length=self.simulation_length, transient=self.transient,
                            plot_write=self.plot_write,
                            **self.model_params)


class TestWilsonCowan(TestModel):
    tvb_to_nest_mode = "rate"

    def __init__(self):
        super(TestWilsonCowan, self).__init__(WilsonCowan, [33, 34], WilsonCowanBuilder,
                                              InterfaceWilsonCowanBuilder, model_params_wc)


class TestWilsonCowanMultisynapse(TestModel):
    tvb_to_nest_mode = "rate"

    def __init__(self):
        super(TestWilsonCowanMultisynapse, self).__init__(WilsonCowan, [33, 34], WilsonCowanMultisynapseBuilder,
                                                          InterfaceWilsonCowanMultisynapseBuilder, model_params_wc)


class TestReducedWongWangExcIO(TestModel):

    def __init__(self):
        super(TestReducedWongWangExcIO, self).__init__(ReducedWongWangExcIO, [33, 34], WWDeco2013Builder,
                                                       RedWWexcIOBuilder, model_params_redww_exc_io)

    def run(self):
        for tvb_to_nest_mode in ["param", "current", "rate"]:
            self.tvb_to_nest_mode = tvb_to_nest_mode
            super(TestReducedWongWangExcIO, self).run()


class TestReducedWongWangExcIOinhI(TestModel):

    def __init__(self):
        super(TestReducedWongWangExcIOinhI, self).__init__(ReducedWongWangExcIOInhI, [33, 34], WWDeco2014Builder,
                                                           RedWWexcIOinhIBuilder, model_params_redww_exc_io_inn_i)

    def run(self):
        for tvb_to_nest_mode in ["param", "current", "rate"]:
            self.tvb_to_nest_mode = tvb_to_nest_mode
            super(TestReducedWongWangExcIOinhI, self).run()


class TestIzhikevichRedWWexcIO(TestModel):
    tvb_to_nest_mode = "rate"

    def __init__(self):
        super(TestIzhikevichRedWWexcIO, self).__init__(ReducedWongWangExcIO, list(range(10)),
                                                       BasalGangliaIzhikevichBuilder, IzhikevichRedWWexcIOBuilder, {})


# @pytest.mark.skip(reason="These tests are taking too much time")
def test_models(models_to_test=[TestWilsonCowan, TestWilsonCowanMultisynapse,
                                TestIzhikevichRedWWexcIO,
                                TestReducedWongWangExcIOinhI, TestReducedWongWangExcIO]):
    test_models_base(models_to_test=models_to_test)


if __name__ == "__main__":
    test_models()
