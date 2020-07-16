# -*- coding: utf-8 -*-
import os
import gc
import time

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb_nest.examples.example import main_example
from tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_nest.nest_models.builders.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder, WilsonCownMultisynapseBuilder
from tvb_nest.interfaces.builders.models.red_ww_basal_ganglia_izhikevich import \
    RedWWexcIOBuilder as IzhikevichRedWWexcIOBuilder
from tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder as InterfaceWilsonCowanMultisynapseBuilder
from tvb_nest.interfaces.builders.models.red_ww import RedWWexcIOBuilder, RedWWexcIOinhIBuilder

from tvb_multiscale.tests.test_models import TestModel as TestModelBase
from tvb_multiscale.tests.test_models import model_params_wc, model_params_redww_exc_io, model_params_redww_exc_io_inn_i

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


class TestModel(TestModelBase):

    nest_nodes_ids = []
    nest_model_builder = None
    interface_model_builder = None
    nest_populations_order = 100
    tvb_to_nest_mode = "rate"
    nest_to_tvb = True
    exclusive_nodes = True
    delays_flag = True
    simulation_length = 110.0
    transient = 10.0

    def __init__(self, model, nest_nodes_ids, nest_model_builder, interface_model_builder, model_params={}):
        super(TestModel, self).__init__(model, model_params)
        self.nest_nodes_ids = nest_nodes_ids
        self.nest_model_builder = nest_model_builder
        self.interface_model_builder = interface_model_builder
        self.results_path = os.path.join(os.getcwd(), "outputs",
                                         self.interface_model_builder.__name__.split("Builder")[0])

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(self.model, self.nest_model_builder, self.interface_model_builder,
                            self.nest_nodes_ids, nest_populations_order=self.nest_populations_order,
                            tvb_to_nest_mode=self.tvb_to_nest_mode, nest_to_tvb=self.nest_to_tvb,
                            exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                            simulation_length=self.simulation_length, transient=self.transient,
                            **self.model_params)


def test_models():
    for model, model_params, nest_model_builder, interface_model_builder \
            in zip([# WilsonCowan,
                    # WilsonCowan,
                    ReducedWongWangExcIO,
                    ReducedWongWangExcIOInhI,
                    ReducedWongWangExcIO],
                    [# model_params_wc,
                     # model_params_wc,
                     model_params_redww_exc_io,
                     model_params_redww_exc_io_inn_i,
                     {}],
                    [# WilsonCowanBuilder,
                     # WilsonCownMultisynapseBuilder,
                     WWDeco2013Builder,
                     WWDeco2014Builder,
                     BasalGangliaIzhikevichBuilder],
                   [# InterfaceWilsonCowanBuilder,
                    # InterfaceWilsonCowanMultisynapseBuilder,
                    RedWWexcIOBuilder,
                    RedWWexcIOinhIBuilder,
                    IzhikevichRedWWexcIOBuilder]
                  ):
        if interface_model_builder == IzhikevichRedWWexcIOBuilder:
            nest_nodes_ids = list(range(10))
        else:
            nest_nodes_ids = [33, 34]

        test_model = TestModel(model, nest_nodes_ids, nest_model_builder, interface_model_builder, model_params)

        if interface_model_builder in [RedWWexcIOBuilder,
                                       RedWWexcIOinhIBuilder]:
            for tvb_to_nest_mode in ["param", "current", "rate"]:
                test_model.tvb_to_nest_mode = tvb_to_nest_mode
                print(test_model.run())
                del test_model
                gc.collect()
                time.sleep(5)
        else:
            print(test_model.run())
            del test_model
            gc.collect()
            time.sleep(5)


if __name__ == "__main__":
    test_models()
