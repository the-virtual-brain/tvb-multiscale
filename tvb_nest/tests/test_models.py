# -*- coding: utf-8 -*-
import os
import gc
from time import sleep

import pytest
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb_nest.examples.example import main_example
from tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_nest.nest_models.builders.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder, WilsonCowanMultisynapseBuilder
from tvb_nest.interfaces.builders.models.red_ww_basal_ganglia_izhikevich import \
    RedWWexcIOBuilder as IzhikevichRedWWexcIOBuilder
from tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder as InterfaceWilsonCowanMultisynapseBuilder
from tvb_nest.interfaces.builders.models.red_ww import RedWWexcIOBuilder, RedWWexcIOinhIBuilder

from tvb_multiscale.tests.test_models import model_params_wc, model_params_redww_exc_io, model_params_redww_exc_io_inn_i

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

    def __init__(self, model, nest_nodes_ids, nest_model_builder, interface_model_builder, model_params={}):
        self.model = model
        self.nest_nodes_ids = nest_nodes_ids
        self.nest_model_builder = nest_model_builder
        self.interface_model_builder = interface_model_builder
        self.results_path = os.path.join(os.getcwd(), "outputs",
                                         self.interface_model_builder.__name__.split("Builder")[0])
        self.model_params = model_params

    def run(self):
        delete_folder_safely(self.results_path)
        return main_example(self.model, self.nest_model_builder, self.interface_model_builder,
                            self.nest_nodes_ids, nest_populations_order=self.nest_populations_order,
                            tvb_to_nest_mode=self.tvb_to_nest_mode, nest_to_tvb=self.nest_to_tvb,
                            exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                            simulation_length=self.simulation_length, transient=self.transient,
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


@pytest.mark.skip(reason="These tests are taking too much time")
def test_models():
    import numpy as np
    from collections import OrderedDict
    # TODO: find out why it fails if I run first the WilsonCowan tests and then the ReducedWongWang ones...
    success = OrderedDict()
    for test_model_class in [TestReducedWongWangExcIOinhI, TestReducedWongWangExcIO,
                             TestWilsonCowan, TestWilsonCowanMultisynapse,
                             TestIzhikevichRedWWexcIO]:
        test_model = test_model_class()
        try:
            print(test_model.run())
            success[test_model_class.__name__] = True
        except Exception as e:
            success[test_model_class.__name__] = e
        del test_model
        gc.collect()
        sleep(5)
    if not np.all([result is True for result in list(success.values())]):
        print(success)
        raise Exception("Test models failed! Details: \n %s" % str(success))


if __name__ == "__main__":
    test_models()
