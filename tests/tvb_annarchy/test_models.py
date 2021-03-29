# -*- coding: utf-8 -*-
import os
import shutil

from tvb.basic.profile import TvbProfile
from tvb_multiscale.core.config import Config
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from examples.tvb_annarchy.example import main_example, results_path_fun
from tvb_multiscale.tvb_annarchy.annarchy_models.models import WilsonCowanBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models import \
    BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_annarchy.old_interfaces.builders.models.wilson_cowan \
    import WilsonCowanBuilder as InterfaceWilsonCowanBuilder
from tvb_multiscale.tvb_annarchy.old_interfaces.builders.models.red_ww_basal_ganglia_izhikevich import \
    RedWWexcIOBuilder as IzhikevichRedWWexcIOBuilder

from tests.core.test_models import loop_all, TestModel
from tests.core.test_models import model_params_wc, model_params_redww_exc_io

from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


class TestModelAnnarchy(TestModel):
    annarchy_nodes_ids = []
    annarchy_model_builder = None
    interface_model_builder = None
    annarchy_populations_order = 10
    tvb_to_annarchy_mode = "rate"
    annarchy_to_tvb = True
    exclusive_nodes = True
    delays_flag = True

    @property
    def results_path(self):
        return results_path_fun(self.annarchy_model_builder, self.interface_model_builder,
                                self.tvb_to_annarchy_mode, self.annarchy_to_tvb)

    def run(self, ):
        os.chdir(os.path.dirname(__file__))
        results_path = self.results_path
        delete_folder_safely(os.path.join(results_path, "res"))
        delete_folder_safely(os.path.join(results_path, "figs"))
        return main_example(self.model, self.annarchy_model_builder, self.interface_model_builder,
                            self.annarchy_nodes_ids, annarchy_populations_order=self.annarchy_populations_order,
                            tvb_to_annarchy_mode=self.tvb_to_annarchy_mode, annarchy_to_tvb=self.annarchy_to_tvb,
                            exclusive_nodes=self.exclusive_nodes, delays_flag=self.delays_flag,
                            simulation_length=self.simulation_length, transient=self.transient,
                            plot_write=self.plot_write,
                            **self.model_params)


class TestWilsonCowan(TestModelAnnarchy):
    model = WilsonCowan
    model_params = model_params_wc
    tvb_to_annarchy_mode = "rate"
    annarchy_nodes_ids = [33, 34]
    annarchy_model_builder = WilsonCowanBuilder
    interface_model_builder = InterfaceWilsonCowanBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


class TestIzhikevichRedWWexcIO(TestModelAnnarchy):
    model = ReducedWongWangExcIO
    model_params = model_params_redww_exc_io
    tvb_to_annarchy_mode = "rate"
    annarchy_nodes_ids = list(range(10))
    annarchy_model_builder = BasalGangliaIzhikevichBuilder
    interface_model_builder = IzhikevichRedWWexcIOBuilder

    # @pytest.mark.skip(reason="These tests are taking too much time")
    def test(self):
        self.run()


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    loop_all(use_numba=False, models_to_test=[TestWilsonCowan, TestIzhikevichRedWWexcIO])
