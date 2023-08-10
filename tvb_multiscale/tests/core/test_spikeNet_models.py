# -*- coding: utf-8 -*-
import os
import shutil

from tvb.basic.profile import TvbProfile
from tvb_multiscale.core.config import Config
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from examples.example import results_path_fun
from tvb_multiscale.tests.core.test_models import TestModel

from tvb.contrib.scripts.utils.file_utils import delete_folder_safely


class TestSpikeNetModel(TestModel):
    spiking_proxy_inds = [0, 1]
    tvb_spikeNet_model_builder = None
    spikeNet_model_builder = None
    population_order = 10
    tvb_to_spikeNet_mode = "rate"
    spikeNet_to_tvb = True
    exclusive_nodes = True
    delays_flag = True

    @property
    def results_path(self):
        return results_path_fun(self.spikeNet_model_builder, self.tvb_to_spikeNet_mode, self.spikeNet_to_tvb)

    def run_fun(self):
        pass

    def run(self):
        delete_folder_safely(self.results_path)
        return self.run_fun()
