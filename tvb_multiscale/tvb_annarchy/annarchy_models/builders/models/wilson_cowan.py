# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        self.w_ee = self.tvb_model.c_ee[0].item()
        self.w_ei = self.tvb_model.c_ei[0].item()
        self.w_ie = self.tvb_model.c_ie[0].item()
        self.w_ii = self.tvb_model.c_ii[0].item()

        if set_defaults:
            self.set_defaults()
