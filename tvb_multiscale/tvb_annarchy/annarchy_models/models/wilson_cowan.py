# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.config import CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.models import DefaultExcIOInhIBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        self.w_ee = kwargs.get("c_ee", kwargs.get("c_ee", self.tvb_serial_sim["model.c_ee"][0].item()))
        self.w_ei = kwargs.get("c_ei", kwargs.get("c_ei", self.tvb_serial_sim["model.c_ei"][0].item()))
        self.w_ie = kwargs.get("c_ie", kwargs.get("c_ie", self.tvb_serial_sim["model.c_ie"][0].item()))
        self.w_ii = kwargs.get("c_ii", kwargs.get("c_ii", self.tvb_serial_sim["model.c_ii"][0].item()))

        if set_defaults:
            self.set_defaults()
