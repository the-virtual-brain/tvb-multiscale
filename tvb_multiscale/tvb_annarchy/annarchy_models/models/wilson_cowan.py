# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.tvb_annarchy.annarchy_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):
    w_ee = 10.0
    w_ei = 6.0
    w_ie = -10.0
    w_ii = -1.0

    def __init__(self, tvb_simulator={}, spiking_nodes_inds=[], spiking_simulator=None,
                 config=None, logger=None):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

    def set_defaults(self, **kwargs):
        self.w_ee = kwargs.get("w_ee", self.tvb_serial_sim.get("model.c_ee", np.array([self.w_ee])))[0].item()
        self.w_ei = kwargs.get("w_ei", self.tvb_serial_sim.get("model.c_ei", np.array([self.w_ei])))[0].item()
        self.w_ie = kwargs.get("w_ie", self.tvb_serial_sim.get("model.c_ie", np.array([self.w_ie])))[0].item()
        self.w_ii = kwargs.get("w_ii", self.tvb_serial_sim.get("model.c_ii", np.array([self.w_ii])))[0].item()
        super(WilsonCowanBuilder, self).set_defaults()
