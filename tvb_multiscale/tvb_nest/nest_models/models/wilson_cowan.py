# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.models.default_exc_io_inh_i import \
    DefaultExcIOInhIBuilder, DefaultExcIOInhIMultisynapseBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):

    w_ee = 10.0
    w_ei = 6.0
    w_ie = -10.0
    w_ii = -1.0

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(WilsonCowanBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

        self.w_ee = 10.0
        self.w_ei = 6.0
        self.w_ie = -10.0
        self.w_ii = -1.0

    def set_defaults(self, **kwargs):
        self.w_ee = np.abs(kwargs.get("w_ee", self.tvb_serial_sim.get("model.c_ee", np.array([self.w_ee])))[0].item())
        self.w_ei = np.abs(kwargs.get("w_ei", self.tvb_serial_sim.get("model.c_ei", np.array([self.w_ei])))[0].item())
        self.w_ie = -np.abs(kwargs.get("w_ie", self.tvb_serial_sim.get("model.c_ie", np.array([self.w_ie]))))[0].item()
        self.w_ii = -np.abs(kwargs.get("w_ii", self.tvb_serial_sim.get("model.c_ii", np.array([self.w_ii]))))[0].item()
        super(WilsonCowanBuilder, self).set_defaults()


class WilsonCowanMultisynapseBuilder(DefaultExcIOInhIMultisynapseBuilder):

    model = "aeif_cond_alpha_multisynapse"

    w_ee = 10.0
    w_ei = 6.0
    w_ie = 10.0
    w_ii = 1.0

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):

        super(WilsonCowanMultisynapseBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                             spiking_simulator, config, logger)

        self.model = "aeif_cond_alpha_multisynapse"
        self.w_ee = 10.0
        self.w_ei = 6.0
        self.w_ie = 10.0
        self.w_ii = 1.0

    def set_defaults(self, **kwargs):
        self.w_ee = np.abs(kwargs.get("w_ee", self.tvb_serial_sim.get("model.c_ee", np.array([self.w_ee])))[0].item())
        self.w_ei = np.abs(kwargs.get("w_ei", self.tvb_serial_sim.get("model.c_ei", np.array([self.w_ei])))[0].item())
        self.w_ie = np.abs(kwargs.get("w_ie", self.tvb_serial_sim.get("model.c_ie", np.array([self.w_ie])))[0].item())
        self.w_ii = np.abs(kwargs.get("w_ii", self.tvb_serial_sim.get("model.c_ii", np.array([self.w_ii])))[0].item())
        super(WilsonCowanMultisynapseBuilder, self).set_defaults()

    def build(self, set_defaults=True):
        return super(WilsonCowanMultisynapseBuilder, self).build(set_defaults)
