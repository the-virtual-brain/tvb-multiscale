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

    def __init__(self, tvb_simulator={}, nest_nodes_ids=[], nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults)
        self.w_ee = np.array([kwargs.get("w_ee", kwargs.get("c_ee", self.w_ee))])
        self.w_ei = np.array([kwargs.get("w_ei", kwargs.get("c_ei", self.w_ei))])
        self.w_ie = np.array([kwargs.get("w_ie", kwargs.get("c_ie", self.w_ie))])
        self.w_ii = np.array([kwargs.get("w_ii", kwargs.get("c_ii", self.w_ii))])

    def configure(self):
        super(WilsonCowanBuilder, self).configure()
        self.w_ee = self.weight_fun(self.tvb_serial_sim.get("model.c_ee", self.w_ee)[0].item())
        self.w_ei = self.weight_fun(self.tvb_serial_sim.get("model.c_ei", self.w_ei)[0].item())
        self.w_ie = self.weight_fun(-np.abs(self.tvb_serial_sim.get("model.c_ie", self.w_ie)[0].item()))
        self.w_ii = self.weight_fun(-np.abs(self.tvb_serial_sim.get("model.c_ii", self.w_ii)[0].item()))


class WilsonCowanMultisynapseBuilder(DefaultExcIOInhIMultisynapseBuilder):

    w_ee = 10.0
    w_ei = 6.0
    w_ie = -10.0
    w_ii = -1.0

    def __init__(self, tvb_simulator={}, nest_nodes_ids=[], nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):

        super(WilsonCowanMultisynapseBuilder, self).__init__(
            tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults, **kwargs)

        self.default_population["model"] = "aeif_cond_alpha_multisynapse"

        self.w_ee = np.array([kwargs.get("w_ee", kwargs.get("c_ee", self.w_ee))])
        self.w_ei = np.array([kwargs.get("w_ei", kwargs.get("c_ei", self.w_ei))])
        self.w_ie = -np.abs(np.array([kwargs.get("w_ie", kwargs.get("c_ie", self.w_ie))]))
        self.w_ii = -np.abs(np.array([kwargs.get("w_ii", kwargs.get("c_ii", self.w_ii))]))

    def configure(self):
        super(WilsonCowanMultisynapseBuilder, self).configure()
        self.w_ee = self.weight_fun(self.tvb_serial_sim.get("model.c_ee", self.w_ee)[0].item())
        self.w_ei = self.weight_fun(self.tvb_serial_sim.get("model.c_ei", self.w_ei)[0].item())
        self.w_ie = self.weight_fun(-np.abs(self.tvb_serial_sim.get("model.c_ie", self.w_ie)[0].item()))
        self.w_ii = self.weight_fun(-np.abs(self.tvb_serial_sim.get("model.c_ii", self.w_ii)()[0].item()))
