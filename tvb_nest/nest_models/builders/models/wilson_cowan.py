# -*- coding: utf-8 -*-

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.models.default_exc_io_inh_i import \
    DefaultExcIOInhIBuilder, DefaultExcIOInhIMultisynapseBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)

        self.w_ee = self.weight_fun(self.tvb_model.c_ee[0].item())
        self.w_ei = self.weight_fun(self.tvb_model.c_ei[0].item())
        self.w_ie = self.weight_fun(-self.tvb_model.c_ie[0].item())
        self.w_ii = self.weight_fun(-self.tvb_model.c_ii[0].item())

        if set_defaults:
            self.set_defaults()


class WilsonCownMultisynapseBuilder(DefaultExcIOInhIMultisynapseBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, set_defaults=True,
                 **kwargs):

        super(WilsonCownMultisynapseBuilder, self).__init__(
            tvb_simulator, nest_nodes_ids, nest_instance, config, set_defaults=False, **kwargs)

        self.default_population["model"] = "aeif_cond_alpha_multisynapse"

        self.w_ee = self.weight_fun(self.tvb_model.c_ee[0].item())
        self.w_ei = self.weight_fun(self.tvb_model.c_ei[0].item())
        self.w_ie = self.weight_fun(self.tvb_model.c_ie[0].item())
        self.w_ii = self.weight_fun(self.tvb_model.c_ii[0].item())

        if set_defaults:
            self.set_defaults()
