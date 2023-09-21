import numpy as np

from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder


class WilsonCowanBuilder(DefaultExcIOInhIBuilder):

    def __init__(self, tvb_simulator={}, spiking_nodes_inds=[], spiking_simulator=None,
                 config=None, logger=None):
        super(WilsonCowanBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator, config, logger)

    def set_defaults(self, **kwargs):
        self.w_ee = self._get_weight_from_model("c_ee", self.w_ee)
        self.w_ei = self._get_weight_from_model("c_ei", self.w_ei)
        self.w_ie = self._get_weight_from_model("c_ie", self.w_ie)
        self.w_ii = self._get_weight_from_model("c_ii", self.w_ii)
        super(WilsonCowanBuilder, self).set_defaults()

    def _get_weight_from_model(self, param, default):
        scale = 1e-3 # TODO: de-hardcode
        weight = self.tvb_serial_sim.get(f"model.{param}")
        if weight:
            weight *= scale
        else:
            weight = default
        return weight

