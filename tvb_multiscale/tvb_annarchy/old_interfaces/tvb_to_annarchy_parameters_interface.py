# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.interfaces.tvb_to_spikeNet_parameter_interface import TVBtoSpikeNetParameterInterface


class TVBtoANNarchyParameterInterface(TVBtoSpikeNetParameterInterface):

    _available_input_parameters = {"current": "I", "potential": "v"}  #

    def __init__(self, spiking_network, name, model, parameter="", tvb_coupling_id=0, nodes_ids=[],
                 scale=np.array([1.0]), neurons=None):
        super(TVBtoANNarchyParameterInterface, self).__init__(spiking_network, name, model, parameter,
                                                               tvb_coupling_id, nodes_ids, scale, neurons)
        self._available_input_parameters = {"current": "I", "potential": "v"}   #

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance
