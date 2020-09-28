# -*- coding: utf-8 -*-
from itertools import cycle
import numpy as np
from pandas import Series

from tvb_multiscale.core.interfaces.tvb_to_spikeNet_parameter_interface import TVBtoSpikeNetParameterInterface

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class TVBtoNESTParameterInterface(TVBtoSpikeNetParameterInterface):

    _available_input_parameters = {"current": "I_e", "potential": "V_m"}  #

    def __init__(self, spiking_network, name, model, parameter="", tvb_coupling_id=0, nodes_ids=[],
                 scale=np.array([1.0]), neurons=None):
        super(TVBtoNESTParameterInterface, self).__init__(spiking_network, name, model, parameter,
                                                          tvb_coupling_id, nodes_ids, scale, neurons)
        self._available_input_parameters = {"current": "I_e", "potential": "V_m"}  #

    @property
    def nest_instance(self):
        return self.spiking_network.annarchy_instance

    def set(self, values):
        values = ensure_list(values)
        n_vals = len(values)
        if n_vals not in [1, self.n_nodes]:
            raise ValueError("Values' number %d is neither equal to 1 "
                             "nor equal to nodes' number %d!" % (n_vals, self.n_nodes))
        for node, value in zip(self.nodes, cycle(values)):
            self.nest_instance.NodeCollection(self[node].neurons).set({self.parameter: value})