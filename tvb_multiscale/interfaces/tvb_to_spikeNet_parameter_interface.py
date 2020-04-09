# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np
from pandas import Series
from six import add_metaclass
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error

LOG = initialize_logger(__name__)


@add_metaclass(ABCMeta)
class TVBtoSpikeNetParameterInterface(Series):
    # This class implements an interface that sends TVB state to the Spiking Network
    # by directly setting a specific parameter of its neurons (e.g., external current I_e in NEST)

    _available_input_parameters = {}

    def __init__(self, spiking_network, name, model, parameter="", tvb_coupling_id=0, nodes_ids=[],
                 scale=np.array([1.0]), neurons=Series()):
        super(TVBtoSpikeNetParameterInterface, self).__init__(neurons)
        self.spiking_network = spiking_network
        self.name = str(name)
        self.model = str(model)
        if self.model not in self._available_input_parameters.keys():
            raise_value_error("model %s is not one of the available parameter interfaces!" % self.model)
        self.parameter = str(parameter)  # The string of the target parameter name
        if len(parameter) == 0:
            self.parameter = self._available_input_parameters[self.model]
        else:
            if self.parameter != self._available_input_parameters[self.model]:
                LOG.warning("Parameter %s is different to the default one %s "
                            "for parameter interface model %s"
                            % (self.parameter, self._available_input_parameters[self.model], self.model))
        self.tvb_coupling_id = int(tvb_coupling_id)
        # The target Spiking Network region nodes which coincide with the source TVB region nodes
        # (i.e., region i of TVB modifies a parameter in region i implemented in Spiking Network):
        self.nodes_ids = nodes_ids
        self.scale = scale  # a scaling weight
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    @property
    def nodes(self):
        return list(self.index)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @abstractmethod
    def set(self, values):
        pass
