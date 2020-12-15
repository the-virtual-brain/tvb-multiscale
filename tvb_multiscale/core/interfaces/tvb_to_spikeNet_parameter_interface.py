# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from six import add_metaclass
from pandas import Series, unique
import numpy as np

from tvb_multiscale.core.config import initialize_logger
from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


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

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str()

    def print_str(self, **kwargs):
        return "\n" + self.__repr__() + \
               "\nName: %s, TVB coupling indice: %d, " \
               "\nspikeNet target parameter: %s " \
               "\nInterface weights: %s " \
               "\nTarget NEST Nodes: %s" % \
                (self.name, self.tvb_coupling_id, self.parameter, str(unique(self.scale).tolist()),
                 str(["%d. %s" % (node_id, node_label)
                      for node_id, node_label in zip(self.nodes_ids, self.nodes)]))

    @property
    def nodes(self):
        return list(self.index)

    @property
    def number_of_nodes(self):
        return len(self.nodes)

    @abstractmethod
    def set(self, values):
        pass

    def _assert_input_size(self, values):
        values = ensure_list(values)
        n_vals = len(values)
        if n_vals not in [1, self.number_of_nodes]:
            raise ValueError("Values' number %d is neither equal to 1 "
                             "nor equal to nodes' number %d!" % (n_vals, self.number_of_nodes))
        elif n_vals == 1:
            values *= self.number_of_nodes
        return values