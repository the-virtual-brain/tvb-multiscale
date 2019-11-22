# -*- coding: utf-8 -*-

from itertools import cycle
from pandas import Series
from numpy import array
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


# Possible NEST input parameters for the interface
NEST_INPUT_PARAMETERS = {"current": "I_e", "potential": "V_m"}  #


class TVBNESTParameterInterface(Series):

    def __init__(self, nest_instance, name, model, parameter="", tvb_coupling_id=0, nodes_ids=[],
                 interface_weights=array([1.0]), neurons=Series()):
        super(TVBNESTParameterInterface, self).__init__(neurons)
        self.nest_instance = nest_instance
        self.name = str(name)
        self.model = str(model)
        if self.model not in NEST_INPUT_PARAMETERS.keys():
            raise ValueError("model %s is not one of the available parameter interfaces!" % self.model)
        self.parameter = str(parameter)
        if len(parameter) == 0:
            self.parameter = NEST_INPUT_PARAMETERS[self.model]
        else:
            if self.parameter != NEST_INPUT_PARAMETERS[self.model]:
                LOG.warning("Parameter %s is different to the default one %s "
                            "for parameter interface model %s"
                            % (self.parameter, NEST_INPUT_PARAMETERS[self.model], self.model))
        self.tvb_coupling_id = int(tvb_coupling_id)
        self.nodes_ids = nodes_ids
        self.interface_weights = interface_weights
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def _input_nodes(self, nodes=None):
        if nodes is None:
            # no input
            return list(self.index)
        else:
            if nodes in list(self.index) or nodes in list(range(len(self))):
                # input is a single index or label
                return [nodes]
            else:
                # input is a sequence of indices or labels
                return list(nodes)

    def set(self, weights, coupling, nodes=None):
        nodes = self._input_nodes(nodes)
        values = ensure_list(self.interface_weights *
                             weights[self.nodes_ids] *
                             coupling[self.tvb_coupling_id, self.nodes_ids].squeeze())
        n_vals = len(values)
        n_nodes = len(nodes)
        if n_vals not in [1, n_nodes]:
            raise ValueError("Values' number %d is neither equal to 1 "
                             "nor equal to nodes' number %d!" % (n_vals, n_nodes))
        for node, value in zip(ensure_list(nodes), cycle(values)):
            self.nest_instance.SetStatus(self[node], {self.parameter: value})
