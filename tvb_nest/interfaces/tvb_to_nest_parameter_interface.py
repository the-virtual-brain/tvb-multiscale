# -*- coding: utf-8 -*-

from itertools import cycle
import numpy as np
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict, OrderedDict

LOG = initialize_logger(__name__)

# Possible NEST input parameters for the interface
NEST_INPUT_PARAMETERS = {"current": "I_e", "potential": "V_m"}  #


class TVBNESTParameterInterface(IndexedOrderedDict):

    def __init__(self, nest_instance, name, model, parameter="", neurons=OrderedDict({}), tvb_coupling_id=0, sign=1):
        self.nest_instance = nest_instance
        if not (isinstance(neurons, dict) and
                np.all([isinstance(neuron, tuple())
                        for neuron in neurons.values()])):
            raise ValueError("Input neurons is not a IndexedOrderedDict of tuples!:\n" %
                             str(neurons))
        super(TVBNESTParameterInterface, self).__init__(neurons)
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
        self.sign = int(sign)
        if sign not in [-1, 1]:
            raise ValueError("Sign %s is neither 1 nor -1!" % str(self.sign))
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def set_values(self, values, nodes=None):
        if nodes is None or len(nodes) == 0:
            nodes = self._dict.keys()
        nodes = ensure_list(nodes)
        values = ensure_list(values)
        n_vals = len(values)
        n_nodes = len(nodes)
        if n_vals not in [1, n_nodes]:
            raise ValueError("Values' number %d is neither equal to 1 "
                             "nor equal to nodes' number %d!" % (n_vals, n_nodes))
        for node, value in zip(ensure_list(nodes), cycle(ensure_list(values))):
            self.nest_instance.SetStatus(self._dict[node], {self.parameter: self.sign * value})
