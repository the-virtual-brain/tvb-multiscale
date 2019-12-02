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
                 scale=array([1.0]), neurons=Series()):
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
        self.scale = scale
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    @property
    def nodes(self):
        return list(self.index)

    @property
    def n_nodes(self):
        return len(self.nodes)

    def set(self, values):
        values = ensure_list(values)
        n_vals = len(values)
        if n_vals not in [1, self.n_nodes]:
            raise ValueError("Values' number %d is neither equal to 1 "
                             "nor equal to nodes' number %d!" % (n_vals, self.n_nodes))
        for node, value in zip(self.nodes, cycle(values)):
            self.nest_instance.SetStatus(self[node], {self.parameter: value})
