# -*- coding: utf-8 -*-
from six import string_types

from pandas import Series
import numpy as np
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_nest.simulator_nest.models.devices import NESTDeviceSet

LOG = initialize_logger(__name__)


class TVBtoNESTDeviceInterface(NESTDeviceSet):

    def __init__(self, nest_instance, name="", model="", dt=0.1, tvb_sv_id=None,
                 nodes_ids=[], target_nodes=[], scale=np.array([1.0]), device_set=Series()):
        super(TVBtoNESTDeviceInterface, self).__init__(name, model, device_set)
        self.nest_instance = nest_instance
        self.dt = dt
        self.tvb_sv_id = tvb_sv_id
        self.nodes_ids = nodes_ids
        self.target_nodes = target_nodes
        self.scale = scale
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    @property
    def n_target_nodes(self):
        return len(self.target_nodes)

    def _return_unique(self, attr):
        dummy = self.do_for_all_devices(attr)
        shape = (self.n_target_nodes, int(len(dummy[0])/self.n_target_nodes))
        for ii, dum in enumerate(dummy):
            dummy[ii] = np.reshape(dum, shape).mean(axis=1)
        return np.array(dummy)

    @property
    def weights(self):
        return self._return_unique("weights")

    @property
    def delays(self):
        return self._return_unique("delays")

    @property
    def receptors(self):
        return self._return_unique("receptors")

    def from_device_set(self, device_set, tvb_sv_id=0, name=None):
        if isinstance(device_set, NESTDeviceSet):
            super(TVBtoNESTDeviceInterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a NESTDeviceSet!: %s" % str(device_set))
        self.tvb_sv_id = tvb_sv_id
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self

    def set(self, values):
        if self.model == "dc_generator":
            self.SetStatus({"amplitude": values,
                            "origin": self.nest_instance.GetKernelStatus("time"),
                            "start": self.nest_instance.GetKernelStatus("min_delay"),
                            "stop": self.dt})
        elif self.model in ["poisson_generator"]:
            # ...and transmit it to the corresponding NEST device,
            # ...which represents that TVB node
            self.SetStatus({"rate": np.maximum(0, values),
                            "origin": self.nest_instance.GetKernelStatus("time"),
                            "start": self.nest_instance.GetKernelStatus("min_delay"),
                            "stop": self.dt})
        elif self.model in ["spike_generator"]:
            # TODO: change this so that rate corresponds to number of spikes instead of spikes' weights
            self.SetStatus({"spikes_times": np.ones((len(values),)) *
                                           self.nest_instance.GetKernelStatus("min_delay"),
                            "origin": self.nest_instance.GetKernelStatus("time"),
                            "spike_weights": values})
        else:
            raise ValueError("Interface model %s is not supported yet!" % self.model)
