# -*- coding: utf-8 -*-
from six import string_types
from pandas import Series, unique
import numpy as np

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.devices import DeviceSet

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals


LOG = initialize_logger(__name__)


class TVBtoSpikeNetDeviceInterface(DeviceSet):

    # This class implements an interface that sends TVB state to the Spiking Network
    # via input/stimulating devices that play the role of TVB region node proxies

    def __init__(self, spiking_network, name="", model="", dt=0.1, tvb_sv_id=None,
                 nodes_ids=[], target_nodes=[], scale=np.array([1.0]), device_set=None):
        super(TVBtoSpikeNetDeviceInterface, self).__init__(name, model, device_set)
        self.spiking_network = spiking_network
        self.dt = dt  # TVB time step
        self.tvb_sv_id = tvb_sv_id  # TVB state variable index linked to this interface
        self.nodes_ids = nodes_ids  # TVB region nodes' (proxies') indices
        self.target_nodes = target_nodes  # Spiking Network target region nodes' indices
        self.scale = scale  # a scaling weight
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str()

    def print_str(self, detailed_output=False, connectivity=False):
        output = "\n" + self.__repr__() + \
                 "\nName: %s, TVB state variable indice: %d, " \
                 "\nInterface weights: %s"  \
                 "\nTarget NEST Nodes indices:\n%s " % \
                 (self.name, self.tvb_sv_id, str(unique(self.scale).tolist()),
                  extract_integer_intervals(self.nodes_ids, print=True))
        if detailed_output:
            output += super(TVBtoSpikeNetDeviceInterface, self).print_str(connectivity)
        return output

    @property
    def n_target_nodes(self):
        return len(self.target_nodes)

    def from_device_set(self, device_set, tvb_sv_id=0, name=None):
        # Generate the interface from a DeviceSet (that corresponds to a collection of devices => proxy-nodes)
        if isinstance(device_set, DeviceSet):
            super(TVBtoSpikeNetDeviceInterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a DeviceSet!: %s" % str(device_set))
        self.tvb_sv_id = tvb_sv_id
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self
