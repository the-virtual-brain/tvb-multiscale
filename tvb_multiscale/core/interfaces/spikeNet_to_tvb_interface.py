# -*- coding: utf-8 -*-
from six import string_types

import numpy as np
from pandas import unique

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.devices import DeviceSet

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals


LOG = initialize_logger(__name__)


class SpikeNetToTVBinterface(DeviceSet):

    # This class implements an interface that sends Spiking Network state to TVB
    # via output/measuring devices

    def __init__(self, spiking_network, tvb_sv_id, name="", model="",
                 nodes_ids=[], scale=np.array([1.0]), device_set=None):
        super(SpikeNetToTVBinterface, self).__init__(name, model, device_set)
        self.spiking_network = spiking_network
        self.tvb_sv_id = tvb_sv_id  # The index of the TVB state variable linked to this interface
        # The indices of the Spiking Nodes which coincide with the TVB region nodes
        # (i.e., region i implemented in Spiking Network updates the region i in TVB):
        self.nodes_ids = nodes_ids
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
                 "\nSource NEST / Target TVB Nodes indices:%s " % \
                 (self.name, self.tvb_sv_id, str(unique(self.scale).tolist()),
                  extract_integer_intervals(self.nodes_ids, print=True))
        if detailed_output:
            output += super(SpikeNetToTVBinterface, self).print_str(connectivity)
        return output

    def from_device_set(self, device_set, name=None):
        if isinstance(device_set, DeviceSet):
            super(SpikeNetToTVBinterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a DeviceSet!: %s" % str(device_set))
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self


