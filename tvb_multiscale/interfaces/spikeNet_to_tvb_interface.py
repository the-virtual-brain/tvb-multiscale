# -*- coding: utf-8 -*-
from six import string_types

from pandas import Series
from numpy import array
from tvb_multiscale.spiking_models.devices import DeviceSet
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error


LOG = initialize_logger(__name__)


class SpikeNetToTVBinterface(DeviceSet):

    # This class implements an interface that sends Spiking Network state to TVB
    # via output/measuring devices

    def __init__(self, spiking_network, name="", model="",
                 tvb_sv_id=None, nodes_ids=[], scale=array([1.0]), device_set=Series()):
        super(SpikeNetToTVBinterface, self).__init__(name, model, device_set)
        self.spiking_network = spiking_network
        self.tvb_sv_id = tvb_sv_id  # The index of the TVB state variable linked to this interface
        # The indices of the Spiking Nodes which coincide with the TVB region nodes
        # (i.e., region i implemented in Spiking Network updates the region i in TVB):
        self.nodes_ids = nodes_ids
        self.scale = scale  # a scaling weight
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def from_device_set(self, device_set, tvb_sv_id=None, name=None):
        if isinstance(device_set, DeviceSet):
            super(SpikeNetToTVBinterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a DeviceSet!: %s" % str(device_set))
        self.tvb_sv_id = tvb_sv_id
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self

    @property
    def population_mean_spikes_number(self):
        return array(self.do_for_all_devices("mean_number_of_spikes")).flatten()

    @property
    def population_mean_spikes_activity(self):
        return array(self.do_for_all_devices("mean_spikes_activity")).flatten()

    @property
    def current_population_mean_values(self):
        return array(self.do_for_all_devices("current_data_mean_values")).flatten()

    @property
    def reset(self):
        return array(self.do_for_all_devices("reset")).flatten()
