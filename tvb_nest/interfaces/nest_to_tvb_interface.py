# -*- coding: utf-8 -*-
from six import string_types

from pandas import Series
from numpy import array
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_nest.simulator_nest.models.devices import NESTDeviceSet


LOG = initialize_logger(__name__)


class NESTtoTVBinterface(NESTDeviceSet):

    def __init__(self, name="", model="", tvb_sv_id=None, nodes_ids=[], interface_weights=array([1.0]), device_set=Series()):
        super(NESTtoTVBinterface, self).__init__(name, model, device_set)
        self.tvb_sv_id = tvb_sv_id
        self.nodes_ids = nodes_ids
        self.interface_weights = interface_weights
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def from_device_set(self, device_set, tvb_sv_id=None, name=None):
        if isinstance(device_set, NESTDeviceSet):
            super(NESTtoTVBinterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a NESTDeviceSet!: %s" % str(device_set))
        self.tvb_sv_id = tvb_sv_id
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self

    @property
    def population_spikes_number(self):
        return self.interface_weights * array(self.do_for_all_devices("mean_number_of_spikes")).flatten()

    @property
    def population_spikes_activity(self):
        return self.interface_weights * array(self.do_for_all_devices("mean_spikes_activity")).flatten()

    @property
    def current_population_mean_values(self):
        return self.interface_weights * array(self.do_for_all_devices("current_data_mean_values")).flatten()

    @property
    def reset(self):
        return array(self.do_for_all_devices("reset")).flatten()
