# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from tvb_nest.simulator_nest.models.devices import NESTDeviceSet
from tvb_scripts.utils.log_error_utils import initialize_logger

LOG = initialize_logger(__name__)


class NESTtoTVBinterface(NESTDeviceSet):

    def __init__(self, name="", model="", device_set=OrderedDict({}), tvb_sv_id=0):
        super(NESTtoTVBinterface, self).__init__(name, model, device_set)
        self.tvb_sv_id = int(tvb_sv_id)
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def from_device_set(self, device_set, tvb_sv_id=None):
        if isinstance(device_set, NESTDeviceSet):
            for attr in ["name", "model", "_dict"]:
                setattr(self, attr, getattr(device_set, attr))
        self.tvb_sv_id = tvb_sv_id
        return self

    @property
    def mean_spikes_rate(self):
        return np.array(self.mean_spikes_rate).flatten()

    @property
    def mean_spikes_activity(self):
        return np.array(self.mean_spikes_activity).flatten()

    @property
    def current_population_mean_values(self):
        return np.array(self.current_data_mean_values).flatten()
