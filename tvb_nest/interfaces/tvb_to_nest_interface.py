# -*- coding: utf-8 -*-
from six import string_types

from pandas import Series
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_nest.simulator_nest.models.devices import NESTDeviceSet

LOG = initialize_logger(__name__)


class TVBtoNESTinterface(NESTDeviceSet):

    def __init__(self, name="", model="", tvb_sv_id=0, device_set=Series()):
        super(TVBtoNESTinterface, self).__init__(name, model, device_set)
        self.tvb_sv_id = tvb_sv_id
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def from_device_set(self, device_set, tvb_sv_id=0, name=None):
        if isinstance(device_set, NESTDeviceSet):
            super(TVBtoNESTinterface, self).__init__(device_set.name, device_set.model, device_set)
        else:
            raise_value_error("Input device_set is not a NESTDeviceSet!: %s" % str(device_set))
        self.tvb_sv_id = tvb_sv_id
        if isinstance(name, string_types):
            self.name = name
        self.update_model()
        return self
