# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.simulator_nest.models.devices import NESTDeviceSet
from tvb_scripts.utils.log_error_utils import initialize_logger

LOG = initialize_logger(__name__)


class TVBtoNESTinterface(NESTDeviceSet):

    def __init__(self, name="", model="", device_set=OrderedDict({}), tvb_sv_id=0):
        super(TVBtoNESTinterface, self).__init__(name, model, device_set)
        self.tvb_sv_id = tvb_sv_id
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def from_device_set(self, device_set, tvb_sv_id=0):
        if isinstance(device_set, NESTDeviceSet):
            for attr in ["name", "model", "_dict"]:
                setattr(self, attr, getattr(device_set, attr))
        self.tvb_sv_id = int(tvb_sv_id)
        return self
