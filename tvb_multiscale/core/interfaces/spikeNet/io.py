from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Float

from tvb_multiscale.core.interfaces.base.io import SetToMemory, GetFromMemory
# rom tvb_multiscale.core.spiking_models.devices import InputDevice, DeviceSet, OutputDevice


class SpikeNetInputDevice(SetToMemory):
    __metaclass__ = ABCMeta

    """
        SpikeNetInputDevice class to set data directly to an InputDevice instance,
        or a DeviceSet of such instances in memory.
        It comprises of:
            - a target attribute, i.e., the InputDevice or DeviceSet instance to send data to,
            - an abstract method to set data to the target, depending on the specific InputDevice.
    """

    model = "input_device"

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    def transform_time(self, time):
        return self.dt * np.arange(time[0], time[-1] + 1)

    @abstractmethod
    def send(self, data):
        pass


class SpikeNetEventsFromOutpuDevice(GetFromMemory):

    """
        SpikeNetEventsFromOutpuDevice class to read events' data
         (times, senders and values from NEST Multimeters-like devices) from an OutputDevice,
         or a DeviceSet of such instances
        It comprises of:
            - a source attribute, i.e., the OutputDevice or DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    model = "output_device"

    def receive(self):
        events = self.source.get_events()
        self.source.reset
        return events


class SpikeNetSenders(Enum):
    SPIKE_NET_INPUT_DEVICE = SpikeNetInputDevice


class SpikeNetReceivers(Enum):
    SPIKE_NET_EVENTS_FROM_OUTPUT_DEVICE = SpikeNetEventsFromOutpuDevice
