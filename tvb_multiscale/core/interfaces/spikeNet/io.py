from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, List

from tvb_multiscale.core.interfaces.base.io import SetToMemory, GetFromMemory
from tvb_multiscale.core.spiking_models.devices import DeviceSet, InputDevice, OutputDevice


class SpikeNetInputDeviceSet(SetToMemory):
    __metaclass__ = ABCMeta

    """
        SpikeNetInputDeviceSet class to set data directly to an Input DeviceSet instance in memory.
        It comprises of:
            - a target attribute, i.e., the Input DeviceSet instance to send data to,
            - an abstract method to set data to the target, depending on the specific Input DeviceSet.
    """

    model = "input_device"

    target = Attr(field_type=DeviceSet,
                  required=True,
                  label="Target of Spiking Network",
                  doc="""Spiking Network DeviceSet of InputDevice instances to set data to.""")

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    _spikeNet_input_device_type = InputDevice

    @property
    @abstractmethod
    def spiking_time(self):
        pass

    @property
    def next_time_step(self):
        return self.spiking_time + self.dt

    def transform_time(self, time):
        return np.maximum(self.next_time_step, self.dt * np.arange(time[0], time[-1] + 1))

    def configure(self):
        super(SpikeNetInputDeviceSet, self).configure()

    @abstractmethod
    def send(self, data):
        pass


class SpikeNetOutputDeviceSet(GetFromMemory):
    __metaclass__ = ABCMeta
    """
        SpikeNetOutputDeviceSet class to read events' data
        (times, senders, and, possibly, values from NEST Multimeters-like devices) from an Output DeviceSet in memory.
        It comprises of:
            - a source attribute, i.e., the Output DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    model = "output_device"

    source = Attr(field_type=DeviceSet,
                  required=True,
                  label="Source of Spiking Network events",
                  doc="""Spiking Network DeviceSet of OutputDevice instances to get events from.""")

    variables = List(of=list,
                     default=(),
                     label="Variables",
                     doc="""List of lists of variables (str) recorded by this SpikeNetOutputDeviceSet instance
                            per device node (proxy)""")

    _spikeNet_output_device_type = OutputDevice

    @abstractmethod
    def device_variables(self, device):
        pass

    def configure(self):
        super(SpikeNetOutputDeviceSet, self).configure()
        self.variables = []
        if self.source.size:
            for device in self.source:
                assert isinstance(device, self._spikeNet_output_device_type)
                self.variables.append(self.device_variables(device))

    @property
    @abstractmethod
    def data(self):
        pass

    def receive(self):
        return self.data

    @abstractmethod
    def reset(self):
        pass


class SpikeNetSenders(Enum):
    SPIKE_NET_INPUT_DEVICE = SpikeNetInputDeviceSet


class SpikeNetReceivers(Enum):
    SPIKE_NET_EVENTS_FROM_OUTPUT_DEVICE = SpikeNetOutputDeviceSet
