from abc import ABCMeta, abstractmethod, ABC
from enum import Enum

import numpy as np
from pandas import Series

from tvb.basic.neotraits.api import Attr, Float, List
from tvb.contrib.scripts.utils.data_structures_utils import concatenate_heterogeneous_DataArrays

from tvb_multiscale.core.interfaces.base.io import SetToMemory, GetFromMemory
from tvb_multiscale.core.spiking_models.devices import DeviceSet, InputDevice, OutputDevice, Multimeter


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
    @abstractmethod
    def spiking_dt(self):
        pass

    @property
    def next_spiking_time_step(self):
        return self.spiking_time + self.spiking_dt

    @property
    def next_time_step(self):
        return self.spiking_time + self.dt

    def transform_time(self, time):
        return self.dt * np.arange(time[0]-1, time[-1])

    def configure(self):
        super(SpikeNetInputDeviceSet, self).configure()

    @abstractmethod
    def send(self, data):
        pass


class SpikeNetOutputDeviceSet(GetFromMemory):

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

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    _spikeNet_output_device_type = OutputDevice

    def device_variables(self, *args):
        return ["times", "senders"]

    def reset(self):
        pass

    def configure(self):
        super(SpikeNetOutputDeviceSet, self).configure()
        self.variables = list()
        if self.source.size:
            for device in self.source:
                assert isinstance(device, self._spikeNet_output_device_type)
                self.variables.append(self.device_variables(device))
        self.reset()

    @property
    def data(self):
        data = list()
        # We need to get only the newly recorded events since last time:
        for i_node, node in enumerate(self.source.devices()):
            data.append(dict)
            for var, val in self.source[node].get_new_events(self.variables).items():
                data[i_node][var] = val
        return [list(), data]

    def receive(self):
        return self.data


class SpikeNetSpikeRecorderDeviceSet(SpikeNetOutputDeviceSet):

    """
        SpikeNetSpikeRecorderDeviceSet class to read spike events' data
        (times, senders) from a Spike Recording DeviceSet in memory.
        It comprises of:
            - a source attribute, i.e., the Spike Recording DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    model = "spike_recorder"

    @property
    def data(self):
        data = list()
        # We need to get only the newly recorded events since last time:
        for i_node, node in enumerate(self.source.devices()):
            data.append(dict())
            for var, val in self.source[node].get_new_events(self.variables).items():
                data[i_node][var] = val
        return [list(), data]


class SpikeNetSpikeRecorderTotalDeviceSet(SpikeNetSpikeRecorderDeviceSet):

    """
        SpikeNetSpikeRecorderTotalDeviceSet class to read mean population spike events' data
        (times, ) from a Spike Recording DeviceSet in memory.
        It comprises of:
            - a source attribute, i.e., the Spike Recording DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    @property
    def data(self):
        spike_data = list()
        for node in self.source.devices():
            spike_data.append(list())
            spike_data[-1] = self.source[node].new_spikes_times
        return [list(), np.array(spike_data, dtype='object')]


class SpikeNetMultimeterDeviceSetBase(SpikeNetOutputDeviceSet, ABC):

    """
        SpikeNetMultimeterDeviceSetBase class to read events' data
        (times, senders, and, values from Multimeters-like devices) from a Multimeter DeviceSet in memory.
        It comprises of:
            - a source attribute, i.e., the Multimeter DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = Multimeter

    def device_variables(self, device):
        return super(SpikeNetMultimeterDeviceSetBase, self).device_variables() + list(device.record_from)

    @property
    def data(self):
        data = Series()
        for node in self.source.devices():
            data[node] = self.source[node].get_new_data(flatten_neurons_inds=True)
        data = concatenate_heterogeneous_DataArrays(data, "Proxy",
                                                    data_keys=None, name=self.source.name,
                                                    fill_value=np.nan, transpose_dims=None)
        if data.size:
            time = data.coords["Time"].values
            # data[0] will be start and end times
            # data[1] will be values array in (time x variables x proxies) shape
            return [np.array([time[0], time[-1]]), data]
        else:
            return [np.array([0.0, 0.0]), data]


class SpikeNetMultimeterDeviceSet(SpikeNetMultimeterDeviceSetBase, ABC):

    """
        SpikeNetMultimeterDeviceSet class to read events' data
        (times, senders, and, values from Multimeters-like devices) from a Multimeter DeviceSet in memory,
        and compute the mean across Neuron dimension.
        It comprises of:
            - a source attribute, i.e., the Multimeter DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    @property
    def data(self):
        data = super(SpikeNetMultimeterDeviceSet, self).data
        return [data[0], data[1].values]


class SpikeNetMultimeterMeanDeviceSet(SpikeNetMultimeterDeviceSetBase, ABC):

    """
        SpikeNetMultimeterMeanDeviceSet class to read events' data
        (times, senders, and, values from Multimeters-like devices) from a Multimeter DeviceSet in memory,
        and compute the mean across Neuron dimension.
        It comprises of:
            - a source attribute, i.e., the Multimeter DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    @property
    def data(self):
        data = super(SpikeNetMultimeterMeanDeviceSet, self).data
        data[1] = data[1].mean(dim="Neuron")
        return [data[0], data[1].values]


class SpikeNetMultimeterTotalDeviceSet(SpikeNetMultimeterDeviceSetBase, ABC):

    """
        SpikeNetMultimeterTotalDeviceSet class to read events' data
        (times, senders, and, values from Multimeters-like devices) from a Multimeter DeviceSet in memory,
        and compute the sum across Neuron dimension.
        It comprises of:
            - a source attribute, i.e., the Multimeter DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    @property
    def data(self):
        data = super(SpikeNetMultimeterTotalDeviceSet, self).data
        data[1] = data[1].sum(dim="Neuron")
        return [data[0], data[1].values]


class SpikeNetSenders(Enum):
    SPIKE_NET_INPUT_DEVICE = SpikeNetInputDeviceSet


class SpikeNetReceivers(Enum):
    SPIKE_NET_EVENTS_FROM_OUTPUT_DEVICE = SpikeNetOutputDeviceSet
    SPIKE_NET_EVENTS_FROM_SPIKE_RECORDER_DEVICE = SpikeNetSpikeRecorderDeviceSet
    SPIKE_NET_TOTAL_EVENTS_FROM_SPIKE_RECORDER_DEVICE = SpikeNetSpikeRecorderTotalDeviceSet
    SPIKE_NET_EVENTS_FROM_MULTIMETER_DEVICE = SpikeNetMultimeterDeviceSet
    SPIKE_NET_MEAN_EVENTS_FROM_MULTIMETER_DEVICE = SpikeNetMultimeterMeanDeviceSet
    SPIKE_NET_TOTAL_EVENTS_FROM_MULTIMETER_DEVICE = SpikeNetMultimeterTotalDeviceSet
