from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet.io import \
    SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet, \
    SpikeNetSpikeRecorderDeviceSet, SpikeNetSpikeRecorderTotalDeviceSet, \
    SpikeNetMultimeterDeviceSet, SpikeNetMultimeterMeanDeviceSet, SpikeNetMultimeterTotalDeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_netpyne.netpyne_models.devices import \
    NetpyneInputDevice, NetpynePoissonGenerator, \
    NetpyneOutputDevice, NetpyneSpikeRecorder


class NetpyneInputDeviceSet(SpikeNetInputDeviceSet):
    __metaclass__ = ABCMeta

    """
        NetpyneInputDeviceSet class to set data directly to a DeviceSet of NetpyneInputDevice instances in memory.
        It comprises of:
            - a target attribute, i.e., the NetpyneInputDeviceSet of NetpyneInputDevice instances to send data to,
            - an abstract method to set data to the target, depending on the specific NetpyneInputDeviceSet.
    """

    _spikeNet_input_device_type = NetpyneInputDevice

    @property
    def spiking_time(self):
        return self.target[0].netpyne_instance.time

    @property
    def spiking_dt(self):
        return self.target[0].netpyne_instance.dt

    @abstractmethod
    def send(self, data):
        pass


class NetpynePoissonGeneratorSet(NetpyneInputDeviceSet):

    """
        NetpynePoissonGeneratorSet class to set data directly to a DeviceSet
        of NESTInhomogeneousPoissonGenerator instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet, of NESTInhomogeneousPoissonGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "poisson_generator"

    _spikeNet_input_device_type = NetpynePoissonGenerator

    def send(self, data):
        # Assuming data is of shape (proxy, time)
        nodesNumber = data[1].shape[0]
        self.target.Set({"rate_times": [self.transform_time(data[0]).tolist()] * nodesNumber,
                         "rate_values": np.maximum([0.0], data[1]).tolist()})


class NetpyneOutputDeviceSet(SpikeNetOutputDeviceSet):

    """
        NetpyneOutputDeviceSet class to read events' data
         (times, senders and values from NetPyNE Multimeters-like devices) from a DeviceSet
        of NetpyneOutputDevice instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneOutputDevice instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "nest_output_device"

    _spikeNet_output_device_type = NetpyneOutputDevice


class NetpyneSpikeRecorderSet(SpikeNetSpikeRecorderDeviceSet, NetpyneOutputDeviceSet):

    """
        NetpyneSpikeRecorderSet class to read events' data (spike times and senders)
        from a DeviceSet of NetpyneSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NetpyneSpikeRecorder


class NetpyneSpikeRecorderTotalSet(SpikeNetSpikeRecorderTotalDeviceSet, NetpyneOutputDeviceSet):

    """
        NetpyneSpikeRecorderSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of NetpyneSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NetpyneSpikeRecorder
