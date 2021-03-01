# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_nest.nest_models.devices import \
    NESTInputDevice, NESTSpikeGenerator, NESTInhomogeneousPoissonGenerator, NESTStepCurrentGenerator, \
    NESTParrotSpikeGenerator, NESTParrotInhomogeneousPoissonGenerator, \
    NESTOutputDevice, NESTSpikeRecorder, NESTMultimeter, NESTVoltmeter


class NESTInputDeviceSet(SpikeNetInputDeviceSet):
    __metaclass__ = ABCMeta

    """
        NESTInputDeviceSet class to set data directly to a DeviceSet of NESTInputDevice instances in memory.
        It comprises of:
            - a target attribute, i.e., the NESTInputDeviceSet of NESTInputDevice instances to send data to,
            - an abstract method to set data to the target, depending on the specific NESTInputDeviceSet.
    """

    _spikeNet_input_device_type = NESTInputDevice

    @abstractmethod
    def send(self, data):
        pass


class NESTInhomogeneousPoissonGeneratorSet(NESTInputDeviceSet):

    """
        NESTInhomogeneousPoissonGeneratorSet class to set data directly to a DeviceSet
        of NESTInhomogeneousPoissonGenerator instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet, of NESTInhomogeneousPoissonGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "inhomogeneous_poisson_generator"

    _spikeNet_input_device_type = NESTInhomogeneousPoissonGenerator

    def send(self, data):
        # Assuming data is of shape (proxy, time)
        self.target.set({"rate_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
                         "rate_values": np.maximum([0.0], data[1]).tolist()})


class NESTParrotInhomogeneousPoissonGeneratorSet(NESTInhomogeneousPoissonGeneratorSet):
    """
        NESTParrotInhomogeneousPoissonGeneratorSet class to set data directly to a DeviceSet
        of NESTParrotInhomogeneousPoissonGenerator instances in memory
        It comprises of:
            - a target attribute,
              i.e., a DeviceSet, of NESTParrotInhomogeneousPoissonGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "parrot_inhomogeneous_poisson_generator"

    _spikeNet_input_device_type = NESTParrotInhomogeneousPoissonGenerator

    pass


class NESTSpikeGeneratorSet(NESTInputDeviceSet):

    """
        NESTSpikeGeneratorSet class to set data directly to a DeviceSet of NESTSpikeGenerator instances in memory.
        It comprises of:
            - a target attribute, i.e., the DeviceSet of NESTSpikeGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "spike_generator"

    _spikeNet_input_device_type = NESTSpikeGenerator

    def send(self, data):
        self.target.set({"spikes_times": np.maximum([0.0], data[-1]).tolist()})


class NESTParrotSpikeGeneratorSet(NESTSpikeGeneratorSet):

    """
        NESTParrotSpikeGeneratorSet class to set data directly to
        a DeviceSet of NESTParrotSpikeGenerator instances in memory.
        It comprises of:
            - a target attribute, i.e., the DeviceSet of NESTParrotSpikeGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "parrot_spike_generator"

    _spikeNet_input_device_type = NESTParrotSpikeGenerator

    pass


class NESTStepCurrentGeneratorSet(NESTInputDeviceSet):

    """
        NESTStepCurrentGeneratorSet class to set data directly to a DeviceSet
        of NESTStepCurrentGenerator instances in memory.
        It comprises of:
            - a target attribute, i.e., the DeviceSet of NESTStepCurrentGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "step_current_generator"

    _spikeNet_input_device_type = NESTStepCurrentGenerator

    def send(self, data):
        self.target.set({"amplitude_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
                         "amplitude_values": data[1].tolist()})


class NESTOutputDeviceSet(SpikeNetOutputDeviceSet):

    """
        NESTOutputDeviceSet class to read events' data
         (times, senders and values from NEST Multimeters-like devices) from a DeviceSet
        of NESTOutputDevice instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTOutputDevice instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "nest_output_device"

    _spikeNet_output_device_type = NESTOutputDevice

    @property
    def reset(self):
        # TODO: find how to reset NEST recorders!
        pass
        # self.source.n_events = 0


class NESTSpikeRecorderSet(SpikeNetOutputDeviceSet):

    """
        NESTSpikeRecorderSet class to read events' data (spike times and senders)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NESTSpikeRecorder

    def receive(self):
        events = self.get("events")
        return [events["times"],    # data[0] will be spike times, a list of spike times per proxy node
                events["senders"]]  # data[1] will be spike senders, a list of spike sender neurons per proxy node


class NESTMultimeterSet(SpikeNetOutputDeviceSet):

    """
        NESTMultimeterSet class to read events' data (times, senders and variable values)
        from a DeviceSet of NESTMultimeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = NESTMultimeter

    def receive(self):
        data = self.source.do_for_all_devices("get_mean_data", return_type="DataArray",
                                              concatenation_index_name="Region").transpose("Time", "Variable", "Region")
        time = np.array(data.coords["Time"])
        time = np.array([time[0], time[-1]])
        return [time, data.values]


class NESTVoltmeterSet(NESTMultimeterSet):

    """
        NESTVoltmeterSet class to read events' data (times, senders and variable values)
         from a DeviceSet of NESTVoltmeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTVoltmeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "voltmeter"

    _spikeNet_output_device_type = NESTVoltmeter


class NESTOutputDeviceGetters(Enum):
    SPIKE_RECORDER = NESTSpikeRecorderSet
    MULTIMETER = NESTMultimeterSet
    VOLTMETER = NESTVoltmeterSet


class NESTInputDeviceSetters(Enum):
    INHOMOGENEOUS_POISSON_GENERATOR = NESTInhomogeneousPoissonGeneratorSet
    SPIKE_GENERATOR = NESTSpikeGeneratorSet
    STEP_CURRENT_GENERATOR = NESTStepCurrentGeneratorSet


NESTSenders = NESTOutputDeviceGetters
NESTReceivers = NESTInputDeviceSetters
NESTCommunicators = combine_enums("NESTCommunicators", NESTSenders, NESTReceivers)
