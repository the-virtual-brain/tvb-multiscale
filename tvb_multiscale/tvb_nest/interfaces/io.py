# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet.io import \
    SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet, \
    SpikeNetSpikeRecorderDeviceSet, SpikeNetSpikeRecorderTotalDeviceSet, \
    SpikeNetMultimeterDeviceSet, SpikeNetMultimeterMeanDeviceSet, SpikeNetMultimeterTotalDeviceSet
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

    @property
    def spiking_time(self):
        return self.target[0].nest_instance.GetKernelStatus("biological_time")

    @property
    def spiking_dt(self):
        return self.target[0].nest_instance.GetKernelStatus("resolution")

    @abstractmethod
    def send(self, data):
        pass

    def transform_time(self, time):
        # TODO: Find a solution for NEST first time step
        return np.maximum(self.next_spiking_time_step,
                          SpikeNetInputDeviceSet.transform_time(self, time) + self.spiking_dt)


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
        self.target.Set({"rate_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
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

    # def send(self, data):
    #     # Assuming data is of shape (proxy, time)
    #     # TODO: Decide if this is necessary, given that we can reduce the delays to the target nodes by resolution time


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
        # TODO: Decide whether to check for the values being in the future...:
        # data[-1] >= self.next_time_step
        for reg, spikes in zip(self.target, data[-1]):
            reg.add_spikes(spikes, time_shift=self.spiking_dt, sort=False)


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

    # def send(self, data):
    #     # TODO: Decide whether to check for the values being in the future...:


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
        self.target.Set({"amplitude_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
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


class NESTSpikeRecorderSet(SpikeNetSpikeRecorderDeviceSet, NESTOutputDeviceSet):

    """
        NESTSpikeRecorderSet class to read events' data (spike times and senders)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NESTSpikeRecorder


class NESTSpikeRecorderTotalSet(SpikeNetSpikeRecorderTotalDeviceSet, NESTOutputDeviceSet):

    """
        NESTSpikeRecorderSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NESTSpikeRecorder


class NESTMultimeterSet(SpikeNetMultimeterDeviceSet, NESTOutputDeviceSet):

    """
        NESTMultimeterSet class to read events' data (times, senders and variable values)
        from a DeviceSet of NESTMultimeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = NESTMultimeter


class NESTMultimeterMeanSet(SpikeNetMultimeterMeanDeviceSet, NESTOutputDeviceSet):
    """
            NESTMultimeterMeanSet class to read population mean events' data (times and variable values)
            from a DeviceSet of NESTMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    _spikeNet_output_device_type = NESTMultimeter


class NESTMultimeterTotalSet(SpikeNetMultimeterTotalDeviceSet, NESTOutputDeviceSet):
    """
            NESTMultimeterTotalSet class to read population total (summed across neurons) events' data
            (times and variable values) from a DeviceSet of NESTMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    _spikeNet_output_device_type = NESTMultimeter


class NESTVoltmeterSet(NESTMultimeterSet):

    """
        NESTVoltmeterSet class to read events' data (times, senders, and variable values)
         from a DeviceSet of NESTVoltmeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTVoltmeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "voltmeter"

    _spikeNet_output_device_type = NESTVoltmeter


class NESTVoltmeterMeanSet(NESTMultimeterMeanSet):

    """
        NESTVoltmeterMeanSet class to read events' mean data (times and variable values)
         from a DeviceSet of NESTVoltmeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTVoltmeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "voltmeter"

    _spikeNet_output_device_type = NESTVoltmeter


class NESTVoltmeterTotalSet(NESTMultimeterTotalSet):

    """
        NESTVoltmeterTotalSet class to read events' total (summed) data (times, and variable values)
         from a DeviceSet of NESTVoltmeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTVoltmeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "voltmeter"

    _spikeNet_output_device_type = NESTVoltmeter


class NESTOutputDeviceGetters(Enum):
    SPIKE_RECORDER = NESTSpikeRecorderSet
    SPIKE_RECORDER_TOTAL = NESTSpikeRecorderTotalSet
    MULTIMETER = NESTMultimeterSet
    MULTIMETER_MEAN = NESTMultimeterMeanSet
    MULTIMETER_TOTAL = NESTMultimeterTotalSet
    VOLTMETER = NESTVoltmeterSet
    VOLTMETER_MEAN = NESTVoltmeterMeanSet
    VOLTMETER_TOTAL = NESTVoltmeterTotalSet


class NESTInputDeviceSetters(Enum):
    INHOMOGENEOUS_POISSON_GENERATOR = NESTInhomogeneousPoissonGeneratorSet
    NEST_PARROT_INHOMOGENEOUS_POISSON_GENERATOR = NESTParrotInhomogeneousPoissonGeneratorSet
    SPIKE_GENERATOR = NESTSpikeGeneratorSet
    STEP_CURRENT_GENERATOR = NESTStepCurrentGeneratorSet


NESTSenders = NESTOutputDeviceGetters
NESTReceivers = NESTInputDeviceSetters
NESTCommunicators = combine_enums("NESTCommunicators", NESTSenders, NESTReceivers)
