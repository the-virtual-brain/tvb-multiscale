# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.base.io import ReaderFromFile
from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDevice, SpikeNetEventsFromOutputDevice
from tvb_multiscale.core.spiking_models.devices import DeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_nest.nest_models.devices import read_nest_output_device_data_from_ascii_to_dict, \
    NESTSpikeGenerator, NESTInhomogeneousPoissonGenerator, NESTStepCurrentGenerator, \
    NESTSpikeRecorder, NESTMultimeter, NESTVoltmeter


class NESTCommunicator(object):
    from nest import NodeCollection
    _node_collection_class = NodeCollection


class NESTInputDeviceSetter(SpikeNetInputDevice, NESTCommunicator):
    __metaclass__ = ABCMeta

    """
        NESTInputDeviceSetter class to set data directly to a NESTInputDeviceSetter instance, 
        a DeviceSet of such instances, or a nest.NodeCollection instance corresponding 
        to a NESTInputDeviceSetter in memory.
        It comprises of:
            - a target attribute, i.e., the NESTInputDeviceSetter instance to send data to,
            - an abstract method to set data to the target, depending on the specific NESTInputDeviceSetter.
    """

    def configure(self, nest_input_device_class):
        assert isinstance(self.target, (nest_input_device_class, DeviceSet, super()._node_collection_class))
        if isinstance(self.target, DeviceSet):
            assert isinstance(self.target[0], nest_input_device_class)
        elif isinstance(self.target, self._node_collection_class):
            assert self.target.get("model") == nest_input_device_class.model
        super(NESTInputDeviceSetter, self).configure()

    @abstractmethod
    def send(self, data):
        pass


class NESTInhomogeneousPoissonGeneratorSetter(NESTInputDeviceSetter):

    """
        NESTInhomogeneousPoissonGeneratorSetter class to set data directly to a NESTInhomogeneousPoissonGenerator 
        instance or a nest.NodeCollection instance corresponding 
        to a NEST inhomogeneous_poisson_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTInhomogeneousPoissonGenerator, DeviceSet, or 
               inhomogeneous_poisson_generator instance to send data to,
            - a method to set data to the target.
    """

    model = "inhomogeneous_poisson_generator"

    def configure(self):
        super(NESTInhomogeneousPoissonGeneratorSetter, self).configure(NESTInhomogeneousPoissonGenerator)

    def send(self, data):
        # Assuming data is of shape (proxy, time)
        self.target.set({"rate_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
                         "rate_values": np.maximum([0.0], data[1]).tolist()})


class NESTSpikeGeneratorSetter(NESTInputDeviceSetter):

    """
        NESTSpikeGenerator class to set data directly to a NESTSpikeGenerator instance 
        or a nest.NodeCollection instance corresponding to a NEST spike_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTSpikeGenerator, DeviceSet, or 
               spike_generator instance to send data to,
            - a method to set data to the target.
    """

    model = "spike_generator"

    def configure(self):
        super(NESTSpikeGeneratorSetter, self).configure(NESTSpikeGenerator)

    def send(self, data):
        self.target.set({"spikes_times": np.maximum([0.0], data[-1]).tolist()})


class NESTStepCurrentGeneratorSetter(NESTInputDeviceSetter):

    """
        NESTStepCurrentGenerator class to set data directly to a NESTStepCurrentGenerator instance, 
        a DeviceSet of such instances, or a nest.NodeCollection instance 
        corresponding to a NEST step_current_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTStepCurrentGenerator, DeviceSet,
             or step_current_generator instance to send data to,
            - a method to set data to the target.
    """

    model = "step_current_generator"

    def configure(self):
        super(NESTStepCurrentGeneratorSetter, self).configure(NESTStepCurrentGenerator)

    def send(self, data):
        self.target.set({"amplitude_times": [self.transform_time(data[0]).tolist()] * data[1].shape[0],
                         "amplitude_values": data[1].tolist()})


class NESTEventsFromOutputDevice(SpikeNetEventsFromOutputDevice, NESTCommunicator):

    """
        NESTEventsFromOutputDevice class to read events' data
         (times, senders and values from NEST Multimeters-like devices) from a NESTOutputDevice device,
         a DeviceSet of NESTOutputDevices a nest.NodeCollection instance corresponding to a NEST recording device.
        It comprises of:
            - a source attribute, i.e., the NESTOutputDevice to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "nest_output_device"

    def configure(self, nest_output_device_class):
        assert isinstance(self.source, (nest_output_device_class, DeviceSet, super()._node_collection_class))
        if isinstance(self.source, DeviceSet):
            assert isinstance(self.source[0], nest_output_device_class)
        elif isinstance(self.source, self._node_collection_class):
            assert self.souce.get("model") == nest_output_device_class.model
        super(NESTEventsFromOutputDevice, self).configure()

    @property
    def reset(self):
        # TODO: find how to reset NEST recorders!
        pass
        # self.source.n_events = 0

    def receive(self):
        events = self.source.get("events")
        self.reset
        return events


class NESTEventsFromSpikeRecorder(SpikeNetEventsFromOutputDevice):

    """
        NESTEventsFromSpikeRecorder class to read events' data
         (spike times and senders) from a NESTSpikeRecorder device,
         a DeviceSet of NESTSpikeRecorder instances 
         or a nest.NodeCollection instance corresponding to a NEST spike_recorder device.
        It comprises of:
            - a source attribute, i.e., the NESTSpikeRecorder to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    def configure(self):
        super(NESTEventsFromSpikeRecorder, self).configure(NESTSpikeRecorder)


class NESTEventsFromMultimeter(SpikeNetEventsFromOutputDevice):

    """
        NESTEventsFromMultimeter class to read events' data
         (times, senders and variable values) from a NESTMultimeter device,
         a DeviceSet of NESTMultimeter instances 
         or a nest.NodeCollection instance corresponding to a NEST multimeter device.
        It comprises of:
            - a source attribute, i.e., the NESTMultimeter to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    def configure(self):
        super(NESTEventsFromMultimeter, self).configure(NESTMultimeter)


class NESTEventsFromVoltmeter(SpikeNetEventsFromOutputDevice):

    """
        NESTEventsFromVoltmeter class to read events' data
         (times, senders and variable values) from a NESTVoltmeter device,
         a DeviceSet of NESTVoltmeter instances 
         or a nest.NodeCollection instance corresponding to a NEST voltmeter device.
        It comprises of:
            - a source attribute, i.e., the NESTMultimeter to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "voltmeter"

    def configure(self):
        super(NESTEventsFromVoltmeter, self).configure(NESTVoltmeter)


class NESTEventsReaderFromRecorderFile(ReaderFromFile, NESTCommunicator):

    """
        NESTEventsReaderFromRecorderFile class to read events' data (times, senders and values from Multimeters)
        from a NEST recorder device file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the file to read data from,
            - an abstract method to read data from the source.
    """

    def receive(self):
        return read_nest_output_device_data_from_ascii_to_dict(self.source)


class NESTEventsFromSpikeRecorderFile(NESTEventsReaderFromRecorderFile):

    """
        NESTSpikeRecorderFile class to read events' data (spike times and senders)
        from a NEST spike_recorder device file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the file to read data from,
            - an abstract method to read data from the source.
    """

    model = "spike_recorder"

    pass


class NESTEventsFromMultimeterFile(NESTEventsReaderFromRecorderFile):

    """
        NESTMultimeterFile class to read events' data (times and values)
        from a NEST multimeter device file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the file to read data from,
            - an abstract method to read data from the source.
    """

    model = "multimeter"

    pass


class NESTEventsFromVoltmeterFile(NESTEventsReaderFromRecorderFile):

    """
        NESTVoltmeterFile class to read voltage events' data (times and values)
        from a NEST voltmeter device file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the file to read data from,
            - an abstract method to read data from the source.
    """

    model = "voltmeter"

    pass


class NESTOutputDeviceGetters(Enum):
    SPIKE_RECORDER = NESTEventsFromSpikeRecorder
    MULTIMETER = NESTEventsFromMultimeter
    VOLTMETER = NESTEventsFromVoltmeter


class NESTFileRecorders(Enum):
    SPIKE_RECORDER_FILE = NESTEventsFromSpikeRecorderFile
    MULTIMETER_FILE = NESTEventsFromMultimeterFile
    VOLTMETER_FILE = NESTEventsFromVoltmeterFile


class NESTInputDeviceSetters(Enum):
    INHOMOGENEOUS_POISSON_GENERATOR = NESTInhomogeneousPoissonGeneratorSetter
    SPIKE_GENERATOR = NESTSpikeGeneratorSetter
    STEP_CURRENT_GENERATOR = NESTStepCurrentGeneratorSetter


NESTSenders = combine_enums("NESTSenders", NESTFileRecorders, NESTOutputDeviceGetters)
NESTReceivers = combine_enums("NESTReceivers", NESTInputDeviceSetters)
NESTCommunicators = combine_enums("NESTCommunicators", NESTSenders, NESTReceivers)
