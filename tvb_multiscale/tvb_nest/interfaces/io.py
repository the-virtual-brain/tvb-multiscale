# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.io import \
    ReaderFromFile, SpikeNetInputDeviceToSet, SpikeNetEventsFromOutpuDevice
from tvb_multiscale.core.spiking_models.devices import DeviceSet
from tvb_multiscale.tvb_nest.nest_models.devices import read_nest_output_device_data_from_ascii_to_dict, \
    NESTInputDevice, NESTOutputDevice, NESTSpikeGenerator, NESTInhomogeneousPoissonGenerator, NESTStepCurrentGenerator


class NESTInputDeviceToSet(SpikeNetInputDeviceToSet):
    from nest import NodeCollection
    __metaclass__ = ABCMeta
    """
        NESTInputDeviceToSet class to set data directly to a NESTInputDevice instance, a DeviceSet of such instances,
        or a nest.NodeCollection instance corresponding to a NESTInputDevice in memory.
        It comprises of:
            - a target attribute, i.e., the NESTInputDevice instance to send data to,
            - an abstract method to set data to the target, depending on the specific NESTInputDevice.
    """

    target = Attr(field_type=(NESTInputDevice, DeviceSet, NodeCollection),
                  required=True,
                  label="Target of Spiking Network",
                  doc="""NESTInputDevice, DeviceSet or nest.NodeCollection instances to set data to.""")

    @abstractmethod
    def send(self, data):
        pass


class NESTInhomogeneousPoissonGeneratorToSet(NESTInputDeviceToSet):
    from nest import NodeCollection

    """
        NESTInhomogeneousPoissonGeneratorToSet class to set data directly to a NESTInhomogeneousPoissonGenerator 
        instance or a nest.NodeCollection instance corresponding 
        to a NEST inhomogeneous_poisson_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTInhomogeneousPoissonGenerator, DeviceSet, or 
               inhomogeneous_poisson_generator instance to send data to,
            - a method to set data to the target.
    """

    target = Attr(field_type=(NESTInhomogeneousPoissonGenerator, DeviceSet, NodeCollection),
                  required=True,
                  label="Target of Spiking Network",
                  doc="""NESTInhomogeneousPoissonGenerator, DeviceSet of such instances, or
                         inhomogeneous_poisson_generator nest.NodeCollection to set data to.""")

    def send(self, data):
        # Assuming data is of shape (proxy, time)
        self.target.set({"rate_times": [data[0].tolist()] * data[1].shape[0],
                         "rate_values": np.maximum([0.0], data[1]).tolist()})


class NESTSpikeGeneratorToSet(NESTInputDeviceToSet):
    from nest import NodeCollection

    """
        NESTSpikeGenerator class to set data directly to a NESTSpikeGenerator instance 
        or a nest.NodeCollection instance corresponding to a NEST spike_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTSpikeGenerator, DeviceSet, or 
               spike_generator instance to send data to,
            - a method to set data to the target.
    """

    target = Attr(field_type=(NESTSpikeGenerator, DeviceSet, NodeCollection),
                  required=True,
                  label="Target of Spiking Network",
                  doc="""NESTSpikeGenerator, DeviceSet of such instances, or 
                         spike_generator nest.NodeCollection to set data to.""")

    def set(self, data):
        self.target.set({"spikes_times": np.maximum([0.0], data[-1]).tolist()})


class NESTStepCurrentGeneratorToSet(NESTInputDeviceToSet):
    from nest import NodeCollection

    """
        NESTStepCurrentGenerator class to set data directly to a NESTStepCurrentGenerator instance, 
        a DeviceSet of such instances, or a nest.NodeCollection instance 
        corresponding to a NEST step_current_generator device in memory.
        It comprises of:
            - a target attribute, i.e., the NESTStepCurrentGenerator, DeviceSet,
             or step_current_generator instance to send data to,
            - a method to set data to the target.
    """

    target = Attr(field_type=(NESTStepCurrentGenerator, DeviceSet, NodeCollection),
                  required=True,
                  label="Target of Spiking Network",
                  doc="""NESTStepCurrentGenerator, DeviceSet of such instances,
                         or step_current_generator nest.NodeCollection to set data to.""")

    def send(self, data):
        self.target.set({"amplitude_times": [data[0].tolist()] * data[1].shape[0],
                         "amplitude_values": np.maximum([0.0], data[1]).tolist()})
        
        
class NESTEventsReaderFromRecorderFile(ReaderFromFile):

    """
        NESTEventsReaderFromRecorderFile class to read events' data (times, senders and values from Multimeters)
        from a NEST recorder device file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the file to read data from,
            - an abstract method to read data from the source.
    """

    def receive(self):
        return read_nest_output_device_data_from_ascii_to_dict(self.source)


class NESTEventsFromOutpuDevice(SpikeNetEventsFromOutpuDevice):
    from nest import NodeCollection
    """
        NESTEventsFromOutpuDevice class to read events' data
         (times, senders and values from NEST Multimeters-like devices) from a NESTOutputDevice device,
         a DeviceSet of NESTOutputDevices a nest.NodeCollection instance corresponding to a NEST recording device.
        It comprises of:
            - a source attribute, i.e., the NESTOutputDevice to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    source = Attr(field_type=(NESTOutputDevice, DeviceSet, NodeCollection),
                  required=True,
                  label="Source of Spiking Network events",
                  doc="""NESTOutputDevice, DeviceSet or nest.NodeCollection types to get events data from.""")

    @property
    def reset(self):
        # TODO: find how to reset NEST recorders!
        pass
        # self.source.n_events = 0

    def receive(self):
        events = self.source.get("events")
        self.reset
        return events
