# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet.io import \
    SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet, \
    SpikeNetSpikeRecorderDeviceSet, SpikeNetSpikeRecorderTotalDeviceSet, \
    SpikeNetMultimeterDeviceSet, SpikeNetMultimeterMeanDeviceSet, SpikeNetMultimeterTotalDeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDevice, ANNarchySpikeSourceArray, \
ANNarchyTimedArray, ANNarchyTimedPoissonPopulation, ANNarchyHomogeneousCorrelatedSpikeTrains, \
    ANNarchyOutputDevice, ANNarchyMonitor, ANNarchySpikeMonitor


class ANNarchyInputDeviceSet(SpikeNetInputDeviceSet):
    __metaclass__ = ABCMeta

    """
        ANNarchyInputDeviceSet class to set data directly to a DeviceSet of ANNarchyInputDevice instances in memory.
        It comprises of:
            - a target attribute, i.e., the ANNarchyInputDeviceSet of ANNarchyInputDevice instances to send data to,
            - an abstract method to set data to the target, depending on the specific ANNarchyInputDeviceSet.
    """

    _spikeNet_input_device_type = ANNarchyInputDevice

    @property
    def spiking_time(self):
        return self.target[0].annarchy_instance.get_time()

    @property
    def spiking_dt(self):
        return self.target[0].dt

    @abstractmethod
    def send(self, data):
        pass


class ANNarchyTimedArraySet(ANNarchyInputDeviceSet):

    """
        ANNarchyTimedArraySet class to set data directly to a DeviceSet of ANNarchyTimedArray instances in memory.
        It comprises of:
            - a target attribute, i.e., the DeviceSet of ANNarchyTimedArray instances to send data to,
            - a method to set data to the target.
    """

    model = "TimedArray"

    _spikeNet_input_device_type = ANNarchyTimedArray

    def send(self, data):
        # Assuming data is of shape (proxy, time), we convert it to (proxy, time, 1)
        self.target.Set({"schedule": self.transform_time(data[0])[np.newaxis],
                         "rates": np.maximum([0.0], data[1])})


class ANNarchyTimedPoissonPopulationSet(ANNarchyTimedArraySet):

    """
        ANNarchyTimedPoissonPopulationSet class to set data directly to a DeviceSet
        of ANNarchyTimedPoissonPopulation instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet, of ANNarchyTimedPoissonPopulation instances to send data to,
            - a method to set data to the target.
    """

    model = "TimedPoissonPopulation"

    _spikeNet_input_device_type = ANNarchyTimedPoissonPopulation


class ANNarchyHomogeneousCorrelatedSpikeTrainsSet(ANNarchyTimedArraySet):

    """
        ANNarchyHomogeneousCorrelatedSpikeTrainsSet class to set data directly to a DeviceSet
        of ANNarchyHomogeneousCorrelatedSpikeTrains instances in memory
        It comprises of:
            - a target attribute,
              i.e., a DeviceSet, of ANNarchyHomogeneousCorrelatedSpikeTrains instances to send data to,
            - a method to set data to the target.
    """

    model = "HomogeneousCorrelatedSpikeTrains"

    _spikeNet_input_device_type = ANNarchyHomogeneousCorrelatedSpikeTrains


class ANNarchySpikeSourceArraySet(ANNarchyInputDeviceSet):

    """
        ANNarchySpikeSourceArraySet class to set data directly
        to a DeviceSet of ANNarchySpikeSourceArray instances in memory.
        It comprises of:
            - a target attribute, i.e., the DeviceSet of ANNarchySpikeSourceArray instances to send data to,
            - a method to set data to the target.
    """

    model = "SpikeSourceArray"

    _spikeNet_input_device_type = ANNarchySpikeSourceArray

    def send(self, data):
        # TODO: Decide whether to check for the values being in the future...:
        # data[-1] >= self.next_time_step
        # Spike times do not need to be advanced by dt,
        # because they anyway count from the start time of the simulation = dt
        for reg, spikes in zip(self.target, data[-1]):
            reg.add_spikes(spikes, time_shift=None, sort=False)


class ANNarchyOutputDeviceSet(SpikeNetOutputDeviceSet):

    """
        ANNarchyOutputDeviceSet class to read events' data
         (times, senders and values from ANNarchy Multimeters-like devices) from a DeviceSet
        of ANNarchyOutputDevice instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchyOutputDevice instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "ANNarchy_output_device"

    _spikeNet_output_device_type = ANNarchyOutputDevice


class ANNarchySpikeMonitorSet(SpikeNetSpikeRecorderDeviceSet, ANNarchyOutputDeviceSet):

    """
        ANNarchySpikeMonitorSet class to read events' data (spike times and senders)
        from a DeviceSet of ANNarchySpikeMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchySpikeMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "SpikeMonitor"

    _spikeNet_output_device_type = ANNarchySpikeMonitor


class ANNarchySpikeMonitorTotalSet(SpikeNetSpikeRecorderTotalDeviceSet, ANNarchyOutputDeviceSet):

    """
        ANNarchySpikeMonitorTotalSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of ANNarchySpikeMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchySpikeMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "SpikeMonitor"

    _spikeNet_output_device_type = ANNarchySpikeMonitor


class ANNarchyMonitorSet(SpikeNetMultimeterDeviceSet, ANNarchyOutputDeviceSet):

    """
        ANNarchyMonitorSet class to read events' data (times, senders and variable values)
        from a DeviceSet of ANNarchyMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "monitor"

    _spikeNet_output_device_type = ANNarchyMonitor


class ANNarchyMonitorMeanSet(SpikeNetMultimeterMeanDeviceSet, ANNarchyOutputDeviceSet):
    """
            ANNarchyMonitorMeanSet class to read population mean events' data (times and variable values)
            from a DeviceSet of ANNarchyMonitor instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    model = "monitor"

    _spikeNet_output_device_type = ANNarchyMonitor


class ANNarchyMonitorTotalSet(SpikeNetMultimeterTotalDeviceSet, ANNarchyOutputDeviceSet):
    """
            ANNarchyMonitorTotalSet class to read population total (summed across neurons) events' data
            (times and variable values) from a DeviceSet of ANNarchyMonitor instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    model = "monitor"

    _spikeNet_output_device_type = ANNarchyMonitor


class ANNarchyOutputDeviceGetters(Enum):
    SPIKE_MONITOR = ANNarchySpikeMonitorSet
    SPIKE_MONITOR_TOTAL = ANNarchySpikeMonitorTotalSet
    MONITOR = ANNarchyMonitorSet
    MONITOR_MEAN = ANNarchyMonitorMeanSet
    MONITOR_TOTAL = ANNarchyMonitorTotalSet


class ANNarchyInputDeviceSetters(Enum):
    TIMED_ARRAY_POISSON_POPULATION = ANNarchyTimedPoissonPopulationSet
    SPIKE_SOURCE_ARRAY = ANNarchySpikeSourceArraySet
    # TIMED_ARRAY = ANNarchyTimedArraySet


ANNarchySenders = ANNarchyOutputDeviceGetters
ANNarchyReceivers = ANNarchyInputDeviceSetters
ANNarchyCommunicators = combine_enums("ANNarchyCommunicators", ANNarchySenders, ANNarchyReceivers)
