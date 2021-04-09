# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
from pandas import Series

from tvb.contrib.scripts.utils.data_structures_utils import \
    data_xarray_from_continuous_events, concatenate_heterogeneous_DataArrays

from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyInputDevice, ANNarchySpikeSourceArray, ANNarchyTimedPoissonPopulation, ANNarchyTimedArray, \
    ANNarchyOutputDevice, ANNarchyMonitor, ANNarchySpikeMonitor  # ANNarchyTimedHomogeneousCorrelatedSpikeTrains


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

    def transform_time(self, time):
        # We need to add a TVB time step to get ANNarchy time in synchronization with TVB time.
        return self.dt * (np.arange(time[0], time[-1] + 1) + 1)

    @abstractmethod
    def send(self, data):
        pass


class ANNarchyTimedPoissonPopulationSet(ANNarchyInputDeviceSet):

    """
        ANNarchyTimedPoissonPopulationSet class to set data directly to a DeviceSet
        of ANNarchyTimedPoissonPopulation instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet, of ANNarchyTimedPoissonPopulation instances to send data to,
            - a method to set data to the target.
    """

    model = "TimedPoissonPopulation"

    _spikeNet_input_device_type = ANNarchyTimedPoissonPopulation

    def send(self, data):
        # Assuming data is of shape (proxy, time), we convert it to (proxy, time, 1)
        # Rate times do not need to be advanced by dt,
        # because they anyway count from the start time of the simulation = dt
        self.target.set({"rates": np.maximum([0.0], data[1][:, :, None]), "schedule": self.transform_time(data[0])})


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
        self.target.set({"spike_times": data[-1]})


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
        self.target.set({"rates": np.maximum([0.0], data[1][:, :, None])})


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

    def device_variables(self, *args):
        return ["times", "senders"]

    def reset(self):
       pass

    def configure(self):
        super(ANNarchyOutputDeviceSet, self).configure()
        self.reset()

    @property
    def data(self):
        data = []
        # We need to get only the newly recorded events since last time:
        for i_node, node in enumerate(self.source.devices()):
            data.append({})
            for var, val in self.source[node].get_new_events(self.variables).items():
                data[i_node][var] = val
        return [[], data]


class ANNarchySpikeMonitorSet(ANNarchyOutputDeviceSet):

    """
        ANNarchySpikeMonitorSet class to read events' data (spike times and senders)
        from a DeviceSet of ANNarchySpikeMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchySpikeMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "SpikeMonitor"

    _spikeNet_output_device_type = ANNarchySpikeMonitor

    @property
    def data(self):
        spike_data = []
        for node in self.source.devices():
            spike_data.append([])
            spike_data[-1] = self.source[node].get_spikes_times_by_neurons(full_senders=True, new=True)
        return [[], spike_data]


class ANNarchySpikeMonitorMeanSet(ANNarchySpikeMonitorSet):

    """
        ANNarchySpikeMonitorSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of ANNarchySpikeMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchySpikeMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    @property
    def data(self):
        spike_data = []
        for node in self.source.devices():
            spike_data.append([])
            spike_data[-1] = self.source[node].new_spikes_times
        return [[], spike_data]


class ANNarchySpikeMonitorTotalSet(ANNarchySpikeMonitorMeanSet):

    """
        ANNarchySpikeMonitorSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of ANNarchySpikeMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchySpikeMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """


class ANNarchyMonitorSet(ANNarchyOutputDeviceSet):

    """
        ANNarchyMonitorSet class to read events' data (times, senders and variable values)
        from a DeviceSet of ANNarchyMonitor instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = ANNarchyMonitor

    def device_variables(self, device):
        return super(ANNarchyMonitorSet, self).device_variables() + list(device.record_from)

    @property
    def data(self):
        data = Series()
        for node in self.source.devices():
            data[node.label] = self.source[node].get_new_data(flatten_neurons_inds=True)
        data = concatenate_heterogeneous_DataArrays(data, "Proxy",
                                                    data_keys=None, name=self.source.name,
                                                    fill_value=np.nan, transpose_dims=None)
        # Unlike spikes, the continuous variable's times are recorded at time t for the time step t-dt -> t
        # Therefore, we need to subtract the TVB dt to be at the same time as TVB
        time = data.coords["Time"].values - self.dt
        # data[0] will be start and end times
        # data[1] will be values array in (time x variables x proxies) shape
        return [np.array([time[0], time[-1]]), data.values]


class ANNarchyMonitorMeanSet(ANNarchyMonitorSet):
    """
            ANNarchyMonitorMeanSet class to read population mean events' data (times and variable values)
            from a DeviceSet of ANNarchyMonitor instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    @property
    def data(self):
        data = super(ANNarchyMonitorMeanSet, self).data
        data[1] = data[1].mean(dim="Neuron")
        return data


class ANNarchyMonitorTotalSet(ANNarchyMonitorSet):
    """
            ANNarchyMonitorTotalSet class to read population total (summed across neurons) events' data
            (times and variable values) from a DeviceSet of ANNarchyMonitor instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of ANNarchyMonitor instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    @property
    def data(self):
        data = super(ANNarchyMonitorTotalSet, self).data
        data[1] = data[1].sum(dim="Neuron")
        return data


class ANNarchyOutputDeviceGetters(Enum):
    SPIKE_MONITOR = ANNarchySpikeMonitorSet
    SPIKE_MONITOR_MEAN = ANNarchySpikeMonitorMeanSet
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
