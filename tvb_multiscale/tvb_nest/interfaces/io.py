# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
from pandas import Series

from tvb.contrib.scripts.utils.data_structures_utils import \
    data_xarray_from_continuous_events, concatenate_heterogeneous_DataArrays

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

    @property
    def spiking_time(self):
        return self.target[0].nest_instance.GetKernelStatus("time")

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
        # TODO: Decide whether to check for the values being in the future...:
        # data[-1] >= self.next_time_step
        self.target.set({"spike_times": data[-1]})


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

    def device_variables(self, *args):
        return ["times", "senders"]

    def reset(self):
       pass

    def configure(self):
        super(NESTOutputDeviceSet, self).configure()
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


class NESTSpikeRecorderSet(NESTOutputDeviceSet):

    """
        NESTSpikeRecorderSet class to read events' data (spike times and senders)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NESTSpikeRecorder

    @property
    def data(self):
        spike_data = []
        for node in self.source.devices():
            spike_data.append([])
            spike_data[-1] = self.source[node].get_spikes_times_by_neurons(full_senders=True, new=True)
        return [[], spike_data]


class NESTSpikeRecorderMeanSet(NESTSpikeRecorderSet):

    """
        NESTSpikeRecorderSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    @property
    def data(self):
        spike_data = []
        for node in self.source.devices():
            spike_data.append([])
            spike_data[-1] = self.source[node].new_spikes_times
        return [[], spike_data]


class NESTSpikeRecorderTotalSet(NESTSpikeRecorderMeanSet):

    """
        NESTSpikeRecorderSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of NESTSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """


class NESTMultimeterSet(NESTOutputDeviceSet):

    """
        NESTMultimeterSet class to read events' data (times, senders and variable values)
        from a DeviceSet of NESTMultimeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = NESTMultimeter

    def device_variables(self, device):
        return super(NESTMultimeterSet, self).device_variables() + list(device.record_from)

    @property
    def data(self):
        data = Series()
        for node in self.source.devices():
            data[node.label] = self.source[node].get_new_data(flatten_neurons_inds=True)
        data = concatenate_heterogeneous_DataArrays(data, "Proxy",
                                                    data_keys=None, name=self.source.name,
                                                    fill_value=np.nan, transpose_dims=None)
        time = data.coords["Time"].values
        # data[0] will be start and end times
        # data[1] will be values array in (time x variables x proxies) shape
        return [np.array([time[0], time[-1]]), data.values]


class NESTMultimeterMeanSet(NESTMultimeterSet):
    """
            NESTMultimeterMeanSet class to read population mean events' data (times and variable values)
            from a DeviceSet of NESTMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    @property
    def data(self):
        data = super(NESTMultimeterMeanSet, self).data
        data[1] = data[1].mean(dim="Neuron")
        return data


class NESTMultimeterTotalSet(NESTMultimeterSet):
    """
            NESTMultimeterTotalSet class to read population total (summed across neurons) events' data
            (times and variable values) from a DeviceSet of NESTMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NESTMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    @property
    def data(self):
        data = super(NESTMultimeterTotalSet, self).data
        data[1] = data[1].sum(dim="Neuron")
        return data


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
    SPIKE_RECORDER_MEAN = NESTSpikeRecorderMeanSet
    SPIKE_RECORDER_TOTAL = NESTSpikeRecorderTotalSet
    MULTIMETER = NESTMultimeterSet
    MULTIMETER_MEAN = NESTMultimeterMeanSet
    MULTIMETER_TOTAL = NESTMultimeterTotalSet
    VOLTMETER = NESTVoltmeterSet
    VOLTMETER_MEAN = NESTVoltmeterMeanSet
    VOLTMETER_TOTAL = NESTVoltmeterTotalSet


class NESTInputDeviceSetters(Enum):
    INHOMOGENEOUS_POISSON_GENERATOR = NESTInhomogeneousPoissonGeneratorSet
    SPIKE_GENERATOR = NESTSpikeGeneratorSet
    STEP_CURRENT_GENERATOR = NESTStepCurrentGeneratorSet


NESTSenders = NESTOutputDeviceGetters
NESTReceivers = NESTInputDeviceSetters
NESTCommunicators = combine_enums("NESTCommunicators", NESTSenders, NESTReceivers)
