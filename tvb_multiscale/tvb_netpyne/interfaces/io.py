from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet.io import \
    SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet, \
    SpikeNetSpikeEventRecorderDeviceSet, SpikeNetSpikeRecorderDeviceSet, \
    SpikeNetMultimeterDeviceSet, SpikeNetMultimeterMeanDeviceSet, SpikeNetMultimeterTotalDeviceSet
from tvb_multiscale.core.utils.data_structures_utils import combine_enums
from tvb_multiscale.tvb_netpyne.netpyne_models.devices import \
    NetpyneInputDevice, NetpynePoissonGenerator, NetpyneParameterInput, \
    NetpyneOutputDevice, NetpyneSpikeRecorder, NetpyneMultimeter

from tvb.basic.neotraits.api import List


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
        of NetpyneInhomogeneousPoissonGenerator instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet, of NetpyneInhomogeneousPoissonGenerator instances to send data to,
            - a method to set data to the target.
    """

    model = "poisson_generator"

    _spikeNet_input_device_type = NetpynePoissonGenerator

    def send(self, data):
        # Assuming data is of shape (proxy, time)
        nodesNumber = data[1].shape[0]
        self.target.Set({"rate_times": [self.transform_time(data[0]).tolist()] * nodesNumber,
                         "rate_values": np.maximum([0.0], data[1]).tolist()})


class NetpyneParameterInputSet(NetpyneInputDeviceSet):

    """
        NetpyneParameterInputSet class to set data directly to a DeviceSet
        of NetpyneParameterInput instances in memory
        It comprises of:
            - a target attribute, i.e., a DeviceSet of NetpyneParameterInput instances to send data to,
            - a method to set data to the target.
    """

    model = "parameter_input"

    _spikeNet_input_device_type = NetpyneParameterInput

    parameters = List(required=True,
                      of=str,
                      default=list(),
                      label="Parameters to receive time resolved input.",
                      doc="""List of parameters' names (of type str) to receive time resolved input.""")

    def configure(self):
        super(NetpyneParameterInputSet, self).configure()
        self.n_parameters = len(self.parameters)
        assert self.n_parameters > 0

    def send(self, data):
        # Assuming data is of shape (proxy, time, parameters)
        nodesNumber = data[1].shape[0]
        if data[1].ndim == 2:
            data[1] = data[1][:, :, np.newaxis]  # make sure that data is 3D
        assert data[1].shape[2] == self.n_parameters
        values_dict = dict()
        # Set times, although in principle this should be optional:
        values_dict["times"] = [self.transform_time(data[0]).tolist()] * nodesNumber
        for iP, param in enumerate(self.parameters):
            values_dict[param] = data[1][:, :, iP].tolist()
        self.target.Set(values_dict)


class NetpyneOutputDeviceSet(SpikeNetOutputDeviceSet):

    """
        NetpyneOutputDeviceSet class to read events' data
         (times, senders and values from NetPyNE Multimeters-like devices) from a DeviceSet
        of NetpyneOutputDevice instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneOutputDevice instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "Netpyne_output_device"

    _spikeNet_output_device_type = NetpyneOutputDevice


class NetpyneSpikeEventRecorderSet(SpikeNetSpikeEventRecorderDeviceSet, NetpyneOutputDeviceSet):

    """
        NetpyneSpikeEventRecorderSet class to read events' data (spike times and senders)
        from a DeviceSet of NetpyneSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NetpyneSpikeRecorder


class NetpyneSpikeRecorderSet(SpikeNetSpikeRecorderDeviceSet, NetpyneOutputDeviceSet):

    """
        NetpyneSpikeEventRecorderSet class to read events' data with no reference to spike senders (i.e., only spike times)
        from a DeviceSet of NetpyneSpikeRecorder instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneSpikeRecorder instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "spike_recorder"

    _spikeNet_output_device_type = NetpyneSpikeRecorder
    
    
class NetpyneMultimeterSet(SpikeNetMultimeterDeviceSet, NetpyneOutputDeviceSet):

    """
        NetpyneMultimeterSet class to read events' data (times, senders and variable values)
        from a DeviceSet of NetpyneMultimeter instances in memory.
        It comprises of:
            - a source attribute, i.e., the DeviceSet of NetpyneMultimeter instances to get (i.e., copy) data from,
            - an abstract method to get data from the source.
    """

    model = "multimeter"

    _spikeNet_output_device_type = NetpyneMultimeter


class NetpyneMultimeterMeanSet(SpikeNetMultimeterMeanDeviceSet, NetpyneOutputDeviceSet):
    """
            NetpyneMultimeterMeanSet class to read population mean events' data (times and variable values)
            from a DeviceSet of NetpyneMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NetpyneMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    _spikeNet_output_device_type = NetpyneMultimeter


class NetpyneMultimeterTotalSet(SpikeNetMultimeterTotalDeviceSet, NetpyneOutputDeviceSet):
    """
            NetpyneMultimeterTotalSet class to read population total (summed across neurons) events' data
            (times and variable values) from a DeviceSet of NetpyneMultimeter instances in memory.
            It comprises of:
                - a source attribute, i.e., the DeviceSet of NetpyneMultimeter instances to get (i.e., copy) data from,
                - an abstract method to get data from the source.
        """

    _spikeNet_output_device_type = NetpyneMultimeter


class NetpyneOutputDeviceGetters(Enum):
    SPIKE_RECORDER = NetpyneSpikeRecorderSet
    SPIKE_EVENT_RECORDER = NetpyneSpikeEventRecorderSet
    MULTIMETER = NetpyneMultimeterSet
    MULTIMETER_MEAN = NetpyneMultimeterMeanSet
    MULTIMETER_TOTAL = NetpyneMultimeterTotalSet


class NetpyneInputDeviceSetters(Enum):
    POISSON_GENERATOR = NetpynePoissonGeneratorSet
    PARAMETER_INPUT_SET = NetpyneParameterInputSet


NetpyneSenders = NetpyneOutputDeviceGetters
NetpyneReceivers = NetpyneInputDeviceSetters
NetpyneCommunicators = combine_enums("NetpyneCommunicators", NetpyneSenders, NetpyneReceivers)
