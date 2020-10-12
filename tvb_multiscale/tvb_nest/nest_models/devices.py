# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.devices import \
    Device, InputDevice, OutputDevice, SpikeRecorder, SpikeRecorder, Multimeter, Voltmeter, SpikeMultimeter

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, list_of_dicts_to_dicts_of_ndarrays

# These classes wrap around NEST commands.


class NESTDevice(Device):
    __metaclass__ = ABCMeta

    """NESTDevice class to wrap around a NEST output (recording) or input (stimulating) device"""

    nest_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, device, nest_instance, *args, **kwargs):
        self.nest_instance = nest_instance
        super(NESTDevice, self).__init__(device, *args, **kwargs)
        self.model = "device"

    def _assert_nest(self):
        if self.nest_instance is None:
            raise ValueError("No NEST instance associated to this %s of model %s!" %
                             (self.__class__.__name__, self.model))

    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        self._assert_nest()
        try:
            self.device.get("element_type")
        except:
            raise ValueError("Failed to Get device %s!" % str(self.device))

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    @property
    def nest_model(self):
        self._assert_nest()
        return str(self.device.get("model"))

    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        self._assert_nest()
        self.device.get(values_dict)

    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
           Returns:
            Dictionary of attributes.
        """
        self._assert_nest()
        if attrs is None:
            return self.device.get()
        else:
            return self.device.get(attrs)

    def _GetConnections(self, **kwargs):
        """Method to get attributes of the connections from/to the device
           Return:
            connections' objects
        """
        self._assert_nest()
        for kw in ["source", "target"]:
            kwval = kwargs.get(kw, None)
            if kwval is not None:
                if len(kwval) == 0:
                    kwval = None
                elif not isinstance(kwval, self.nest_instance.NodeCollection):
                    kwval = self.nest_instance.NodeCollection(kwval)
                kwargs[kw] = kwval
        connections = self.nest_instance.GetConnections(**kwargs)
        if len(connections) == 0:
            return ()
        else:
            return connections

    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the device
            Arguments:
             connections: connections' objects.
             values_dict: dictionary of attributes names' and values.
            Returns:
             Dictionary of lists of connections' attributes.
        """
        self._assert_nest()
        connections.set(values_dict)

    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of lists of connections' attributes.
        """
        self._assert_nest()
        if attrs is None:
            return connections.get()
        else:
            return connections.get(ensure_list(attrs))

    def GetConnections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            SynapseCollection.
        """
        return self._GetConnections(source=self.device)

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            SynapseCollection..
        """
        return self._GetConnections(source=self.device)

    def get_neurons(self, source_or_target="target"):
        """Method to get the indices of all the neurons the device is connected from/to.
           Mind that for all input and all out output devices, except for spike detector,
           the devices connects to the neurons, and not vice-versa,
           i.e., neurons are the target of the device connection.
        """
        neurons = []
        for conn in self.connections:
            neuron = getattr(conn, source_or_target)
            if neuron is not None:
                neurons.append(neuron)
        return tuple(np.unique(neurons).tolist())

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.get_neurons("target")


class NESTInputDevice(NESTDevice, InputDevice):

    """NESTInputDevice class to wrap around a NEST input (stimulating) device"""

    model = "input_device"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTInputDevice, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "input_device"


class NESTPoissonGenerator(NESTInputDevice):

    """NESTPoissonGenerator class to wrap around a NEST poisson_generator device"""

    model = "poisson_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTPoissonGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "poisson_generator"


class NESTSinusoidalPoissonGenerator(NESTInputDevice):

    """NESTSinusoidalPoissonGenerator class to wrap around a NEST sinusoidal_poisson_generator device"""

    model = "sinusoidal_poisson_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTSinusoidalPoissonGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "sinusoidal_poisson_generator"


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):

    """NESTInhomogeneousPoissonGenerator class to wrap around a NEST inhomogeneous_poisson_generator device"""

    model = "inhomogeneous_poisson_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTInhomogeneousPoissonGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "inhomogeneous_poisson_generator"


class NESTMIPGenerator(NESTInputDevice):

    """NESTMIPGenerator class to wrap around a NEST mip_generator device"""

    model = "mip_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTMIPGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "mip_generator"


class NESTGammaSupGenerator(NESTInputDevice):

    """NESTGammaSupGenerator class to wrap around a NEST gamma_sup_generator device"""

    model = "gamma_sup_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTGammaSupGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "gamma_sup_generator"


class NESTDPPDSupGenerator(NESTInputDevice):

    """NESTDPPDSupGenerator class to wrap around a NEST ppd_sup_generator device"""

    model = "ppd_sup_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTDPPDSupGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "ppd_sup_generator"


class NESTSpikeGenerator(NESTInputDevice):

    """NESTSpikeGenerator class to wrap around a NEST spike_generator device"""

    model = "spike_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTSpikeGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "spike_generator"


class NESTPulsePacketGenerator(NESTInputDevice):

    """NESTPulsePacketGenerator class to wrap around a NEST pulse_packet_generator device"""

    model = "pulse_packet_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTPulsePacketGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "pulse_packet_generator"


class NESTDCGenerator(NESTInputDevice):

    """NESTDCGenerator class to wrap around a NEST dc_generator device"""

    model = "dc_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTDCGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "dc_generator"


class NESTStepCurrentGenerator(NESTInputDevice):

    """NESTStepCurrentGenerator class to wrap around a NEST step_current_generator device"""

    model = "step_current_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTStepCurrentGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "step_current_generator"


class NESTACGenerator(NESTInputDevice):

    """NESTACGenerator class to wrap around a NEST ac_generator device"""

    model = "ac_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTACGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "ac_generator"


class NESTStepRateGenerator(NESTInputDevice):

    """NESTStepRateGenerator class to wrap around a NEST step_rate_generator device"""

    model = "step_rate_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTStepRateGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "step_rate_generator"


class NESTNoiseGenerator(NESTInputDevice):

    """NESTNoiseGenerator class to wrap around a NEST noise_generator device"""

    model = "noise_generator"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTNoiseGenerator, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "noise_generator"


NESTInputDeviceDict = {"poisson_generator": NESTPoissonGenerator,
                       "sinusoidal_poisson_generator": NESTSinusoidalPoissonGenerator,
                       "inhomogeneous_poisson_generator": NESTInhomogeneousPoissonGenerator,
                       "mip_generator": NESTMIPGenerator,
                       "gamma_sup_generator": NESTGammaSupGenerator,
                       "ppd_sup_generator": NESTDPPDSupGenerator,
                       "spike_generator": NESTSpikeGenerator,
                       "pulse_packet_generator": NESTPulsePacketGenerator,
                       "dc_generator": NESTDCGenerator,
                       "step_current_generator": NESTStepCurrentGenerator,
                       "ac_generator": NESTACGenerator,
                       "step_rate_generator": NESTStepRateGenerator,
                       "noise_generator": NESTNoiseGenerator
                       }


class NESTOutputDevice(NESTDevice, OutputDevice):

    """NESTOutputDevice class to wrap around a NEST output (recording) device"""

    model = "output_device"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTOutputDevice, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "output_device"

    @property
    def events(self):
        self._assert_nest()
        return self.device.get("events")

    @property
    def number_of_events(self):
        self._assert_nest()
        return self.device.get("n_events")

    @property
    def n_events(self):
        return self.number_of_events
    
    @property
    def reset(self):
        # TODO: find how to reset recorders!
        self._assert_nest()
        pass
        # self.device.n_events = 0


class NESTSpikeRecorder(NESTOutputDevice, SpikeRecorder):

    """NESTSpikeRecorder class to wrap around a NEST spike_detector device"""

    model = "spike_detector"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTSpikeRecorder, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "spike_detector"

    # Only SpikeRecorder is the target of connections with neurons in NEST:

    def GetConnections(self):
        """Method to get connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(target=self.device)

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(target=self.device)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.get_neurons("source")


class NESTMultimeter(NESTOutputDevice, Multimeter):

    """NESTMultimeter class to wrap around a NEST multimeter device"""

    model = "multimeter"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTMultimeter, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "multimeter"

    @property
    def record_from(self):
        self._assert_nest()
        return [str(name) for name in self.device.get('record_from')]
    
    
class NESTVoltmeter(NESTMultimeter, Voltmeter):

    """NESTVoltmeter class to wrap around a NEST voltmeter device"""

    model = "voltmeter"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTVoltmeter, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "voltmeter"
        assert self.var in self.record_from
        
    @property
    def var(self):
        return "V_m"
    
    @property
    def get_V_m(self):
        return self.get_var()
    
    @property
    def V_m(self):
        return self.get_var()
    
    
class NESTSpikeMultimeter(NESTMultimeter, NESTSpikeRecorder, SpikeMultimeter):

    """NESTSpikeMultimeter class to wrap around a NEST multimeter device
       that records only from continuous time spike weights variable."""

    model = "spike_multimeter"
    spike_var = "spikes"

    def __init__(self, device, nest_instance, *args, **kwargs):
        super(NESTSpikeMultimeter, self).__init__(device, nest_instance, *args, **kwargs)
        self.model = "spike_multimeter"


NESTOutputDeviceDict = {"spike_recorder": NESTSpikeRecorder,
                        "multimeter": NESTMultimeter,
                        "spike_multimeter": NESTSpikeMultimeter,
                        "voltmeter": NESTVoltmeter}


NESTOutputSpikeDeviceDict = {"spike_recorder": NESTSpikeRecorder,
                             "spike_multimeter": NESTSpikeMultimeter}

