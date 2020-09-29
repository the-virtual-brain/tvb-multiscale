# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

from tvb_multiscale.core.spiking_models.devices import \
    Device, InputDevice, OutputDevice, SpikeDetector, SpikeRecorder, Multimeter, Voltmeter, SpikeMultimeter

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, list_of_dicts_to_dicts_of_ndarrays

# These classes wrap around NEST commands.


class NESTDevice(Device):
    __metaclass__ = ABCMeta

    nest_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, device, nest_instance):
        self.nest_instance = nest_instance
        super(NESTDevice, self).__init__(device)
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
    model = "input_device"

    def __init__(self, device, nest_instance):
        super(NESTInputDevice, self).__init__(device, nest_instance)
        self.model = "input_device"


class NESTPoissonGenerator(NESTInputDevice):
    model = "poisson_generator"

    def __init__(self, device, nest_instance):
        super(NESTPoissonGenerator, self).__init__(device, nest_instance)
        self.model = "poisson_generator"


class NESTSinusoidalPoissonGenerator(NESTInputDevice):
    model = "sinusoidal_poisson_generator"

    def __init__(self, device, nest_instance):
        super(NESTSinusoidalPoissonGenerator, self).__init__(device, nest_instance)
        self.model = "sinusoidal_poisson_generator"


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):
    model = "inhomogeneous_poisson_generator"

    def __init__(self, device, nest_instance):
        super(NESTInhomogeneousPoissonGenerator, self).__init__(device, nest_instance)
        self.model = "inhomogeneous_poisson_generator"


class NESTMIPGenerator(NESTInputDevice):
    model = "mip_generator"

    def __init__(self, device, nest_instance):
        super(NESTMIPGenerator, self).__init__(device, nest_instance)
        self.model = "mip_generator"


class NESTGammaSupGenerator(NESTInputDevice):
    model = "gamma_sup_generator"

    def __init__(self, device, nest_instance):
        super(NESTGammaSupGenerator, self).__init__(device, nest_instance)
        self.model = "gamma_sup_generator"


class NESTDPPDSupGenerator(NESTInputDevice):
    model = "ppd_sup_generator"

    def __init__(self, device, nest_instance):
        super(NESTDPPDSupGenerator, self).__init__(device, nest_instance)
        self.model = "ppd_sup_generator"


class NESTSpikeGenerator(NESTInputDevice):
    model = "spike_generator"

    def __init__(self, device, nest_instance):
        super(NESTSpikeGenerator, self).__init__(device, nest_instance)
        self.model = "spike_generator"


class NESTPulsePacketGenerator(NESTInputDevice):
    model = "pulse_packet_generator"

    def __init__(self, device, nest_instance):
        super(NESTPulsePacketGenerator, self).__init__(device, nest_instance)
        self.model = "pulse_packet_generator"


class NESTDCGenerator(NESTInputDevice):
    model = "dc_generator"

    def __init__(self, device, nest_instance):
        super(NESTDCGenerator, self).__init__(device, nest_instance)
        self.model = "dc_generator"


class NESTStepCurrentGenerator(NESTInputDevice):
    model = "step_current_generator"

    def __init__(self, device, nest_instance):
        super(NESTStepCurrentGenerator, self).__init__(device, nest_instance)
        self.model = "step_current_generator"


class NESTACGenerator(NESTInputDevice):
    model = "ac_generator"

    def __init__(self, device, nest_instance):
        super(NESTACGenerator, self).__init__(device, nest_instance)
        self.model = "ac_generator"


class NESTStepRateGenerator(NESTInputDevice):
    model = "step_rate_generator"

    def __init__(self, device, nest_instance):
        super(NESTStepRateGenerator, self).__init__(device, nest_instance)
        self.model = "step_rate_generator"


class NESTNoiseGenerator(NESTInputDevice):
    model = "noise_generator"

    def __init__(self, device, nest_instance):
        super(NESTNoiseGenerator, self).__init__(device, nest_instance)
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
    model = "output_device"

    def __init__(self, device, nest_instance):
        super(NESTOutputDevice, self).__init__(device, nest_instance)
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


class NESTSpikeDetector(NESTOutputDevice, SpikeDetector):
    model = "spike_detector"

    def __init__(self, device, nest_instance):
        super(NESTSpikeDetector, self).__init__(device, nest_instance)
        self.model = "spike_detector"

    # Only SpikeDetector is the target of connections with neurons in NEST:

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


class NESTSpikeRecorder(NESTSpikeDetector, SpikeRecorder):
    model = "spike_recorder"

    def __init__(self, device, nest_instance):
        super(NESTSpikeRecorder, self).__init__(device, nest_instance)
        self.model = "spike_recorder"


class NESTMultimeter(NESTOutputDevice, Multimeter):
    model = "multimeter"

    def __init__(self, device, nest_instance):
        super(NESTMultimeter, self).__init__(device, nest_instance)
        self.model = "multimeter"

    @property
    def record_from(self):
        self._assert_nest()
        return [str(name) for name in self.device.get('record_from')]
    
    
class NESTVoltmeter(NESTMultimeter, Voltmeter):
    model = "voltmeter"

    def __init__(self, device, nest_instance):
        super(NESTVoltmeter, self).__init__(device, nest_instance)
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
    
    
class NESTSpikeMultimeter(NESTMultimeter, NESTSpikeDetector, SpikeMultimeter):
    model = "spike_multimeter"
    spike_var = "spikes"

    def __init__(self, device, nest_instance):
        super(NESTSpikeMultimeter, self).__init__(device, nest_instance)
        self.model = "spike_multimeter"


NESTOutputDeviceDict = {"spike_detector": NESTSpikeDetector,
                        "spike_recorder": NESTSpikeRecorder,
                        "multimeter": NESTMultimeter,
                        "spike_multimeter": NESTSpikeMultimeter,
                        "voltmeter": NESTVoltmeter}


NESTOutputSpikeDeviceDict = {"spike_detector": NESTSpikeDetector,
                             "spike_recorder": NESTSpikeRecorder,
                             "spike_multimeter": NESTSpikeMultimeter}

