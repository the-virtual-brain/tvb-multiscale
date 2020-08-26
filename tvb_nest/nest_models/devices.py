# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from tvb_multiscale.spiking_models.devices import \
    Device, InputDevice, OutputDevice, SpikeDetector, Multimeter, Voltmeter, SpikeMultimeter

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, list_of_dicts_to_dicts_of_ndarrays

# These classes wrap around NEST commands.


class NESTDevice(Device):

    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor_type"

    def __init__(self, device, nest_instance):
        super(NESTDevice, self).__init__(device)
        self.nest_instance = nest_instance
        self.model = "device"

    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        try:
            self.nest_instance.GetStatus(self.device)[0]["element_type"]
        except:
            raise ValueError("Failed to Get device %s!" % str(self.device))

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    @property
    def nest_model(self):
        return str(self.nest_instance.GetStatus(self.device)[0]["model"])

    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        self.nest_instance.SetStatus(self.device, values_dict)

    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
           Returns:
            Dictionary of attributes.
        """
        if attrs is None:
            return self.nest_instance.GetStatus(self.device)[0]
        else:
            return self.nest_instance.GetStatus(self.device, attrs)[0]

    def _GetConnections(self, **kwargs):
        """Method to get attributes of the connections from/to the device
           Return:
            connections' objects
        """
        return self.nest_instance.GetConnections(**kwargs)

    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the device
            Arguments:
             connections: connections' objects.
             values_dict: dictionary of attributes names' and values.
            Returns:
             Dictionary of arrays of connections' attributes.
        """
        self.nest_instance.SetStatus(connections, values_dict)

    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of arrays of connections' attributes.
        """
        if attrs is None:
            return list_of_dicts_to_dicts_of_ndarrays(self.nest_instance.GetStatus(connections))
        else:
            attrs = ensure_list(attrs)
            return OrderedDict(zip(attrs, np.array(self.nest_instance.GetStatus(connections, attrs))))

    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.device,
                                    target=self.filter_neurons(neurons, exclude_neurons))

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(source=self.device)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected from."""
        return tuple([conn[1] for conn in self.connections])


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

    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.device,
                                    target=self.filter_neurons(neurons, exclude_neurons))

    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(source=self.device)
        
    @property
    def events(self):
        return self.nest_instance.GetStatus(self.device)[0]["events"]

    @property
    def number_of_events(self):
        return self.nest_instance.GetStatus(self.device, "n_events")[0]

    @property
    def n_events(self):
        return self.number_of_events
    
    @property
    def reset(self):
        self.nest_instance.SetStatus(self.device, {'n_events': 0})


class NESTSpikeDetector(NESTOutputDevice, SpikeDetector):
    model = "spike_detector"

    def __init__(self, device, nest_instance):
        super(NESTSpikeDetector, self).__init__(device, nest_instance)
        self.model = "spike_detector"

    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.filter_neurons(neurons, exclude_neurons), 
                                    target=self.device)

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(target=self.device)

    @property
    def neurons(self):
        return tuple([conn[0] for conn in self.connections])
    
    
class NESTMultimeter(NESTOutputDevice, Multimeter):
    model = "multimeter"

    def __init__(self, device, nest_instance):
        super(NESTMultimeter, self).__init__(device, nest_instance)
        self.model = "multimeter"

    @property
    def record_from(self):
        return [str(name) for name in self.nest_instance.GetStatus(self.device)[0]['record_from']]
    
    
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
                        "multimeter": NESTMultimeter,
                        "spike_multimeter": NESTSpikeMultimeter,
                        "voltmeter": NESTVoltmeter}


NESTOutputSpikeDeviceDict = {"spike_detector": NESTSpikeDetector,
                             "spike_multimeter": NESTSpikeMultimeter}

