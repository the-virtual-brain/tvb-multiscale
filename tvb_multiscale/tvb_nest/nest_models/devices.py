# -*- coding: utf-8 -*-

import os
from abc import ABCMeta
import glob

import numpy as np
from numpy.lib.recfunctions import rename_fields
import xarray as xr

from tvb_multiscale.core.spiking_models.devices import \
    Device, InputDevice, SpikeRecorder, Multimeter, Voltmeter, SpikeMultimeter  # OutputDevice,
from tvb_multiscale.core.utils.data_structures_utils import flatten_neurons_inds_in_DataArray
from tvb_multiscale.core.utils.file_utils import truncate_ascii_file_after_header

from tvb_multiscale.tvb_nest.nest_models.node import _NESTNodeCollection
from tvb_multiscale.tvb_nest.nest_models.population import NESTParrotPopulation

from tvb.basic.neotraits.api import Attr, Int, List

from tvb.contrib.scripts.utils.data_structures_utils \
    import ensure_list, extract_integer_intervals, data_xarray_from_continuous_events


# These classes wrap around NEST commands.


class NESTDevice(_NESTNodeCollection):
    __metaclass__ = ABCMeta

    """NESTDevice class to wrap around a NEST output (recording) or input (stimulating) device"""

    from nest import NodeCollection

    device = Attr(field_type=NodeCollection, default=NodeCollection(), required=False,
                  label="NEST device ", doc="""Device NodeCollection instance""")

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        _NESTNodeCollection.__init__(self, device, nest_instance, **kwargs)

    def _assert_device(self):
        self._assert_nodes()
        """Method to assert that the node of the network is a device"""
        try:
            self.device.get("element_type")
        except:
            raise ValueError("Failed to Get device %s!" % str(self.device))

    @property
    def virtual_process_id(self):
        if self.device:
            return self.device.get("vp")
        else:
            return None

    def GetConnections(self, *args, **kwargs):
        """Method to get all connections of the device from neurons.
           Returns:
            SynapseCollection.
        """
        return self._GetConnections(self.device, source_or_target="source")

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

    def _print_neurons(self, neurons):
        return "%d neurons: %s" % (self.number_of_neurons, extract_integer_intervals(neurons, print=True))


class NESTInputDevice(NESTDevice, InputDevice):

    """NESTInputDevice class to wrap around a NEST input (stimulating) device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        NESTDevice.__init__(self, device, nest_instance, **kwargs)
        Device.__init__(self, device, **kwargs)

    def print_str(self, connectivity=False):
        return InputDevice.print_str(self, connectivity, "source")


class NESTPoissonGenerator(NESTInputDevice):

    """NESTPoissonGenerator class to wrap around a NEST poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTSinusoidalPoissonGenerator(NESTInputDevice):

    """NESTSinusoidalPoissonGenerator class to wrap around a NEST sinusoidal_poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "sinusoidal_poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):

    """NESTInhomogeneousPoissonGenerator class to wrap around a NEST inhomogeneous_poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "inhomogeneous_poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTMIPGenerator(NESTInputDevice):

    """NESTMIPGenerator class to wrap around a NEST mip_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "mip_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTGammaSupGenerator(NESTInputDevice):

    """NESTGammaSupGenerator class to wrap around a NEST gamma_sup_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "gamma_sup_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTDPPDSupGenerator(NESTInputDevice):

    """NESTDPPDSupGenerator class to wrap around a NEST ppd_sup_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ppd_sup_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTSpikeGenerator(NESTInputDevice):

    """NESTSpikeGenerator class to wrap around a NEST spike_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)

    @property
    def number_of_devices(self):
        if self.device:
            return len(self.device)
        else:
            return 0

    def _assert_value_size(self, val):
        try:
            n_vals = len(val)
            if n_vals != self.number_of_devices:
                if n_vals in [0, 1]:
                    val = [val] * self.number_of_devices
                else:
                    raise ValueError("The number of input values %s to set() method of the NESTSpikeGenerator"
                                     "is neither 0, nor 1 or equal to the number of devices %s!"
                                     % (n_vals, self.number_of_devices))
        except:
            val = [val] * self.number_of_devices
        return val

    def Set(self, values_dict, nodes=None):
        if nodes is None:
            nodes = self.device
        for key, vals in values_dict.items():
            for i_dev, val in enumerate(self._assert_value_size(vals)):
                nodes[i_dev].set({key: val})


class NESTPulsePacketGenerator(NESTInputDevice):

    """NESTPulsePacketGenerator class to wrap around a NEST pulse_packet_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "pulse_packet_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTDCGenerator(NESTInputDevice):

    """NESTDCGenerator class to wrap around a NEST dc_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "dc_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTStepCurrentGenerator(NESTInputDevice):

    """NESTStepCurrentGenerator class to wrap around a NEST step_current_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "step_current_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTACGenerator(NESTInputDevice):

    """NESTACGenerator class to wrap around a NEST ac_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ac_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTStepRateGenerator(NESTInputDevice):

    """NESTStepRateGenerator class to wrap around a NEST step_rate_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "step_rate_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTNoiseGenerator(NESTInputDevice):

    """NESTNoiseGenerator class to wrap around a NEST noise_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "noise_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTParrotInputDevice(NESTInputDevice, NESTParrotPopulation):

    """NESTParrotInputDevice class to combine a NEST InputDevice with a parrot_neuron population"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        label = str(kwargs.get("label", ""))
        model = str(kwargs.get("model", "nest_parrot_input_device"))
        NESTInputDevice.__init__(self, device=device, nest_instance=nest_instance, **kwargs)
        NESTParrotPopulation.__init__(self, nodes=population, nest_instance=nest_instance, **kwargs)
        self.device = device
        self._nodes = population
        self.label = label
        self.model = model
        self._number_of_connections = 0
        self._number_of_neurons = self.get_size()

    def Set(self, values_dict, neurons=None):
        if neurons is None:
            NESTInputDevice.Set(self, values_dict, self.device)
        else:
            NESTParrotPopulation.Set(self, values_dict, neurons)

    def Get(self, attrs=None, neurons=None, summary=None):
        if neurons is None:
            return NESTInputDevice.Get(self, attrs, self.device, summary)
        else:
            return NESTParrotPopulation.Get(self, attrs, neurons, summary)

    def get_attributes(self, neurons=None, summary=False):
        if neurons is None:
            return NESTInputDevice.get_attributes(self, self.device, summary)
        else:
            return NESTParrotPopulation.get_attributes(self, neurons, summary)

    def _default_neurons_and_source_or_target(self, neurons=None, source_or_target=None):
        if neurons is None:
            neurons = self._nodes
        if source_or_target is None:
            source_or_target = "source"
        return neurons, source_or_target

    def GetConnections(self, neurons=None, source_or_target=None):
        return NESTParrotPopulation.GetConnections(self,
                                                   *self._default_neurons_and_source_or_target(neurons,
                                                                                                source_or_target))

    def SetToConnections(self, values_dict, neurons=None, source_or_target=None):
        NESTParrotPopulation.SetToConnections(self, values_dict,
                                              *self._default_neurons_and_source_or_target(neurons,
                                                                                           source_or_target))

    def GetFromConnections(self, attrs=None, neurons=None, source_or_target=None, summary=None):
        return NESTParrotPopulation.GetFromConnections(self, attrs=attrs, summary=summary,
                                                       *self._default_neurons_and_source_or_target(neurons,
                                                                                                    source_or_target))

    def get_weights(self, neurons=None, source_or_target=None, summary=None):
        return NESTParrotPopulation.get_weights(self, summary=summary,
                                                *self._default_neurons_and_source_or_target(neurons,
                                                                                             source_or_target))

    def get_delays(self, neurons=None, source_or_target=None, summary=None):
        return NESTParrotPopulation.get_delays(self, summary=summary,
                                               *self._default_neurons_and_source_or_target(neurons,
                                                                                            source_or_target))

    def get_receptors(self, neurons=None, source_or_target=None, summary=None):
        return NESTParrotPopulation.get_receptors(self, summary=summary,
                                                  *self._default_neurons_and_source_or_target(neurons,
                                                                                               source_or_target))

    def _GetConnections(self, neurons=None, source_or_target=None):
        return NESTParrotPopulation._GetConnections(self,
                                                    *self._default_neurons_and_source_or_target(neurons,
                                                                                                 source_or_target))

    def _GetFromConnections(self, attrs=None, connections=None):
        return NESTParrotPopulation._GetFromConnections(self, attrs, connections)

    def _SetToConnections(self, values_dict, connections=None):
        NESTParrotPopulation._SetToConnections(self, values_dict, connections)

    def get_neurons(self, source_or_target="source"):
        return NESTInputDevice.get_neurons(self, source_or_target)

    @property
    def neurons(self):
        return self._nodes.global_id

    def get_size(self):
        return NESTParrotPopulation.get_size(self)

    def _print_nodes(self):
        return NESTParrotPopulation._print_nodes(self)

    def print_str(self, connectivity=False):
        output = ""
        output += NESTInputDevice.print_str(self, connectivity=False)
        output += "\n"
        output += NESTParrotPopulation.print_str(self, connectivity)
        return output


class NESTParrotPoissonGenerator(NESTParrotInputDevice):

    """NESTPoissonGenerator class to wrap around a NEST poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotSinusoidalPoissonGenerator(NESTParrotInputDevice):

    """NESTParrotSinusoidalPoissonGenerator class to wrap around a NEST sinusoidal_poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_sinusoidal_poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotInhomogeneousPoissonGenerator(NESTParrotInputDevice):

    """NESTParrotInhomogeneousPoissonGenerator class to wrap around a NEST inhomogeneous_poisson_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_inhomogeneous_poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotMIPGenerator(NESTParrotInputDevice):

    """NESTParrotMIPGenerator class to wrap around a NEST mip_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_mip_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotGammaSupGenerator(NESTParrotInputDevice):

    """NESTParrotGammaSupGenerator class to wrap around a NEST gamma_sup_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_gamma_sup_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotDPPDSupGenerator(NESTParrotInputDevice):

    """NESTParrotDPPDSupGenerator class to wrap around a NEST ppd_sup_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_ppd_sup_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotSpikeGenerator(NESTParrotInputDevice, NESTSpikeGenerator):

    """NESTParrotSpikeGenerator class to wrap around a NEST spike_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_spike_generator")
        NESTSpikeGenerator.__init__(self, device, nest_instance, **kwargs)
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)

    def Set(self, values_dict, neurons=None):
        if neurons is None:
            NESTSpikeGenerator.Set(self, values_dict, self.device)
        else:
            NESTParrotPopulation.Set(self, values_dict, neurons)


class NESTParrotPulsePacketGenerator(NESTParrotInputDevice):

    """NESTParrotPulsePacketGenerator class to wrap around a NEST pulse_packet_generator device"""

    from nest import NodeCollection

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_pulse_packet_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


NESTSpikeInputDeviceDict = {"poisson_generator": NESTPoissonGenerator,
                            "sinusoidal_poisson_generator": NESTSinusoidalPoissonGenerator,
                            "inhomogeneous_poisson_generator": NESTInhomogeneousPoissonGenerator,
                            "mip_generator": NESTMIPGenerator,
                            "gamma_sup_generator": NESTGammaSupGenerator,
                            "ppd_sup_generator": NESTDPPDSupGenerator,
                            "spike_generator": NESTSpikeGenerator,
                            "pulse_packet_generator": NESTPulsePacketGenerator
                            }


NESTParrotSpikeInputDeviceDict = {"parrot_poisson_generator": NESTParrotPoissonGenerator,
                                  "parrot_sinusoidal_poisson_generator": NESTParrotSinusoidalPoissonGenerator,
                                  "parrot_inhomogeneous_poisson_generator": NESTParrotInhomogeneousPoissonGenerator,
                                  "parrot_mip_generator": NESTParrotMIPGenerator,
                                  "parrot_gamma_sup_generator": NESTParrotGammaSupGenerator,
                                  "parrot_ppd_sup_generator": NESTParrotDPPDSupGenerator,
                                  "parrot_spike_generator": NESTParrotSpikeGenerator,
                                  "parrot_pulse_packet_generator": NESTParrotPulsePacketGenerator
                                  }


NESTCurrentInputDeviceDict = {"dc_generator": NESTDCGenerator,
                              "step_current_generator": NESTStepCurrentGenerator,
                              "ac_generator": NESTACGenerator,
                              "step_rate_generator": NESTStepRateGenerator,
                              "noise_generator": NESTNoiseGenerator
                              }


NESTInputDeviceDict = {}
NESTSpikeInputDeviceDict.update(NESTParrotSpikeInputDeviceDict)
NESTInputDeviceDict.update(NESTSpikeInputDeviceDict)
NESTInputDeviceDict.update(NESTCurrentInputDeviceDict)


def read_nest_output_device_data_from_ascii_to_dict(filepath):
    """This function reads data from a NEST recording device ascii file into an events dictionary
       Arguments:
        - filepath: absolute or relative path to the file (string)
       Returns:
        the events dictionary of the recorded data
    """
    recarray = rename_fields(np.genfromtxt(filepath, names=True, skip_header=2),
                             {"sender": "senders", "time_ms": "times"})
    return {name: ensure_list(recarray[name]) for name in recarray.dtype.names}


class NESTOutputDevice(NESTDevice):

    """NESTOutputDevice class to wrap around a NEST output (recording) device"""

    _output_events_counter = Int(field_type=int, default=0, required=True, label="Number of output events",
                                 doc="""The number of recorded events that 
                                          have been given to the output via a get_events() call.""")

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "nest_output_device")
        NESTDevice.__init__(self, device, nest_instance, **kwargs)
        if kwargs.get("record_to", "ascii") == "ascii":
            self._get_events = self._get_events_from_ascii
            self._reset = self._delete_events_in_ascii_files
        else:
            self._get_events = self._get_events_from_memory
            self._reset = self._delete_events_in_memory

    @property
    def record_from(self):
        return []

    def _get_filenames(self):
        return glob.glob(os.path.join(self.nest_instance.GetKernelStatus("data_path"), "%s*" % self.label))

    @property
    def _empty_events(self):
        keys = ["times", "senders"] + self.record_from
        return dict(zip(keys, [[]]*len(keys)))

    def _get_events_from_ascii(self):
        events = self._empty_events
        filenames = self._get_filenames()
        for filepath in filenames:
            this_file_events = read_nest_output_device_data_from_ascii_to_dict(filepath)
            for key in events.keys():
                events[key] = events[key] + this_file_events[key]
        return events

    def _get_events_from_memory(self):
        if self.device:
            return self.device.get("events")
        else:
            return {"times": [], "senders": []}

    def get_events(self, **kwargs):
        if self.device:
            events = super(NESTOutputDevice, self).get_events(**kwargs)
            # Advance the _output_events_counter
            self._output_events_counter = self.device.get("n_events")
            return events
        else:
            return {"times": [], "senders": []}

    def get_new_events(self, variables=None, **filter_kwargs):
        return self.get_events(variables=variables, events_inds=self._output_events_counter, **filter_kwargs)

    @property
    def events(self):
        return self._get_events()

    @property
    def number_of_events(self):
        if self.device:
            return self.device.get("n_events")
        else:
            return 0

    @property
    def number_of_new_events(self):
        return self.number_of_events - self._output_events_counter

    @property
    def n_events(self):
        return self.number_of_events

    def _delete_events_in_ascii_files(self):
        for filepath in self._get_filenames():
            truncate_ascii_file_after_header(filepath, header_chars="#")

    def _delete_events_in_memory(self):
        # TODO: find how to reset recorders!
        pass

    def reset(self):
        self._reset()


class NESTSpikeRecorder(NESTOutputDevice, SpikeRecorder):

    """NESTSpikeRecorder class to wrap around a NEST spike_recorder device"""

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_recorder")
        NESTOutputDevice.__init__(self, device, nest_instance, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)

    # Only SpikeRecorder is the target of connections with neurons in NEST:

    def GetConnections(self, *args, **kwargs):
        """Method to get connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(self.device, source_or_target="target")

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.get_neurons("source")

    def print_str(self, connectivity=False):
        return SpikeRecorder.print_str(self, connectivity, "target")


class NESTMultimeter(NESTOutputDevice, Multimeter):

    """NESTMultimeter class to wrap around a NEST multimeter device"""

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "multimeter")
        NESTOutputDevice.__init__(self, device, nest_instance, **kwargs)
        Multimeter.__init__(self, device, **kwargs)

    @property
    def record_from(self):
        if self.device:
            return [str(name) for name in self.device.get('record_from')]
        else:
            return []

    def get_data(self, variables=None, name=None, dims_names=["Time", "Variable", "Neuron"],
                 flatten_neurons_inds=True, new=False):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None, which defaults to the label of the Device
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
            flatten_neurons_inds: if true, neurons coordinates are arranged from 1 to number of neurons,
                                  instead for neurons_inds
            new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
          Returns:
            a xarray DataArray with the output data
        """
        if name is None:
            name = self.label
        if new:
            events = self.get_new_events()
        else:
            events = self.events
        times = events.get("times")
        senders = events.get("senders")
        if len(times) + len(senders):
            data = data_xarray_from_continuous_events(events, times, senders,
                                                      variables=self._determine_variables(variables),
                                                      name=name, dims_names=dims_names)
            if flatten_neurons_inds:
                data = flatten_neurons_inds_in_DataArray(data, data.dims[2])
        else:
            vars = self._determine_variables(variables)
            data = xr.DataArray(np.empty((len(times), len(vars), len(senders))), name=name, dims=dims_names,
                                coords={dims_names[0]: times, dims_names[1]: vars, dims_names[2]: senders})
        return data

    def print_str(self, connectivity=False):
        return Multimeter.print_str(self, connectivity, "source")


class NESTVoltmeter(NESTMultimeter, Voltmeter):

    """NESTVoltmeter class to wrap around a NEST voltmeter device"""

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "voltmeter")
        NESTMultimeter.__init__(self, device, nest_instance, **kwargs)
        Voltmeter.__init__(self, device, **kwargs)
        assert self.var in self.record_from
        
    @property
    def var(self):
        return "V_m"
    
    @property
    def get_V_m(self):
        return self.var

    @property
    def V_m(self):
        return self.var
    
    def print_str(self, connectivity=False):
        return Voltmeter.print_str(self, connectivity, "source")


class NESTSpikeMultimeter(NESTMultimeter, NESTSpikeRecorder, SpikeMultimeter):

    """NESTSpikeMultimeter class to wrap around a NEST multimeter device
       that records only from continuous time spike weights variable."""

    spike_vars = List(of=str, default=("spikes"),
                      label="Spike variables' labels list",
                      doc="""List of labels of spike variables""")

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_multimeter")
        NESTMultimeter.__init__(self, device, nest_instance, **kwargs)
        NESTSpikeRecorder.__init__(self, device, nest_instance, **kwargs)
        SpikeMultimeter.__init__(self, device, **kwargs)

    def print_str(self, connectivity=False):
        return NESTMultimeter.print_str(self, connectivity)


NESTOutputSpikeDeviceDict = {"spike_recorder": NESTSpikeRecorder}


NESTOutputContinuousTimeDeviceDict = {"multimeter": NESTMultimeter,
                                      "spike_multimeter": NESTSpikeMultimeter,
                                      "voltmeter": NESTVoltmeter}


NESTOutputDeviceDict = {}
NESTOutputDeviceDict.update(NESTOutputSpikeDeviceDict)
NESTOutputDeviceDict.update(NESTOutputContinuousTimeDeviceDict)
