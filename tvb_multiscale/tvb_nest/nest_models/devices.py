# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import xarray as xr

from tvb.basic.neotraits.api import Int, List, Attr

from tvb.contrib.scripts.utils.log_error_utils import warning
from tvb.contrib.scripts.utils.data_structures_utils \
    import ensure_list, extract_integer_intervals, data_xarray_from_continuous_events

from nest import NodeCollection

from tvb_multiscale.core.spiking_models.devices import \
    Device, InputDevice, SpikeRecorder, Multimeter, Voltmeter, SpikeMultimeter
from tvb_multiscale.core.utils.data_structures_utils import flatten_neurons_inds_in_DataArray
from tvb_multiscale.core.utils.file_utils import truncate_ascii_file_after_header, \
    read_nest_output_device_data_from_ascii_to_dict
from tvb_multiscale.tvb_nest.nest_models.node import _NESTNodeCollection
from tvb_multiscale.tvb_nest.nest_models.population import NESTParrotPopulation


# These classes wrap around NEST commands.


class NESTDevice(_NESTNodeCollection):
    __metaclass__ = ABCMeta

    """NESTDevice class to wrap around a NEST output (recording) or input (stimulating) device"""


    # device = Attr(field_type=NodeCollection, default=NodeCollection(), required=False,
    #               label="NEST device ", doc="""Device NodeCollection instance""")

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        self.device = device
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
        connections = self.connections
        if len(connections):
            return tuple(np.unique(getattr(connections, source_or_target)).tolist())
        else:
            return ()

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.get_neurons("target")


class NESTInputDevice(NESTDevice, InputDevice):

    """NESTInputDevice class to wrap around a NEST input (stimulating) device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        NESTDevice.__init__(self, device, nest_instance, **kwargs)
        Device.__init__(self, device, **kwargs)

    def __str__(self):
        return InputDevice.__str__(self)

    def info(self, recursive=0):
        return InputDevice.info(self, recursive=recursive)

    def info_neurons(self, **kwargs):
        return self.info_connections("source")

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return InputDevice.info_details(self, recursive=recursive,
                                        connectivity=connectivity, source_or_target="source")


class NESTPoissonGenerator(NESTInputDevice):

    """NESTPoissonGenerator class to wrap around a NEST poisson_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTSinusoidalPoissonGenerator(NESTInputDevice):

    """NESTSinusoidalPoissonGenerator class to wrap around a NEST sinusoidal_poisson_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "sinusoidal_poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):

    """NESTInhomogeneousPoissonGenerator class to wrap around a NEST inhomogeneous_poisson_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "inhomogeneous_poisson_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTMIPGenerator(NESTInputDevice):

    """NESTMIPGenerator class to wrap around a NEST mip_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "mip_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTGammaSupGenerator(NESTInputDevice):

    """NESTGammaSupGenerator class to wrap around a NEST gamma_sup_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "gamma_sup_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTDPPDSupGenerator(NESTInputDevice):

    """NESTDPPDSupGenerator class to wrap around a NEST ppd_sup_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ppd_sup_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTSpikeGenerator(NESTInputDevice):

    """NESTSpikeGenerator class to wrap around a NEST spike_generator device"""

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

    def add_spikes(self, spikes, time_shift=None, nodes=None, sort=False):
        if len(spikes):
            if nodes is None:
                nodes = self.device
            n_neurons = len(nodes)
            current_time = self.nest_instance.GetKernelStatus("biological_time")
            if time_shift:
                # Apply time_shift, if any
                new_spikes = []
                for i_sp, spike in enumerate(spikes):
                    new_spike = np.array(spike) + time_shift
                    if len(new_spike.shape):
                        # if spikes is a sequence of sequences for each neuron
                        new_spikes.append(ensure_list(new_spike[new_spike > current_time]))
                    else:
                        # if spikes is a sequence of the same spikes for all neurons
                        if new_spike > current_time:
                            new_spikes.append(new_spike.item())
                spikes = new_spikes
            if len(spikes) != n_neurons:
                # Assume we have to add the same spikes to all neurons
                spikes = ensure_list(spikes) * len(nodes)
            for neuron, new_spikes in zip(nodes, spikes):
                old_spikes = np.array(neuron.get("spike_times"))
                old_spikes = old_spikes[old_spikes > current_time].tolist()
                these_spikes = old_spikes + ensure_list(new_spikes)
                if len(these_spikes) and sort:
                    these_spikes = np.sort(these_spikes).tolist()
                neuron.set({"spike_times": these_spikes})


class NESTPulsePacketGenerator(NESTInputDevice):

    """NESTPulsePacketGenerator class to wrap around a NEST pulse_packet_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "pulse_packet_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTDCGenerator(NESTInputDevice):

    """NESTDCGenerator class to wrap around a NEST dc_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "dc_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTStepCurrentGenerator(NESTInputDevice):

    """NESTStepCurrentGenerator class to wrap around a NEST step_current_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "step_current_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTACGenerator(NESTInputDevice):

    """NESTACGenerator class to wrap around a NEST ac_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ac_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTStepRateGenerator(NESTInputDevice):

    """NESTStepRateGenerator class to wrap around a NEST step_rate_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "step_rate_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTNoiseGenerator(NESTInputDevice):

    """NESTNoiseGenerator class to wrap around a NEST noise_generator device"""

    def __init__(self, device=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "noise_generator")
        NESTInputDevice.__init__(self, device, nest_instance, **kwargs)


class NESTParrotInputDevice(NESTInputDevice, NESTParrotPopulation):

    """NESTParrotInputDevice class to combine a NEST InputDevice with a parrot_neuron population"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        label = str(kwargs.get("label", ""))
        model = str(kwargs.get("model", "nest_parrot_input_device"))
        NESTInputDevice.__init__(self, device=device, nest_instance=nest_instance, **kwargs)
        NESTParrotPopulation.__init__(self, nodes=population, nest_instance=nest_instance, **kwargs)
        self.device = device
        self._nodes = population
        self.label = label
        self.model = model

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

    @property
    def neurons(self):
        return self.get_neurons("target")

    def get_devices_neurons(self):
        return self._nodes

    @property
    def parrot_neurons(self):
        return self._nodes

    @property
    def number_of_neurons(self):
        return NESTInputDevice.number_of_neurons.fget(self)

    @property
    def number_of_connected_neurons(self):
        return self.number_of_neurons

    @property
    def number_of_device_neurons(self):
        return self.get_size()

    def info_nodes(self):
        return NESTParrotPopulation.info_nodes(self)

    def info_neurons(self, **kwargs):
        return NESTParrotPopulation.info_connections(self, "source")

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return NESTInputDevice.info_details(recursive=recursive, connectivity=False) + \
               NESTParrotPopulation.info_details(self, recursive=recursive,
                                                 connectivity=connectivity, source_or_target="source")


class NESTParrotPoissonGenerator(NESTParrotInputDevice):

    """NESTPoissonGenerator class to wrap around a NEST poisson_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotSinusoidalPoissonGenerator(NESTParrotInputDevice):

    """NESTParrotSinusoidalPoissonGenerator class to wrap around a NEST sinusoidal_poisson_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_sinusoidal_poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotInhomogeneousPoissonGenerator(NESTParrotInputDevice):

    """NESTParrotInhomogeneousPoissonGenerator class to wrap around a NEST inhomogeneous_poisson_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_inhomogeneous_poisson_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotMIPGenerator(NESTParrotInputDevice):

    """NESTParrotMIPGenerator class to wrap around a NEST mip_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_mip_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotGammaSupGenerator(NESTParrotInputDevice):

    """NESTParrotGammaSupGenerator class to wrap around a NEST gamma_sup_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_gamma_sup_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotDPPDSupGenerator(NESTParrotInputDevice):

    """NESTParrotDPPDSupGenerator class to wrap around a NEST ppd_sup_generator device"""

    def __init__(self, device=NodeCollection(), population=NodeCollection(), nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "parrot_ppd_sup_generator")
        NESTParrotInputDevice.__init__(self, device, population, nest_instance, **kwargs)


class NESTParrotSpikeGenerator(NESTParrotInputDevice, NESTSpikeGenerator):

    """NESTParrotSpikeGenerator class to wrap around a NEST spike_generator device"""

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


class NESTOutputDevice(NESTDevice):
    __metaclass__ = ABCMeta

    """NESTOutputDevice class to wrap around a NEST output (recording) device"""

    _total_num_virtual_procs = Int(field_type=int, default=1, required=True,
                                   label="Total number of NEST virtual processes",
                                   doc="""Total number of NEST virtual processes""")

    _record_to = None
    reset_upon_record = False

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "nest_output_device")
        NESTDevice.__init__(self, device, nest_instance, **kwargs)
        self._total_num_virtual_procs = self.nest_instance.GetKernelStatus("total_num_virtual_procs")
        self._update_record_to()
        self.reset_upon_record = kwargs.get('reset_upon_record', self.reset_upon_record)

    def __getstate__(self):
        d = super(NESTOutputDevice, self).__getstate__()
        d["_total_num_virtual_procs"] = self._total_num_virtual_procs
        d["_record_to"] = self.record_to
        return d

    def __setstate__(self, d):
        super(NESTOutputDevice, self).__setstate__(d)
        self._total_num_virtual_procs = d.get("_total_num_virtual_procs", 1)
        self._record_to = d["_record_to"]
        self._update_record_to()

    @property
    def record_from(self):
        return []

    @property
    def record_to(self):
        if self._record_to is None and self.device is not None:
            self._record_to = self.device.get("record_to")
        return self._record_to

    def _update_record_to(self):
        if self.device:
            if self.record_to == "ascii":
                self._get_events = self._get_events_from_ascii
                self._number_of_events = self._number_of_events_in_ascii_files
                self.delete_events = self._delete_events_in_ascii_files
                self._get_new_events = self._get_new_events_from_ascii
                self._output_events_counter = [0] * self._total_num_virtual_procs
            else:
                self._get_events = self._get_events_from_memory
                self._number_of_events = self._number_of_events_in_memory
                self.delete_events = self._delete_events_in_memory
                self._get_new_events = self._get_new_events_from_memory
                self._output_events_counter = 0

    @property
    def _empty_events(self):
        keys = ["times", "senders"] + self.record_from
        return dict(zip(keys, [np.array([])]*len(keys)))

    def _get_events_from_ascii(self):
        events = self._empty_events
        for iF, filepath in enumerate(self.Get("filenames")["filenames"]):
            # Reading all events:
            this_file_events = read_nest_output_device_data_from_ascii_to_dict(filepath,
                                                                               n_lines_to_skip=0,
                                                                               empty_file=self.reset_upon_record)
            n_events = len(this_file_events["senders"])
            # Compute the new number of total events for this process...
            if n_events:
                self._output_events_counter[iF] = np.where(self.reset_upon_record,
                                                           self._output_events_counter[iF] + n_events,
                                                           n_events).item()
            if self._output_events_counter[iF]:
                # Merge file data, if any:
                for key in events.keys():
                    events[key] = np.concatenate([events[key], this_file_events[key]])
        return events

    def _get_events_from_memory(self):
        if self.device:
            return self.device.get("events")
        else:
            return self._empty_events

    def get_events(self, **kwargs):
        if self.device:
            events = super(NESTOutputDevice, self).get_events(**kwargs)
            return events
        else:
            return self._empty_events

    def _get_new_events_from_ascii(self):
        events = self._empty_events
        for iF, filepath in enumerate(self.Get("filenames")["filenames"]):  #
            # We read only new events from files:
            this_file_new_events = read_nest_output_device_data_from_ascii_to_dict(
                filepath,
                n_lines_to_skip=np.where(self.reset_upon_record, 0, self._output_events_counter[iF]).item(),
                empty_file=self.reset_upon_record)
            number_of_new_events = len(this_file_new_events.get("senders", []))
            if number_of_new_events:
                # Advance the _output_events_counter
                self._output_events_counter[iF] += number_of_new_events
                # Merge file data:
                for key in events.keys():
                    events[key] = np.concatenate([events[key], this_file_new_events[key]])
        return events

    @abstractmethod
    def _get_new_events_from_memory(self):
        pass

    def get_new_events(self, variables=None, **filter_kwargs):
        return super(NESTOutputDevice, self).get_events(events=self._get_new_events(),
                                                        variables=variables, **filter_kwargs)

    @property
    def events(self):
        return self._get_events()

    def _number_of_events_in_ascii_files(self):
        return len(self._get_events()["senders"])

    def _number_of_events_in_memory(self):
        return self.device.get("n_events")

    @property
    def number_of_events(self):
        if self.device:
            return self._number_of_events()
        else:
            return 0

    @property
    def number_of_new_events(self):
        if self.device:
            return len(self.get_new_events()["senders"])
        else:
            return 0

    @property
    def n_events(self):
        return self.number_of_events

    def _delete_events_in_ascii_files(self):
        for filepath in self.Get("filenames")["filenames"]:  #
            truncate_ascii_file_after_header(filepath, header="#")
        self._output_events_counter = [0] * self._total_num_virtual_procs

    def _delete_events_in_memory(self):
        try:
            self.device.n_events = 0
            self._output_events_counter = 0
        except Exception as e:
            warning(e)

    def reset(self):
        self.delete_events()


class NESTSpikeRecorder(NESTOutputDevice, SpikeRecorder):

    """NESTSpikeRecorder class to wrap around a NEST spike_recorder device"""

    _last_spike_time = 0.0

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_recorder")
        NESTOutputDevice.__init__(self, device, nest_instance, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)
        self._last_spike_time = self.nest_instance.GetKernelStatus("biological_time")

    # Only SpikeRecorder is the target of connections with neurons in NEST:

    def __getstate__(self):
        d = SpikeRecorder.__getstate__(self)
        d.update(NESTOutputDevice.__getstate__(self))
        d["_last_spike_time"] = self._last_spike_time
        return d

    def __setstate__(self, d):
        SpikeRecorder.__setstate__(self, d)
        NESTOutputDevice.__setstate__(self, d)
        self._last_spike_time = d["_last_spike_time"]

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

    def __get_events_from_memory(self):
        events = NESTOutputDevice._get_events_from_memory(self)
        n_total_events = self.device.n_events
        number_of_new_events = n_total_events - self._output_events_counter
        if number_of_new_events:
            self._output_events_counter = n_total_events
            self._last_spike_time = np.max(events["times"])
        return events, number_of_new_events

    def _get_events_from_memory(self):
        return self.__get_events_from_memory()[0]

    def _get_new_events_from_memory(self):
        old_last_spike_time = float(self._last_spike_time)
        events, number_of_new_events = self.__get_events_from_memory()
        if number_of_new_events:
            if self._total_num_virtual_procs > 1:
                new_spike_inds = np.where(np.array(events["times"]) > old_last_spike_time)
                for key, vals in events.items():
                    events[key] = vals[new_spike_inds]
            else:
                for key, vals in events.items():
                    events[key] = vals[-number_of_new_events:]
            return events
        else:
            return self._empty_events

    def info(self, recursive=0):
        return SpikeRecorder.info(self, recursive=recursive)

    def info_neurons(self, **kwargs):
        return SpikeRecorder.info_connections(self, "target")

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return SpikeRecorder.info_details(self, recursive=recursive,
                                          connectivity=connectivity, source_or_target="target")

    def reset(self):
        NESTOutputDevice.reset(self)
        self._last_spike_time = self.nest_instance.GetKernelStatus("biological_time")


class NESTMultimeter(NESTOutputDevice, Multimeter):

    """NESTMultimeter class to wrap around a NEST multimeter device"""

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "multimeter")
        NESTOutputDevice.__init__(self, device, nest_instance, **kwargs)
        Multimeter.__init__(self, device, **kwargs)

    def __getstate__(self):
        d = Multimeter.__getstate__(self)
        d.update(NESTOutputDevice.__getstate__(self))
        d["_output_events_counter"] = self._output_events_counter
        return d

    def __setstate__(self, d):
        Multimeter.__setstate__(self, d)
        NESTOutputDevice.__setstate__(self, d)
        self._output_events_counter = d.get("_output_events_counter", [0])

    def __str__(self):
        return Multimeter.__str__(self)

    @property
    def record_from(self):
        if self.device:
            return [str(name) for name in self.device.get('record_from')]
        else:
            return []

    def _get_events_from_memory(self):
        events = NESTOutputDevice._get_events_from_memory(self)
        self._output_events_counter = self.device.n_events
        return events

    def _get_new_events_from_memory(self):
        events = NESTOutputDevice._get_events_from_memory(self)
        n_total_events = self.device.n_events
        if n_total_events:
            number_of_new_events_per_proc = \
                (n_total_events - self._output_events_counter) / self._total_num_virtual_procs
            self._output_events_counter = n_total_events
            new_events_indices = []
            n_total_events_per_proc = n_total_events / self._total_num_virtual_procs
            for ii in range(self._total_num_virtual_procs):
                proc_end = (ii + 1)*n_total_events_per_proc
                new_events_indices += \
                    np.arange(proc_end - number_of_new_events_per_proc, proc_end).astype('i').tolist()
            if number_of_new_events_per_proc:
               # Select new data:
               for key in events.keys():
                   events[key] = events[key][new_events_indices]
            else:
                keys = list(events.keys())
                events = dict(zip(keys, []*len(keys)))
        return events

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

    def info(self, recursive=0):
        return Multimeter.info(self, recursive=recursive)

    def info_neurons(self, **kwargs):
        return Multimeter.info_connections(self, "source")

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return Multimeter.info_details(self, recursive=recursive,
                                       connectivity=connectivity, source_or_target="source")

    def reset(self):
        NESTOutputDevice.reset(self)
        self._output_events_counter = [0] * self._total_num_virtual_procs


class NESTVoltmeter(NESTMultimeter, Voltmeter):

    """NESTVoltmeter class to wrap around a NEST voltmeter device"""

    def __init__(self, device=None, nest_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "voltmeter")
        NESTMultimeter.__init__(self, device, nest_instance, **kwargs)
        Voltmeter.__init__(self, device, **kwargs)
        assert self.var in self.record_from

    def __getstate__(self):
        d = Voltmeter.__getstate__(self)
        d.update(NESTMultimeter.__getstate__(self))
        return d

    def __setstate__(self, d):
        Voltmeter.__setstate__(self, d)
        NESTMultimeter.__setstate__(self, d)

    @property
    def var(self):
        return "V_m"
    
    @property
    def get_V_m(self):
        return self.var

    @property
    def V_m(self):
        return self.var


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

    def __getstate__(self):
        d = SpikeMultimeter.__getstate__(self)
        d.update(NESTSpikeRecorder.__getstate__(self))
        d.update(NESTMultimeter.__getstate__(self))
        d["spike_vars"] = self.spike_vars
        return d

    def __setstate__(self, d):
        SpikeMultimeter.__setstate__(self, d)
        NESTSpikeRecorder.__setstate__(self, d)
        NESTMultimeter.__setstate__(self, d)
        self.spike_vars = d.get("spike_vars", self.spike_vars)


NESTOutputSpikeDeviceDict = {"spike_recorder": NESTSpikeRecorder}


NESTOutputContinuousTimeDeviceDict = {"multimeter": NESTMultimeter,
                                      "spike_multimeter": NESTSpikeMultimeter,
                                      "voltmeter": NESTVoltmeter}


NESTOutputDeviceDict = {}
NESTOutputDeviceDict.update(NESTOutputSpikeDeviceDict)
NESTOutputDeviceDict.update(NESTOutputContinuousTimeDeviceDict)
