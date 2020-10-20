# -*- coding: utf-8 -*-
from abc import ABCMeta
from collections import OrderedDict
from xarray import DataArray, combine_by_coords
import numpy as np

from tvb_multiscale.core.spiking_models.devices import \
    InputDevice, OutputDevice, SpikeRecorder, Multimeter, SpikeMultimeter
from tvb_multiscale.tvb_annarchy.annarchy_models.population import ANNarchyPopulation

from tvb.contrib.scripts.utils.data_structures_utils import flatten_list


# These classes wrap around ANNarchy commands.


class ANNarchyInputDevice(InputDevice, ANNarchyPopulation):
    __metaclass__ = ABCMeta

    """ANNarchyInputDevice class to wrap around an ANNarchy.Population, acting as an input (stimulating) device"""

    model = "input_device"

    params = {}

    def __init__(self, device,  label="", model="input_device", annarchy_instance=None, **kwargs):
        self.params = kwargs.get("params", {})
        ANNarchyPopulation.__init__(self, device, label, model, annarchy_instance, **kwargs)
        InputDevice.__init__(self, device, model=model, label=label)

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @property
    def annarchy_model(self):
        self._assert_annarchy()
        return str(self.device.get("model"))

    @property
    def device_ind(self):
        return self.population_ind

    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        ANNarchyPopulation.Set(self, values_dict)

    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all device's attributes.
           Returns:
            Dictionary of attributes.
        """
        ANNarchyPopulation.Get(self, attrs)

    def _GetConnections(self):
        """Method to get attributes of the connections from the device
           Return:
            Projections' objects
        """
        self._assert_annarchy()
        connections = ANNarchyPopulation._GetConnections(self, neurons=None, source_or_target="source")
        return connections

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from the device.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections of the device.
        """
        if connections is None:
            connections = ANNarchyPopulation._GetConnections(self, neurons=None, source_or_target="source")
        ANNarchyPopulation._SetToConnections(self, values_dict, connections)

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from the device
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            connections: Projection' object or collection (list, tuple, array) thereof.
                         If connections is a list of Projections,
                         we assume that all Projections have the same attributes.
                         Default = None, corresponding to all connections of the device.
            Returns:
             Dictionary of lists (for the possible different Projection objects) of arrays of connections' attributes.
        """
        if connections is None:
            connections = ANNarchyPopulation._GetConnections(self, neurons=None, source_or_target="source")
        ANNarchyPopulation._GetFromConnections(connections, attrs)

    def GetConnections(self):
        """Method to get connections of the device to neurons.
           Returns:
            list of Projection objects.
        """
        return self._GetConnections()

    @property
    def connections(self):
        """Method to get all connections of the device to neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections()

    def get_neurons(self):
        """Method to get the indices of all the neurons the device is connected to.
        """
        neurons = []
        for conn in self.connections:
            neuron = conn.post
            if neuron is not None and neuron not in neurons:
                neurons.append(neuron)
        return tuple(neurons)

    @property
    def neurons(self):
        return self.get_neurons()

    def get_number_of_neurons(self):
        """Method to compute the total number of ANNarchyPopulation's neurons.
            Returns:
                int: number of neurons.
        """
        if self._population is None:
            return 0
        return InputDevice.get_number_of_neurons(self)

    @property
    def number_of_neurons(self):
        """Method to get the number of all neurons connected from/to the device."""
        if not self._number_of_neurons:
            self._number_of_neurons = self.get_number_of_neurons()
        return self._number_of_neurons

    @property
    def number_of_connected_neurons(self):
        return self.number_of_neurons

    @property
    def number_of_devices_neurons(self):
        return self._population.size


"""
Input devices for spiking populations
Not yet implemented: Input devices for rate-coded populations
"""


class ANNarchySpikeSourceArray(ANNarchyInputDevice, InputDevice):

    """ANNarchySpikeSourceArray class to wrap around an ANNarchy.SpikeSourceArray,
       acting as an input (stimulating) device, by sending spikes to target neurons."""

    model = "SpikeSourceArray"

    def __init__(self, device, label="", annarchy_instance=None, **kwargs):
        super(ANNarchySpikeSourceArray, self).__init__(device, label, "SpikeSourceArray",
                                                       annarchy_instance, **kwargs)


class ANNarchyPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an ANNarchy.PoissonPopulation,
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    model = "PoissonPopulation"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyPoissonPopulation, self).__init__(device,  label, "PoissonPopulation",
                                                        annarchy_instance, **kwargs)


class ANNarchyHomogeneousCorrelatedSpikeTrains(ANNarchyInputDevice):

    """ANNarchyHomogeneousCorrelatedSpikeTrains class to wrap around
       an ANNarchy.HomogeneousCorrelatedSpikeTrains,
       acting as an input (stimulating) device, by generating and sending
       correlated Poisson spikes to target neurons."""

    model = "HomogeneousCorrelatedSpikeTrains"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyHomogeneousCorrelatedSpikeTrains, self).__init__(device,  label,
                                                                       "HomogeneousCorrelatedSpikeTrains",
                                                                       annarchy_instance, **kwargs)


class ANNarchyCurrentInjector(InputDevice):

    """ANNarchyCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a continuous quantity interpreted as a current (or potentially rate)."""

    model = "CurrentInjector"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyCurrentInjector, self).__init__(device,  label, "CurrentInjector",
                                                      annarchy_instance, **kwargs)


class ANNarchyDCCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyDCCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a constant continuous quantity interpreted as a DC current (or potentially rate)."""

    model = "DCCurrentInjector"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyDCCurrentInjector, self).__init__(device,  label, "DCCurrentInjector",
                                                        annarchy_instance, **kwargs)


class ANNarchyACCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyACCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a sinusoidaly varying continuous quantity interpreted as a AC current (or potentially rate)."""

    model = "ACCurrentInjector"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyACCurrentInjector, self).__init__(device,  label, "ACCurrentInjector",
                                                        annarchy_instance, **kwargs)


class ANNarchyTimedArray(InputDevice):

    """ANNarchyTimedArray class to wrap around a rate ANNarchy.TimedArray,
       acting as an input (stimulating) device, by generating and sending
       a set of continuous quantities interpreted as a current or rate values."""

    model = "TimedArray"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyTimedArray, self).__init__(device,  label, "TimedArray",
                                                 annarchy_instance, **kwargs)


class ANNarchyPoissonNeuron(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an PoissonNeuron model
       (from Maith et al 2020, see tvb_annarchy.annarchy.izhikevich_maith_etal),
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    model = "Poisson_neuron"

    def __init__(self, device,  label="", annarchy_instance=None, **kwargs):
        super(ANNarchyPoissonNeuron, self).__init__(device,  label, "Poisson_neuron",
                                                    annarchy_instance, **kwargs)


ANNarchyInputDeviceDict = {"PoissonPopulation": ANNarchyPoissonPopulation,
                           "HomogeneousCorrelatedSpikeTrains": ANNarchyHomogeneousCorrelatedSpikeTrains,
                           "SpikeSourceArray": ANNarchySpikeSourceArray,
                           "CurrentInjector": ANNarchyCurrentInjector,
                           "DCCurrentInjector": ANNarchyDCCurrentInjector,
                           "ACCurrentInjector": ANNarchyACCurrentInjector,
                           "TimedArray": ANNarchyTimedArray,
                           # From Maith et al 2020, see anarchy.izhikevich_maith_etal.py:
                           "Poisson_neuron": ANNarchyPoissonNeuron
                           }


class ANNarchyOutputDeviceConnection(object):

    """ANNarchyOutputDeviceConnection class holds properties of connections
       between ANNarchyOutputDevice instances and ANNarchyPopulation ones"""

    def __init__(self, pre=None, post=None):
        self.pre = pre
        self.post = post


class ANNarchyOutputDevice(OutputDevice):

    """ANNarchyOutputDevice class to wrap around ANNarchy.Monitor instances, acting as an output device"""

    _data = DataArray(np.empty((0, 0, 0, 0)),
                      dims=["Time", "Variable", "Population", "Neuron"])
    monitors = OrderedDict()
    _monitors_inds = None
    model = "output_device"
    label = ""
    annarchy_instance = None

    params = {}

    _default_connection_attrs = ["pre", "post"]
    _default_attrs = ["variables", "period", "period_offset", "start"]

    _dt = None

    def __init__(self, monitors, label="", model="output_device", annarchy_instance=None,
                 run_tvb_multiscale_init=True, **kwargs):
        if isinstance(monitors, dict):
            self.monitors = OrderedDict(monitors)
        else:
            self.monitors = OrderedDict()
        self.model = model
        self.params = kwargs.pop("params", {})
        self.label = label
        self.annarchy_instance = annarchy_instance
        self._data = DataArray(np.empty((0, 0, 0, 0)),
                               dims=["Time", "Variable", "Population", "Neuron"])
        if run_tvb_multiscale_init:
            OutputDevice.__init__(self, monitors, model=self.model, label=label)
        if self.annarchy_instance is not None:
            self._monitors_inds = self._get_monitors_inds()

    def _assert_annarchy(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _get_monitors_inds(self):
        """Method to get the indices of the devices' Monitors from list of all Monitors of the ANNarchy network."""
        monitors_inds = []
        for monitor in self.monitors.keys():
            monitors_inds.append(self.annarchy_instance.Global._network[0]["monitors"].index(monitor))
        return monitors_inds

    @property
    def monitors_inds(self):
        """Method to get the indices of the devices' Monitors from list of all Monitors of the ANNarchy network,
           for the first time and set the respective protected property_monitors_inds."""
        if self._monitors_inds is None:
            self._monitors_inds = self._get_monitors_inds()
        return self._monitors_inds

    @property
    def populations(self):
        """Method to get the ANNarchy.Population instances this device records from."""
        populations = list(self.monitors.values())
        if len(populations) == 1:
            populations = populations[0]
        return populations

    @property
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        for monitor in self.monitors.keys():
            for key, val in values_dict.items:
                setattr(monitor, key, val)

    def _set_attributes_to_dict(self, dictionary, monitor, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(monitor.get(attribute))
        else:
            dictionary[attribute] = [monitor.get(attribute)]

    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponding to all devices' attributes.
           Returns:
            Dictionary of attributes.
        """
        dictionary = {}
        for monitor in self.monitors.keys():
            if attrs is None:
                for attr in self._default_attrs:
                    self._set_attributes_to_dict(dictionary, monitor, attr)
            else:
                for attr in attrs:
                    self._set_attributes_to_dict(dictionary, monitor, attr)
        return dictionary

    def _GetConnections(self):
        """Method to get attributes of the connections from the device
           Return:
            ANNarchyOutputDeviceConnection' objects
        """
        connections = []
        for monitor, population in self.monitors.items():
            connections.append(ANNarchyOutputDeviceConnection(pre=population, post=monitor))
        return connections

    def _SetToConnections(self, values_dict, connections=None):
        pass

    def _GetFromConnections(self, attrs=None, connections=None):
        pass

    def GetConnections(self):
        """Method to get connections of the device from neurons.
           Returns:
            list of ANNarchyOutputDeviceConnection objects.
        """
        return self._GetConnections()

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            ANNarchyOutputDeviceConnection objects.
        """
        return self._GetConnections()

    def get_neurons(self):
        """Method to get the indices of all the neurons the device monitors.
        """
        neurons = []
        for pop in self.populations:
            if pop is not None and pop not in neurons:
                neurons.append(pop)
        return tuple(neurons)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device monitors."""
        return self.get_neurons()

    @property
    def record_from(self):
        return np.unique(flatten_list([str(m.variables) for m in self.monitors.keys()])).tolist()

    def _compute_times(self, times):
        """Method to merge the time vectors of ANNarchy.Monitor instances"""
        output_times = []
        for var_times in times.values():
            output_times = np.union1d(output_times,
                                      np.arange(var_times["start"], var_times["stop"] + self.dt, self.dt))
        return np.unique(output_times)

    def _record(self):
        """Method to get data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer of xarray.DataArray type."""
        for monitor, population in self.monitors.items():
            times = self._compute_times(monitor.times())
            data = monitor.get()
            data = DataArray(np.array(data.values()),
                             dims=["Time", "Variable", "Population", "Neuron"],
                             coords={"Time": times,
                                     "Variable": data.keys(),
                                     "Population": [population.name],
                                     "Neuron": population.ranks})
            self._data = combine_by_coords([self._data, data], fill_value=np.nan)

    @property
    def events(self):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        self._record()
        data = self._data.stack(Var=tuple(self._data.dims))
        coords = dict(data.coords)
        events = dict()
        events["times"] = data.coords["Time"].values
        events["senders"] = np.array([(pop, neuron)
                                      for pop, neuron in zip(data.coords["Population"].values.tolist(),
                                                             data.coords["Neuron"].values.tolist())])
        for var in coords["Variable"]:
            events[var] = data.loc[var].values
        return events

    @property
    def number_of_events(self):
        self._record()
        return self._data.size

    @property
    def n_events(self):
        return self.number_of_events

    @property
    def reset(self):
        self._record()
        self._data = DataArray(np.empty((0, 0, 0, 0)),
                               dims=["Time", "Variable", "Population", "Neuron"])


class ANNarchyMonitor(ANNarchyOutputDevice, Multimeter):

    """ANNarchyMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time quantities."""

    model = "Monitor"

    def __init__(self, monitors, label="", model="Monitor",
                 annarchy_instance=None, run_tvb_multiscale_init=True, **kwargs):
        self.model = model
        ANNarchyOutputDevice.__init__(self, monitors, label, self.model, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        if run_tvb_multiscale_init:
            Multimeter.__init__(self, monitors, model=self.model, label=self.label)


class ANNarchySpikeMonitor(ANNarchyOutputDevice, SpikeRecorder):

    """ANNarchySpikeMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of spike discrete events."""

    model = "SpikeMonitor"

    _data = []

    def __init__(self, monitors, label="", annarchy_instance=None, run_tvb_multiscale_init=True, **kwargs):
        self.model = "SpikeMonitor"
        ANNarchyOutputDevice.__init__(self, monitors, label, self.model, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        if run_tvb_multiscale_init:
            SpikeRecorder.__init__(self, monitors, model=self.model, label=self.label)

    def _record(self):
        """Method to get discrete spike events' data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer of xarray.DataArray type."""
        for i_m, (monitor, population) in enumerate(self.monitors.items()):
            if len(self._data) <= i_m:
                self._data.append(OrderedDict())
            for neuron, spikes_times in monitor.get("spike"):
                self._data[i_m].update({neuron: spikes_times})

    @property
    def events(self):
        """Method to record discrete spike events' data from ANNarchy.Monitor instances,
           and to return them in a events dictionary."""
        self._record()
        events = OrderedDict()
        events["times"] = []
        events["senders"] = []
        for i_m, (monitor, population) in enumerate(self.monitors.items()):
            population_ind = self.annarchy_instance.Global._network[0]["populations"].index(population)
            for neuron, spikes_times in self.data[i_m].items():
                events["times"] += spikes_times
                events["senders"] += [tuple(population_ind, neuron)] * len(spikes_times)
        return events


class ANNarchySpikeMultimeter(ANNarchyMonitor, ANNarchySpikeMonitor, SpikeMultimeter):

    """ANNarchySpikeMultimeter class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time spike weights' variables."""

    model = "spike_multimeter"

    def __init__(self, monitors, label="", annarchy_instance=None, **kwargs):
        self.model = "spike_multimeter"
        ANNarchyMonitor.__init__(self, monitors, label, self.model, annarchy_instance,
                                 run_tvb_multiscale_init=False, **kwargs)
        ANNarchySpikeMonitor.__init__(self, monitors, label, self.model, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        SpikeMultimeter.__init__(self, monitors, model=self.model, label=self.label)

    @property
    def events(self):
        """Method to record continuous time spike weights' data from ANNarchy.Monitor instances,
           and to return them in a discrete events dictionary."""
        self._record()
        data = self._data.stack(Var=tuple(self._data.dims))
        coords = dict(data.coords)
        events = dict()
        inds = []
        for var in coords["Variable"]:
            var_inds = np.where(data.loc[var].values != 0)[0]
            events[var] = data.loc[var].values[var_inds]
            inds += var_inds
        events["times"] = data.coords["Time"].values[inds]
        events["senders"] = np.array([(pop, neuron)
                                      for pop, neuron in zip(data.coords["Population"].values.tolist(),
                                                             data.coords["Neuron"].values.tolist())])[inds]
        return events


ANNarchyOutputDeviceDict = {"Monitor": ANNarchyMonitor,
                            "SpikeMonitor": ANNarchySpikeMonitor,
                            "spike_multimeter": ANNarchySpikeMultimeter}

ANNarchyOutputSpikeDeviceDict = {"SpikeMonitor": ANNarchySpikeMonitor,
                                 "spike_multimeter": ANNarchySpikeMultimeter}
