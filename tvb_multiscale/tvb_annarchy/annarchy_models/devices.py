# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from xarray import DataArray, combine_by_coords
import numpy as np


from tvb.basic.neotraits.api import Attr, Int, List

from tvb.contrib.scripts.utils.data_structures_utils import \
    flatten_list, ensure_list, extract_integer_intervals, is_integer

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.spiking_models.devices import InputDevice, SpikeRecorder, Multimeter, SpikeMultimeter
from tvb_multiscale.core.utils.data_structures_utils import flatten_neurons_inds_in_DataArray
from tvb_multiscale.tvb_annarchy.annarchy_models.population import _ANNarchyPopulation


# These classes wrap around ANNarchy commands.

class ANNarchyDevice(HasTraits):

    __metaclass__ = ABCMeta

    """Abstract ANNarchyDevice class"""

    device = None

    model = Attr(field_type=str, default="device", required=True,
                 label="ANNarchyDevice model", doc="""Label of ANNarchyDevice model""")

    label = Attr(field_type=str, default="", required=True,
                 label="ANNarchyDevice label", doc="""Label of ANNarchyDevice""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the ANNarchyDevice resides or connects to""")

    _number_of_connections = None
    _number_of_neurons = None

    annarchy_instance = None

    _dt = None

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        self.device = device
        self._dt = None
        self._number_of_connections = None
        self._number_of_neurons = None
        self.label = str(kwargs.get("label", self.__class__.__name__))
        self.model = str(kwargs.get("model", self.__class__.__name__))
        self.brain_region = str(kwargs.get("brain_region", ""))
        self.annarchy_instance = annarchy_instance
        super(ANNarchyDevice, self).__init__()

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @abstractmethod
    def _assert_annarchy(self):
        pass

    @property
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

    @abstractmethod
    def _GetConnections(self, **kwargs):
        """Method to get connections of the device from/to populations.
            Returns:
             connections' objects.
        """
        pass

    def GetConnections(self, **kwargs):
        """Method to get connections of the device from/to populations.
            Returns:
             connections' objects.
        """
        return self._GetConnections()

    @property
    def connections(self):
        """Method to get all connections of the device from/to populations.
           Returns:
            connections' objects.
        """
        return self._GetConnections()

    def get_neurons(self):
        """Method to get the indices of all the neurons the device connects to/from."""
        from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import get_population_ind
        neurons = []
        for pop in self.populations:
            population_ind = get_population_ind(pop, self.annarchy_instance)
            local_inds = pop.ranks
            neurons += tuple(zip([population_ind] * len(local_inds), local_inds))
        return tuple(neurons)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device connects from/to."""
        return self.get_neurons()


class ANNarchyInputDevice(_ANNarchyPopulation, ANNarchyDevice, InputDevice):

    __metaclass__ = ABCMeta

    """ANNarchyInputDevice class to wrap around an ANNarchy.Population, acting as an input (stimulating) device"""

    from ANNarchy import Population

    _nodes = Attr(field_type=Population, default=None, required=False,
                  label="Specific ANNarchy.Population",
                  doc="""Instance of specific ANNarchyInputDevice's ANNarchy.Population""")

    params = {}

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        self.params = kwargs.get("params", {})
        _ANNarchyPopulation.__init__(self, device, annarchy_instance, **kwargs)
        ANNarchyDevice.__init__(self, device, annarchy_instance, **kwargs)
        InputDevice.__init__(self, device, **kwargs)

    def _assert_annarchy(self):
        _ANNarchyPopulation._assert_annarchy(self)

    def _assert_device(self):
        if self.annarchy_instance is not None and self._nodes is not None:
            from ANNarchy import Population
            assert isinstance(self._nodes, Population)

    @property
    def annarchy_model(self):
        if self.device:
            return str(self.device.neuron_type.name)
        else:
            return ""

    @property
    def device_ind(self):
        return self.population_ind

    def Set(self, values_dict, **kwargs):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        InputDevice.Set(self, values_dict)

    def Get(self, attrs=None, **kwargs):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all device's attributes.
           Returns:
            Dictionary of attributes.
        """
        return InputDevice.Get(self, attrs)

    def _GetConnections(self, **kwargs):
        """Method to get attributes of the connections from the device
           Return:
            Projections' objects
        """
        connections = _ANNarchyPopulation._GetConnections(self, source_or_target="source")
        return connections

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from the device.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections of the device.
        """
        if connections is None:
            connections = _ANNarchyPopulation._GetConnections(self, neurons=None, source_or_target="source")
        _ANNarchyPopulation._SetToConnections(self, values_dict, connections)

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
            connections = _ANNarchyPopulation._GetConnections(self, neurons=None, source_or_target="source")
        return _ANNarchyPopulation._GetFromConnections(self, attrs, connections)

    @property
    def populations(self):
        populations = []
        for conn in self.connections:
            if conn.post not in populations:
                populations.append(conn.post)
        return populations

    @property
    def number_of_devices_neurons(self):
        if self._nodes is None:
            return 0
        return self._nodes.size

    @property
    def number_of_neurons(self):
        return InputDevice.number_of_neurons.fget(self)

    @property
    def number_of_connected_neurons(self):
        return self.number_of_neurons

    def info_neurons(self):
        return _ANNarchyPopulation.info_neurons(self)

    def info(self, recursive=0):
        return InputDevice.info(self, recursive=recursive)

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return InputDevice.info_details(self, recursive=recursive,
                                        connectivity=connectivity, source_or_target="source")


"""
Input devices for spiking populations
Not yet implemented: Input devices for rate-coded populations
"""


class ANNarchySpikeSourceArray(ANNarchyInputDevice):

    """ANNarchySpikeSourceArray class to wrap around an ANNarchy.SpikeSourceArray,
       acting as an input (stimulating) device, by sending spikes to target neurons."""

    from ANNarchy import SpikeSourceArray

    _nodes = Attr(field_type=SpikeSourceArray, default=SpikeSourceArray([[]]), required=False,
                  label="Specific ANNarchy.Population",
                  doc="""Instance of specific ANNarchyInputDevice's ANNarchy.Population""")

    def __init__(self, device=SpikeSourceArray([[]]), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "SpikeSourceArray")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)

    def add_spikes(self, spikes, time_shift=None, nodes=None, sort=False):
        if len(spikes):
            current_time = self.annarchy_instance.get_time()
            if time_shift:
                # Apply time_shift, if any
                new_spikes = []
                for i_sp, spike in enumerate(spikes):
                    new_spike = np.array(spike) + time_shift
                    if len(new_spike.shape):
                        # if spikes is a sequence of sequences for each neuron
                        new_spikes.append(ensure_list(new_spike[new_spike >= current_time]))
                    else:
                        # if spikes is a sequence of the same spikes for all neurons
                        if new_spike > current_time:
                            new_spikes.append(new_spike.item())
                spikes = new_spikes
            if len(spikes) != self.number_of_connected_neurons:
                # Assume we have to add the same spikes to all neurons
                spikes = ensure_list(spikes) * len(nodes)
            old_spikes = self.device.get("spike_times")
            new_spikes = []
            for old_spike, new_spike in zip(old_spikes, spikes):
                old_spike = np.array(old_spike)
                old_spike = old_spike[old_spike >= current_time].tolist()
                new_spike = old_spike + ensure_list(new_spike)
                if len(new_spike) and sort:
                    new_spike = np.sort(new_spike).tolist()
                new_spikes.append(new_spike)
            self.device.set({"spike_times": new_spikes})

    def reset(self):
        self._nodes.reset()


class ANNarchyPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an ANNarchy.PoissonPopulation,
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    from ANNarchy import PoissonPopulation

    def __init__(self, device=PoissonPopulation(geometry=0, rates=0.0), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "PoissonPopulation")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyHomogeneousCorrelatedSpikeTrains(ANNarchyInputDevice):

    """ANNarchyHomogeneousCorrelatedSpikeTrains class to wrap around
       an ANNarchy.HomogeneousCorrelatedSpikeTrains,
       acting as an input (stimulating) device, by generating and sending
       correlated Poisson spikes to target neurons."""

    from ANNarchy import HomogeneousCorrelatedSpikeTrains

    def __init__(self, device=HomogeneousCorrelatedSpikeTrains(geometry=0, rates=0.0, corr=0.0, tau=1.0),
                 annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "HomogeneousCorrelatedSpikeTrains")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyContinuousInputDevice(ANNarchyInputDevice):

    """ANNarchyContinuousInputDevice class to wrap around a ANNarchy rate neuron, or TimedArray specific population,
        acting as an input (stimulating) device, by generating and sending
        a set of continuous quantities interpreted as a current or rate values."""

    from ANNarchy import TimedArray
    from tvb_multiscale.tvb_annarchy.annarchy.input_devices import CurrentProxy

    proxy = Attr(field_type=bool, label="proxy", default=True, required=True,
                 doc="""Flag to store data after reading from ANNarchy monitor.""")

    proxy_type = CurrentProxy

    proxy_target = Attr(field_type=str, label="proxy target", default="exc", required=True,
                        doc="""Proxy target label (string).""")

    def __init__(self, device=TimedArray(np.array([[]])), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedArray")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyTimedArray(ANNarchyContinuousInputDevice):

    """ANNarchyTimedArray class to wrap around a rate ANNarchy.TimedArray,
       acting as an input (stimulating) device, by generating and sending
       a set of continuous quantities interpreted as a current or rate values."""

    from ANNarchy import TimedArray

    def __init__(self, device=TimedArray(np.array([[]])), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedArray")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyCurrentInjector(ANNarchyContinuousInputDevice):

    """ANNarchyCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a continuous quantity interpreted as a current (or potentially rate)."""

    from ANNarchy import Population
    from tvb_multiscale.tvb_annarchy.annarchy.input_devices import CurrentInjector

    def __init__(self, device=Population(0, neuron=CurrentInjector), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "CurrentInjector")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyDCCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyDCCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a constant continuous quantity interpreted as a DC current (or potentially rate)."""

    from ANNarchy import Population
    from tvb_multiscale.tvb_annarchy.annarchy.input_devices import DCCurrentInjector

    def __init__(self, device=Population(0, neuron=DCCurrentInjector), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "DCCurrentInjector")
        ANNarchyCurrentInjector.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyACCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyACCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a sinusoidaly varying continuous quantity interpreted as a AC current (or potentially rate)."""

    from ANNarchy import Population
    from tvb_multiscale.tvb_annarchy.annarchy.input_devices import ACCurrentInjector

    def __init__(self, device=Population(0, neuron=ACCurrentInjector), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ACCurrentInjector")
        ANNarchyCurrentInjector.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyTimedPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyTimedPoissonPopulation class to wrap around a rate ANNarchy.TimedPoissonPopulation,
       in order to act as an input (stimulating) device."""

    from ANNarchy import TimedPoissonPopulation

    def __init__(self, device=TimedPoissonPopulation(0, [], []), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedPoissonPopulation")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


ANNarchyInputDeviceDict = {}


ANNarchyTimedSpikeInputDeviceDict = \
    {"TimedPoissonPopulation": ANNarchyTimedPoissonPopulation}

ANNarchySpikeInputDeviceDict = {"PoissonPopulation": ANNarchyPoissonPopulation,
                                "HomogeneousCorrelatedSpikeTrains": ANNarchyHomogeneousCorrelatedSpikeTrains,
                                "SpikeSourceArray": ANNarchySpikeSourceArray}
ANNarchySpikeInputDeviceDict.update(ANNarchyTimedSpikeInputDeviceDict)


ANNarchyCurrentInputDeviceDict = {"TimedArray": ANNarchyTimedArray,
                                  "CurrentInjector": ANNarchyCurrentInjector,
                                  "DCCurrentInjector": ANNarchyDCCurrentInjector,
                                  "ACCurrentInjector": ANNarchyACCurrentInjector
                                  }

ANNarchyInputDeviceDict.update(ANNarchySpikeInputDeviceDict)
ANNarchyInputDeviceDict.update(ANNarchyCurrentInputDeviceDict)


class ANNarchyOutputDeviceConnection(HasTraits):

    """ANNarchyOutputDeviceConnection class holds properties of connections
       between ANNarchyOutputDevice instances and _ANNarchyPopulation ones"""

    from ANNarchy import Population, Monitor

    pre = Attr(field_type=Population, default=None, required=True,
               label="Population connection source ", doc="""The ANNarchy.Population as the connection's source.""")

    post = Attr(field_type=Monitor, default=None, required=True,
                label="Monitor connection target", doc="""The ANNarchy.Monitor as the connection's target.""")

    @property
    def attributes(self):
        return ["pre", "post"]


class ANNarchyOutputDevice(ANNarchyDevice):
    __metaclass__ = ABCMeta

    """ANNarchyOutputDevice class to wrap around ANNarchy.Monitor instances, acting as an output device"""

    _data = None

    monitors = Attr(field_type=dict, default=lambda: OrderedDict(), required=True,
                    label="Device's Monitors' dictionary",
                    doc="""A dictionary of the ANNarchy.Monitor instances of the ANNarchyOutputDevice""")

    _monitors_inds = List(of=int, label="Device's Monitors' indices", default=(),
                          doc="""A list of the indices of the ANNarchy.Monitor instances of the ANNarchyOutputDevice""")

    store_data = Attr(field_type=bool, label="Store Data",
                      default=True, required=True,
                      doc="""Flag to store data after reading from ANNarchy monitor.""")

    annarchy_instance = None

    params = {}

    _weight_attr = "w"
    _delay_attr = "delay"
    _receptor_attr = "target"

    _default_attrs = ["variables", "period", "period_offset", "start"]

    _record_from = []

    _period = None

    def __init__(self, device=OrderedDict(), annarchy_instance=None, **kwargs):
        if isinstance(device, dict):
            monitors = OrderedDict(device)
        else:
            monitors = OrderedDict()
        ANNarchyDevice.__init__(self, monitors, annarchy_instance, **kwargs)
        self.params = kwargs.pop("params", {})
        self.annarchy_instance = annarchy_instance
        if self.annarchy_instance is not None:
            self._monitors_inds = self._get_monitors_inds()

    def _assert_spiking_simulator(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _assert_annarchy(self):
        self._assert_spiking_simulator()

    def _assert_device(self):
        if self.annarchy_instance is not None and self.monitors is not None:
            assert isinstance(self.monitors, dict)
            from ANNarchy import Monitor, Population, PopulationView
            for monitor, pop in self.monitors.items():
                assert isinstance(monitor, Monitor)
                assert isinstance(pop, (Population, PopulationView))

    def _assert_nodes(self):
        self._assert_device()

    @property
    def annarchy_model(self):
        return "Monitor"

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
    def number_of_monitors(self):
        if isinstance(self.monitors, dict):
            return len(self.monitors)
        else:
            return 0

    @property
    def period(self):
        if self._period is None:
            for monitor, population in self.monitors.items():
                # TODO: Find out why it fails for monitor.period!!!
                if self._period is None:
                    self._period = monitor._period
                else:
                    if self._period != monitor._period:
                        raise ValueError("Monitor %s of Device %s has period %g,\n"
                                         "which is different from the previously set period %g!"
                                         % (population.name, self.label, monitor.period, self._period))
        return self._period

    def _Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        for monitor in self.monitors.keys():
            for key, val in values_dict.items:
                setattr(monitor, key, val)

    def Set(self, values_dict):
        self._Set(values_dict)

    @property
    def gids(self):
        return self.monitors_inds

    def _set_attributes_to_dict(self, dictionary, monitor, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(getattr(monitor, attribute))
        else:
            dictionary[attribute] = [getattr(monitor, attribute)]

    def _Get(self, attrs=None, **kwargs):
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

    def Get(self, attrs=None, **kwargs):
        return self._Get(attrs)

    def _GetConnections(self, **kwargs):
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

    def _set_attributes_of_connection_to_dict(self, dictionary, connection, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(connection.get(attribute))
        else:
            dictionary[attribute] = [connection.get(attribute)]

    def _GetFromConnections(self, attrs=None, connections=None, **kwargs):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections to/from the present population.
             attrs: collection (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponds to all attributes
            Returns:
             Dictionary of lists (for the possible different Projection objects) of arrays of connections' attributes.
        """
        # if connections is None:
        #     connections = self._GetConnections()
        # own_connections = self._GetConnections()
        # dictionary = {}
        # for connection in ensure_list(connections):
        #     dictionary = {}
        #     if connection in own_connections:
        #         if attrs is None:
        #             attrs = self._default_connection_attrs
        #         else:
        #             attrs = np.intersect1d(attrs, connection.attributes)
        #         for attribute in attrs:
        #            self._set_attributes_of_connection_to_dict(dictionary, connection, attribute)
        return {}

    @property
    def populations(self):
        """Method to get the ANNarchy.Population instances this device records from."""
        populations = []
        for pop in self.monitors.values():
            if pop not in populations:
                populations.append(pop)
        return populations

    @property
    def record_from(self):
        if self._record_from:
            return self._record_from
        else:
            self._record_from = []
            for monitor, population in self.monitors.items():
                if len(self._record_from):
                    if np.any(self._record_from != monitor.variables):
                        raise ValueError("Monitor %s of Device %s records from variables %s,\n"
                                         "which are different from the previously set ones %s!"
                                         % (population.name, self.label,
                                            str(monitor.variables), str(self._record_from)))

                else:
                    self._record_from = ensure_list(monitor.variables)
        return self._record_from

    def _get_senders(self, population, neurons_ranks, str_flag=False):
        population_ind = self.annarchy_instance.Global._network[0]["populations"].index(population)
        if str_flag:
            senders = ["%d_%d" % (population_ind, neuron_rank) for neuron_rank in ensure_list(neurons_ranks)]
        else:
            senders = [(population_ind, neuron_rank) for neuron_rank in ensure_list(neurons_ranks)]
        if len(senders) == 1:
            senders = senders[0]
        return senders

    @abstractmethod
    def _record(self):
        """Method to get data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer of xarray.DataArray type."""
        pass

    @property
    @abstractmethod
    def events(self):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        pass

    @property
    def n_events(self):
        return self.number_of_events

    @abstractmethod
    def reset(self):
        pass

    @property
    def nodes(self):
        return tuple(self.monitors.keys())

    def pause(self):
        for monitor in self.monitors.keys():
            monitor.pause()

    def resume(self):
        for monitor in self.monitors.keys():
            monitor.resume()


class ANNarchyMonitor(ANNarchyOutputDevice, Multimeter):

    """ANNarchyMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time quantities."""

    _data = Attr(field_type=DataArray, label="Data buffer",
                 default=DataArray(np.empty((0, 0, 0)), dims=["Time", "Variable", "Neuron"]),
                 required=True,
                 doc="""A DataArray buffer holding the data read from the Monitors""")

    _output_events_index = Int(field_type=int, default=0, required=True, label="Index of output events",
                                 doc="""The number of recorded events that 
                                        have been given to the output via a get_events() call.""")

    def __init__(self, device=OrderedDict(), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "Monitor")
        ANNarchyOutputDevice.__init__(self, device, annarchy_instance, **kwargs)
        Multimeter.__init__(self, device, **kwargs)

    def _compute_times(self, times, data_time_length=None):
        """Method to compute the time vector of ANNarchy.Monitor instances"""
        times_lims = []
        dt = self.dt
        period = self.period
        current_step = self.annarchy_instance.get_current_step()
        for var, var_times in times.items():
            this_steps = [var_times["start"][1], var_times["stop"][-1]]
            if this_steps[0] == this_steps[1]:
                this_steps[1] = current_step
            if len(times_lims):
                if np.any(this_steps != times_lims):
                    raise ValueError("Start and stop times %s of variable %s\n"
                                     "are not equal to the ones previously set %s for Device %s!"
                                     % (str(this_steps), var, str(times_lims), self.label))
            else:
                times_lims = this_steps
        if len(times_lims):
            n_times = int(np.ceil((times_lims[1] - times_lims[0]) * dt / period))
            if data_time_length:
                if data_time_length != n_times:
                    raise ValueError("Adding data of time length %d different than the time vector length %d!" %
                                     (data_time_length, n_times))
                else:
                    n_times = data_time_length
            start_time = times_lims[0] * dt
            return start_time + np.arange(n_times) * period
        else:
            return np.array(times_lims).astype("f")

    def _record(self):
        """Method to get data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer of xarray.DataArray type."""
        data = DataArray(np.empty((0, 0, 0)), dims=["Time", "Variable", "Neuron"], name=self.label)
        for monitor, population in self.monitors.items():
            m_data = monitor.get()
            variables = list(m_data.keys())
            m_data = np.array(list(m_data.values()))
            if m_data.size > 0:
                m_data = m_data.transpose((1, 0, 2))
                m_data = DataArray(m_data,
                                 dims=["Time", "Variable", "Neuron"],
                                 coords={"Time": self._compute_times(monitor.times(), m_data.shape[0]),
                                         "Variable": variables,
                                         "Neuron": self._get_senders(population, population.ranks, True)},
                                 name=self.label)
                if data.size > 0:
                    data = combine_by_coords([data, m_data], fill_value=np.nan)
                else:
                    data = m_data.copy()
        if self.store_data:
            if self._data.size > 0:
                self._data = combine_by_coords([self._data, data], fill_value=np.nan)
            else:
                self._data = data.copy()
        return data

    def _get_data(self, data=None, variables=None, events_inds=None, name=None,
                  dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        if data is None:
            data = self._data
        if events_inds is None:
            _data = data
        else:
            _data = data[events_inds]
        if variables:
            _data = _data.loc[:, variables]

        if np.any(_data.dims != dims_names):
            _data = _data.rename(dict(zip(_data.dims, dims_names)))
        if flatten_neurons_inds:
            _data = flatten_neurons_inds_in_DataArray(_data, _data.dims[2])
        else:
            _data = DataArray(_data)
        if name:
            _data.name = name
        self._output_events_index = self._data.shape[0]
        return _data

    def get_new_data(self, variables=None, name=None,
                     dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        return self._get_data(self._record(), variables, slice(self._output_events_index, None),
                              name, dims_names, flatten_neurons_inds)

    def get_data(self, variables=None, events_inds=None,
                 name=None, dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None, which defaults to the label of the Device
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        self._record()
        return self._get_data(None, variables, events_inds, name, dims_names, flatten_neurons_inds)

    def _get_events(self, data):
        variables = data.coords["Variable"].values
        data = self._data.stack(Var=("Time", "Neuron"))
        times_senders = np.array([[float(var[0]), var[1]] for var in data.coords["Var"].values]).astype("O")
        events = dict()
        events["times"] = np.array(times_senders[:, 0]).astype("f")
        events["senders"] = times_senders[:, 1]
        del times_senders
        for i_var, var in enumerate(variables):
            events[var] = data[i_var].values
        return events

    def get_new_events(self, variables=None, name=None,
                      dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        return self._get_events(self.get_new_data(variables, flatten_neurons_inds, dims_names, flatten_neurons_inds))

    @property
    def events(self):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        return self._get_events(self.get_data())

    @property
    def new_events(self):
        return self.get_new_events()

    def _number_of_recorded_events(self):
        if self._data is None:
            return 0
        return self._data.shape[0] * self._data.shape[-1]  # times x neurons

    @property
    def number_of_recorded_events(self):
        return self._number_of_recorded_events()

    def _number_of_events(self):
        self._record()
        return self._number_of_recorded_events()

    @property
    def number_of_events(self):
        return self._number_of_events()

    def _number_of_new_events(self):
        return (self._data.shape[0] - self._output_events_index) * self._data.shape[-1]

    @property
    def number_of_new_events(self):
        return self._number_of_new_events()

    def reset(self):
        self._record()
        self._data = DataArray(np.empty((0, 0, 0)),
                               dims=["Time", "Variable", "Neuron"])
        self._output_events_index = 0

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return Multimeter.info_details(self, recursive=recursive,
                                       connectivity=connectivity, source_or_target="target")


class ANNarchySpikeMonitor(ANNarchyOutputDevice, SpikeRecorder):

    """ANNarchySpikeMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of spike discrete events."""

    _data = Attr(field_type=OrderedDict, label="Data buffer",
                 default=None,
                 required=True,
                 doc="""An OrderedDict buffer holding the spikes' times and senders read from the Monitors""")

    _output_events_counter = Int(field_type=int, default=0, required=True, label="Index of output events",
                                 doc="""The number of recorded events that 
                                      have been given to the output via a get_events() call.""")

    def __init__(self, device=OrderedDict(), annarchy_instance=None, **kwargs):
        # Initialize it immediately because it is required!:
        self._data = OrderedDict({"times": [], "senders": []})
        kwargs["model"] = kwargs.get("model", "SpikeMonitor")
        ANNarchyOutputDevice.__init__(self, device, annarchy_instance, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)

    def _record(self):
        """Method to get discrete spike events' data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer."""
        events = OrderedDict()
        events["times"] = []
        events["senders"] = []
        for monitor, population in self.monitors.items():
            # TODO: report error when calling raster_plot without args!!!
            spike = {"spike": monitor.get('spike')}
            if len(spike['spike']):
                spike_times, spike_ranks = monitor.raster_plot(spike)
                if spike_times.size:
                    events["times"] += spike_times.tolist()
                    population_ind = self.annarchy_instance.Global._network[0]["populations"].index(population)
                    spike_senders = list(zip([population_ind] * spike_times.size, spike_ranks))
                    events["senders"] += spike_senders
        events['times'] = np.array(events['times'])
        events['senders'] = np.array(events['senders'])
        if len(events["times"]):
            inds = np.argsort(events["times"])
            events["times"] = events["times"][inds]
            events["senders"] = events["senders"][inds]
            if self.store_data:
                self._data["times"] += events["times"].tolist()
                self._data["senders"] += events["senders"].tolist()
        return events

    def _get_events(self, data=None):
        if data is None:
            events = self._record()
            self._output_events_counter += len(events['times'])
            if self.store_data:
                events = OrderedDict()
                for key, val in self._data.items():
                    events[key] = np.array(val)
                self._output_events_counter = len(events['times'])
        return events

    def get_new_events(self):
        return self._record()

    @property
    def events(self):
        """Method to record discrete spike events' data from ANNarchy.Monitor instances,
           and to return them in a events dictionary."""
        return self._get_events()

    @property
    def new_events(self):
        return self.get_new_events()

    def _number_of_recorded_events(self):
        return self._output_events_counter

    @property
    def number_of_recorded_events(self):
        return self._number_of_recorded_events()

    def _number_of_events(self):
        self._record()
        return self._number_of_recorded_events()

    @property
    def number_of_events(self):
        return self._number_of_events()

    def _number_of_new_events(self):
        number_of_recorded_events = self._number_of_recorded_events
        return self.number_of_events - number_of_recorded_events

    @property
    def number_of_new_events(self):
        return self._number_of_new_events()

    def reset(self):
        self._record()
        self._data = OrderedDict()
        self._data["times"] = []
        self._data["senders"] = []
        self._output_events_counter = 0

    def info_details(self, recursive=0, connectivity=False, **kwargs):
        return SpikeRecorder.info_details(self, recursive=recursive,
                                          connectivity=connectivity, source_or_target="target")


class ANNarchySpikeMultimeter(ANNarchyMonitor, ANNarchySpikeMonitor, SpikeMultimeter):

    """ANNarchySpikeMultimeter class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time spike weights' variables."""

    _data = Attr(field_type=DataArray, label="Data buffer",
                 default=DataArray(np.empty((0, 0, 0, 0)),
                                   dims=["Time", "Variable", "Population", "Neuron"]),
                 required=True,
                 doc="""A DataArray buffer holding the data read from the Monitors""")

    def __init__(self, monitors=OrderedDict(), annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_multimeter")
        ANNarchyMonitor.__init__(self, monitors, annarchy_instance, **kwargs)
        ANNarchySpikeMonitor.__init__(self, monitors, annarchy_instance, **kwargs)
        SpikeMultimeter.__init__(self, monitors, **kwargs)

    def _record(self):
        return ANNarchyMonitor._record(self)

    def _get_events(self, data):
        """Method to record continuous time spike weights' data from ANNarchy.Monitor instances,
           and to return them in a discrete events dictionary."""
        data = data.stack(Var=tuple(data.dims))
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

    @property
    def events(self):
        """Method to record continuous time spike weights' data from ANNarchy.Monitor instances,
           and to return them in a discrete events dictionary."""
        self._record()
        return self._get_events(ANNarchyMonitor.get_data())

    def get_new_events(self):
        return self._get_events(ANNarchyMonitor.get_new_data())

    @property
    def new_events(self):
        return self.get_new_events()

    def _number_of_recorded_events(self):
       return ANNarchyMonitor._number_of_recorded_events(self)

    def _number_of_events(self):
        return ANNarchyMonitor._number_of_events(self)

    def _number_of_new_events(self):
        return ANNarchyMonitor._number_of_new_events(self)

    def reset(self):
        ANNarchyMonitor.reset(self)


ANNarchyOutputDeviceDict = {}


ANNarchyOutputSpikeDeviceDict = {"SpikeMonitor": ANNarchySpikeMonitor}


ANNarchyOutputContinuousTimeDeviceDict = {"Monitor": ANNarchyMonitor,
                                          "spike_multimeter": ANNarchySpikeMultimeter}


ANNarchyOutputDeviceDict.update(ANNarchyOutputSpikeDeviceDict)
ANNarchyOutputDeviceDict.update(ANNarchyOutputContinuousTimeDeviceDict)
