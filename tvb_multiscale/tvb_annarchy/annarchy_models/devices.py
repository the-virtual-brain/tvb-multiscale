# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from xarray import DataArray, combine_by_coords
import numpy as np

from tvb_multiscale.core.spiking_models.devices import \
   Device, InputDevice, OutputDevice, SpikeRecorder, Multimeter, SpikeMultimeter
from tvb_multiscale.core.utils.data_structures_utils import flatten_neurons_inds_in_DataArray

from tvb_multiscale.tvb_annarchy.annarchy_models.population import ANNarchyPopulation

from tvb.basic.neotraits.api import HasTraits, Attr, Int, List

from tvb.contrib.scripts.utils.data_structures_utils import \
    flatten_list, ensure_list, extract_integer_intervals, is_integer


# These classes wrap around ANNarchy commands.

class ANNarchyDevice(Device):

    __metaclass__ = ABCMeta

    """Abstract ANNarchyDevice class"""

    annarchy_instance = None

    _dt = None

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        super(ANNarchyDevice, self).__init__(device=device, **kwargs)
        self.annarchy_instance = annarchy_instance

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
    def _GetConnections(self):
        """Method to get connections of the device from/to populations.
            Returns:
             connections' objects.
        """
        pass

    def GetConnections(self):
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

    @property
    @abstractmethod
    def populations(self):
        pass

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

    def _print_neurons(self, neurons):
        neurons = np.array(neurons)
        number_of_neurons = len(neurons)
        output = "%d neurons: [" % number_of_neurons
        populations_inds = np.unique(neurons[:, 0])
        for pop_ind in populations_inds:
            pop_neurons = neurons[neurons[:, 0] == pop_ind, 1]
        output += "(%d, %s)" % (pop_ind, extract_integer_intervals(pop_neurons, print=True))
        output += "]"
        return output


class ANNarchyInputDevice(ANNarchyDevice, InputDevice, ANNarchyPopulation):
    __metaclass__ = ABCMeta

    """ANNarchyInputDevice class to wrap around an ANNarchy.Population, acting as an input (stimulating) device"""

    params = {}

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        self.params = kwargs.get("params", {})
        ANNarchyDevice.__init__(self, device, annarchy_instance=annarchy_instance, **kwargs)
        InputDevice.__init__(self, device, **kwargs)
        ANNarchyPopulation.__init__(self, device, annarchy_instance, **kwargs)

    def _assert_annarchy(self):
        ANNarchyPopulation._assert_annarchy(self)

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
        return ANNarchyPopulation.Get(self, attrs)

    def _GetConnections(self):
        """Method to get attributes of the connections from the device
           Return:
            Projections' objects
        """
        connections = ANNarchyPopulation._GetConnections(self, source_or_target="source")
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
        return ANNarchyPopulation._GetFromConnections(self, attrs, connections)

    @property
    def populations(self):
        populations = []
        for conn in self.connections:
            if conn.post not in populations:
                populations.append(conn.post)
        return populations

    def get_size(self):
        """Method to compute the total number of ANNarchyDevice's connected neurons.
            Returns:
                int: number of neurons.
        """
        if self._nodes is None:
            return 0
        return InputDevice.get_number_of_neurons(self)

    @property
    def number_of_devices_neurons(self):
        if self._nodes is None:
            return 0
        return self._nodes.size

    @property
    def number_of_connected_neurons(self):
        return self.get_size()


"""
Input devices for spiking populations
Not yet implemented: Input devices for rate-coded populations
"""


class ANNarchySpikeSourceArray(ANNarchyInputDevice):

    """ANNarchySpikeSourceArray class to wrap around an ANNarchy.SpikeSourceArray,
       acting as an input (stimulating) device, by sending spikes to target neurons."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "SpikeSourceArray")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)

    def reset(self):
        self._nodes.reset()


class ANNarchyPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an ANNarchy.PoissonPopulation,
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "PoissonPopulation")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyHomogeneousCorrelatedSpikeTrains(ANNarchyInputDevice):

    """ANNarchyHomogeneousCorrelatedSpikeTrains class to wrap around
       an ANNarchy.HomogeneousCorrelatedSpikeTrains,
       acting as an input (stimulating) device, by generating and sending
       correlated Poisson spikes to target neurons."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "HomogeneousCorrelatedSpikeTrains")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyContinuousInputDevice(ANNarchyInputDevice):

    """ANNarchyContinuousInputDevice class to wrap around a ANNarchy rate neuron, or TimedArray specific population,
        acting as an input (stimulating) device, by generating and sending
        a set of continuous quantities interpreted as a current or rate values."""

    from tvb_multiscale.tvb_annarchy.annarchy.input_devices import CurrentProxy

    proxy = Attr(field_type=bool, label="proxy", default=True, required=True,
                 doc="""Flag to store data after reading from ANNarchy monitor.""")

    proxy_type = CurrentProxy

    proxy_target = Attr(field_type=str, label="proxy target", default="exc", required=True,
                        doc="""Proxy target label (string).""")

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedArray")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyTimedArray(ANNarchyContinuousInputDevice):

    """ANNarchyTimedArray class to wrap around a rate ANNarchy.TimedArray,
       acting as an input (stimulating) device, by generating and sending
       a set of continuous quantities interpreted as a current or rate values."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedArray")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyPoissonNeuron(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an PoissonNeuron model
       (from Maith et al 2020, see tvb_annarchy.annarchy.izhikevich_maith_etal),
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "Poisson_neuron")
        ANNarchyInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyCurrentInjector(ANNarchyContinuousInputDevice):

    """ANNarchyCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a continuous quantity interpreted as a current (or potentially rate)."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "CurrentInjector")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyDCCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyDCCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a constant continuous quantity interpreted as a DC current (or potentially rate)."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "DCCurrentInjector")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyACCurrentInjector(ANNarchyCurrentInjector):

    """ANNarchyACCurrentInjector class to wrap around a rate ANNarchy.Population,
       acting as an input (stimulating) device, by generating and sending
       a sinusoidaly varying continuous quantity interpreted as a AC current (or potentially rate)."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "ACCurrentInjector")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


class ANNarchyTimedPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyTimedPoissonPopulation class to wrap around a rate ANNarchy.TimedPoissonPopulation,
       in order to act as an input (stimulating) device."""

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "TimedPoissonPopulation")
        ANNarchyContinuousInputDevice.__init__(self, device, annarchy_instance, **kwargs)


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
       between ANNarchyOutputDevice instances and ANNarchyPopulation ones"""

    from ANNarchy import Population, Monitor

    pre = Attr(field_type=Population, default=None, required=True,
               label="Population connection source ", doc="""The ANNarchy.Population as the connection's source.""")

    post = Attr(field_type=Monitor, default=None, required=True,
                label="Monitor connection target", doc="""The ANNarchy.Monitor as the connection's target.""")

    @property
    def attributes(self):
        return ["pre", "post"]


class ANNarchyOutputDevice(ANNarchyDevice, OutputDevice):
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

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        run_tvb_multiscale_init = kwargs.pop("run_tvb_multiscale_init", True)
        if isinstance(device, dict):
            monitors = OrderedDict(device)
        else:
            monitors = OrderedDict()
        ANNarchyDevice.__init__(self, monitors, annarchy_instance, **kwargs)
        if run_tvb_multiscale_init:
            OutputDevice.__init__(self, monitors, **kwargs)
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
                if self._period is None:
                    self._period = monitor.period
                else:
                    if self._period != monitor.period:
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
        return None

    def _set_attributes_to_dict(self, dictionary, monitor, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(getattr(monitor, attribute))
        else:
            dictionary[attribute] = [getattr(monitor, attribute)]

    def _Get(self, attrs=None):
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

    def Get(self, attrs=None):
        return self._Get(attrs)

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

    def _set_attributes_of_connection_to_dict(self, dictionary, connection, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(connection.get(attribute))
        else:
            dictionary[attribute] = [connection.get(attribute)]

    def _GetFromConnections(self, attrs=None, connections=None):
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

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "Monitor")
        if kwargs.pop("run_tvb_multiscale_init", True):
            Multimeter.__init__(self, device, **kwargs)
        ANNarchyOutputDevice.__init__(self, device, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)

    def _compute_times(self, times, data_time_length=None):
        """Method to compute the time vector of ANNarchy.Monitor instances"""
        times_lims = []
        current_step = self.annarchy_instance.get_current_step()
        for var, var_times in times.items():
            this_steps = [var_times["start"][0], var_times["stop"][-1]]
            if this_steps[0] == this_steps[1]:
                this_steps[1] = current_step
            if len(times_lims):
                if np.any(this_steps != times_lims):
                    raise ValueError("Start and stop times %s of variable %s\n"
                                     "are not equal to the ones previously set %s for Device %s!"
                                     % (str(this_steps), var, str(times_lims), self.label))
            else:
                times_lims = this_steps
        dt = self.dt
        period = self.period
        if len(times_lims):
            n_times = int(np.ceil((times_lims[1] - times_lims[0]) * dt / period))
            if data_time_length:
                if data_time_length > n_times:
                    raise ValueError("Adding data of time length %d bigger than the time vector length %d!" %
                                     (data_time_length, n_times))
                else:
                    n_times = data_time_length
            stop_time = times_lims[1] * dt + period
            start_time = stop_time - n_times * period
            return np.arange(start_time, stop_time, period)
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
                                 coords={"Time": self._compute_times(monitor.times(), data.shape[0]),
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


class ANNarchySpikeMonitor(ANNarchyOutputDevice, SpikeRecorder):

    """ANNarchySpikeMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of spike discrete events."""

    _data = List(of=dict, label="SpikeMonitor data buffer", default=(),
                 doc="""A list of dictionaries (one per Monitor) for holding the spike events
                       read from the Monitors""")

    _output_events_counter = List(of=OrderedDict, label="Number of output events", default=(),
                                  doc="""A list of lists of numbers of recorded events per sender neuron and monitor 
                                         that have been given to the output via a get_events() call.""")

    def __init__(self, device=None, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "SpikeMonitor")
        if kwargs.pop("run_tvb_multiscale_init", True):
            SpikeRecorder.__init__(self, device, **kwargs)
        ANNarchyOutputDevice.__init__(self, device, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)

    def _record(self):
        """Method to get discrete spike events' data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer."""
        dt = self.dt
        number_of_monitors = self.number_of_monitors
        data = tuple([OrderedDict()] * number_of_monitors)
        if self.store_data:
            while len(self._data) < number_of_monitors:
                self._data += (OrderedDict(),)
        for i_m, (monitor, population) in enumerate(self.monitors.items()):
            spikes = monitor.get("spike")
            senders = self._get_senders(population, list(spikes.keys()), False)
            for sender, spikes_times in zip(senders, list(spikes.values())):
                data[i_m].update({sender: (np.array(spikes_times) * dt).tolist()})
                if self.store_data:
                    self._data[i_m].update({sender: self._data[i_m].get(sender, []) + data[i_m].get(sender, [])})
        return data

    def _get_events(self, data=None):
        if data is None:
            data = self._record()
            if self.store_data:
                data = self._data
        if self.store_data:
            while len(self._output_events_counter) < self.number_of_monitors:
                self._output_events_counter += (OrderedDict(),)
        events = OrderedDict()
        events["times"] = []
        events["senders"] = []
        for i_m, monitor_data in enumerate(data):
            for sender, spikes_times in monitor_data.items():
                events["times"] += spikes_times
                events["senders"] += [sender] * len(spikes_times)
                if self.store_data:
                    self._output_events_counter[i_m][sender] = len(events["times"])
        inds = np.argsort(events["times"])
        events["times"] = np.array(events["times"])[inds].tolist()
        events["senders"] = np.array(events["senders"])[inds].tolist()
        return events

    def get_new_events(self):
        return self._get_events(self._record())

    @property
    def events(self):
        """Method to record discrete spike events' data from ANNarchy.Monitor instances,
           and to return them in a events dictionary."""
        return self._get_events()

    @property
    def new_events(self):
        return self.get_new_events()

    def _number_of_recorded_events(self):
        self._record()
        n_events = 0
        for i_m, monitor_data in enumerate(self._data):
            for sender, spikes_times in monitor_data.items():
                n_events += len(spikes_times)
        return n_events

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
        return self._number_of_recorded_events() - np.prod(self._output_events_counter)

    @property
    def number_of_new_events(self):
        return self._number_of_new_events()

    def reset(self):
        self._record()
        self._data = ()
        self._output_events_counter = ()


class ANNarchySpikeMultimeter(ANNarchyMonitor, ANNarchySpikeMonitor, SpikeMultimeter):

    """ANNarchySpikeMultimeter class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time spike weights' variables."""

    _data = Attr(field_type=DataArray, label="Data buffer",
                 default=DataArray(np.empty((0, 0, 0, 0)),
                                   dims=["Time", "Variable", "Population", "Neuron"]),
                 required=True,
                 doc="""A DataArray buffer holding the data read from the Monitors""")

    def __init__(self, monitors, annarchy_instance=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_multimeter")
        SpikeMultimeter.__init__(self, monitors, **kwargs)
        ANNarchyMonitor.__init__(self, monitors, annarchy_instance,
                                 run_tvb_multiscale_init=False, **kwargs)
        ANNarchySpikeMonitor.__init__(self, monitors, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)

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
