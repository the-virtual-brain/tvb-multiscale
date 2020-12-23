# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from xarray import DataArray, combine_by_coords
import numpy as np

from tvb_multiscale.core.spiking_models.devices import \
   Device, InputDevice, OutputDevice, SpikeRecorder, Multimeter, SpikeMultimeter
from tvb_multiscale.core.utils.data_structures_utils import flatten_neurons_inds_in_DataArray

from tvb_multiscale.tvb_annarchy.annarchy_models.population import ANNarchyPopulation

from tvb.basic.neotraits.api import HasTraits, Attr, List

from tvb.contrib.scripts.utils.data_structures_utils import flatten_list, ensure_list, extract_integer_intervals


# These classes wrap around ANNarchy commands.

class ANNarchyDevice(Device):

    __metaclass__ = ABCMeta

    """Abstract ANNarchyDevice class"""

    annarchy_instance = None

    def __init__(self, device=None, label="", model="annarchy_device", annarchy_instance=None):
        super(ANNarchyDevice, self).__init__(device=device, label=label, model=model)
        self.annarchy_instance = annarchy_instance

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @abstractmethod
    def _assert_annarchy(self):
        pass

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
            neurons += list(zip([population_ind] * len(local_inds), local_inds))
        return tuple(np.unique(neurons, axis=0).tolist())

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

    def __init__(self, device=None,  label="", model="annarchy_input_device", annarchy_instance=None, **kwargs):
        self.params = kwargs.get("params", {})
        ANNarchyDevice.__init__(self, device, label=label, model=model, annarchy_instance=annarchy_instance)
        InputDevice.__init__(self, device, model=model, label=label)
        ANNarchyPopulation.__init__(self, device, label, model, annarchy_instance)

    def _assert_annarchy(self):
        ANNarchyPopulation._assert_annarchy(self)

    def _assert_device(self):
        if self.annarchy_instance is not None and self._population is not None:
            from ANNarchy import Population
            assert isinstance(self._population, Population)

    @property
    def annarchy_model(self):
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

    def get_number_of_neurons(self):
        """Method to compute the total number of ANNarchyDevice's connected neurons.
            Returns:
                int: number of neurons.
        """
        if self._population is None:
            return 0
        return InputDevice.get_number_of_neurons(self)

    @property
    def number_of_devices_neurons(self):
        return self._population.size

    @property
    def number_of_connected_neurons(self):
        return self.get_number_of_neurons()


"""
Input devices for spiking populations
Not yet implemented: Input devices for rate-coded populations
"""


class ANNarchySpikeSourceArray(ANNarchyInputDevice, InputDevice):

    """ANNarchySpikeSourceArray class to wrap around an ANNarchy.SpikeSourceArray,
       acting as an input (stimulating) device, by sending spikes to target neurons."""

    def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
        super(ANNarchySpikeSourceArray, self).__init__(device, label, "SpikeSourceArray",
                                                       annarchy_instance, **kwargs)

    def reset(self):
        self._population.reset()


class ANNarchyPoissonPopulation(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an ANNarchy.PoissonPopulation,
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
        super(ANNarchyPoissonPopulation, self).__init__(device,  label, "PoissonPopulation",
                                                        annarchy_instance, **kwargs)


class ANNarchyHomogeneousCorrelatedSpikeTrains(ANNarchyInputDevice):

    """ANNarchyHomogeneousCorrelatedSpikeTrains class to wrap around
       an ANNarchy.HomogeneousCorrelatedSpikeTrains,
       acting as an input (stimulating) device, by generating and sending
       correlated Poisson spikes to target neurons."""

    def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
        super(ANNarchyHomogeneousCorrelatedSpikeTrains, self).__init__(device,  label,
                                                                       "HomogeneousCorrelatedSpikeTrains",
                                                                       annarchy_instance, **kwargs)


# class ANNarchyCurrentInjector(InputDevice):
#
#     """ANNarchyCurrentInjector class to wrap around a rate ANNarchy.Population,
#        acting as an input (stimulating) device, by generating and sending
#        a continuous quantity interpreted as a current (or potentially rate)."""
#
#     def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
#         super(ANNarchyCurrentInjector, self).__init__(device,  label, "CurrentInjector",
#                                                       annarchy_instance, **kwargs)
#
#
# class ANNarchyDCCurrentInjector(ANNarchyCurrentInjector):
#
#     """ANNarchyDCCurrentInjector class to wrap around a rate ANNarchy.Population,
#        acting as an input (stimulating) device, by generating and sending
#        a constant continuous quantity interpreted as a DC current (or potentially rate)."""
#
#     def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
#         super(ANNarchyDCCurrentInjector, self).__init__(device,  label, "DCCurrentInjector",
#                                                         annarchy_instance, **kwargs)
#
#
# class ANNarchyACCurrentInjector(ANNarchyCurrentInjector):
#
#     """ANNarchyACCurrentInjector class to wrap around a rate ANNarchy.Population,
#        acting as an input (stimulating) device, by generating and sending
#        a sinusoidaly varying continuous quantity interpreted as a AC current (or potentially rate)."""
#
#     def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
#         super(ANNarchyACCurrentInjector, self).__init__(device,  label, "ACCurrentInjector",
#                                                         annarchy_instance, **kwargs)
#
#
class ANNarchyTimedArray(InputDevice):

    """ANNarchyTimedArray class to wrap around a rate ANNarchy.TimedArray,
       acting as an input (stimulating) device, by generating and sending
       a set of continuous quantities interpreted as a current or rate values."""

    def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
        super(ANNarchyTimedArray, self).__init__(device,  label, "TimedArray",
                                                 annarchy_instance, **kwargs)


class ANNarchyPoissonNeuron(ANNarchyInputDevice):

    """ANNarchyPoissonPopulation class to wrap around an PoissonNeuron model
       (from Maith et al 2020, see tvb_annarchy.annarchy.izhikevich_maith_etal),
       acting as an input (stimulating) device, by generating and sending
       uncorrelated Poisson spikes to target neurons."""

    def __init__(self, device=None, label="", annarchy_instance=None, **kwargs):
        super(ANNarchyPoissonNeuron, self).__init__(device,  label, "Poisson_neuron",
                                                    annarchy_instance, **kwargs)


ANNarchyInputDeviceDict = {}


ANNarchySpikeInputDeviceDict = {"PoissonPopulation": ANNarchyPoissonPopulation,
                                 "HomogeneousCorrelatedSpikeTrains": ANNarchyHomogeneousCorrelatedSpikeTrains,
                                 "SpikeSourceArray": ANNarchySpikeSourceArray,
                                 # From Maith et al 2020, see anarchy.izhikevich_maith_etal.py:
                                 "Poisson_neuron": ANNarchyPoissonNeuron
                               }


ANNarchyCurrentInputDeviceDict = {"TimedArray": ANNarchyTimedArray
                                  # "CurrentInjector": ANNarchyCurrentInjector,
                                  # "DCCurrentInjector": ANNarchyDCCurrentInjector,
                                  # "ACCurrentInjector": ANNarchyACCurrentInjector,
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

    annarchy_instance = None

    params = {}

    _weight_attr = "w"
    _delay_attr = "delay"
    _receptor_attr = "target"

    _default_attrs = ["variables", "period", "period_offset", "start"]

    _record_from = []

    _dt = None

    _period = None

    def __init__(self, monitors=None, label="", model="annarchy_output_device", annarchy_instance=None,
                 run_tvb_multiscale_init=True, **kwargs):
        if isinstance(monitors, dict):
            monitors = OrderedDict(monitors)
        else:
            monitors = OrderedDict()
        ANNarchyDevice.__init__(self, monitors, label, model, annarchy_instance)
        if run_tvb_multiscale_init:
            OutputDevice.__init__(self, monitors, model=model, label=label)
        self.params = kwargs.pop("params", {})
        self.annarchy_instance = annarchy_instance
        if self.annarchy_instance is not None:
            self._monitors_inds = self._get_monitors_inds()

    def _assert_annarchy(self):
        if self.annarchy_instance is None:
            raise ValueError("No ANNarchy instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _assert_device(self):
        if self.annarchy_instance is not None and self.monitors is not None:
            assert isinstance(self.monitors, dict)
            from ANNarchy import Monitor, Population, PopulationView
            for monitor, pop in self.monitors.items():
                assert isinstance(monitor, Monitor)
                assert isinstance(pop, (Population, PopulationView))

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
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

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
            dictionary[attribute].append(getattr(monitor, attribute))
        else:
            dictionary[attribute] = [getattr(monitor, attribute)]

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

    def _get_senders(self, population, neurons_ranks):
        population_ind = self.annarchy_instance.Global._network[0]["populations"].index(population)
        if len(self.monitors) > 1:
            senders = ["%d_%d" % (population_ind, neuron_rank) for neuron_rank in ensure_list(neurons_ranks)]
            if len(senders) == 1:
                senders = senders[0]
            return senders
        else:
            return neurons_ranks

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

    def __init__(self, monitors=None, label="", model="Monitor",
                 annarchy_instance=None, run_tvb_multiscale_init=True, **kwargs):
        if run_tvb_multiscale_init:
            Multimeter.__init__(self, monitors, model=str(model), label=str(label))
        ANNarchyOutputDevice.__init__(self, monitors, label, self.model, annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        self.model = str(model)

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
        for monitor, population in self.monitors.items():
            data = monitor.get()
            variables = list(data.keys())
            data = np.array(list(data.values()))
            if data.size > 0:
                data = data.transpose((1, 0, 2))
                data = DataArray(data,
                                 dims=["Time", "Variable", "Neuron"],
                                 coords={"Time": self._compute_times(monitor.times(), data.shape[0]), "Variable": variables,
                                         "Neuron": self._get_senders(population, population.ranks)},
                                 name=self.label)
                if self._data.size:
                    self._data = combine_by_coords([self._data, data], fill_value=np.nan)
                else:
                    self._data = data

    def get_data(self, variables=None, name=None, dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
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
        if variables:
            data = self._data.loc[:, variables]
        else:
            data = self._data
        if np.any(data.dims != dims_names):
            data = data.rename(dict(zip(data.dims, dims_names)))
        if flatten_neurons_inds:
            data = flatten_neurons_inds_in_DataArray(data, data.dims[2])
        else:
            data = DataArray(data)
        if name:
            data.name = name
        return data

    @property
    def events(self):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        self._record()
        variables = self._data.coords["Variable"].values
        data = self._data.stack(Var=("Time", "Neuron"))
        times_senders = np.array([[float(var[0]), var[1]] for var in data.coords["Var"].values]).astype("O")
        events = dict()
        events["times"] = np.array(times_senders[:, 0]).astype("f")
        events["senders"] = times_senders[:, 1]
        del times_senders
        for i_var, var in enumerate(variables):
            events[var] = data[i_var].values
        return events

    @property
    def number_of_events(self):
        self._record()
        if self._data is None:
            return 0
        return self._data.shape[0] * self._data.shape[-1]  # times x neurons

    def reset(self):
        self._record()
        self._data = DataArray(np.empty((0, 0, 0)),
                               dims=["Time", "Variable", "Neuron"])


class ANNarchySpikeMonitor(ANNarchyOutputDevice, SpikeRecorder):

    """ANNarchySpikeMonitor class to wrap around ANNarchy.Monitor instances,
       acting as an output device of spike discrete events."""

    _data = List(of=dict, label="SpikeMonitor data buffer", default=(),
                 doc="""A list of dictionaries (one per Monitor) for holding the spike events
                       read from the Monitors""")

    def __init__(self, monitors=None, label="", annarchy_instance=None, run_tvb_multiscale_init=True, **kwargs):
        if run_tvb_multiscale_init:
            SpikeRecorder.__init__(self, monitors, label=self.label)
        ANNarchyOutputDevice.__init__(self, monitors, label, "SpikeMonitor", annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        self.model = "SpikeMonitor"

    def _record(self):
        """Method to get discrete spike events' data from ANNarchy.Monitor instances,
           and merge and store them to the _data buffer."""
        dt = self.dt
        for i_m, (monitor, population) in enumerate(self.monitors.items()):
            if len(self._data) <= i_m:
                self._data += (OrderedDict(), )
            spikes = monitor.get("spike")
            senders = self._get_senders(population, list(spikes.keys()))
            for sender, spikes_times in zip(senders, list(spikes.values())):
                self._data[i_m].update({sender:
                                            self._data[i_m].get(sender, []) + (np.array(spikes_times) * dt).tolist()})

    @property
    def events(self):
        """Method to record discrete spike events' data from ANNarchy.Monitor instances,
           and to return them in a events dictionary."""
        self._record()
        events = OrderedDict()
        events["times"] = []
        events["senders"] = []
        for i_m, monitor_data in enumerate(self._data):
            for sender, spikes_times in monitor_data.items():
                events["times"] += spikes_times
                events["senders"] += [sender] * len(spikes_times)
        return events

    @property
    def number_of_events(self):
        self._record()
        n_events = 0
        for i_m, monitor_data in enumerate(self._data):
            for sender, spikes_times in monitor_data.items():
                n_events += len(spikes_times)
        return n_events

    def reset(self):
        self._record()
        self._data = ()


class ANNarchySpikeMultimeter(ANNarchyMonitor, ANNarchySpikeMonitor, SpikeMultimeter):

    """ANNarchySpikeMultimeter class to wrap around ANNarchy.Monitor instances,
       acting as an output device of continuous time spike weights' variables."""

    _data = Attr(field_type=DataArray, label="Data buffer",
                 default=DataArray(np.empty((0, 0, 0, 0)),
                                   dims=["Time", "Variable", "Population", "Neuron"]),
                 required=True,
                 doc="""A DataArray buffer holding the data read from the Monitors""")

    def __init__(self, monitors, label="", annarchy_instance=None, **kwargs):
        SpikeMultimeter.__init__(self, monitors, model="spike_multimeter", label=self.label)
        ANNarchyMonitor.__init__(self, monitors, label, "spike_multimeter", annarchy_instance,
                                 run_tvb_multiscale_init=False, **kwargs)
        ANNarchySpikeMonitor.__init__(self, monitors, label, "spike_multimeter", annarchy_instance,
                                      run_tvb_multiscale_init=False, **kwargs)
        self.model = "spike_multimeter"

    def _record(self):
        return ANNarchyMonitor._record(self)

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

    def reset(self):
        ANNarchyMonitor.reset(self)


ANNarchyOutputDeviceDict = {}


ANNarchyOutputSpikeDeviceDict = {"SpikeMonitor": ANNarchySpikeMonitor}


ANNarchyOutputContinuousTimeDeviceDict = {"Monitor": ANNarchyMonitor,
                                          "spike_multimeter": ANNarchySpikeMultimeter}


ANNarchyOutputDeviceDict.update(ANNarchyOutputSpikeDeviceDict)
ANNarchyOutputDeviceDict.update(ANNarchyOutputContinuousTimeDeviceDict)
