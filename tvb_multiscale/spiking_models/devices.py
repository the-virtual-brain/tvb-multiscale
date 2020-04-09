# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from tvb.simulator.plot.utils.computations_utils import spikes_rate_convolution, compute_spikes_counts
from tvb.simulator.plot.utils.data_structures_utils \
    import ensure_list, flatten_list, list_of_dicts_to_dict_of_lists, \
    sort_events_by_x_and_y, data_xarray_from_continuous_events
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error

LOG = initialize_logger(__name__)


# Classes for creating:
# - output devices that can measure and summarize the activity of a whole neuronal population
# - input devices that induce some activity by stimulation to a whole neuronal population.
# Output devices assume that data is structured in discrete events tagged by time and sender neuron.


class Device(object):
    __metaclass__ = ABCMeta

    device = None
    model = "device"

    def __init__(self, device, *args, **kwargs):
        self.model = "device"
        self.device = device

    @abstractmethod
    def _assert_device(self):
        pass

    # Methods to get or set methods for devices or their connections:

    @abstractmethod
    def Get(self, attr=None):
        pass

    @abstractmethod
    def Set(self, values_dict):
        pass

    @abstractmethod
    def _get_connections(self):
        pass

    @abstractmethod
    def GetFromConnections(self, connections, attr=None):
        pass

    @abstractmethod
    def SetToConnections(self, connections, values_dict):
        pass

    def filter_neurons(self, neurons=None, exclude_neurons=[]):
        # Method to select or exclude some of the connected neurons to the device:
        temp_neurons = self.neurons
        if neurons is not None:
            temp_neurons = list(self.neurons)
        for neuron in exclude_neurons:
            if neuron in temp_neurons:
                temp_neurons.remove(neuron)
        return np.array(temp_neurons)

    def get_number_of_neurons(self, neurons=None, exclude_neurons=[]):
        return len(self.filter_neurons(neurons=neurons, exclude_neurons=exclude_neurons))

    def get_connections(self, neurons=None, exclude_neurons=[]):
        # Method to get all connections of the device to/from neurons
        if neurons is not None:
            if len(exclude_neurons) > 0:
                neurons = list(neurons)
                for neuron in exclude_neurons:
                    neurons.remove(neuron)
            return self._get_connections(source=self.device, target=tuple(neurons))
        else:
            neurons = self.neurons
            if len(exclude_neurons) > 0:
                neurons = list(neurons)
                for neuron in exclude_neurons:
                    neurons.remove(neuron)
            return self._get_connections(source=self.device, target=tuple(neurons))

    def get_weights(self, neurons=None, exclude_neurons=[]):
        return np.array([self.GetFromConnections((conn,), "weight")
                         for conn in self.get_connections(neurons, exclude_neurons)])

    def get_delays(self, neurons=None, exclude_neurons=[]):
        return np.array([self.GetFromConnections((conn,), "delay")
                         for conn in self.get_connections(neurons, exclude_neurons)])

    def get_receptors(self, neurons=None, exclude_neurons=[]):
        return np.array([self.GetFromConnections((conn,), "receptor")
                         for conn in self.get_connections(neurons, exclude_neurons)])

    def get_node_weight(self, neurons=None, exclude_neurons=[]):
        return np.mean(self.get_weights(neurons, exclude_neurons))

    def get_node_delay(self, neurons=None, exclude_neurons=[]):
        return np.mean(self.get_delays(neurons, exclude_neurons))

    def get_node_receptors(self, neurons=None, exclude_neurons=[]):
        return pd.unique(self.get_receptors(neurons, exclude_neurons))  # pd.unique is faster than np.unique

    # Properties of device connections across all neurons connected to it

    @property
    def connections(self):
        return self._get_connections(source=self.device)

    @property
    def neurons(self):
        return tuple([conn[1] for conn in self.connections])

    @property
    def weights(self):
        return np.array([self.GetFromConnections((conn,), "weight") for conn in self.connections])

    @property
    def delays(self):
        return np.array([self.GetFromConnections((conn,), "delay") for conn in self.connections])

    @property
    def receptors(self):
        return np.array([self.GetFromConnections((conn,), "receptor") for conn in self.connections])

    # Summary properties of device across all neurons connected to it

    @property
    def node_weight(self):
        return np.mean(self.weights)

    @property
    def node_delay(self):
        return np.mean(self.delays)

    @property
    def node_receptors(self):
        return np.unique(self.receptors)

    @property
    def number_of_neurons(self):
        return self.get_number_of_neurons()

    @property
    def number_of_connections(self):
        return self.get_number_of_neurons()

    # def update_number_of_connections(self):
    #     neurons = self.neurons
    #     for neuron in neurons:
    #         element_type = self.Get((neuron,))["element_type"]
    #         if element_type != "neuron":
    #             raise_value_error("Node %d is not a neuron but a %s!" % (neuron, element_type))
    #     self.number_of_connections = len(neurons)


class InputDevice(Device):
    model = "input_device"

    def __init__(self, device, *args, **kwargs):
        super(InputDevice, self).__init__(device, *args, **kwargs)
        self.model = "input_device"


InputDeviceDict = {}


class OutputDevice(Device):
    model = "output_device"

    def __init__(self, device, *args, **kwargs):
        super(OutputDevice, self).__init__(device, *args, **kwargs)
        self.model = "output_device"

    def filter_events(self, events, variables=None, neurons=None, times=None,
                      exclude_neurons=[], exclude_times=[]):

        def in_fun(x, values):
            in_flag = False
            if len(values) > 0:
                in_flag = (x in values)
                if len(values) == 2:
                    if values[0] is not None:
                        in_flag = in_flag or (x > values[0])
                    if values[1] is not None:
                        in_flag = in_flag or (x < values[1])
            return in_flag

        # This method will select/exclude part of the measured events, depending on user inputs
        if variables is None:
            variables = events.keys()
        output_events = OrderedDict()
        if events is None:
            events = self.events
        events_times = np.array(events["times"])
        senders = np.array(events["senders"])
        inds = np.ones((self.n_events,))
        if len(inds) > 0:
            if times is not None:
                if neurons is not None:
                    inds = np.logical_and(inds, [in_fun(time, flatten_list(times)) and
                                                 (not in_fun(time, flatten_list(exclude_times))) and
                                                 in_fun(sender, flatten_list(neurons)) and
                                                 (not in_fun(sender, flatten_list(exclude_neurons)))
                                                 for time, sender in zip(events_times, senders)])
                else:
                    inds = np.logical_and(inds, [in_fun(time, flatten_list(times)) and
                                                 (not in_fun(time, flatten_list(exclude_times))) and
                                                 (not in_fun(sender, flatten_list(exclude_neurons)))
                                                 for time, sender in zip(events_times, senders)])
            else:
                if neurons is not None:
                    inds = np.logical_and(inds, [(not in_fun(time, flatten_list(exclude_times))) and
                                                 in_fun(sender, flatten_list(neurons)) and
                                                 (not in_fun(sender, flatten_list(exclude_neurons)))
                                                 for time, sender in zip(events_times, senders)])
                else:
                    inds = np.logical_and(inds, [(not in_fun(time, flatten_list(exclude_times))) and
                                                 (not in_fun(sender, flatten_list(exclude_neurons)))
                                                 for time, sender in zip(events_times, senders)])
            for var in ensure_list(variables):
                output_events[var] = events[var][inds]
        else:
            for var in ensure_list(variables):
                output_events[var] = np.array([])
        return output_events

    @property
    @abstractmethod
    def events(self):
        pass

    @property
    @abstractmethod
    def number_of_events(self):
        pass

    @property
    @abstractmethod
    def reset(self):
        pass

    @property
    def senders(self):
        return self.events['senders']

    @property
    def times(self):
        return self.events['times']


class SpikeDetector(OutputDevice):
    model = "spike_detector"

    def __init__(self, device, *args, **kwargs):
        super(SpikeDetector, self).__init__(device, *args, **kwargs)
        self.model = "spike_detector"

    def get_connections(self, neurons=None, exclude_neurons=[]):
        if neurons is not None:
            if len(exclude_neurons) > 0:
                neurons = list(neurons)
                for neuron in exclude_neurons:
                    neurons.remove(neuron)
            return self._get_connections(target=self.device, source=tuple(neurons))
        else:
            neurons = self.neurons
            if len(exclude_neurons) > 0:
                neurons = list(neurons)
                for neuron in exclude_neurons:
                    neurons.remove(neuron)
            return self._get_connections(target=self.device, source=tuple(neurons))

    @property
    def connections(self):
        return self._get_connections(target=self.device)

    # The following properties are time summaries without taking into consideration spike timing:

    def get_spikes_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.filter_events(None, "times", neurons, times, exclude_neurons, exclude_times)["times"]

    def get_spikes_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.filter_events(None, "senders", neurons, times, exclude_neurons, exclude_times)["senders"]

    def get_number_of_spikes(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return len(self.get_spikes_times(neurons, times, exclude_neurons, exclude_times))

    def get_mean_number_of_spikes(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        n_neurons = self.get_number_of_neurons(neurons, exclude_neurons)
        if n_neurons > 0:
            return len(self.get_spikes_times(neurons, times, exclude_neurons, exclude_times)) / n_neurons
        else:
            return 0.0

    def get_spikes_rate(self, dt=1.0, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_mean_number_of_spikes(neurons, times, exclude_neurons, exclude_times) / dt

    def get_spikes_times_by_neurons(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[],
                                    full_senders=False):
        sorted_events = sort_events_by_x_and_y(self.events, x="senders", y="times",
                                               filter_x=neurons, filter_y=times,
                                               exclude_x=exclude_neurons, exclude_y=exclude_times)
        if full_senders:
            # In this case we also include neurons with 0 spikes in the output
            sender_neurons = self.filter_neurons(neurons=neurons, exclude_neurons=exclude_neurons)
            output = OrderedDict()
            for neuron in sender_neurons:
                output[neuron] = np.array([])
            output.update(sorted_events)
            return output
        else:
            return sorted_events

    def get_spikes_neurons_by_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return sort_events_by_x_and_y(self.events, x="times", y="senders",
                                      filter_x=times, filter_y=neurons,
                                      exclude_x=exclude_times, exclude_y=exclude_neurons)

    @property
    def spikes_times(self):
        return self.times

    @property
    def spikes_senders(self):
        return self.senders

    @property
    def number_of_spikes(self):
        return self.number_of_events

    @property
    def mean_number_of_spikes(self):
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        return self.get_spikes_rate()

    # The following properties are computed across time:

    def compute_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                        spikes_kernel=None, mode="per_neuron",
                                        name=None, **kwargs):
        if spikes_kernel is None:
            # Default spikes' kernel is just a rectangular one, normalized with its width.
            spikes_kernel = np.ones((spikes_kernel_width_in_points,)) / spikes_kernel_width

        if name is None:
            name = self.model + " - Total spike rate across time"

        if mode == "per_neuron":
            senders_neurons = []
            rates = []
            # Computing separately per neuron
            for neuron, spikes_times in self.get_spikes_times_by_neurons(full_senders=True, **kwargs).items():
                senders_neurons.append(neuron)
                if len(spikes_times) > 0:
                    # Getting spikes counts per time interval
                    spikes_counts = compute_spikes_counts(spikes_times, time)
                    # Computing rate as some kind of convolution with spikes_kernel
                    rates.append(spikes_rate_convolution(spikes_counts, spikes_kernel))
                else:
                    rates.append(np.zeros(time.shape))
            return xr.DataArray(rates, dims=["Neuron", "Time"], coords={"Neuron": senders_neurons,
                                                                        "Time": time})
        else:
            # Computing for the whole population(s) alltogether
            spikes_times = self.get_spikes_times(**kwargs)
            if len(spikes_times) > 0:
                # Getting spikes counts per time interval
                spikes_counts = compute_spikes_counts(spikes_times, time)
                # Computing rate as some kind of convolution with spikes_kernel
                rates = spikes_rate_convolution(spikes_counts, spikes_kernel)
            else:
                rates = np.zeros(time.shape)
            return xr.DataArray(rates, dims=["Time"], coords={"Time": time}, name=name)

    def compute_mean_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                             spikes_kernel=None, name=None, **kwargs):
        if name is None:
            name = self.model + " - Mean spike rate accross time"
        n_neurons = self.get_number_of_neurons(**kwargs)
        if n_neurons > 0:
            return self.compute_spikes_rate_across_time(time, spikes_kernel_width, spikes_kernel_width_in_points,
                                                        spikes_kernel, "total", name, **kwargs) / n_neurons
        else:
            return 0.0 * time


class Multimeter(OutputDevice):
    model = "multimeter"

    def __init__(self, device, *args, **kwargs):
        super(Multimeter, self).__init__(device)
        self.model = "multimeter"

    @property
    @abstractmethod
    def record_from(self):
        pass

    @property
    def time(self):
        return np.unique(self.times)

    def _determine_variables(self, variables=None):
        if variables is not None:
            variables = ensure_list(variables)
            for variable in variables:
                assert variable in self.record_from
        else:
            variables = self.record_from
        return variables

    def get_data(self, variables=None, neurons=None, exclude_neurons=[],
                 name=None, dims_names=["Variable", "Neuron", "Time"]):
        if name is None:
            name = self.model
        events = dict(self.events)
        times = events.pop("times")
        senders = events.pop("senders")
        # We assume that the multimeter captures events even for continuous variables as it is the case in NEST.
        # Therefore, we have to re-arrange the output to get all variables separated following time order.
        return data_xarray_from_continuous_events(events, times, senders,
                                                  variables=self._determine_variables(variables),
                                                  filter_senders=neurons, exclude_senders=exclude_neurons,
                                                  name=name, dims_names=dims_names)

    def get_mean_data(self, variables=None, neurons=None, exclude_neurons=[]):
        data = self.get_data(variables, neurons, exclude_neurons)
        return data.mean(dim="Neuron")

    def get_total_data(self, variables=None, neurons=None, exclude_neurons=[]):
        data = self.get_data(variables, neurons, exclude_neurons)
        return data.sum(dim="Neuron")

    @property
    def data(self):
        return self.get_data()

    @property
    def data_mean(self):
        return self.get_mean_data()

    def current_data(self, variables=None, neurons=None, exclude_neurons=[],
                     name=None, dims_names=["Variable", "Neuron"]):
        # This method will return current time data
        if name is None:
            name = self.model
        coords = OrderedDict()
        variables = self._determine_variables(variables)
        events = self.events
        times = events["times"]
        coords[dims_names[0]] = variables
        if len(times) > 0:
            # Get only the last time stamp events:
            output_inds = events["times"] == events["times"][-1]
            # Optionally filter sender neurons
            senders = events["senders"]
            if neurons is not None:
                output_inds = np.logical_and(output_inds,
                                             [sender in flatten_list(neurons) for sender in senders])
            else:
                neurons = ensure_list(self.neurons)
            if len(exclude_neurons) > 0:
                output_inds = np.logical_and(output_inds,
                                             [sender not in flatten_list(exclude_neurons) for sender in senders])
            coords[dims_names[1]] = neurons
            data = np.empty((len(variables), len(neurons)))
            for i_var, var in enumerate(variables):
                data[i_var] = events[var][output_inds]
        else:
            # The multimeter is still empty, so return zeros
            if neurons is None:
                neurons = ensure_list(self.neurons)
            else:
                neurons = flatten_list(neurons)
            for neuron in flatten_list(exclude_neurons):
                try:
                    neurons.remove(neuron)
                except:
                    pass
            n_neurons = len(neurons)
            coords[dims_names[1]] = neurons
            data = np.empty((len(variables), len(neurons)))
            for i_var in range(len(variables)):
                data[i_var] = np.zeros((n_neurons,))
        return xr.DataArray(data, coords=coords, dims=list(coords.keys()), name=name)

    def current_data_mean(self, variables=None, neurons=None, exclude_neurons=[],
                          name=None, dims_names=["Variable", "Neuron"]):
        output = self.current_data(variables, neurons, exclude_neurons, name, dims_names)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, variables=None, neurons=None, exclude_neurons=[],
                                 name=None, dims_names=["Variable", "Neuron"]):
        return self.current_data_mean(variables, neurons, exclude_neurons, name, dims_names).values.tolist()


class Voltmeter(Multimeter):
    # The Voltmer is just a Mutlimeter measuring only a voltage quantity
    model = "voltmeter"

    def __init__(self, device, *args, **kwargs):
        super(Voltmeter, self).__init__(device)
        self.model = "voltmeter"

    @property
    @abstractmethod
    def var(self):
        # A method to return the string of the voltage variable's name, e.g, "V_m" for membrane potential in NEST
        pass

    def get_data(self, neurons=None, exclude_neurons=[],
                 name=None, dims_names=["Variable", "Neuron", "Time"]):
        return super(Voltmeter, self).get_data(self.var, neurons, exclude_neurons,
                                               name, dims_names)

    def get_mean_data(self, neurons=None, exclude_neurons=[]):
        data = self.get_data(neurons, exclude_neurons)
        return data.mean(dim="Neuron")

    @property
    def data(self):
        return self.get_data()

    @property
    def data_mean(self):
        return self.get_mean_data()

    def current_data(self, neurons=None, exclude_neurons=[],
                     name=None, dims_names=["Variable", "Neuron"]):
        return super(Voltmeter, self).current_data(self.var, neurons, exclude_neurons,
                                                   name, dims_names)

    def current_data_mean(self, neurons=None, exclude_neurons=[],
                          name=None, dims_names=["Variable", "Neuron"]):
        output = self.current_data(neurons, exclude_neurons, name, dims_names)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, neurons=None, exclude_neurons=[],
                                 name=None, dims_names=["Variable", "Neuron"]):
        return self.current_data_mean(neurons, exclude_neurons, name, dims_names).values.tolist()

    def get_var(self, neurons=None, exclude_neurons=[]):
        return self.get_data(self.var, neurons, exclude_neurons)[0][self.var]


class SpikeMultimeter(Multimeter, SpikeDetector):
    # This is a multimeter that measure a continuous spike variable,
    # i.e., a variable that for time moments where there is no spike emitted/received, and
    # equal to a positive or negative spike weight, where there is a spike.

    model = "spike_multimeter"
    spike_var = "spikes"

    def __init__(self, device, *args, **kwargs):
        super(SpikeMultimeter, self).__init__(device)
        self.model = "spike_multimeter"

    def get_spikes(self, neurons=None, exclude_neurons=[]):
        return self.get_data(self.spike_var, neurons, exclude_neurons)[self.spike_var]

    def get_spikes_inds(self, neurons=None, exclude_neurons=[]):
        # This method returns time moments where there is a spike for every sender neuron.
        spikes = self.events[self.spike_var]
        spikes_inds = spikes != 0.0
        senders = self.senders
        if neurons is not None:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender in flatten_list(neurons) for sender in senders])
        if len(exclude_neurons) > 0:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender not in flatten_list(exclude_neurons) for sender in senders])
        return np.where(spikes_inds)

    def get_spikes_events(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        # This method returns an event structure similar to a spike_detectors,
        # i.e., where there are events only for spike times, not for continuous time
        events = dict(self.events)
        inds = events[self.spike_var] != 0
        for var, val in events.items():
            events[var] = val[inds]
        events["weights"] = np.array(events[self.spike_var])
        del events[self.spike_var]
        return self.filter_events(events, None, neurons, times, exclude_neurons, exclude_times)

    def get_spikes_weights(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["weights"]

    def get_spikes_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["times"]

    def get_spikes_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["senders"]

    # The following properties are time summaries without taking into consideration spike timing:

    def get_mean_spikes_activity(self, neurons=None, exclude_neurons=[]):
        # This method returns mean activity by adding spike weights
        # and dividing by the total number of neurons, after optional filtering
        n_neurons = self.get_number_of_neurons(neurons, exclude_neurons)
        if n_neurons > 0:
            return np.sum(self.get_spikes_weights(neurons, exclude_neurons)) / n_neurons
        else:
            return 0.0

    def get_total_spikes_activity(self, neurons=None, exclude_neurons=[]):
        # This method returns mean activity by adding spike weights
        return np.sum(self.get_spikes_weights(neurons, exclude_neurons))

    @property
    def spikes(self):
        return self.get_spikes()

    @property
    def spikes_times(self):
        return self.get_spikes_times()

    @property
    def spikes_senders(self):
        return self.get_spikes_senders()

    @property
    def spikes_weights(self):
        return self.get_spikes_weights()

    @property
    def number_of_spikes(self):
        return np.sum(self.spikes > 0)

    @property
    def mean_number_of_spikes(self):
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        return self.get_spikes_rate()

    @property
    def mean_spikes_activity(self):
        return self.get_mean_spikes_activity()

    @property
    def total_spikes_activity(self):
        return self.get_total_spikes_activity()

    # The following properties are computed across time:

    def compute_spikes_activity_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                            spikes_kernel=None, mode="per_neuron",
                                            name=None, rate_mode="activity", **kwargs):

        if name is None:
            name = self.model + " - Total spike activity accross time"
        spikes = self.get_spikes(**kwargs).values
        if rate_mode == "rate":
            for i_spike, spike in spikes:
                spikes[i_spike] = np.heaviside(spike, 0.0)

        if spikes_kernel is None:
            spikes_kernel = np.ones((spikes_kernel_width_in_points,))
            if rate_mode.find("rate") > -1:
                # For spike rate computation we have to normalize with the kernel width in time units
                spikes_kernel /= spikes_kernel_width
            else:
                # For "activity" computation we have to normalize with the kernel width in time steps
                spikes_kernel /= spikes_kernel_width_in_points

        if mode == "per_neuron":
            # Returning output per neuron
            activity = []
            for spike in spikes:
                activity.append(spikes_rate_convolution(spike, spikes_kernel))
            return xr.DataArray(np.array(activity), dims=["Neuron", "Time"],
                                coords={"Neuron": spikes.coords[spikes.dims[0]].item(),
                                        "Time": time})
        else:
            # Returning output as for all neurons together
            spikes = np.sum(spikes, axis=0).squeeze()
            activity = spikes_rate_convolution(spikes, spikes_kernel)

        return xr.DataArray(activity, dims=["Time"], coords={"Time": time}, name=name)

    def compute_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                        spikes_kernel=None, mode="per_neuron",
                                        name=None, **kwargs):
        return self.compute_spikes_activity_across_time(time, spikes_kernel_width, spikes_kernel_width_in_points,
                                                        spikes_kernel=spikes_kernel, mode=mode,
                                                        name=name, rate_mode="rate", **kwargs)

    def compute_mean_spikes_activity_across_time(self, time, spike_kernel_width,
                                                 spikes_kernel=None, name=None, **kwargs):
        if name is None:
            name = self.model + " - Mean spike activity accross time"
        n_neurons = self.get_number_of_neurons(**kwargs)
        if n_neurons > 0:
            return self.compute_spikes_activity_across_time(time, spike_kernel_width, spikes_kernel,
                                                            "total", name, **kwargs) / n_neurons
        else:
            return 0.0 * time


OutputDeviceDict = {"spike_detector": SpikeDetector,
                    "multimeter": Multimeter,
                    "spike_multimeter": SpikeMultimeter,
                    "voltmeter": Voltmeter}

OutputSpikeDeviceDict = {"spike_detector": SpikeDetector,
                         "spike_multimeter": SpikeMultimeter}


class DeviceSet(pd.Series):

    def __init__(self, name="", model="", device_set=pd.Series(), **kwargs):
        super(DeviceSet, self).__init__(device_set, **kwargs)
        if np.any([not isinstance(device, Device) for device in self]):
            raise ValueError("Input device_set is not a Series of Device objects!:\n%s" %
                             str(device_set))
        self.name = str(name)
        self.model = str(model)
        self.update_model()
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def _input_nodes(self, nodes=None):
        if nodes is None:
            # no input nodes
            return list(self.index)
        else:
            if nodes in list(self.index) or nodes in list(range(len(self))):
                # input nodes is a single index or label
                return [nodes]
            else:
                # input nodes is a sequence of indices or labels
                return list(nodes)

    def _return_by_type(self, values_dict, return_type="dict", concatenation_index_name="Region", name=None):
        if return_type == "values":
            return list(values_dict.values())
        elif return_type == "dict":
            return values_dict
        elif return_type == "Series":
            return pd.Series(values_dict, name=name)
        elif return_type == "xarray":
            if name is None:
                name = self.name
            for key, val in values_dict.items():
                if isinstance(val, xr.DataArray):
                    val.name = key
                else:
                    raise_value_error("DataArray concatenation not possible! "
                                      "Not all outputs are DataArrays!:\n %s" % str(values_dict))
            dims = list(values_dict.keys())
            values = list(values_dict.values())
            if len(values) == 0:
                return xr.DataArray([])
            output = xr.concat(values, dim=pd.Index(dims, name=concatenation_index_name))
            output.name = name
            return output
        else:
            return values_dict

    def do_for_all_devices(self, attr, *args, nodes=None, return_type="values",
                           concatenation_index_name="Region", name=None, **kwargs):
        values_dict = OrderedDict()
        for node in self._input_nodes(nodes):
            val = getattr(self[node], attr)
            if hasattr(val, "__call__"):
                values_dict.update({node: val(*args, **kwargs)})
            else:
                values_dict.update({node: val})
        return self._return_by_type(values_dict, return_type, concatenation_index_name, name)

    @property
    def number_of_neurons(self):
        number_of_neurons = self.do_for_all_devices("number_of_neurons")
        if len(number_of_neurons) == 0:
            number_of_neurons = 0
        return number_of_neurons

    @property
    def times(self):
        return self.do_for_all_devices("times", return_type="values")

    @property
    def time(self):
        return self.do_for_all_devices("time", return_type="values")

    @property
    def senders(self):
        return self.do_for_all_devices("senders", return_type="values")

    @property
    def weights(self):
        return self.do_for_all_devices("node_weight", return_type="values")

    @property
    def delays(self):
        return self.do_for_all_devices("node_delay", return_type="values")

    @property
    def receptors(self):
        return self.do_for_all_devices("node_receptors", return_type="values")

    def record_from_per_node(self, nodes=None, return_type="values"):
        # A method to get the attribute "record_from" from all devices of the set
        # This method will cause an error if the device set does not consist of OutputDevices
        values_dict = self.do_for_all_devices("record_from", nodes=nodes, return_type="dict")
        if return_type == "dict":
            return values_dict
        values = np.array(list(values_dict.values()))
        unique_values, indices = np.unique(values, return_inverse=True)
        if return_type == "values":
            return unique_values.squeeze().tolist()
        nodes = np.array(list(values_dict.keys()))
        output = OrderedDict()
        for i_output, val in enumerate(unique_values):
            key = np.where(indices == i_output)[0]
            output[tuple(nodes[key])] = val
        return output

    @property
    def record_from(self):
        return self.record_from_per_node()

    def update_model(self):
        if len(self) > 0:
            models = self.do_for_all_devices("model")
            if np.any([model != self.model for model in models]):
                raise ValueError("Not all devices of the DeviceSet are of the same model!:\n %s" % str(models))

    def update(self, device_set=None):
        if device_set:
            super(DeviceSet, self).update(device_set)
        self.update_model()

    def Get(self, attrs=None, nodes=None, return_type="dict", name=None):
        # A method to get attributes from all devices of the set
        if attrs is None:
            # Get dictionary of all attributes
            values_dict = []
            for node in self._input_nodes(nodes):
                values_dict.append(self[node].Get())
            values_dict = list_of_dicts_to_dict_of_lists(values_dict)
        else:
            values_dict = OrderedDict({})
            for attr in ensure_list(attrs):
                this_attr = []
                for node in self._input_nodes(nodes):
                    this_attr.append(self[node].Get(attr))
                values_dict.update({attr: this_attr})
        return self._return_by_type(values_dict, return_type, name)

    def Set(self, value_dict, nodes=None):
        # A method to set attributes to all devices of the set
        # TODO: find a nice way to treat arrays or lists attributes.
        #  Some NEST device parameters have to be arrays and some scalars,
        #  but we don't know what is what beforehand...
        def get_scalar_dict1(din, index):
            dout = dict()
            for key, val in din.items():
                try:
                    dout[key] = val[index] * np.ones(1)
                except:
                    dout[key] = val
            return dout

        def get_scalar_dict2(din, index):
            dout = dict()
            for key, val in din.items():
                try:
                    dout[key] = val[index]
                except:
                    dout[key] = val
            return dout

        for i_n, node in enumerate(self._input_nodes(nodes)):
            try:
                # Good for spike times and weights of spike generator
                value_dict_i_n = get_scalar_dict1(value_dict, i_n)
                self[node].Set(value_dict_i_n)
            except:
                # Good for amplitude of dc generator and rate of poisson generator
                value_dict_i_n = get_scalar_dict2(value_dict, i_n)
                self[node].Set(value_dict_i_n)
