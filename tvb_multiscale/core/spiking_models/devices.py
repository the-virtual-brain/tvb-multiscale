# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from six import string_types
from copy import deepcopy
from collections import OrderedDict

import pandas as pd
import xarray as xr
import numpy as np

from tvb_multiscale.core.config import initialize_logger, LINE
from tvb_multiscale.core.utils.data_structures_utils import filter_events, summarize, flatten_neurons_inds_in_DataArray

from tvb.basic.neotraits.api import HasTraits, Attr, Int, List

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, list_of_dicts_to_dict_of_lists, sort_events_by_x_and_y, data_xarray_from_continuous_events, is_integer


LOG = initialize_logger(__name__)


class Device(HasTraits):
    __metaclass__ = ABCMeta

    """Device class to wrap around an 
       output (recording/measuring/monitoring) or input (stimulating) device.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
    """

    device = None  # a device object, depending on its simulator implementation

    model = Attr(field_type=str, default="device", required=True,
                 label="Device model", doc="""Label of Device model""")

    label = Attr(field_type=str, default="", required=True,
                 label="Device label", doc="""Label of Device""")

    _number_of_connections = Int(field_type=int, default=0, required=True, label="Number of connections",
                                 doc="""The number of total device's connections""")

    _number_of_neurons = Int(field_type=int, default=0, required=True, label="Number of neurons",
                             doc="""The number of total neurons connected to the device""")

    # Modify accordingly for other simulators than NEST, by settin in the inheriting class:
    # _weight_attr = "weight"
    # _delay_attr = "delay"
    # _receptor_attr = "receptor"

    def __init__(self, device=None, *args, **kwargs):
        self.device = device   # a device object, depending on its simulator implementation
        super(Device, self).__init__()
        self.label = kwargs.pop("label", "")
        self.model = kwargs.pop("model", "device")
        self._number_of_connections = 0
        self._number_of_neurons = 0

    def __repr__(self):
        output = "%s - Model: %s\n%s" % (self.__class__.__name__, self.model, self.device.__str__())
        if len(self.label):
            output = "%s: %s" % (self.label, output)
        return output

    def __str__(self):
        return self.print_str()

    @property
    def _print_from_to(self):
        return "from/to"

    @abstractmethod
    def _print_neurons(self):
        pass

    def print_str(self, connectivity=False):
        output = "\n" + self.__repr__() + "\nparameters: %s" % str(self.get_attributes())
        if connectivity:
            neurons = ensure_list(self.neurons)
            conn_attrs = self.GetFromConnections(attrs=[self._weight_attr, self._delay_attr, self._receptor_attr],
                                                 summary=3)
            output += ",\nconnections %s\n%s," \
                      "\nweights: %s," \
                      "\ndelays: %s," \
                      "\nreceptors: %s" % \
                      (self._print_from_to,
                       self._print_neurons(neurons),
                       str(conn_attrs.get(self._weight_attr, "")),
                       str(conn_attrs.get(self._delay_attr, "")),
                       str(conn_attrs.get(self._receptor_attr, "")))
        return output

    @abstractmethod
    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        pass

    # Methods to get or set attributes for devices and/or their connections:

    @abstractmethod
    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        pass

    @abstractmethod
    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all device's attributes.
           Returns:
            Dictionary of attributes.
        """
        pass

    @abstractmethod
    def GetConnections(self):
        """Method to get connections of the device to/from neurons.
           Returns:
            connections' objects.
        """
        pass

    @abstractmethod
    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the device
            Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects. Default = None, corresponding to all device's connections
        """
        pass

    @abstractmethod
    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all device's attributes
            connections: connections' objects. Default = None, corresponding to all device's connections
           Returns:
            Dictionary of sequences (tuples, lists, arrays) of connections' attributes.
        """
        pass

    def get_attributes(self):
        """Method to get all attributes of the device.
           Returns:
            Dictionary of sequences (tuples, lists, arrays) of neurons' attributes.
        """
        return self.Get()

    def get_number_of_connections(self):
        """Method to get the number of  connections of the device to/from neurons.
           Returns:
            int: number of connections
        """
        return len(self.GetConnections())

    def get_number_of_neurons(self):
        """Method to get the number of  neurons connected to/from the device.
           Returns:
            int: number of connections
        """
        return len(self.neurons)

    def SetToConnections(self, values_dict):
        """Method to set attributes of the connections from/to the device.
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        self.SetToConnections(values_dict)

    def GetFromConnections(self, attrs=None, summary=None):
        """Method to get attributes of the connections from/to the device.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of sequences (tuples, lists, arrays) of connections' attributes.
        """
        attributes = self._GetFromConnections(attrs)
        if summary:
            return summarize(attributes, summary)
        else:
            return attributes

    def get_weights(self, summary=None):
        """Method to get weights of the connections from/to the device.
           Arguments:
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequences (tuples, lists, arrays) of connections' weights
        """
        return self.GetFromConnections(self._weight_attr, summary)[self._weight_attr]

    def get_delays(self, summary=None):
        """Method to get delays of the connections from/to the device.
           Arguments:
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequences (tuples, lists, arrays) of connections' delays
        """
        return self.GetFromConnections(self._delay_attr, summary)[self._delay_attr]

    def get_receptors(self, summary=None):
        """Method to get the receptors of the connections from/to the device.
           Arguments:
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequences (tuples, lists, arrays) of connections' receptors
        """
        return self.GetFromConnections(self._receptor_attr, summary)[self._receptor_attr]

    # attributes of device connections across all neurons connected to it

    @property
    @abstractmethod
    def connections(self):
        """Method to get all connections of the device to/from the device.
           Returns:
            connections' objects.
        """
        pass

    @property
    @abstractmethod
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        pass

    @property
    def weights(self):
        """Method to get all connections' weights of the device to/from the device."""
        return self.get_weights()

    @property
    def delays(self):
        """Method to get all connections' delays of the device to/from the device."""
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get all connections' receptors of the device to/from the device."""
        return self.get_receptors()

    # Summary attributes of device across all neurons connected to it

    @property
    def number_of_connections(self):
        """Method to get the number of all connections from/to the device."""
        if not self._number_of_connections:
            self._number_of_connections = self.get_number_of_connections()
        return self._number_of_connections

    @property
    def number_of_neurons(self):
        """Method to get the number of all neurons connected from/to the device."""
        if not self._number_of_neurons:
            self._number_of_neurons = self.get_number_of_neurons()
        return self._number_of_neurons

    @property
    def node_weight(self):
        """Method to get the mean of all connections' weights of the device to/from neurons."""
        return np.mean(self.weights)

    @property
    def node_delay(self):
        """Method to get the mean of all connections' delays of the device to/from neurons."""
        return np.mean(self.delays)

    @property
    def node_receptors(self):
        """Method to get all unique connections' receptors of the device to/from neurons."""
        return np.unique(self.receptors)


class InputDevice(Device):

    """InputDevice class to wrap around an input (stimulating) device"""

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "input_device")
        super(InputDevice, self).__init__(device, *args, **kwargs)

    def GetConnections(self):
        """Method to get connections of the device to neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.device)

    @property
    def connections(self):
        """Method to get all connections of the device to/from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(source=self.device)

    @property
    def _print_from_to(self):
        return "to"


InputDeviceDict = {}


class OutputDevice(Device):

    """OutputDevice class to wrap around an output (recording/measuring/monitoring) device"""

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "output_device")
        super(OutputDevice, self).__init__(device, *args, **kwargs)

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
        return self.GetConnections(target=self.device)

    @property
    def _print_from_to(self):
        return "from"

    def filter_events(self, events, variables=None, times=None, exclude_times=[]):
        """This method will select/exclude part of the measured events, depending on user inputs
            Arguments:
                events: dictionary of events
                variables: sequence (list, tuple, array) of variables to be included in the output,
                           assumed to correspond to keys of the events dict.
                           Default=None, corresponds to all keys of events.
                times: sequence (list, tuple, array) of times the events of which should be included in the output.
                         Default = None, corresponds to all events' times.
                exclude_times: sequence (list, tuple, array) of times
                               the events of which should be excluded from the output. Default = [].
            Returns:
              the filtered dictionary of events
        """
        return filter_events(events, variables, times, exclude_times)

    def get_events(self, events=None, variables=None, events_inds=None, **filter_kwargs):
        """This method will select/exclude part of the measured events, depending on user inputs
            Arguments:
                events: dictionary of events
                variables: sequence (list, tuple, array) of variables to be included in the output,
                           assumed to correspond to keys of the events dict.
                           Default=None, corresponds to all keys of events.
                events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                             to slice the event attributes. Default = None, i.e., it doesn't apply.
                filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
              the filtered dictionary of events
        """
        if events is None:
            # The events of the device:
            events = deepcopy(self.events)
        if variables is None:
            variables = events.keys()
        else:
            variables = ensure_list(variables)
        n_events = len(events["times"])
        if n_events > 0:
            if events_inds is not None:
                if hasattr(events_inds, "__len__") or isinstance(events_inds, slice):
                    # events_inds are numerical or boolean indices, or a slice:
                    select_fun = lambda x, events_inds: np.array(x)[events_inds].tolist()
                else: # events_inds is a scalar to start indexing from:
                    select_fun = lambda x, events_inds: np.array(x)[events_inds:].tolist()
                for var in variables:
                    events[var] = select_fun(events[var], events_inds)
            if len(filter_kwargs) > 0:
                return filter_events(events, **filter_kwargs)
        else:
            for var in variables:
                events[var] = []
        return events

    @property
    @abstractmethod
    def events(self):
        """This method returns the dictionary of events"""
        pass

    @property
    @abstractmethod
    def number_of_events(self):
        """This method returns the number (integer) of events"""
        pass

    @abstractmethod
    def reset(self):
        """This method resets the device by deleting all events"""
        pass

    @property
    def senders(self):
        """This method returns the senders neurons of events"""
        return self.events['senders']

    @property
    def times(self):
        """This method returns the times of events"""
        return self.events['times']


class SpikeRecorder(OutputDevice):

    """OutputDevice class to wrap around a spike recording device"""

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "spike_recorder")
        super(SpikeRecorder, self).__init__(device, *args, **kwargs)

    def get_spikes_events(self, events_inds=None, **filter_kwargs):
        """This method will select/exclude part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the filtered dictionary of spikes' events
        """
        return self.get_events(events_inds=events_inds, **filter_kwargs)

    def get_spikes_times(self, events_inds=None, **filter_kwargs):
        """This method will return spikes' times
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the spikes' times in an array
        """
        return self.get_spikes_events(events_inds=events_inds, variables="times", **filter_kwargs)["times"]

    def get_spikes_senders(self, events_inds=None, **filter_kwargs):
        """This method will return spikes' times
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
           Returns:
            the spikes' times in an array
        """
        return self.get_spikes_events(events_inds=events_inds, variables="senders", **filter_kwargs)["senders"]

    # The following attributes are time summaries without taking into consideration spike timing:

    def get_number_of_spikes(self, events_inds=None, **filter_kwargs):
        """This method will return the total number of spikes
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                         to slice the event attributes. Default = None, i.e., it doesn't apply.
            filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             int: the spikes' total number
        """
        return len(self.get_spikes_times(events_inds=events_inds, **filter_kwargs))

    def get_mean_number_of_spikes(self, events_inds=None, **filter_kwargs):
        """This method will return the total number of spikes divided by the total number of connected neurons
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             float: total number of spikes divided by the total number of connected neurons
        """
        n_neurons = self.get_number_of_neurons()
        if n_neurons > 0:
            return len(self.get_spikes_times(events_inds=events_inds, **filter_kwargs)) / n_neurons
        else:
            return 0.0

    def get_spikes_rate(self, dt=1.0, events_inds=None, **filter_kwargs):
        """This method will return the mean spike rate,
           as the total number of spikes divided by the total number of connected neurons and time duration
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            dt: total time duration, Default = 1.0 (in msec)
            events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                         to slice the event attributes. Default = None, i.e., it doesn't apply.
            filter_kwargs: see filter_events method for its possible keyword arguments
           Returns:
            float: total number of spikes divided by the total number of connected neurons
        """
        return self.get_mean_number_of_spikes(events_inds=events_inds, **filter_kwargs) / dt

    def get_spikes_times_by_neurons(self, events_inds=None, times=None, exclude_times=[], full_senders=False):
        """This method will return the spikes' times per neuron,
             after selecting/excluding part of the detected spikes' events, depending on user inputs
             Arguments:
              events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                           to slice the event attributes. Default = None, i.e., it doesn't apply.
              times: sequence (list, tuple, array) of times the events of which should be included in the output.
                     Default = None, corresponds to all events' times.
              exclude_times: sequence (list, tuple, array) of times
                             the events of which should be excluded from the output. Default = [].
              full_senders: if True, we include neurons' indices that have sent no spike at all. Default=False
             Returns:
              dictionary of spike events sorted by sender neuron
         """
        sorted_events = sort_events_by_x_and_y(self.get_events(events_inds=events_inds),
                                               x="senders", y="times",
                                               filter_y=times, exclude_y=exclude_times)
        if full_senders:
            # In this case we also include neurons with 0 spikes in the output
            sender_neurons = self.neurons
            output = OrderedDict()
            for neuron in sender_neurons:
                output[neuron] = np.array([])
            output.update(sorted_events)
            return output
        else:
            return sorted_events

    def get_spikes_neurons_by_times(self, events_inds=None, times=None, exclude_times=[]):
        """This method will return the spikes' senders per spike time,
             after selecting/excluding part of the detected spikes' events, depending on user inputs
             Arguments:
              events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                           to slice the event attributes. Default = None, i.e., it doesn't apply.
              times: sequence (list, tuple, array) of times the events of which should be included in the output.
                     Default = None, corresponds to all events' times.
              exclude_times: sequence (list, tuple, array) of times
                             the events of which should be excluded from the output. Default = [].
             Returns:
              dictionary of spike events sorted by time
         """
        return sort_events_by_x_and_y(self.get_events(events_inds=events_inds),
                                      x="times", y="senders", filter_x=times, exclude_x=exclude_times)

    @property
    def spikes_times(self):
        """This method will return spikes' times in an array"""
        return self.times

    @property
    def spikes_senders(self):
        """This method will return spikes' sender neurons in an array"""
        return self.senders

    @property
    def number_of_spikes(self):
        """This method will return the total number (int) of spike events"""
        return self.number_of_events

    @property
    def mean_number_of_spikes(self):
        """This method will return the total number of spikes divided by the total number of connected neurons"""
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        """This method will return the mean spike rate (float) as the
           total number of spikes divided by the total number of connected neurons, assuimng a duration of 1 ms
        """
        return self.get_spikes_rate()


class Multimeter(OutputDevice):

    """OutputDevice class to wrap around an output device
       that records continuous time data only."""

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "multimeter")
        super(Multimeter, self).__init__(device, *args, **kwargs)

    @property
    @abstractmethod
    def record_from(self):
        """This method returns a list of the recorded variabls of the device"""
        pass

    @property
    def time(self):
        """This method returns the time vector (array) of the device."""
        return np.unique(self.times)

    def _determine_variables(self, variables=None):
        """This method selects (a subset of) variables after asserting that the multimeter records from them.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
           Returns:
            a list of (selected) variables' names the multimeter records from
        """
        if variables is not None:
            variables = ensure_list(variables)
            for variable in variables:
                assert variable in self.record_from
        else:
            variables = self.record_from
        return variables

    @abstractmethod
    def get_data(self, variables=None, name=None, dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        pass

    def get_mean_data(self, variables=None, name=None, dims_names=["Time", "Variable"]):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None, which defaults to the label of the Device
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], dims_names[1], "Neuron"]
        data = self.get_data(variables, name, dims_names)
        return data.mean(dim="Neuron")

    def get_total_data(self, variables=None, name=None, dims_names=["Time", "Variable"]):
        """This method returns time series' data recorded by the multimeter, summed across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
           flatten_neurons_inds: if true, neurons coordinates are arranged from 1 to number of neurons,
                                  instead for neurons_inds
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], dims_names[1], "Neuron"]
        data = self.get_data(variables, name, dims_names)
        return data.sum(dim="Neuron")

    @property
    def data(self):
        """This method returns time series' data recorded by the multimeter.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data()

    @property
    def data_mean(self):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data()

    @property
    def data_total(self):
        """This method returns time series' data recorded by the multimeter, summed across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data()

    def current_data(self, variables=None, name=None, dims_names=["Variable", "Neuron"], flatten_neurons_inds=True):
        """This method returns the last time point of the data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron"]
            flatten_neurons_inds: if true, neurons coordinates are arranged from 1 to number of neurons,
                                  instead for neurons_inds
           Returns:
            a xarray DataArray with the output data
        """
        # This method will return current time data
        if name is None:
            name = self.model
        coords = OrderedDict()
        events = self.events
        times = events["times"]
        coords[dims_names[0]] = self._determine_variables(variables)
        if flatten_neurons_inds:
            coords[dims_names[1]] = np.arange(self.number_of_neurons)
        else:
            coords[dims_names[1]] = np.array(self.neurons)
        data = np.zeros((len(coords[dims_names[0]]), self.number_of_neurons))
        if len(times) > 0:
            senders_sorted_inds = np.argsort(events["senders"])
            # Get only the last time stamp events:
            output_inds = events["times"] == events["times"][-1]
            for i_var, var in enumerate(coords[dims_names[0]]):
                data[i_var][:, senders_sorted_inds] = events[var][output_inds]
        return xr.DataArray(data, coords=coords, dims=list(coords.keys()), name=name)

    def current_data_mean(self, variables=None, name=None, dim_name="Variable"):
        """This method returns the last time point of the data recorded by the multimeter, averaged across neurons.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dim_name, "Neuron"]
        output = self.current_data(variables, name, dims_names)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, variables=None):
        """This method returns the last time point of the data recorded by the multimeter, averaged across neurons,
           in a list of values
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
           Returns:
            a list of values with output data
        """
        return self.current_data_mean(variables).values.tolist()


class Voltmeter(Multimeter):

    """OutputDevice class to wrap around an output device
      that records continuous time membrane potential data only."""

    # The Voltmeter is just a Mutlimeter measuring only a voltage quantity

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "voltmeter")
        super(Voltmeter, self).__init__(device, *args, **kwargs)

    @property
    @abstractmethod
    def var(self):
        """A method to return the string of the voltage variable's label, e.g, "V_m" for membrane potential in NEST"""
        pass

    def get_data(self, name=None, dims_names=["Time", "Variable", "Neuron"]):
        """This method returns time series' data of the membrane potential recorded by the voltmeter.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        return super(Voltmeter, self).get_data(self.var, name, dims_names)

    def get_mean_data(self, name=None, dims_names=["Time", "Variable"]):
        """This method returns time series' data of the membrane potential recorded by the voltmeter,
           averaged across neurons.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
           Returns:
            a xarray DataArray with the output data
        """
        data = self.get_data()
        return data.mean(dim="Neuron")

    @property
    def data(self):
        """This method returns time series' data of the membrane potential recorded by the voltmeter.
           Returns:
            a xarray DataArray with the output data and dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data()

    @property
    def data_mean(self):
        """This method returns time series' data of the membrane potential recorded by the voltmeter,
           averaged across neurons.
           Returns:
            a xarray DataArray with the output data and dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_mean_data()

    def current_data(self, name=None, dims_names=["Variable", "Neuron"]):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        return super(Voltmeter, self).current_data([self.var], name, dims_names)

    def current_data_mean(self, name=None, dim_name="Variable"):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter,
           averaged across neurons.
           Arguments:
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            a xarray DataArray with the output data
        """
        output = self.current_data(name, dim_name)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, name=None, dim_name="Variable"):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter,
           averaged across neurons, in a list of values.
           Arguments:
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            list of values with the output data
        """
        return self.current_data_mean(name, dim_name).values.tolist()


class SpikeMultimeter(Multimeter, SpikeRecorder):

    """This is a SpikeMultimeter device that records continuous time spike weights' variables,
       i.e., variables that equal zero for every time moment when there is no spike emitted/received, and
       a positive or negative spike floating value, where there is a spike.
    """

    def __init__(self, device, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "spike_multimeter")
        super(SpikeMultimeter, self).__init__(device, *args, **kwargs)

    @property
    def spikes_vars(self):
        return self.record_from

    @property
    def number_of_spikes_var(self):
        return len(self.spikes_vars)

    def get_spikes(self, name=None, dims_names=["Time", "Variable", "Neuron"]):
        """This method returns time series' data of spike weights recorded by the spike multimeter.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        return self.get_data(self.spikes_vars, name, dims_names)

    def get_spikes_inds(self):
        """This method returns all events' indices for which there is a spike, i.e., the spike weight is not zero.
           Returns:
            a numpy array with the output indices
        """
        spikes_inds = []
        for spike_var in self.spikes_vars:
            spikes_inds += np.where(np.array(self.events[spike_var]))[0].tolist()
        return spikes_inds

    def get_spikes_events(self, events_inds=None, **filter_kwargs):
        """This method will select/exclude part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the filtered dictionary of spikes' events
        """
        events = deepcopy(self.events)
        if events_inds:
            spike_var = self.spikes_vars[0]
            if (hasattr(events_inds, "__len__") and len(events_inds) > 0) or isinstance(events_inds, slice):
                # events_inds are numerical or boolean indices, or a slice:
                events_inds = np.arange(len(events[spike_var])).astype("i")[events_inds]
            else:  # events_inds is a scalar to start indexing from:
                events_inds = np.arange(len(events[spike_var])).astype("i")[events_inds:]
            events_inds = np.intersect1d(self.get_spikes_inds(), events_inds)
        else:
            events_inds = self.get_spikes_inds()
        return self.get_events(events, events_inds=events_inds, **filter_kwargs)

    def get_spikes_weights(self, events_inds=None, **filter_kwargs):
        """This method will return spikes' weights
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the spikes' weights in a list
        """
        outputs = []
        for spike_var in self.spikes_vars:
            outputs.append(self.get_spikes_events(events_inds=events_inds,
                                                  variables=spike_var, **filter_kwargs)[spike_var])
        return outputs

    def get_spikes_times(self, events_inds=None, **filter_kwargs):
        """This method will return spikes' times
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the spikes' times in a list
        """
        return self.get_spikes_events(events_inds=events_inds, variables="times", **filter_kwargs)["times"]

    def get_spikes_senders(self, events_inds=None, **filter_kwargs):
        """This method will return spikes' senders
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the spikes' times in a list
        """
        return self.get_spikes_events(events_inds=events_inds, variables="senders", **filter_kwargs)["senders"]

    # The following attributes are time summaries without taking into consideration spike timing:

    def get_total_spikes_activity(self, events_inds=None, **filter_kwargs):
        """This method returns the total spikes' activity by adding spike weights
           and dividing by the total number of neurons,
           after filtering spikes' events based on user inputs.
           Arguments:
             events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                          to slice the event attributes. Default = None, i.e., it doesn't apply.
             filter_kwargs: see filter_events method for its possible keyword arguments
            Returns:
             the spikes' weights in a list
           Returns:
            float: total spikes' activity
        """
        spikes_sum = []
        for spike_var_spikes in self.get_spikes_weights(events_inds=events_inds, **filter_kwargs):
            spikes_sum.append(np.sum(spike_var_spikes))
        if len(spikes_sum) == 1:
            return spikes_sum[0]
        else:
            return spikes_sum

    def get_mean_spikes_activity(self, events_inds=None, **filter_kwargs):
        """This method returns the mean spike activity by adding spike weights
           and dividing by the total number of neurons,
           after filtering spikes' events based on user inputs.
           Arguments:
            events_inds: a scalar to start indexing from, or a sequence of indices or a slice,
                         to slice the event attributes. Default = None, i.e., it doesn't apply.
            filter_kwargs: see filter_events method for its possible keyword arguments
           Returns:
            float: mean spikes' activity
        """
        n_neurons = self.get_number_of_neurons()
        if n_neurons > 0:
            spikes_sum = ensure_list(self.get_total_spikes_activity(events_inds=events_inds, **filter_kwargs))
            for ii in range(len(spikes_sum)):
                spikes_sum[ii] /= n_neurons
            if len(spikes_sum) == 1:
                return spikes_sum[0]
            else:
                return spikes_sum
        else:
            if self.number_of_spikes_var > 1:
                return np.array([0.0]*self.number_of_spikes_var)
            else:
                return 0.0

    @property
    def spikes(self):
        """This method returns time series' data of spike weights recorded by the spike multimeter.
           Returns:
            a xarray DataArray with the output data
        """
        return self.get_spikes()

    @property
    def spikes_times(self):
        """This method will return spikes' times.
            Returns:
             the spikes' times in an array
        """
        return self.get_spikes_times()

    @property
    def spikes_senders(self):
        """This method will return spikes' senders neurons' indices,
            Returns:
             the pikes' senders neurons' indices in an array
        """
        return self.get_spikes_senders()

    @property
    def spikes_weights(self):
        """This method will return spikes' weights.
            Returns:
             the spikes' weights in an array
        """
        return self.get_spikes_weights()

    @property
    def number_of_spikes(self):
        """This method will return the total number of spikes with weight not equal to 0."""
        return np.sum(self.spikes != 0)

    @property
    def mean_number_of_spikes(self):
        """This method will return the total number of spikes,
         divided by the total number of neurons recorded by the device."""
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        """This method will return the mean spike rate (float) as the
           total number of spikes divided by the total number of connected neurons, assuimng a duration of 1 ms
        """
        return self.get_spikes_rate()

    @property
    def total_spikes_activity(self):
        """This method returns the total spike activity by adding spike weights.
           Returns:
            float: total spikes' activity
        """
        return self.get_total_spikes_activity()

    @property
    def mean_spikes_activity(self):
        """This method returns the mean spike activity by adding spike weights
           and dividing by the total number of neurons.
           Returns:
            float: mean spikes' activity
        """
        return self.get_mean_spikes_activity()


OutputDeviceDict = {"spike_recorder": SpikeRecorder,
                    "multimeter": Multimeter,
                    "spike_multimeter": SpikeMultimeter,
                    "voltmeter": Voltmeter}

OutputContinuousTimeDeviceDict = {"multimeter": Multimeter,
                                  "spike_multimeter": SpikeMultimeter,
                                  "voltmeter": Voltmeter}

OutputSpikeDeviceDict = {"spike_recorder": SpikeRecorder,
                         "spike_multimeter": SpikeMultimeter}


class DeviceSet(pd.Series, HasTraits):

    """DeviceSet class is a indexed mapping (based on inheritance from a pandas.Series)
       of a set of Device instances, that correspond to input or output devices referencing
       a quantity to be recorded or stimulated from/to a set of neural populations,
       and distributed across brain regions' nodes.
       Therefore, the DeviceSet is labelled according to the quantity and/or population,
       it measrues/stimulates from/to, and indexed by regions' nodes' integer and/or label index,
       e.g. device_set["rh_insula"]."""

    model = Attr(field_type=str, default="", required=True,
                 label="DeviceSet's model", doc="""Label of DeviceSet's devices' model""")

    _number_of_connections = List(of=int, default=(),
                                  label="Number of connections",
                                  doc="""The number of total connections of the DeviceSet""")

    def __init__(self, label="", model="", device_set=None, **kwargs):
        pd.Series.__init__(self, device_set, name=str(label), **kwargs)
        HasTraits.__init__(self)
        self.model = str(model)
        if np.any([not isinstance(device, Device) for device in self]):
            raise ValueError("Input device_set is not a Series of Device objects!:\n%s" %
                             str(device_set))
        self.update_model()
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    @property
    def label(self):
        """The region node label."""
        return self.name

    def __getitem__(self, items):
        """This method will return a subset of the DeviceSet if the argument is a sequence,
        or a single Device if the argument is an integer indice or a string label."""
        if isinstance(items, string_types) or is_integer(items):
            return super(DeviceSet, self).__getitem__(items)
        return DeviceSet(name=self.name, model=self.model, device_set=super(DeviceSet, self).__getitem__(items))

    def _repr(self):
        return "%s - Name: %s, Model: %s" % \
               (self.__class__.__name__, self.name, self.model)

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self.print_str()

    def print_str(self, connectivity=False):
        output = "\n" + self._repr()
        output += ",\nDevices:"
        for node_index, node in self.iteritems():
            output += LINE + node.print_str(connectivity)
        return output

    def devices(self, input_devices=None):
        """This method returns (a subset of) the DeviceSet devices' labels in a list."""
        if input_devices is None:
            # no input nodes
            return list(self.index)
        else:
            if input_devices in list(self.index) or input_devices in list(range(len(self))):
                # input nodes is a single index or label
                return [input_devices]
            else:
                # input nodes is a sequence of indices or labels
                return list(input_devices)

    def _return_by_type(self, values_dict, return_type="dict", concatenation_index_name="Region", name=None):
        """This method returns data collected from the Devices of the DeviceSet in a desired output format, among
           dict (Default), pandas.Series, xarray.DataArray or a list of values, depending on user input.
           Arguments:
            values_dict: dictionary of attributes and values
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "dict". "values" stands for a list of values, without labelling the output.
            concatenation_index_name: The dimension name along which the concatenation across devices happens.
                                      Default = "Region".
            name: Label of the output data. Default = None.
           Returns:
            output data in the selected type.
        """
        if return_type == "values":
            return list(values_dict.values())
        elif return_type == "dict":
            return values_dict
        elif return_type == "Series":
            return pd.Series(values_dict, name=name)
        elif return_type == "DataArray":
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
        """This method will perform the required action (Device method or property),
           and will return the output data collected from (a subset of) the Devices of the DeviceSet,
           in a desired output format, among dict (Default), pandas.Series, xarray.DataArray or a list of values,
           depending on user input.
           Arguments:
            attr: the name of the method/property of Device requested
            *args: possible position arguments of attr
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
            concatenation_index_name: The dimension name along which the concatenation across devices happens.
                                      Default = "Region".
            name: Label of the output data. Default = None.
            **kwargs: possible keyword arguments of attr
           Returns:
            output data in the selected type.
        """
        values_dict = OrderedDict()
        for device in self.devices(nodes):
            val = getattr(self[device], attr)
            if hasattr(val, "__call__"):
                values_dict.update({device: val(*args, **kwargs)})
            else:
                values_dict.update({device: val})
        return self._return_by_type(values_dict, return_type, concatenation_index_name, name)

    @property
    def number_of_connections(self):
        """This method will return the total number of connections of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connections
        """
        self._number_of_connections = self.do_for_all_devices("number_of_connections")
        if np.sum(self._number_of_connections) == 0:
            self._number_of_connections = 0
        return self._number_of_connections

    @property
    def number_of_neurons(self):
        """This method will return the total number of connections of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connections
        """
        self._number_of_neurons = self.do_for_all_devices("number_of_neurons")
        if np.sum(self._number_of_neurons) == 0:
            self._number_of_neurons = 0
        return self._number_of_neurons

    @property
    def times(self):
        """This method will return the event times recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' times
        """
        return self.do_for_all_devices("times", return_type="values")

    @property
    def time(self):
        """This method will return the time arrays of the events recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' time (numpy) arrays
        """
        return self.do_for_all_devices("time", return_type="values")

    @property
    def senders(self):
        """This method will return the event senders neurons' indices recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' senders neurons' indices
        """
        return self.do_for_all_devices("senders", return_type="values")

    @property
    def weights(self):
        """This method will return the weights of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the weights of the connections
        """
        return self.do_for_all_devices("node_weight", return_type="values")

    @property
    def delays(self):
        """This method will return the delays of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the delays of the connections
        """
        return self.do_for_all_devices("node_delay", return_type="values")

    @property
    def receptors(self):
        """This method will return the receptors of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the receptors of the connections
        """
        return self.do_for_all_devices("node_receptors", return_type="values")

    def record_from_per_node(self, nodes=None, return_type="values"):
        """A method to get the attribute "record_from" from (a subset of) all Devices of the DeviceSet
           This method will lead to an error if the device set does not consist of OutputDevices.
           Arguments:
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
           Returns:
            the output in the selected type
        """
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
        """A method to get the attribute "record_from" from all Devices of the DeviceSet
           This method will lead to an error if the device set does not consist of OutputDevices.
           Returns:
            the outputs of each device in a list
        """
        return self.record_from_per_node()

    def update_model(self):
        """Assert that all Devices of the set are of the same model."""
        if len(self) > 0:
            models = self.do_for_all_devices("model")
            if np.any([model != self.model for model in models]):
                raise ValueError("Not all devices of the DeviceSet are of the same model!:\n %s" % str(models))

    def update(self, device_set=None):
        if device_set:
            super(DeviceSet, self).update(device_set)
        self.update_model()
        self._number_of_connections = self.do_for_all_devices("get_number_of_connections")
        self._number_of_neurons = self.do_for_all_devices("get_number_of_neurons")

    def Get(self, attrs=None, nodes=None, return_type="dict", name=None):
        """A method to get attributes from (a subset of) all Devices of the DevoceSet.
            attr: the name of the method/property of Device requested
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
            name: Label of the output data. Default = None.
        """
        if attrs is None:
            # Get dictionary of all attributes
            values_dict = []
            for node in self.devices(nodes):
                values_dict.append(self[node].Get())
            values_dict = list_of_dicts_to_dict_of_lists(values_dict)
        else:
            values_dict = OrderedDict({})
            for attr in ensure_list(attrs):
                this_attr = []
                for node in self.devices(nodes):
                    this_attr.append(self[node].Get(attr))
                values_dict.update({attr: this_attr})
        return self._return_by_type(values_dict, return_type, name)

    def Set(self, value_dict, nodes=None):
        """A method to set attributes to (a subset of) all Devices of the DeviceSet.
            value_dict: dict of attributes and values to be set
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
        """
        # A method to set attributes to all devices of the set
        # TODO: find a nice way to treat arrays or lists attributes.
        #  Some NEST device attributes have to be arrays and some scalars,
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

        for i_n, node in enumerate(self.devices(nodes)):
            try:
                # Good for spike times and weights of spike generator
                value_dict_i_n = get_scalar_dict1(value_dict, i_n)
                self[node].Set(value_dict_i_n)
            except:
                # Good for amplitude of dc generator and rate of poisson generator
                value_dict_i_n = get_scalar_dict2(value_dict, i_n)
                self[node].Set(value_dict_i_n)
