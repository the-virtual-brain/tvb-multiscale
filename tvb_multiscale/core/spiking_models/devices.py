# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import OrderedDict

import pandas as pd
import xarray as xr
import numpy as np

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.utils.data_structures_utils import filter_events
from tvb_multiscale.core.spiking_models.node import SpikingNodeCollection
from tvb_multiscale.core.spiking_models.node_set import SpikingNodesSet

from tvb.basic.neotraits.api import Attr, Int, List

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, list_of_dicts_to_dict_of_lists, sort_events_by_x_and_y


LOG = initialize_logger(__name__)


class Device(SpikingNodeCollection):
    __metaclass__ = ABCMeta

    """Device class to wrap around an 
       output (recording/measuring/monitoring) or input (stimulating) device.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
    """

    device = None  # a device object, depending on its simulator implementation

    label = Attr(field_type=str, default="", required=True,
                 label="Device label", doc="""Label of Device""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the Device resides or connects to""")
    
    _number_of_connections = None
    _number_of_neurons = None

    _weight_attr = ""
    _delay_attr = ""
    _receptor_attr = ""

    def __init__(self, device=None, **kwargs):
        self.device = device    # a device object, depending on its simulator implementation
        SpikingNodeCollection.__init__(self, device, **kwargs)

    def __getstate__(self):
        d = super(Device, self).__getstate__()
        d["device"] = self.device
        return d

    def __setstate__(self, d):
        super(Device, self).__setstate__(d)
        self.device = d.get("device", None)

    @abstractmethod
    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        return self._assert_node

    # Methods to get or set attributes for devices and/or their connections:

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
        if self.neurons is not None:
            return len(ensure_list(self.neurons))
        else:
            return 0

    @abstractmethod
    def get_neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        pass
    
    @property
    @abstractmethod
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        return self.get_neurons()

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
    def number_of_connected_neurons(self):
        return self.number_of_neurons

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

    def info_neurons(self):
        return {"connected_nodes_gids": np.array(self.neurons)}


class InputDevice(Device):

    """InputDevice class to wrap around an input (stimulating) device"""

    def GetConnections(self, **kwargs):
        """Method to get connections of the device to neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(self.device, source_or_target="source")


InputDeviceDict = {}


class OutputDevice(Device):

    """OutputDevice class to wrap around an output (recording/measuring/monitoring) device"""

    def GetConnections(self, **kwars):
        """Method to get connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self._GetConnections(self.device, source_or_target="target")

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
            if filter_kwargs.pop("new", False):
                events = self.get_new_events()
            else:
                events = self.events.copy()
        if variables is None:
            variables = events.keys()
        else:
            variables = ensure_list(variables)
        n_events = len(events["times"])
        if n_events > 0:
            if events_inds is not None:
                if hasattr(events_inds, "__len__") or isinstance(events_inds, slice):
                    # events_inds are numerical or boolean indices, or a slice:
                    select_fun = lambda x, events_inds: np.array(x)[events_inds]
                else: # events_inds is a scalar to start indexing from:
                    select_fun = lambda x, events_inds: np.array(x)[events_inds:]
                for var in variables:
                    events[var] = select_fun(events[var], events_inds)
            if len(filter_kwargs) > 0:
                return filter_events(events, **filter_kwargs)
        else:
            for var in variables:
                events[var] = np.array([])
        return events

    @abstractmethod
    def get_new_events(self, variables=None, **filter_kwargs):
        pass

    @property
    @abstractmethod
    def events(self):
        """This method returns the dictionary of events"""
        pass

    @property
    def new_events(self):
        return self.get_new_events()

    @property
    @abstractmethod
    def number_of_events(self):
        """This method returns the number (integer) of events"""
        pass

    @property
    @abstractmethod
    def number_of_new_events(self):
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

    def __init__(self, device=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_recorder")
        OutputDevice.__init__(self, device, **kwargs)

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
        n_neurons = self.number_of_neurons
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

    def get_spikes_times_by_neurons(self, events_inds=None, times=None, exclude_times=[], full_senders=False,
                                    **filter_kwargs):
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
              filter_kwargs: see filter_events method for its possible keyword arguments
             Returns:
              dictionary of spike events sorted by sender neuron
         """
        if full_senders:
            filter_x = self.neurons
        else:
            filter_x = None
        return sort_events_by_x_and_y(self.get_events(events_inds=events_inds, **filter_kwargs),
                                      x="senders", y="times", filter_x=filter_x,
                                      filter_y=times, exclude_y=exclude_times, hashfun=tuple)

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
                                      x="times", y="senders", filter_x=times, exclude_x=exclude_times, hashfun=str)

    @property
    def spikes_times(self):
        """This method will return spikes' times in an array"""
        return self.times

    @property
    def spikes_senders(self):
        """This method will return spikes' sender neurons in an array"""
        return self.senders

    @property
    def new_spikes_times(self):
        """This method will return newly recorded spikes' times in an array"""
        return self.get_spikes_times(new=True)

    @property
    def new_spikes_senders(self):
        """This method will return newly recorded spikes' sender neurons in an array"""
        return self.get_spikes_senders(new=True)

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

    def __init__(self, device=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "multimeter")
        OutputDevice.__init__(self, device, **kwargs)

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
    def get_data(self, variables=None, name=None, dims_names=["Time", "Variable", "Neuron"],
                 flatten_neurons_inds=True, new=False):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
            flatten_neurons_inds: if true, neurons coordinates are arranged from 1 to number of neurons,
                                  instead for neurons_inds
            new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
           Returns:
            a xarray DataArray with the output data
        """
        pass

    def get_new_data(self, **kwargs):
        kwargs.pop("new", None)
        return self.get_data(new=True, **kwargs)

    def get_mean_data(self, variables=None, name=None, dims_names=["Time", "Variable"], new=False):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None, which defaults to the label of the Device
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
            new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], dims_names[1], "Neuron"]
        data = self.get_data(variables, name, dims_names, new=new)
        return data.mean(dim="Neuron")

    def get_total_data(self, variables=None, name=None, dims_names=["Time", "Variable"], new=False):
        """This method returns time series' data recorded by the multimeter, summed across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
           flatten_neurons_inds: if true, neurons coordinates are arranged from 1 to number of neurons,
                                  instead for neurons_inds
           new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], dims_names[1], "Neuron"]
        data = self.get_data(variables, name, dims_names, new=new)
        return data.sum(dim="Neuron")

    @property
    def data(self):
        """This method returns time series' data recorded by the multimeter.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data()

    @property
    def new_data(self):
        """This method returns time series' data newly recorded by the multimeter.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data(new=True)

    @property
    def data_mean(self):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data()

    @property
    def new_data_total(self):
        """This method returns time series' data newly recorded by the multimeter,
           summed across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data(new=True)

    @property
    def new_data_mean(self):
        """This method returns time series' data newly recorded by the multimeter,
           averaged across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data(new=True)

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
           in a an array of values
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
           Returns:
            a numpy.array of values with output data
        """
        return self.current_data_mean(variables).values


class Voltmeter(Multimeter):

    """OutputDevice class to wrap around an output device
      that records continuous time membrane potential data only."""

    # The Voltmeter is just a Multimeter measuring only a voltage quantity

    def __init__(self, device=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "voltmeter")
        OutputDevice.__init__(self, device, **kwargs)

    @property
    @abstractmethod
    def var(self):
        """A method to return the string of the voltage variable's label, e.g, "V_m" for membrane potential in NEST"""
        pass

    def get_data(self, name=None, dims_names=["Time", "Variable", "Neuron"], new=False):
        """This method returns time series' data of the membrane potential recorded by the voltmeter.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
            new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
           Returns:
            a xarray DataArray with the output data
        """
        return super(Voltmeter, self).get_data(self.var, name, dims_names, new=new)

    def get_mean_data(self, name=None, dims_names=["Time", "Variable"], new=False):
        """This method returns time series' data of the membrane potential recorded by the voltmeter,
           averaged across neurons.
           Arguments:
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable"]
            new: boolean flag. Default = False. If True, we return data only from newly recorded events
                 (e.g., events recorded after the last call to get_data)
           Returns:
            a xarray DataArray with the output data
        """
        data = self.get_data()
        return data.mean(dim="Neuron")

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
           averaged across neurons, in a numpy.array of values.
           Arguments:
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            numpy.array of values with the output data
        """
        return self.current_data_mean(name, dim_name).values


class SpikeMultimeter(Multimeter, SpikeRecorder):

    """This is a SpikeMultimeter device that records continuous time spike weights' variables,
       i.e., variables that equal zero for every time moment when there is no spike emitted/received, and
       a positive or negative spike floating value, where there is a spike.
    """

    def __init__(self, device=None, **kwargs):
        kwargs["model"] = kwargs.get("model", "spike_multimeter")
        Multimeter.__init__(self, device, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)

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
            events_inds = intersect1d(self.get_spikes_inds(), events_inds)
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
             the spikes' times in a numpy.array
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
             the spikes' times in a numpy.array
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
             the spikes' weights in a numpy.array
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
        n_neurons = self.number_of_neurons
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


class DeviceSet(SpikingNodesSet):

    """DeviceSet class is a indexed mapping (based on inheritance from a pandas.Series)
       of a set of Device instances, that correspond to input or output devices referencing
       a quantity to be recorded or stimulated from/to a set of neural populations,
       and distributed across brain regions' nodes.
       Therefore, the DeviceSet is labelled according to the quantity and/or population,
       it measures/stimulates from/to, and indexed by regions' nodes' integer and/or label index,
       e.g. device_set["rh_insula"]."""

    model = Attr(field_type=str, default="", required=True,
                 label="DeviceSet's model", doc="""Label of DeviceSet's devices' model""")

    _number_of_connections = []

    _number_of_neurons = []

    _collection_name = "Device"

    def __init__(self, device_set=pd.Series(dtype='object'), **kwargs):
        self.model = str(kwargs.pop("model", ""))
        SpikingNodesSet.__init__(self, device_set, **kwargs)
        if np.any([not isinstance(device, Device) for device in self]):
            raise ValueError("Input device_set is not a Series of Device objects!:\n%s" %
                             str(device_set))
        self.update_model()
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def __getstate__(self):
        d = super(DeviceSet, self).__getstate__()
        d["model"] = self.model
        d["_collection_name"] = self._collection_name
        return d

    def __setstate__(self, d):
        super(DeviceSet, self).__setstate__(d)
        self.model = d.get("model", self.model)

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

    @property
    def number_of_connections(self):
        """This method will return the total number of connections of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connections
        """
        if len(self._number_of_connections) == 0:
            self._number_of_connections = ensure_list(self.do_for_all("number_of_connections"))
        return self._number_of_connections

    @property
    def number_of_neurons(self):
        """This method will return the total number of connected neurons of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connected neurons
        """
        if len(self._number_of_neurons) == 0:
            self._number_of_neurons = ensure_list(self.do_for_all("number_of_neurons"))
        return self._number_of_neurons

    @property
    def number_of_connected_neurons(self):
        """This method will return the total number of connected neurons of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connected neurons
        """
        return self.number_of_neurons

    @property
    def times(self):
        """This method will return the event times recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' times
        """
        return self.do_for_all("times", return_type="values")

    @property
    def time(self):
        """This method will return the time arrays of the events recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' time (numpy) arrays
        """
        return self.do_for_all("time", return_type="values")

    @property
    def senders(self):
        """This method will return the event senders neurons' indices recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' senders neurons' indices
        """
        return self.do_for_all("senders", return_type="values")

    @property
    def weights(self):
        """This method will return the weights of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the weights of the connections
        """
        return self.do_for_all("node_weight", return_type="values")

    @property
    def delays(self):
        """This method will return the delays of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the delays of the connections
        """
        return self.do_for_all("node_delay", return_type="values")

    @property
    def receptors(self):
        """This method will return the receptors of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the receptors of the connections
        """
        return self.do_for_all("node_receptors", return_type="values")

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
        values_dict = self.do_for_all("record_from", nodes=nodes, return_type="dict")
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
            models = ensure_list(self.do_for_all("model"))
            if np.any([model != models[0] for model in models]):
                raise ValueError("Not all devices of the DeviceSet %s are of the same model!:\n %s"
                                 % (self.name, str(models)))
            self.model = models[0]

    def update(self, device_set=None):
        if device_set:
            super(DeviceSet, self).update(device_set)
        self.update_model()
        self.configure()

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
                    value = self[node].Get(attr)
                    if isinstance(value, dict):
                        value = value[attr]
                    this_attr.append(value)
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
                # Good for spike times and weights of spike generator,
                # as well as rates of inhomogeneous_poisson_generator
                value_dict_i_n = get_scalar_dict1(value_dict, i_n)
                self[node].Set(value_dict_i_n)
            except:
                # Good for amplitude of dc generator and rate of poisson generator
                value_dict_i_n = get_scalar_dict2(value_dict, i_n)
                self[node].Set(value_dict_i_n)


class DeviceSets(SpikingNodesSet):

    """DeviceSets is an indexed mapping (based on inheriting from pandas.Series class)
       between DeviceSet instances' labels and those instances.
    """

    _collection_name = "DeviceSet"
