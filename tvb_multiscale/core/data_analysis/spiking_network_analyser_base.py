# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

import numpy as np
from pandas import Series
from xarray import DataArray

from tvb.basic.neotraits.api import Attr, Float
from tvb.datatypes import connectivity

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, concatenate_heterogeneous_DataArrays, sort_events_by_x_and_y

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.data_structures_utils import get_caller_fun_name


LOG = initialize_logger(__name__)


def _get_safely_list_item(i_item, lst, default_output):
    try:
        output = lst[i_item]
    except:
        try:
            output = lst[0]
        except:
            output = default_output
    return output


class SpikingNetworkAnalyserBase(HasTraits):

    """SpikingNetworkAnalyserBase
       - gets data from the output devices of a SpikingNetwork,
       (TODO: - or reads them from files,)
       - performs computations of mean field quantities, including spikes_train' rates,
       - and returns the results in the desired output type
         (numpy.array, xarray.DataArray, TVB TimeSeries, pandas.Series of xarray.DataArray)
       """

    spikeNet = None
    # spikeNet = Attr(field_type=SpikingNetwork,
    #                 label="Spiking Network",
    #                 default=None,
    #                 required=False,
    #                 doc="""An instance of a SpikingNetwork class""")

    start_time = Float(
        label="Start time",
        default=None,
        required=False,
        doc="""Starting time of data (ms)""")

    end_time = Float(
        label="End time",
        default=None,
        required=False,
        doc="""Ending time of data (ms)""")

    period = Float(
        label="Sampling period",
        default=1.0,
        required=True,
        doc="""Sampling period (ms)""")

    transient = Float(
        label="Transient time",
        default=0.0,
        required=True,
        doc="""Transient time to be excluded from computations, 
               but to not be cut out from data returned (ms). Default = 0.0.
               In case that start_time is set by the first spike time, 
               the start_time of computations is set as the maximum(start_time, transient).
               In all other cases, the start_time of computations is given as start_time + transient.""")

    time_series_output_type = Attr(field_type=str, default="array", required=True,
                                   label="Output type option for time series results.",
                                   doc="""The output type of the results, which can be either 'array' (Default), 
                                          in which case a DataArray or a Series of DataArrays 
                                          for unmerged heterogeneous results is returned, 
                                          or "TVB", in which case a TVB TimeSeries instance is returned.""")

    force_homogeneous_results = Attr(field_type=bool, default=True, required=True,
                                     label="Flag to merge heteregoneous results.",
                                     doc="""If force_homogeneous_results is True, 
                                            DataArrays from different regions and or populations " \
                                           "are merged to a single DataArray. It defaults to True and "
                                           "it is also forced to True, 
                                           if the time_series_output_type = 'TVB' TimeSeries.""")

    connectivity = Attr(
        field_type=connectivity.Connectivity,
        label="Long-range connectivity",
        default=None,
        required=False,
        doc="""A tvb.datatypes.Connectivity object which contains the
         structural long-range connectivity data (i.e., white-matter tracts). In
         combination with the ``Long-range coupling function`` it defines the inter-regional
         connections. These couplings undergo a time delay via signal propagation
         with a propagation speed of ``Conduction Speed``.
         \nIt is necessary if the time_series_output_type="TVB" for TimeSeriesRegion class instance.""")

    flatten_neurons_inds = Attr(field_type=bool, default=True, required=False,
                                label="Flag to flatten neurons indices",
                                doc="""If flatten_neurons_inds is True, "
                                      "we substitute the global neurons' indices by their local rank""")

    return_data = Attr(field_type=bool, default=False, required=True,
                       label="Flag to return data for plotting.",
                       doc="""If return_data is True, 
                              the data used for the computation of the measures are also returned.""")

    _fmax = np.finfo(dtype="f").max
    _fmin_resolution = np.finfo(dtype="f").resolution

    def __init__(self, spikeNet=None, **kwargs):
        if spikeNet:
            self.spikeNet = spikeNet
        super(SpikingNetworkAnalyserBase, self).__init__(**kwargs)

    def from_instance(self, instance, **kwargs):
        for attr in instance._own_declarative_attrs:
            setattr(self, attr, getattr(instance, attr))
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        self.configure()
        return self

    def _get_method_name(self, method=None, caller_id=2):
        """A method to return the name of an input method.
           Argument:
            - method: Default=None, in which case the inspect module is used to identify the caller method
            - caller_id: Default=2, the id of the caller method in the stack.
           Returns:
            - the name of the method (string)
        """
        if method:
            return method.__name__
        else:
            return get_caller_fun_name(caller_id)

    def _get_data_name(self, method=None):
        """A method to return the name of the data loaded by a "get_..." named method
           that returns some kind of data from the SpikingNetwork devices or written files thereof.
           Argument:
            - method: a get_..." named method of this class,
                      Default=None, in which case the _get_method_name() is used
           Returns:
            - the name of the data type
        """
        return self._get_method_name(method, caller_id=3).split("get_")[-1].split("_from")[0]

    def get_spikes_from_device(self, device, **kwargs):
        """Method to load a dictionary of spike' events data
           from a spikes' recording SpikingNetwork device
           into a pandas.Series.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "spikes"
              "spikes": a pandas.Series with keys "times" and "senders" of the spikes,
              "number_of_neurons": the number of the neurons the device records from
        """
        data_name = self._get_data_name()
        return {data_name: Series(device.get_spikes_events(**kwargs)),
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_spikes_times_from_device(self, device, **kwargs):
        """Method to load an array of spike' times data from a spikes' recording SpikingNetwork device.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "spikes"
              "spikes": a numpy.array of spikes' times,
              "number_of_neurons": the number of the neurons the device records from
        """
        data_name = self._get_data_name()
        return {data_name: np.array(np.sort(device.get_spikes_times(**kwargs))),
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_spikes_times_by_neuron_from_device(self, device, **kwargs):
        """Method to load a dictionary of spike' times data, organized by sender neuron,
           from a spikes' recording SpikingNetwork device
           into a pandas.Series.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method,
                        out of which the flag "full_senders" (True by default) will return empty arrays
                        for the neurons the device records from, which have no spikes sent.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "spikes_times_by_neuron"
              "spikes_times_by_neuron": a pandas.Series
                                        with keys the sender neurons indices and values the corresponding spikes' times,
                                        (If the flag flatten_neurons_inds = True,
                                        the indices are just integer counts of the spikes the device records from.
                                        If the flag flatten_neurons_inds = False,
                                        the indices correspond to the neurons' global gids.)
              "number_of_neurons": the number of the neurons the device records from
        """
        kwargs["full_senders"] = True
        spikes_times_by_neuron = device.get_spikes_times_by_neurons(**kwargs)
        number_of_neurons = len(spikes_times_by_neuron)
        if self.flatten_neurons_inds:
            spikes_times_by_neuron = OrderedDict(zip(range(number_of_neurons), spikes_times_by_neuron.values()))
        data_name = self._get_data_name()
        return {data_name: Series(spikes_times_by_neuron),
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_data_by_neuron_from_device(self, device, **kwargs):
        """Method to load a dictionary of continuous time variables' data, organized by sender neuron,
           from a recording SpikingNetwork device
           into a xarray.DataArray.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "data_by_neuron"
              "data_by_neuron": a xarray.DataArray of dimensions (Time, Variable, Neuron),
                                (If the flag flatten_neurons_inds = True,
                                 the indices are just integer counts of the spikes the device records from.
                                 If the flag flatten_neurons_inds = False,
                                 the indices correspond to the neurons' global gids.
                                 The flag can be overwritten by a kwarg.)
              "number_of_neurons": the number of the neurons the device records from
        """
        this_kwargs = deepcopy(kwargs)
        this_kwargs["flatten_neurons_inds"] = this_kwargs.get("flatten_neurons_inds", self.flatten_neurons_inds)
        data = device.get_data(**kwargs)
        data_name = self._get_data_name()
        return {data_name: data,
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_mean_data_from_device(self, device, **kwargs):
        """Method to load a dictionary of population mean field activity of
           continuous time variables' data,
           from a recording SpikingNetwork device
           into a xarray.DataArray.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "mean_data"
              "mean_data": a xarray.DataArray of dimensions (Variable,),
              "number_of_neurons": the number of the neurons the device records from
        """
        data_name = self._get_data_name()
        data = device.get_mean_data(**kwargs)
        return {data_name: data,
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_total_data_from_device(self, device,**kwargs):
        """Method to load a dictionary of population total (i.e., neurons' sum) activity of
           continuous time variables' data,
           from a recording SpikingNetwork device
           into a xarray.DataArray.
           Arguments:
            - device: the Device instance
            - **kwargs: other keyword arguments for the device.get_spikes_events() method.
           Returns:
            - a dictionary of the following key-value pairs:
              "data_name": "total_data"
              "total_data": a xarray.DataArray of dimensions (Variable,),
              "number_of_neurons": the number of the neurons the device records from
        """
        data_name = self._get_data_name()
        data = device.get_total_data(**kwargs)
        return {data_name: data,
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def _assert_input_start_end_times_from_spikes_times(self, spikes, start_time, end_time):
        """Method to set the starting and ending time of a computation,
           given the input start_time and end_time arguments and the input spikes' times, if any.
           Arguments:
            - spikes: a Spike Train or a numpy array of spikes' times
            - start_time: float (ms)
            - end_time: float (ms)
           Returns:
            - start_time: float (ms)
            - end_time: float (ms)
        """
        if start_time is None:
            # If start_time is not set:
            if len(spikes):
                # ...and there are spikes, set start_time equal to the first spike in time:
                start_time = float(np.min(spikes))
            else:
                # ...otherwise, set start_time to 0.0
                start_time = 0.0
            # Modify start_time to be the maximum of start_time and the transient attribute of the class.
            start_time = float(np.maximum(start_time, self.transient))
        else:
            # Add to the start time the transient to be removed from any computations.
            start_time = self.start_time + self.transient
        if end_time is None:
            # If end_time is not set:
            if len(spikes):
                # ...and there are spikes, set end_time equal to the last spike in time:
                end_time = float(np.max(spikes))
            else:
                # ...otheriwse, set end_time equal to start_time plus the sampling period attribute of the class.
                end_time = start_time + self.period
        # Make sure that the end_time is larger than the start_time by at least one sampling period time.
        if end_time < start_time + self.period:
            end_time = start_time + self.period
        return start_time, end_time

    def _assert_start_end_times_from_spikes_times(self, spikes):
        """Method to set the starting and ending time of a computation,
           given the start_time and end_time attributes and the input spikes' times, if any.
           Arguments:
            - spikes: a Spike Train or a numpy array of spikes' times
           Returns:
            - start_time: float (ms)
            - end_time: float (ms)
        """
        # Initialize from the class instance attributes
        start_time = self.start_time
        end_time = self.end_time
        return self._assert_input_start_end_times_from_spikes_times(spikes, start_time, end_time)

    def _assert_start_end_times_from_data_time(self, time):
        """Method to set the starting and ending time of a computation,
           given the time vector of continuous time variable's data.
           Arguments:
            - time: a numpy array of the time vector
           Returns:
            - start_time: float (ms)
            - end_time: float (ms)
        """
        # Initialize from the class instance attributes
        start_time = self.start_time
        end_time = self.end_time
        if start_time is None:
            # If start_time is not set:
            if len(time):
                # ...and the time vector is not empty, i.e., there are some data,
                # set the start_time equal to the first time point
                start_time = float(np.min(time))
            else:
                # ...otheriwse, set start_time to 0.0
                start_time = 0.0
        # Add to the start time the transient to be removed from any computations.
        start_time = start_time + self.transient
        if end_time is None:
            if len(time):
                # ...and the time vector is not empty, i.e., there are some data,
                # set the end_time equal to the last time point
                end_time = float(np.max(time))
            else:
                # ...otheriwse, set end_time equal to start_time plus the sampling period attribute of the class.
                end_time = start_time + self.period
        # Make sure that the end_time is larger than the start_time by at least one sampling period time.
        if end_time < start_time + self.period:
            end_time = start_time + self.period
        return start_time, end_time

    def _get_spikes_times_from_spikes_events(self, spikes):
        """Method to safely get a numpy.array of spikes
           from a spikes input that might be a dict or pandas.Series.
           Arguments:
            - spikes: a sequence (array, list, tuple) of spikes' times or
                      a dict or pandas.Series with a key "times" mapping to values spikes' times.
        """
        if isinstance(spikes, (dict, Series)):
            return np.array(spikes["times"])
        else:
            return np.array(spikes)

    def _get_comput_res_type(self, method=None):
        """A method to return the name of the result computed by a "compute_..." named method of this class.
           Argument:
            - method: a "compute_..." named method of this class,
                      Default=None, in which case the _get_method_name method is used
           Returns:
            - the name of the result
        """
        return self._get_method_name(method, caller_id=3).split("compute_")[-1]

    def _compute_rate_base(self, spikes_times, res_type, number_of_neurons=1, duration=None, **kwargs):
        if not duration:
            start_time, end_time = self._assert_start_end_times_from_spikes_times(spikes_times)
            duration = end_time - start_time
            if duration < 0.0:
                raise ValueError("start_time (=%g) cannot be smaller than end_time (=%g)!" % (start_time, end_time))
            elif duration == 0.0:
                duration = 1.0
        return {res_type: DataArray(
                            np.sum(np.logical_and(spikes_times >= start_time, spikes_times <= end_time)).astype("f")
                            / duration * 1000).squeeze()}

    def compute_rate(self, spikes, number_of_neurons=1, duration=None, **kwargs):
        """A method to compute rate from an input of spikes' events or spikes' times as:
           number_of_spikes/total_duration * 1000 in Hz.
           Arguments:
            - spikes: a Spike Train or
                      an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            - duration: Default=None, in which case it is computed by start_time and end_time
            - kwargs
           Returns:
            - a dictionary of the following key-value pair:
             "rate": xarray.DataArray(rate).squeeze()
        """
        res_type = self._get_comput_res_type()
        spikes_times = np.array(self._get_spikes_times_from_spikes_events(spikes))
        return self._compute_rate_base(spikes_times, res_type, number_of_neurons, duration, **kwargs)

    def compute_mean_rate(self, spikes, number_of_neurons=1, duration=None, **kwargs):
        """A method to compute mean rate from an input of spikes' events or spikes' times as:
           number_of_spikes/total_duration/number_o_neurons * 1000 in Hz.
           Arguments:
            - spikes: a Spike Train or
                      an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            - duration: Default=None, in which case it is computed by start_time and end_time
            - kwargs
           Returns:
            - a dictionary of the following key-value pair:
             "mean_rate": xarray.DataArray(rate).squeeze()
        """
        res_type = self._get_comput_res_type()
        res2_type = self._get_comput_res_type(self.compute_rate)
        results = self.compute_rate(spikes, duration=duration, **kwargs)
        results[res_type] = results[res2_type] / number_of_neurons
        del results[res2_type]
        return results

    def _compute_delta_rate(self, time, spikes_times, t_start, t_stop):
        """A method to compute instantaneous spiking rate when the kernel method fails
           because there are not enough distinct bins, as when there are not enough spikes,
           and/or all the spikes are concentrated in only one time bin.
           The spikes contribute an equal weight of 1 / sampling period,
           to the corresponding sampling period time bin.
           Arguments:
            - time: the array of the time vector
            - spikes: an array of spikes' times
            Returns:
             - an array of instantaneous rate time series
        """
        result = np.zeros(time.shape)
        for spike_time in np.unique(spikes_times):
            if spike_time >= t_start and spike_time <= t_stop:
                result[int(np.floor((spike_time - time[0]) / self.period))] += \
                    np.sum(spikes_times == spike_time) / self.period
        return result

    def compute_spikes_rates_by_neuron(self, spikes, number_of_neurons=1, rate_method=None, **kwargs):
        """A method to compute any type of spiking rate, but separately for each neuron.
           Arguments:
            - spikes: a Spike Train instance or
                      an array of spikes' times
                      or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            -rate_method: Default=None, in which case the instantaneous rate time series for each neuron are computed
            **kwargs: keyword arguments for the method that computes the rate
           Returns:
            - a dictionary of the following key-value pair:
             result_name: a xarray.DataArray of the result of dimensions (Neuron, ) or (Time, Neuron)
        """
        if rate_method is None:
            rate_method = self.compute_rate
        res_type = self._get_comput_res_type(rate_method) + "_by_neuron"
        if "times" in spikes.keys():
            spikes_times_by_neuron = Series(sort_events_by_x_and_y(spikes, x="senders", y="times"))
        else:
            spikes_times_by_neuron = spikes
        neurons = list(spikes_times_by_neuron.keys())
        spikes_times_by_neuron = list(spikes_times_by_neuron.values)
        rates = OrderedDict()
        if len(neurons) < number_of_neurons:
            neurons = np.arange(number_of_neurons)
        for i_neuron, neuron in enumerate(neurons):
            if len(spikes_times_by_neuron):
                spikes_times = spikes_times_by_neuron.pop(0)
            else:
                spikes_times = []
            rates[neuron] = \
                    list(rate_method(spikes_times, number_of_neurons=number_of_neurons, **kwargs).values())[0]
        rates = concatenate_heterogeneous_DataArrays(rates, "Neuron", fill_value=0.0)
        return {res_type: rates}

    def _get_data_time(self, data):
        """A method to get the time vector of time series DataArray,
           assuming tha Time is the first dimension,
           and time points are included in the coordinates.
           Arguments:
            - data: the input time series DataArray
           Returns:
            - a numpy array of the time vector
        """
        if data.size:
            return np.array(data.coords.get(data.dims[0], []))
        return np.array([])

    def _apply_transient_to_data(self, data):
        """A method to remove the transient from a time series DataArray.
           Arguments:
            - data: the input time series DataArray
           Returns:
            - a the output time series DataArray with the transient removed
        """
        if self.transient and data.size:
            time = self._get_data_time(data)
            if time.size:
                t_start = self._assert_start_end_times_from_data_time(time)[0]
                if t_start > time[0]:
                    return data.loc[t_start:]
        return data

    def compute_activity(self, data, **kwargs):
        """A method to compute the activity of continuous time variables
           by taking the mean across the Time dimension.
           Arguments:
            - data: the input time series DataArray
            - kwargs
           Returns:
            - a dictionary of key-value pair:
             "activity": output time series DataArray of dimensions (Variables, Neuron)
        """
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        return {self._get_comput_res_type(): data}

    def compute_mean_field(self, data, **kwargs):
        """A method to compute the mean field activity of continuous time variables
           by taking the mean across both the Time and Neuron dimensions.
           Arguments:
            - data: the input time series DataArray
            - kwargs
           Returns:
            - a dictionary of key-value pair:
             "mean_field": output DataArray of dimensions (Variables, )
        """
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        if data.ndim > 1:
            data = data.mean(dim="Neuron")
        return {self._get_comput_res_type(): data}

    def compute_total_activity(self, data, **kwargs):
        """A method to compute the total activity of continuous time variables
           by taking the mean across the Time dimension and summing across the Neuron one.
           Arguments:
            - data: the input time series DataArray
            - kwargs
           Returns:
            - a dictionary of key-value pair:
             "total_activity": output DataArray of dimensions (Variables, )
        """
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        if data.ndim > 1:
            data = data.sum(dim="Neuron")
        return {self._get_comput_res_type(): data}

    def compute_activity_time_series(self, data, **kwargs):
        """A method to compute the activity time series of continuous time variables
           by taking the mean across the Time dimension.
           Arguments:
            - data: the input time series DataArray
            - kwargs
           Returns:
            - a dictionary of key-value pair:
             "activity_time_series": output time series DataArray of dimensions (Time, Variables, Neuron)
        """
        data = self._apply_transient_to_data(DataArray(data))
        return {self._get_comput_res_type(): data}

    def compute_mean_field_time_series(self, data, **kwargs):
        """A method to compute the mean field time series of continuous time variables
           by taking the mean across the Neuron dimension.
           Arguments:
            - data: the input time series DataArray
            - kwargs
           Returns:
            - a dictionary of key-value pair:
             "mean_field_time_series": output time series DataArray of dimensions (Time, Variables)
        """
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Neuron")
        return {self._get_comput_res_type(): data}

    def compute_total_activity_time_series(self, data, **kwargs):
        """A method to compute the total activity time series of continuous time variables
            by summing across the Neuron dimension.
            Arguments:
             - data: the input time series DataArray
             - kwargs
            Returns:
             - a dictionary of key-value pair:
              "mean_field_time_series": output time series DataArray of dimensions (Time, Variables)
         """
        data = self._apply_transient_to_data(DataArray(data)).sum(dim="Neuron")
        return {self._get_comput_res_type(): data}
