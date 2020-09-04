# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
from scipy.stats import describe
from pandas import unique

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, flatten_list, is_integer, extract_integer_intervals


def flatten_neurons_inds_in_DataArray(data_array, neurons_dim_label="Neuron"):
    dims = list(data_array.dims)
    try:
        dim_id = dims.index(neurons_dim_label)
    except:
        dim_id = -1
    neurons_dim_label = dims[dim_id]
    neuron_labels = np.arange(data_array.shape[dim_id])
    data_array.coords[neurons_dim_label] = neuron_labels
    return data_array


def filter_neurons(neurons, exclude_neurons=[]):
    """This method will select/exclude the connected neurons, depending on user inputs
       Arguments:
        neurons: collection (list, tuple, array) of neurons which should be included in the output.
                 Default = None, corresponds to all neurons the device is connected to.
        exclude_neurons: collection (list, tuple, array) of neurons
                         which should be excluded from the output. Default = [].
       Returns:
        numpy.array of neurons.
    """
    # Method to select or exclude some of the connected neurons to the device:
    temp_neurons = ensure_list(neurons)
    for neuron in exclude_neurons:
        if neuron in temp_neurons:
            temp_neurons.remove(neuron)
    return tuple(temp_neurons)


def filter_events(events, variables=None, neurons=None, times=None,
                  exclude_neurons=[], exclude_times=[]):
    """This method will select/exclude part of the measured events, depending on user inputs
        Arguments:
            events: dictionary of events
            variables: collection (list, tuple, array) of variables to be included in the output,
                       assumed to correspond to keys of the events dict.
                       Default=None, corresponds to all keys of events.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                     Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                             the events of which should be excluded from the output. Default = [].
        Returns:
              the filtered dictionary (of arrays per attribute) of events
    """

    def in_fun(x, values):
        # Function to return a boolean about whether a value is
        # within a collection or an interval (len(values) == 2) of values:
        in_flag = False
        if len(values) > 0:
            in_flag = x in values
            if in_flag:
                return in_flag
            if len(values) == 2:
                if values[0] is not None:
                    in_flag = x > values[0]
                    if in_flag:
                        if values[1] is not None:
                            in_flag = x < values[1]
                    return in_flag
                if values[1] is not None:
                    return x < values[1]
        return in_flag

    # The variables to return:
    if variables is None:
        variables = events.keys()

    # The events:
    output_events = OrderedDict()

    events_times = np.array(events["times"])
    senders = np.array(events["senders"])

    # As long as there are events:
    n_events = len(events["times"])
    if n_events > 0:
        inds = np.ones((n_events,))
        if times is not None:  # If we select times...
            if neurons is not None:  # ...and neurons:
                inds = np.logical_and(inds, [in_fun(time, flatten_list(times)) and
                                             (not in_fun(time, flatten_list(exclude_times))) and
                                             in_fun(sender, flatten_list(neurons)) and
                                             (not in_fun(sender, flatten_list(exclude_neurons)))
                                             for time, sender in zip(events_times, senders)])
            else:  # ...or all neurons
                inds = np.logical_and(inds, [in_fun(time, flatten_list(times)) and
                                             (not in_fun(time, flatten_list(exclude_times))) and
                                             (not in_fun(sender, flatten_list(exclude_neurons)))
                                             for time, sender in zip(events_times, senders)])
        else:  # ...or all times...
            if neurons is not None:  # ...and we select neurons...:
                inds = np.logical_and(inds, [(not in_fun(time, flatten_list(exclude_times))) and
                                             in_fun(sender, flatten_list(neurons)) and
                                             (not in_fun(sender, flatten_list(exclude_neurons)))
                                             for time, sender in zip(events_times, senders)])
            else:  # ...or all neurons as well...:
                inds = np.logical_and(inds, [(not in_fun(time, flatten_list(exclude_times))) and
                                             (not in_fun(sender, flatten_list(exclude_neurons)))
                                             for time, sender in zip(events_times, senders)])
        for var in ensure_list(variables):
            output_events[var] = events[var][inds]
    else:
        for var in ensure_list(variables):
            output_events[var] = np.array([])
    return output_events


def summarize(results, decimals=None):

    def unique_fun(vals):
        return unique(np.around(vals, decimals=decimals))

    def stats_fun(vals):
        d = describe(vals)
        summary = {}
        summary["n"] = d.nobs
        summary["mean"] = d.mean
        summary["minmax"] = d.minmax
        summary["var"] = d.variance
        return summary

    if is_integer(decimals):
        fun = unique_fun
    else:
        fun = stats_fun

    output = {}
    for attr, val in results.items():
        vals = ensure_list(val)
        if len(vals) > 3:
            try:
                if str(np.array(vals).dtype)[0] == "i":
                    output[attr] = extract_integer_intervals(vals)
                else:
                    output[attr] = fun(vals)
            except:
                try:
                    # Try boolean
                    unique_vals = list(unique(vals))
                    if len(unique_vals) < 2:
                        # If they are all True or all False
                        output[attr] = unique_vals
                    else:
                        output[attr] = {"True": extract_integer_intervals(np.where(vals)[0]),
                                        "False": extract_integer_intervals(np.where(np.invert(vals))[0])}
                except:
                    try:
                        # treat the rest as strings
                        output[attr] = list(unique([str(v) for v in vals]).tolist())
                    except:
                        output[attr] = list(vals)
        else:
            output[attr] = vals

    return output