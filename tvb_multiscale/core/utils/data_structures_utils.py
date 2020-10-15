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


def filter_events(events, variables=None, times=None, exclude_times=[]):
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
              the filtered dictionary (of arrays per attribute) of events
    """

    def in_fun(values):
        # Function to return a boolean about whether a value is
        # within a sequence or an interval (len(values) == 2) of values:
        if len(values) == 2:
            if values[0] is not None:
                if values[1] is not None:
                    return lambda x: x >= values[0] and x <= values[1]
                else:
                    return lambda x: x >= values[0]
            elif values[1] is not None:
                return lambda x: x <= values[0]
            else:
                return lambda x: x
        else:
            return lambda x: x in values

    # The variables to return:
    if variables is None:
        variables = events.keys()

    # The events:
    output_events = OrderedDict()

    events_times = np.array(events["times"])

    n_events = len(events["times"])
    if n_events > 0:
        # As long as there are events:
        # If we (un)select times...
        if times is not None and len(times) > 0:
            in_times = in_fun(flatten_list(times))
        else:
            in_times = lambda x: True
        if exclude_times is not None and len(exclude_times) > 0:
            not_in_exclude_times = lambda x: not in_fun(flatten_list(exclude_times))(x)
        else:
            not_in_exclude_times = lambda x: True
        inds = np.logical_and(np.ones((n_events,)),
                              [in_times(time) and not_in_exclude_times(time)
                               for time in events_times])
        for var in ensure_list(variables):
            output_events[var] = events[var][inds]
    else:
        for var in ensure_list(variables):
            output_events[var] = []
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
                val_type = np.array(vals).dtype
                if str(val_type)[0] == "i":
                    output[attr] = extract_integer_intervals(vals)
                else:
                    output[attr] = fun(vals)
                    if isinstance(output[attr], np.ndarray):
                        output[attr] = output[attr].astype(val_type)
            except:
                try:
                    val_type = np.array(vals).dtype
                    # Try boolean
                    unique_vals = list(unique(vals).astype(val_type))
                    if len(unique_vals) < 2:
                        # If they are all True or all False
                        output[attr] = unique_vals
                    else:
                        output[attr] = {"True": extract_integer_intervals(np.where(val)[0]),
                                        "False": extract_integer_intervals(np.where(np.invert(vals))[0])}
                except:
                    try:
                        val_type = np.array(vals).dtype
                        # treat the rest as strings
                        output[attr] = list(unique([str(v) for v in vals]).astype(val_type).tolist())
                    except:
                        output[attr] = list(vals)
        else:
            output[attr] = vals

    return output