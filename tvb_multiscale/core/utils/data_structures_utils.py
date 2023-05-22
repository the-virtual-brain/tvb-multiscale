# -*- coding: utf-8 -*-

from inspect import stack
from itertools import product
from collections import OrderedDict
from six import string_types
from enum import Enum
import typing
from copy import deepcopy

import numpy as np
from scipy.stats import describe
import pandas as pd
from xarray import DataArray

from tvb.basic.neotraits.api import HasTraits


from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, flatten_list, is_integer, extract_integer_intervals


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False


def get_caller_fun_name(caller_id=1):
    return str(stack()[caller_id][3])


def get_ordered_dimensions(dims, dims_order):
    out_dims = []
    dims = ensure_list(dims)
    for dim in dims_order:
        if dim in dims:
            out_dims.append(dim)
            dims.remove(dim)
    return out_dims + dims


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
            output_events[var] = np.array([])
    return output_events


def cross_dimensions_and_coordinates_MultiIndex(dims, pop_labels, all_regions_lbls):
    from pandas import MultiIndex
    stacked_dims = "-".join(dims)
    names = []
    new_dims = []
    for d in ["i", "j"]:
        names.append([dim + "_" + d for dim in dims])
        new_dims.append(stacked_dims + "_" + d)
    new_coords = {new_dims[0]: MultiIndex.from_product([pop_labels, all_regions_lbls], names=names[0]),
                  new_dims[1]: MultiIndex.from_product([pop_labels, all_regions_lbls], names=names[1])}
    return new_dims, new_coords


def combine_DataArray_dims(arr, dims_combinations, join_string=", ", return_array=True):
    new_dims = []
    new_coords = {}
    stacked_dims = {}
    for dim_combin in dims_combinations:
        new_dim = join_string.join(["%s" % arr.dims[i_dim] for i_dim in dim_combin])
        new_dims.append(new_dim)
        stacked_dims[new_dim] =[arr.dims[i_dim] for i_dim in dim_combin]
        new_coords[new_dim] = [join_string.join(coord_combin)
                               for coord_combin in product(*[arr.coords[arr.dims[i_dim]].data for i_dim in dim_combin])]
    if return_array:
        return DataArray(arr.stack(**stacked_dims).data, dims=new_dims, coords=new_coords, name=arr.name)
    else:
        return arr.stack(**stacked_dims).data, new_dims, new_coords


def get_enum_names(en):
    return [val.name for val in en.__members__.values()]


def get_enum_values(en):
    return [val.value for val in en.__members__.values()]


def combine_enums(enum_name, *args):
    d = OrderedDict()
    for enm in args:
        for name, member in enm.__members__.items():
            d[name] = member.value
    return Enum(enum_name, d)


def summarize_value(value, digits=3):

    def unique(values, astype=None):
        values = np.array(values)
        if values.size:
            if astype is None:
                astype = str(values.dtype)
            try:
                unique_vals = pd.unique(values).astype(astype)
            except:
                unique_vals = np.unique(values).astype(astype)
            if unique_vals.ndim < values.ndim:
                return unique_vals[np.newaxis, ...]
            else:
                return unique_vals
        return values

    def unique_dicts(list_of_dicts):
        return [dict(t) for t in {tuple(d.items()) for d in list_of_dicts}]

    def unique_floats_fun(vals):
        scale = 10 ** np.floor(np.log10(np.percentile(np.abs(vals), 95)))
        return scale * unique(np.around(vals / scale, decimals=digits))

    def stats_fun(vals):
        d = describe(vals)
        summary = OrderedDict()
        # summary["n"] = d.nobs
        summary["min"] = d.minmax[0]
        summary["median"] = np.median(vals)
        summary["max"] = d.minmax[1]
        summary["mean"] = d.mean
        summary["var"] = d.variance
        return summary

    vals = ensure_list(value)
    n_vals = len(vals)
    try:
        val_type = str(np.array(vals).dtype)
        if np.all([isinstance(val, dict) for val in vals]):
            # If they are all dicts:
            return np.array(vals)
        else:
            unique_vals = unique(vals, val_type)
            if unique_vals.ndim > 1 and np.prod(unique_vals.shape[1:]) > 5:
                return summarize_value(np.array(vals).flatten(), digits)
            n_unique_vals = unique_vals.shape[0]
            if n_unique_vals < 2:
                # If they are all of the same value, just set this value:
                return unique_vals[0]
            else:
                val_type = str(unique_vals.dtype)
                if val_type[0] == 'f' and n_unique_vals > 5:
                    # If unique_vals is a vector of more than 5 values...
                    try:
                        if is_integer(digits):
                            # ...either try to further reduce it by approximation...
                            temp_unique_vals = unique_floats_fun(unique_vals)
                            n_temp_unique_vals = temp_unique_vals.shape[0]
                            if n_temp_unique_vals.shape[0] == 1:
                                return temp_unique_vals[0]
                        else:
                            temp_unique_vals = unique_vals
                            n_temp_unique_vals = n_unique_vals
                        if n_temp_unique_vals > 5:
                            # ...or compute summary statistics
                            return stats_fun(np.array(vals))
                        else:
                            unique_vals = temp_unique_vals
                            n_unique_vals = n_temp_unique_vals
                    except:
                        pass
            if n_unique_vals < n_vals and n_unique_vals <= 5:
                # If it is not a vector of floats, or there are (now) less than 5 values,
                # return a summary dictionary with the indices of each value:
                output = OrderedDict()
                vals = np.array(vals)
                if val_type[0] == 'f':
                    indices = OrderedDict()
                    for unique_val in unique_vals:
                        indices[unique_val] = []
                    for iV, val in enumerate(vals):
                        indices[unique_vals[np.argmin(np.abs(val - unique_vals))]].append(iV)
                    for unique_val, val_indices in indices.items():
                        intervals = extract_integer_intervals(val_indices, print=True)
                        if len(intervals) <= 50:
                            output["=%s" % str(unique_val)] = "{%s}" % intervals
                        else:
                            output["=%s" % str(unique_val)] = "%d total instances" % len(val_indices)
                else:
                    for unique_val in unique_vals:
                        indices = np.where(vals == unique_val)[0]
                        intervals = extract_integer_intervals(indices, print=True)
                        if len(intervals) <= 48:
                            output["=%s" % str(unique_val)] = "{%s}" % intervals
                        else:
                            output["=%s" % str(unique_val)] = "%d total instances" % len(indices)
                return output
            else:
                # If there are still more than 5 non-float values
                if val_type[0] == "i":
                    # Summarize them if they are integer:
                    return extract_integer_intervals(vals)
                else:
                    # Or just return them...
                    return unique_vals
    except Exception as e:
        # Something went wrong, return the original property
        print(e)
        raise
        # return np.array(vals)


def summarize(results, digits=None):
    outputs = {}
    for attr, val in results.items():
        output = summarize_value(val, digits)
        if output is None:
            outputs[attr] = val
        else:
            outputs[attr] = output
    return outputs


def summary_value_to_string_dict(summary, arname):
    string_dict = OrderedDict()
    string_dict[arname] = "-" * 20
    if isinstance(summary, dict):
        for ikey, ival in summary.items():
            key = str(ikey)
            if isinstance(ival, string_types):
                string_dict[key] = ival
            else:
                try:
                    string_dict[key] = '{:g}'.format(ival)
                except:
                    string_dict[key] = str(ival)
        # key = "[%s]" % ", ".join(key)
        # val = "[%s]" % ", ".join(val)
        # return {key: val}
    else:
        try:
            string_dict["\tunique values"] = '{:g}'.format(summary)
        except:
            string_dict["\tunique values"] = str(summary)
    return string_dict


def narray_summary_info(ar, ar_name='', omit_shape=False):
    # type: (np.ndarray, str, bool) -> typing.Dict[str, Any]
    """
    A 2 column table represented as a dict of str->str
    """
    if ar is None:
        return {'is None': 'True'}

    ret = {}
    if not omit_shape:
        ret.update({'shape': ar.shape, 'dtype': ar.dtype})

    if ar.size == 0:
        ret['is empty'] = 'True'
        return ret

    if ar.dtype.kind in 'iufc':
        has_nan = np.isnan(ar).any()
        if has_nan:
            ret['has NaN'] = 'True'

    summary = summarize_value(ar, digits=3)
    if isinstance(summary, dict):
        ret.update(summary)
    else:
        if np.array(summary, dtype='object').size < ar.size:
            ret.update({'unique_values': summary})
        else:
            ret.update({'values': summary})

    if ar_name:
        return {ar_name + ': ' + str(k): v for k, v in ret.items()}
    else:
        return ret


def narray_describe(ar, arname='', omit_shape=False):
    # type: (numpy.ndarray) -> str
    return summary_value_to_string_dict(narray_summary_info(ar, arname, omit_shape), arname)


def trait_object_str(class_name, summary):
    result = ['{} ('.format(class_name)]
    maxlenk = max(len(k) for k in summary)
    for k in summary:
        result.append('  {:.<{}} {}'.format(k + ' ', maxlenk, summary[k]))
    result.append(')')
    return '\n'.join(result)


def trait_object_repr_html(class_name, summary):
    result = [
        '<table>',
        '<h3>{}</h3>'.format(class_name),
        '<thead><tr><th></th><th style="text-align:left;width:40%">value</th></tr></thead>',
        '<tbody>',
    ]
    for k in summary:
        row_fmt = '<tr><td>{}</td><td style="text-align:left;"><pre>{}</pre></td>'
        result.append(row_fmt.format(k, summary[k]))
    result += ['</tbody></table>']
    return '\n'.join(result)


def summary_info(info, to_string=False):
    """
    A more structured __str__
    A 2 column table represented as a dict of str->str
    The default __str__ and html representations of this object are derived from
    this table.
    Override this method and return such a table filled with instance information
    that informs the user about your instance
    """
    if to_string:
        array_fun = narray_describe
    else:
        array_fun = narray_summary_info
    ret = OrderedDict()
    for aname, attr in dict(info).items():
        try:
            if isinstance(attr, np.ndarray):
                ret.update(array_fun(attr, aname))
            elif isinstance(attr, HasTraits):
                ret[aname] = attr.title
            elif isinstance(attr, (list, tuple)):
                ret.update(array_fun(np.array(attr), aname))
            elif isinstance(attr, dict):
                new_dict = OrderedDict()
                for key, val in attr.items():
                    new_dict["%s[%s]" % (aname, key)] = val
                ret.update(summary_info(new_dict, to_string))
            else:
                ret[aname] = repr(attr)
        except Exception as e:
            print("Failed to summarize: ")
            print(aname)
            print(attr.__class__.__name__)
            print(Warning(e))
    return ret


def property_to_fun(property):
    if hasattr(property, "__call__"):
        return property
    else:
        return lambda *args, **kwargs: deepcopy(property)
