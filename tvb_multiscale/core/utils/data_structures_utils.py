# -*- coding: utf-8 -*-

from inspect import stack
from itertools import product
from collections import OrderedDict
from six import string_types
from enum import Enum
import typing
from copy import deepcopy
import warnings

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


def repeat_nested_lists_to_size(data, size):
    """
    Conditionally expands input data to a list of lists of a target size.

    Logic:
    1. If input is a flat list, it is wrapped and repeated 'size' times.
    2. If input is a single-item list of lists, that item is repeated 'size' times.
    3. If input is already a list of lists of length > 1:
        - If len(data) == size, it is returned as is (or throws error per your rule).
        - If len(data) != size, it raises a ValueError.

    Args:
        data (list): Source data (list or list of lists).
        size (int): The target length for the outer list.

    Returns:
        list[list]: A nested list where len(result) == size.

    Raises:
        ValueError: If the input is already a multi-row list of lists
                    that does not match 'size'.
    """
    # Check if the input is a list of lists
    is_nested = all(isinstance(i, list) for i in data) if data else False
    current_len = len(data)

    # CASE 1: The input is already a list of lists
    if is_nested:
        if current_len == 1:
            # Single inner list: repeat it to reach 'size'
            return [list(data[0]) for _ in range(size)]

        elif current_len > 1:
            # Already expanded: Throw error as requested
            if current_len == size:
                raise ValueError(f"Input is already a list of lists of size {size}.")
            else:
                raise ValueError(f"Input size {current_len} does not match target {size}.")

    # CASE 2: The input is a flat list (real numbers)
    # We wrap it and repeat it to reach 'size'
    return [list(data) for _ in range(size)]

# # --- Test Scenarios ---
#
# # 1. Flat list -> Repeats to size
# print(f"Flat: {repeat_nested_lists_to_size([1.1, 2.2], 3)}")
# # Output: [[1.1, 2.2], [1.1, 2.2], [1.1, 2.2]]
#
# # 2. Single nested list -> Repeats to size
# print(f"Single Nested: {repeat_nested_lists_to_size([[5.5, 6.6]], 2)}")
# # Output: [[5.5, 6.6], [5.5, 6.6]]
#
# # 3. Already correct size -> Throws Error
# try:
#     repeat_nested_lists_to_size([[1, 2], [3, 4]], 2)
# except ValueError as e:
#     print(f"Error (Correct Size): {e}")
#
# # 4. Incorrect multi-row size -> Throws Error
# try:
#     repeat_nested_lists_to_size([[1, 2], [3, 4]], 5)
# except ValueError as e:
#     print(f"Error (Wrong Size): {e}")


def repeat_to_shape_robust(arr, target_shape):
    """
    Enforces a specific total shape, allowing broadcasting only if the
    input's first dimension is 1. Otherwise, dimensions must match.
    """
    if not isinstance(target_shape, tuple):
        raise TypeError(f"target_shape must be a tuple, got {type(target_shape)}")

    arr = np.asanyarray(arr)
    if arr.size == 0:
        raise ValueError("Input array 'arr' cannot be empty.")

    # 1. Standardize to at least 2D (N, 1)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim < 1:
        arr = arr.reshape(1, 1)

    curr_dim0 = arr.shape[0]
    target_dim0 = target_shape[0]

    # 2. Strict First Dimension Enforcement
    if curr_dim0 != target_dim0 and curr_dim0 != 1:
        raise ValueError(
            f"Incompatible first dimension: input has {curr_dim0}, "
            f"but target requires {target_dim0}."
        )

    # 3. Extract the 1D "spine" from the first column
    # This ensures we have exactly 'curr_dim0' elements.
    spine = arr[:, 0]

    # 4. Reshape spine to (curr_dim0, 1, 1, ...)
    # The number of trailing 1s matches the length of the target trailing dims.
    spine_shape = (curr_dim0,) + (1,) * (len(target_shape) - 1)
    base_view = spine.reshape(spine_shape)

    # 5. Broadcast to the full target shape
    return np.broadcast_to(base_view, target_shape)
#
# def run_tests():
#     # TEST 1: 1D Input matching target first dimension
#     # Input: (30,) -> Target: (30, 2, 2)
#     # Result: (30, 2, 2)
#     d1 = np.arange(30)
#     res1 = repeat_to_shape_robust(d1, (30, 2, 2))
#     print(f"Test 1 (Match): Input {d1.shape} -> Output {res1.shape}")
#
#     # TEST 2: 1D Input broadcasting to a larger first dimension
#     # Input: (1,) -> Target: (10, 3)
#     # Result: (10, 3)
#     d2 = np.array([99])
#     res2 = repeat_to_shape_robust(d2, (10, 3))
#     print(f"Test 2 (Broadcast 1 to N): Input {d2.shape} -> Output {res2.shape}")
#
#     # TEST 3: Multi-D input matching target
#     # Input: (5, 100) -> Target: (5, 2, 2, 2)
#     # Result: (5, 2, 2, 2)
#     d3 = np.zeros((5, 100))
#     res3 = repeat_to_shape_robust(d3, (5, 2, 2, 2))
#     print(f"Test 3 (Multi-D Match): Input {d3.shape} -> Output {res3.shape}")
#
#     # TEST 4: Error - First dimension mismatch (non-1)
#     # Input: (2,) -> Target: (5, 1)
#     # Result: ValueError
#     try:
#         print("Test 4 (Mismatch): ", end="")
#         repeat_to_shape_robust([1, 2], (5, 1))
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
#
#     # TEST 5: Error - Target shape is not a tuple
#     try:
#         print("Test 5 (Type Check): ", end="")
#         repeat_to_shape_robust([1], 5)
#     except TypeError as e:
#         print(f"Caught expected error: {e}")
#
# run_tests()


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


# TODO: Revert to tvb_contrib's version once the pull request is accepted:
def sort_events_by_x_and_y(events, x="senders", y="times",
                           filter_x=None, filter_y=None, exclude_x=[], exclude_y=[], hashfun=str):
    xs = np.array(flatten_list(events[x]))
    if filter_x is None:
        xlabels = np.unique(xs, axis=0).tolist()
    else:
        xlabels = np.unique(flatten_list(filter_x), axis=0).tolist()
    for xlbl in exclude_x:
        try:
            xlabels.remove(xlbl)
        except:
            pass
    ys = flatten_list(events[y])
    if filter_y is not None:
        ys = [yy for yy in ys if yy in flatten_list(filter_y)]
    for yy in exclude_y:
        try:
            ys.remove(yy)
        except:
            pass
    ys = np.array(ys)
    keys = []
    for xlbl in xlabels:
        if not isinstance(xlbl, Hashable):
            keys.append(hashfun(xlbl))
        else:
            keys.append(xlbl)
    if len(ys):
        sorted_events = OrderedDict()
        for key, xlbl in zip(keys, xlabels):
            # NEW (Compatible with NumPy 1.x and 2.0+)
            sorted_events[key] = np.sort(ys[(xs == xlbl).all(axis=-1)])
    else:
        sorted_events = OrderedDict(zip(keys, [np.array([])] * len(keys)))
    return sorted_events


# TODO: Revert to tvb_contrib's version once the pull request is accepted:
def concatenate_heterogeneous_DataArrays(data, concat_dim_name,
                                         data_keys=None, name=None, fill_value=np.nan, transpose_dims=None):
    from pandas import Series
    from xarray import concat
    from pandas import Index

    if isinstance(data, (dict, Series)):
        if data_keys is None:
            data_keys = ensure_list(data.keys())
        if isinstance(data, dict):  # dict
            data = ensure_list(data.values())
        else:  # pd.Series
            if name is None:
                name = data.name
            data = ensure_list(data.values)
    # Idiomatic xarray approach: build a dict of new coords and use assign_coords
    cleaned_data = []
    for da in data:
        updated_coords = {}
        for c_name, coord in da.coords.items():
            if "string" in str(coord.dtype).lower():
                updated_coords[c_name] = coord.values.astype(object)
        # assign_coords returns a new DataArray, leaving the original untouched
        if updated_coords:
            da = da.assign_coords(updated_coords)

        cleaned_data.append(da)
    # Pass the newly mapped list of DataArrays to concat
    data = concat(cleaned_data, Index(data_keys, name=concat_dim_name), fill_value=fill_value, join='outer')
    data.name = name
    if transpose_dims:
        data = data.transpose(*transpose_dims)
    return data


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


def safe_dict_copy(din):
    # Used to avoid errors of some NEST related classes such as pynestkernel.SLIDatum
    dout = dict()
    for key, val in din.items():
        try:
            dout[key] = deepcopy(din[key])
        except Exception as e:
            dout[key] = din[key]
            warnings.warn("Dictionary element\n%s\nof type\n%s\ncould not be deepcopied!\n"
                          "See error below: \n%s\n"
                          "Setting the same object to the copied dictionary!" %
                          (str(din[key]), str(type(din[key])), str(e)))

    return dout


def safe_deepcopy(obj):
    if isinstance(obj, dict):
        return safe_dict_copy(obj)
    else:
        try:
            out = deepcopy(obj)
        except Exception as e:
            out = obj
            warnings.warn("Object\n%s\nof type\n%s\ncould not be deepcopied!\n"
                          "See error below:\n%s\n"
                          "Setting the same object to the output!" %
                          (str(obj), str(type(obj)), str(e)))
        return out


def property_to_fun(property):
    if hasattr(property, "__call__"):
        return property
    else:
        return lambda *args, **kwargs: safe_deepcopy(property)
