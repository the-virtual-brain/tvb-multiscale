# coding=utf-8

# Data structure manipulations and conversions
from six import string_types
import re
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from tvb_scripts.utils.log_error_utils import warning, raise_value_error, raise_import_error, initialize_logger
from tvb_scripts.config import CalculusConfig


logger = initialize_logger(__name__)


def is_numeric(value):
    return isinstance(value, (float, np.float, np.float64, np.float32, np.float16, np.float128,
                              int, np.int, np.int0, np.int8, np.int16, np.int32, np.int64,
                              complex, np.complex, np.complex64, np.complex128, np.complex256,
                              np.long, np.number))


def is_integer(value):
    return isinstance(value, (int, np.int, np.int0, np.int8, np.int16, np.int32, np.int64))


def is_float(value):
    return isinstance(value, (float, np.float, np.float64, np.float32, np.float16, np.float128))


def isequal_string(a, b, case_sensitive=False):
    if case_sensitive:
        return a == b
    else:
        try:
            return a.lower() == b.lower()
        except AttributeError:
            logger.warning("Case sensitive comparison!")
            return a == b


def sort_dict(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def dicts_of_lists(dictionary, n=1):
    for key, value in dictionary.items():
        dictionary[key] = ensure_list(dictionary[key])
        if len(dictionary[key]) == 1 and n > 1:
            dictionary[key] = dictionary[key] * n
    return dictionary


def iterable_to_dict(obj):
    d = OrderedDict()
    for ind, value in enumerate(obj):
        d["%02d" % ind] = value
    return d


def dict_to_list_or_tuple(dictionary, output_obj="list"):
    dictionary = sort_dict(dictionary)
    output = dictionary.values()
    if output_obj == "tuple":
        output = tuple(output)
    return output


def list_of_dicts_to_dict_of_tuples(lst):
    return dict(zip(lst[0], zip(*list([d.values() for d in lst]))))


def list_of_dicts_to_dict_of_lists(lst):
    d = list_of_dicts_to_dict_of_tuples(lst)
    for key, val in d.items():
        d[key] = list(val)
    return d


def list_of_dicts_to_dicts_of_ndarrays(lst, shape=None):
    d = list_of_dicts_to_dict_of_tuples(lst)
    if isinstance(shape, tuple):
        for key, val in d.items():
            d[key] = np.reshape(np.stack(d[key]), shape)
    else:
        for key, val in d.items():
            d[key] = np.stack(d[key])
            for sh in d[key].shape[:0:-1]:
                if sh == 1:
                    d[key] = np.squeeze(d[key], axis=-1)
                else:
                    break
    return d


def arrays_of_dicts_to_dicts_of_ndarrays(arr):
    lst = arr.flatten().tolist()
    d = list_of_dicts_to_dicts_of_ndarrays(lst)
    for key, val in d.items():
        d[key] = np.reshape(d[key], arr.shape)
    return d


def dicts_of_lists_to_lists_of_dicts(dictionary):
    return [dict(zip(dictionary, t)) for t in zip(*dictionary.values())]


def ensure_list(arg):
    if not (isinstance(arg, list)):
        try:  # if iterable
            if isinstance(arg, (string_types, dict)):
                arg = [arg]
            elif hasattr(arg, "__iter__"):
                arg = list(arg)
            else: # if not iterable
                arg = [arg]
        except:  # if not iterable
            arg = [arg]
    return arg


def flatten_list(lin, sort=False):
    lout = []
    for sublist in lin:
        if isinstance(sublist, (list, tuple)):
            temp = flatten_list(list(sublist))
        else:
            temp = [sublist]
        for item in temp:
            lout.append(item)
    if sort:
        lout.sort()
    return lout


def flatten_tuple(t, sort=False):
    return tuple(flatten_list(list(t), sort))


def labels_to_inds(labels, target_labels):
    if isinstance(target_labels, string_types):
        return_single_element = True
        target_labels = ensure_list(target_labels)
    else:
        target_labels = list(target_labels)
        return_single_element = False
    inds = []
    for lbl in target_labels:
        inds.append(labels.index(lbl))
    if return_single_element:
        # if there was only one label string input
        return inds[0]
    else:
        return inds


def monopolar_to_bipolar(labels, indices=None, data=None):
    if indices is None:
        indices = range(len(labels))
    bipolar_lbls = []
    bipolar_inds = [[], []]
    for ind in range(len(indices) - 1):
        iS1 = indices[ind]
        iS2 = indices[ind + 1]
        if (labels[iS1][0] == labels[iS2][0]) and \
                int(re.findall(r'\d+', labels[iS1])[0]) == \
                int(re.findall(r'\d+', labels[iS2])[0]) - 1:
            bipolar_lbls.append(labels[iS1] + "-" + labels[iS2])
            bipolar_inds[0].append(iS1)
            bipolar_inds[1].append(iS2)
    if isinstance(data, np.ndarray):
        data = data[bipolar_inds[0]] - data[bipolar_inds[1]]
        return bipolar_lbls, bipolar_inds, data
    else:
        return bipolar_lbls, bipolar_inds


def sort_events_by_x_and_y(events, x="senders", y="times",
                           filter_x=None, filter_y=None, exclude_x=[], exclude_y=[]):
    xs = np.array(flatten_list(events[x]))
    if filter_x is None:
        xlabels = np.unique(xs).tolist()
    else:
        xlabels = np.unique(flatten_list(filter_x)).tolist()
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
    sorted_events = OrderedDict()
    for xlbl in xlabels:
        sorted_events[xlbl] = np.sort(ys[xs == xlbl])
    return sorted_events


def data_xarray_from_continuous_events(events, times, senders, variables=[],
                                       filter_senders=None, exclude_senders=[], name=None,
                                       dims_names=["Variable", "Neuron", "Time"]):
    unique_times = np.unique(times).tolist()
    if filter_senders is None:
        filter_senders = np.unique(senders).tolist()
    else:
        filter_senders = np.unique(flatten_list(filter_senders)).tolist()
    for sender in exclude_senders:
        filter_senders.remove(sender)
    if len(variables) is None:
        variables = list(events.keys())
    coords = OrderedDict()
    coords[dims_names[0]] = variables
    coords[dims_names[1]] = filter_senders
    coords[dims_names[2]] = unique_times
    n_senders = len(filter_senders)
    n_times = len(unique_times)
    data = np.empty((len(variables), n_senders, n_times))
    last_time = times[0]
    i_time = unique_times.index(last_time)
    i_sender = -1
    for id, (time, sender) in enumerate(zip(times, senders)):
        # Try best guess of next sender:
        i_sender += 1
        if i_sender >= n_senders:
            i_sender = 0
        if filter_senders[i_sender] != sender:
            try:
                i_sender = filter_senders.index(sender)
            except:
                break  # This sender is not one of the chosen filter_senders
        if time != last_time:
            last_time = time
            # Try best guess of next time index:
            i_time += 1
            if i_time >= n_times:
                i_time = n_times - 1
            if time != unique_times[i_time]:
                i_time = unique_times.index(time)
        for i_var, var in enumerate(variables):
            data[i_var, i_sender, i_time] = events[var][id]
    try:
        from xarray import DataArray
        return DataArray(data, dims=list(coords.keys()), coords=coords, name=name)
    except:
        return {"data": data, "dims": list(coords.keys()), "coords": coords, "name": name}


def property_to_fun(property):
    if hasattr(property, "__call__"):
        return property
    else:
        return lambda *args, **kwargs: property

