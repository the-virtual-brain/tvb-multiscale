# -*- coding: utf-8 -*-

import sys
import os
import importlib.util

import dill  # , pickle  TODO: decide whether to use one or the other, or make it a configuration choice
from six import string_types

import numpy as np
from numpy.lib.recfunctions import rename_fields

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


import codecs


def remove_bom_inplace(path):
    """Removes BOM mark, if it exists, from a file and rewrites it in-place"""
    buffer_size = 4096
    bom_length = len(codecs.BOM_UTF8)

    with open(path, "r+b") as fp:
        chunk = fp.read(buffer_size)
        if chunk.startswith(codecs.BOM_UTF8):
            i = 0
            chunk = chunk[bom_length:]
            while chunk:
                fp.seek(i)
                fp.write(chunk)
                i += len(chunk)
                fp.seek(bom_length, os.SEEK_CUR)
                chunk = fp.read(buffer_size)
            fp.seek(-bom_length, os.SEEK_CUR)
            fp.truncate()


def truncate_ascii_file_after_header(filepath, header=0):
    """
    This function will truncate an ascii file below its header.
    Arguments:
        filepath: absolute or relative path to the file (string)
        header=0: If it is a string, it is the starting set of characters denoting a header line.
                  Otherwise, it is assume to be an integer signifying the number of header lines.
    """
    with open(filepath, "r+", encoding='ascii') as file:
        if isinstance(header, string_types):
            line = file.readline()
            n_header_chars = len(header)
            # If header is a string, assume that it is the string that denotes a header line
            while len(line) >= n_header_chars and line[:n_header_chars] == header:
                line = file.readline()
            file.seek(file.tell() - len(line), os.SEEK_SET)
        else:
            # Otherwise it is assumed to be an integer, signifying the number of header lines to skip
            for _ in range(int(header)):
                _ = file.readline()
            file.seek(file.tell(), os.SEEK_SET)
        file.truncate()


def read_data_from_ascii_to_dict(filepath, n_lines_to_skip=0, n_header_lines=0,
                                 renaming_labels_dict=None, empty_file=False):
    """This function reads data from an ascii file into an events dictionary using numpy.genfromtxt.
       The last line of the header is assumed to contain the labels of the data in columns.
       Optionally, the dict keys (data labels) can be renamed using numpy.lib.recfunctions.rename_fields
       upon the numpy record array read by numpy.genfromtxt
       Arguments:
        - filepath: absolute or relative path to the file (string).
        - n_lines_to_skip=0: number of events' rows to skip from reading.
        - n_header_lines=0: number of header lines to skip in addition to n_lines_to_skip.
        - renaming_labels_dict=None: if it is a dict, it maps existing labels to their new names to be reassigned.
        - empty_file=False: flag to delete file contents below the header after reading.
       Returns:
        the dictionary of the recorded data
    """
    recarray = \
        np.genfromtxt(filepath, skip_header=n_header_lines+n_lines_to_skip, encoding='ascii',
                      names=ensure_list(np.loadtxt(filepath, skiprows=n_header_lines-1, dtype="str", max_rows=1)))
    if isinstance(renaming_labels_dict, dict):
        recarray = rename_fields(recarray, renaming_labels_dict)
    events = {name: np.array(ensure_list(recarray[name])) for name in recarray.dtype.names}
    if empty_file:
        truncate_ascii_file_after_header(filepath, header=n_header_lines)
    return events


def read_nest_output_device_data_from_ascii_to_dict(filepath, n_lines_to_skip=0, empty_file=False):
    """This function reads data from a NEST recording device ascii file into an events dictionary.
       The following renaming happens to match the labels of the events when recording in "memory" in NEST:
       {"sender": "senders", "time_ms": "times"} (keys (old label) -> values (new label))
       Arguments:
        - filepath: absolute or relative path to the file (string).
        - n_lines_to_skip=0: number of events' rows to skip from reading.
        - empty_file=False: flag to delete file contents below the header after reading.
       Returns:
        the events dictionary of the recorded data
    """
    # TODO: Find a faster way to read and truncate the file without the buggy \x00 character...
    events = {}
    keys = []
    filedata = ""
    with open(filepath, "r+", encoding='ascii') as file:
        for iL, line in enumerate(file):
            if empty_file and iL < 3:
                filedata += line
            if iL == 2:
                keys = line.replace("\n", "").replace('sender', 'senders').replace('time_ms', 'times').split("\t")
                events = dict(zip(keys, [list() for _ in range(len(keys))]))
            elif iL > 2 + n_lines_to_skip:
                for key, val in zip(keys,
                                    list([float(d) for d in line.replace('\x00', '').replace("\n", "").split("\t")])):
                    events[key] += [val]
    for key, val in events.items():
        events[key] = np.array(val)
    if np.any(np.isnan(events["senders"])):
        print(events)
        raise
    if empty_file:
        with open(filepath, "w", encoding='ascii') as file:
            file.write(filedata)
        # truncate_ascii_file_after_header(filepath, header=3)
    return events
    # return read_data_from_ascii_to_dict(filepath, n_lines_to_skip=n_lines_to_skip, n_header_lines=3,
    #                                     renaming_labels_dict={"sender": "senders", "time_ms": "times"},
    #                                     empty_file=empty_file)


def dump_pickled_dict(d, filepath):
    filepath = filepath.split('.pkl')[0] + ".pkl"
    with open(filepath, "wb") as f:
        dill.dump(d, f)


def load_pickled_dict(filepath):
    filepath = filepath.split('.pkl')[0] + ".pkl"
    with open(filepath, "rb") as f:
        d = dill.load(f)
    return d


def load_module_from_file(filepath, module_name="module.name"):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo
