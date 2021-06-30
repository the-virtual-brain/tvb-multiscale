# -*- coding: utf-8 -*-

import os

import numpy as np
from numpy.lib.recfunctions import rename_fields

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def truncate_ascii_file_after_header(filepath, header_chars="#"):
    n_header_chars = len(header_chars)
    with open(filepath, "r+") as file:
        line = file.readline()
        while len(line) >= n_header_chars and line[:n_header_chars] == header_chars:
            line = file.readline()
        file.seek(file.tell() - len(line), os.SEEK_SET)
        file.truncate()
        file.close()


def read_nest_output_device_data_from_ascii_to_dict(filepath, n_row_events_to_skip=0):
    """This function reads data from a NEST recording device ascii file into an events dictionary
       Arguments:
        - filepath: absolute or relative path to the file (string)
        - n_row_events_to_skip: number of events' rows to skip from reading. Default=0
       Returns:
        the events dictionary of the recorded data
    """
    recarray = rename_fields(np.genfromtxt(filepath, skip_header=3+n_row_events_to_skip,
                                           names=ensure_list(np.loadtxt(filepath,
                                                                        skiprows=2, dtype="str", max_rows=1))),
                             {"sender": "senders", "time_ms": "times"})
    return {name: np.array(ensure_list(recarray[name])) for name in recarray.dtype.names}
