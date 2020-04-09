# encoding=utf8

import numpy as np
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, warning

from tvb_scripts.io.r_file_io import rdump


def merge_csv_data(*csvs):
    data_ = {}
    for csv in csvs:
        for key, val in csv.items():
            if key in data_:
                data_[key] = np.concatenate(
                    (data_[key], val),
                    axis=0
                )
            else:
                data_[key] = val
    return data_


def parse_csv(fname, merge=True):
    if '*' in fname:
        import glob
        return parse_csv(glob.glob(fname), merge=merge)
    if isinstance(fname, (list, tuple)):
        csv = []
        for csv_dict in [parse_csv(_) for _ in fname]:
            if len(csv_dict) > 0:
                csv.append(csv_dict)
        if merge:
            csv = merge_csv_data(*csv)
        return csv

    try:
        lines = []
        with open(fname, 'r') as fd:
            for line in fd.readlines():
                if not line.startswith('#'):
                    lines.append(line.strip().split(','))
        names = [field.split('.') for field in lines[0]]
        data = []
        for id_line, line in enumerate(lines[1:]):
            append_data = True
            for iline in range(len(line)):
                try:
                    line[iline] = float(line[iline])
                except:
                    logger = initialize_logger(__name__)
                    logger.warn("Failed to convert string " + line[iline] + " to float!" +
                                "\nSkipping line " + str(id_line) + ":  " + str(line) + "!")
                    append_data = False
                    break
            if append_data:
                data.append(line)
        data = np.array(data)

        namemap = {}
        maxdims = {}
        for i, name in enumerate(names):
            if name[0] not in namemap:
                namemap[name[0]] = []
            namemap[name[0]].append(i)
            if len(name) > 1:
                maxdims[name[0]] = name[1:]

        for name in maxdims.keys():
            dims = []
            for dim in maxdims[name]:
                dims.append(int(dim))
            maxdims[name] = tuple(reversed(dims))

        # data in linear order per Stan, e.g. mat is col maj
        # TODO array is row maj, how to distinguish matrix vs array[,]?
        data_ = {}
        for name, idx in namemap.items():
            new_shape = (-1,) + maxdims.get(name, ())
            data_[name] = data[:, idx].reshape(new_shape)

        return data_
    except:
        warning("Failed to read %s!" % fname)
        return {}


def parse_csv_in_cols(fname):
    names = []
    sdims = {}
    with open(fname, 'r') as fd:
        scols = fd.readline().strip().split(',')[1:]
        output = {key: {} for key in scols}
        for line in fd.readlines():
            if '"' not in line:
                continue
            if line.startswith('#'):
                break
            _, key, vals = line.split('"')
            vals = [np.float(v) for v in vals.split(',')[1:]]
            if '[' in key:
                name, dim = key.replace('[', ']').split(']')[:-1]
                if name not in names:
                    names.append(name)
                    for icol, scol in enumerate(scols):
                        output[scol][name] = []
                sdims[name] = tuple(int(i) for i in dim.split(','))
                for icol, scol in enumerate(scols):
                    output[scol][name].append(vals[icol])
            else:
                for icol, scol in enumerate(scols):
                    output[scol][key] = vals[icol]

    for key in names:
        for icol, scol in enumerate(scols):
            output[scol][key] = np.array(output[scol][key]).reshape(sdims[key])

    return output


def csv2mode(csv_fname, mode=None):
    csv = parse_csv(csv_fname)
    data = {}
    for key, val in csv.items():
        if key.endswith('__'):
            continue
        if mode is None:
            val_ = val[0]
        elif mode == 'mean':
            val_ = val.mean(axis=0)
        elif mode[0] == 'p':
            val_ = np.percentile(val, int(mode[1:]), axis=0)
        data[key] = val_
    return data


def csv2r(csv_fname, r_fname=None, mode=None):
    data = csv2mode(csv_fname, mode=mode)
    r_fname = r_fname or csv_fname + '.R'
    rdump(r_fname, data)
