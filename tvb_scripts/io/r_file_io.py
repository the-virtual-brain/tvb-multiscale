# coding=utf-8
import numpy as np


def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size,) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(
            key=key, c=c, dim=dim)
        return struct


def rdump(filepath, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(filepath, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            else:
                try:
                    val = val.flat[0]
                except:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')


def rload(fname):
    """Load a dict of data from an R dump format file.
    """
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    data = {}
    for line in lines:
        lhs, rhs = [_.strip() for _ in line.split('<-')]
        if rhs.startswith('structure'):
            vals, dim = rhs.replace('(', ' ').replace(')', ' ').split('c')[1:]  # [1:] instead of *_,
            vals = [float(v) for v in vals.split(',')[:-1]]
            dim = [int(v) for v in dim.split(',')]
            val = np.array(vals).reshape(dim[::-1]).T
        elif rhs.startswith('c'):
            val = np.array([float(_) for _ in rhs[2:-1].split(',')])
        else:
            try:
                val = int(rhs)
            except:
                try:
                    val = float(rhs)
                except:
                    raise ValueError(rhs)
        data[lhs] = val
    return data
