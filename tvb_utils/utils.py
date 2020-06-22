# -*- coding: utf-8 -*-

import sys
import os
import logging
import time
from collections import OrderedDict
from logging.handlers import TimedRotatingFileHandler


def initialize_logger(name, target_folder):
    """
    create logger for a given module
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not (os.path.isdir(target_folder)):
        os.makedirs(target_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)

    fh = TimedRotatingFileHandler(os.path.join(target_folder, 'logs.log'), when="d", interval=1, backupCount=2)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # Log errors separately, to have them easy to inspect
    fhe = TimedRotatingFileHandler(os.path.join(target_folder, 'log_errors.log'), when="d", interval=1, backupCount=2)
    fhe.setFormatter(formatter)
    fhe.setLevel(logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fhe)

    return logger


def safe_makedirs(folder):
    # solution based on https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    if not os.path.isdir(folder):
        try:
            # This should work for Python >= 3.2
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:  # Python ≥ 2.5
            try:
                os.makedirs(folder)
            except:
                try:  # just in case errno cannot be imported...
                    import errno
                    if exc.errno == errno.EEXIST and os.path.isdir(folder):
                        pass
                    else:
                        raise
                except:
                    pass


def print_toc_message(tic):
    toc = time.time() - tic
    if toc > 60.0:
        if toc > 3600.0:
            toc /= 3600.0
            unit = "hours"
        else:
            toc /= 60.0
            unit = "mins"
    else:
        unit = "sec"
    print("DONE in %f %s!" % (toc, unit))


def read_dicts_from_h5file_recursively(h5file_or_group):
    d = OrderedDict()
    for k in h5file_or_group.keys():
        try:
            d[k] = h5file_or_group[k][()]
        except:
            d[k] = read_dicts_from_h5file_recursively(h5file_or_group[k])
    return d