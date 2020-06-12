# -*- coding: utf-8 -*-

import sys
import os
import logging
import time
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