# -*- coding: utf-8 -*-
#
# Logging and errors helper functions
#

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from tvb.simulator.plot.config import CONFIGURED


def initialize_logger(name, target_folder=CONFIGURED.FOLDER_LOGS):
    """
    create logger for a given module
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not (os.path.isdir(target_folder)):
        os.makedirs(target_folder)

    logger = logging.getLogger(name)

    if name == "matplotlib":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)

    # We introduce a short delay by going to the end of the log file
    # This allows running Dockerfile on Windows 10 when the user
    # don't stop the container before running it again.
    file = open(os.path.join(target_folder, 'logs.log'), 'rb')
    file.seek(0, 2)

    fh = TimedRotatingFileHandler(os.path.join(target_folder, 'logs.log'), when="d", interval=1, backupCount=2)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # We introduce a short delay by going to the end of the log file
    # This allows running Dockerfile on Windows 10 when the user
    # don't stop the container before running it again.
    file = open(os.path.join(target_folder, 'log_errors.log'), 'rb')
    file.seek(0, 2)

    # Log errors separately, to have them easy to inspect
    fhe = TimedRotatingFileHandler(os.path.join(target_folder, 'log_errors.log'), when="d", interval=1, backupCount=2)
    fhe.setFormatter(formatter)
    fhe.setLevel(logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fhe)

    return logger


def raise_value_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nValueError: " + msg + "\n")
    raise ValueError(msg)


def raise_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nError: " + msg + "\n")
    raise Exception(msg)


def raise_import_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nImportError: " + msg + "\n")
    raise ImportError(msg)


def raise_not_implemented_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nNotImplementedError: " + msg + "\n")
    raise NotImplementedError(msg)


def warning(msg, logger=None):
    if logger is not None:
        logger.warning("\n" + msg + "\n")
