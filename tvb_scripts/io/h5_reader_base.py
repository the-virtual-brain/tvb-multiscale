# -*- coding: utf-8 -*-

import os

import h5py
from tvb.simulator.plot.utils.log_error_utils import initialize_logger

from tvb_scripts.io.h5_writer import H5Writer


class H5ReaderBase(object):
    logger = initialize_logger(__name__)

    H5_TYPE_ATTRIBUTE = H5Writer().H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer().H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    def _open_file(self, name, path=None, h5_file=None):
        if h5_file is None:
            if not os.path.isfile(path):
                raise ValueError("%s file %s does not exist" % (name, path))

            self.logger.info("Starting to read %s from: %s" % (name, path))
            h5_file = h5py.File(path, 'r', libver='latest')
        return h5_file

    def _close_file(self, h5_file, close_file=True):
        if close_file:
            h5_file.close()

    def _log_success(self, name, path=None):
        if path is not None:
            self.logger.info("Successfully read %s from: %s" % (name, path))


class H5GroupHandlers(object):

    def read_dictionary_from_group(self, group, type=None):
        dictionary = dict()
        for dataset in group.keys():
            dictionary.update({dataset: group[dataset][()]})
        for attr in group.attrs.keys():
            dictionary.update({attr: group.attrs[attr]})
        if type is None:
            type = group.attrs[H5Writer().H5_TYPE_ATTRIBUTE]
        else:
            return dictionary
