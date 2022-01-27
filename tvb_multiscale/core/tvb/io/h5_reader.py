# -*- coding: utf-8 -*-

import os

import h5py

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.tvb.io.h5_writer import H5Writer


class H5Reader(object):

    logger = initialize_logger(__name__)

    H5_TYPE_ATTRIBUTE = H5Writer().H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer().H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    def _open_file(self, name, path=None, h5_file=None):
        if h5_file is None:
            if not os.path.isfile(path):
                raise ValueError("%s file %s does not exist" % (name, path))

            self.logger.info_details("Starting to read %s from: %s" % (name, path))
            h5_file = h5py.File(path, 'r', libver='latest')
        return h5_file

    def _close_file(self, h5_file, close_file=True):
        if close_file:
            h5_file.close()

    def _log_success(self, name, path=None):
        if path is not None:
            self.logger.info_details("Successfully read %s from: %s" % (name, path))

    def read_dictionary(self, path=None, h5_file=None, close_file=True):  # type=None,
        """
        :param path: Path towards a dictionary H5 file
        :return: dict
        """
        h5_file = self._open_file("Dictionary", path, h5_file)
        dictionary = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE).read_dictionary_from_group(h5_file)  # , type
        self._close_file(h5_file, close_file)
        self._log_success("Dictionary", path)
        return dictionary

    def read_list_of_dicts(self, path=None, h5_file=None, close_file=True):  # type=None,
        h5_file = self._open_file("List of dictionaries", path, h5_file)
        list_of_dicts = []
        id = 0
        h5_group_handlers = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE)
        while 1:
            try:
                dict_group = h5_file[str(id)]
            except:
                break
            list_of_dicts.append(h5_group_handlers.read_dictionary_from_group(dict_group))  # , type
            id += 1
        self._close_file(h5_file, close_file)
        self._log_success("List of dictionaries", path)
        return list_of_dicts


class H5GroupHandlers(object):

    H5_TYPE_ATTRIBUTE = H5Writer().H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer().H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    def __init__(self, h5_subtype_attribute):
        if h5_subtype_attribute is not None:
            self.H5_SUBTYPE_ATTRIBUTE = h5_subtype_attribute

    def read_dictionary_from_group(self, group):  # , type=None
        dictionary = dict()
        for dataset in group.keys():
            try:
                value = group[dataset][()]
            except:
                try:
                    value = self.read_dictionary_from_group(group[dataset])
                except:
                    value = None
            dictionary.update({dataset: value})
        for attr in group.attrs.keys():
            if attr not in self.H5_TYPES_ATTRUBUTES:
                dictionary.update({attr: group.attrs[attr]})
        # if type is None:
        #     type = group.attrs[H5Reader.H5_SUBTYPE_ATTRIBUTE]
        # else:
        return dictionary
