# -*- coding: utf-8 -*-

import os
from tvb.basic.readers import H5Reader as H5ReaderBase

from .base import Base
from .h5_writer import H5Writer


class H5GroupHandlers(object):

    H5_TYPE_ATTRIBUTE = H5Writer.H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer.H5_SUBTYPE_ATTRIBUTE
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


class H5Reader(H5ReaderBase, Base):

    H5_TYPE_ATTRIBUTE = H5Writer.H5_TYPE_ATTRIBUTE
    H5_SUBTYPE_ATTRIBUTE = H5Writer.H5_SUBTYPE_ATTRIBUTE
    H5_TYPES_ATTRUBUTES = [H5_TYPE_ATTRIBUTE, H5_SUBTYPE_ATTRIBUTE]

    hfd5_source = None

    def __init__(self, h5_path):
        super(H5Reader, self).__init__(h5_path)
        self.h5_path = h5_path

    @property
    def _hdf_file(self):
        return self.hfd5_source

    def _set_hdf_file(self, hfile):
        self.hfd5_source = hfile

    @property
    def _fmode(self):
        return "r"

    @property
    def _mode(self):
        return "read"

    @property
    def _mode_past(self):
        return "read"

    @property
    def _to_from(self):
        return "from"

    def _open_file(self, type_name=""):
        if not os.path.isfile(self.h5_path):
            raise ValueError("%s file %s does not exist" % (type_name, self.h5_path))
        super(H5Reader, self)._open_file(type_name)

    def read_dictionary(self, path=None, close_file=True):  # type=None,
        """
        :return: dict
        """
        dictionary = dict()
        self._assert_file(path, "Dictionary")
        try:
            dictionary = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE).read_dictionary_from_group(h5_file)  # , type
            self._log_success_or_warn(None, "Dictionary")
        except Exception as e:
            self._log_success_or_warn(e, "Dictionary")
        self._close_file(close_file)
        return dictionary

    def read_list_of_dicts(self, path=None, close_file=True):  # type=None,
        self._assert_file(path, "List of Dictionaries")
        list_of_dicts = []
        id = 0
        h5_group_handlers = H5GroupHandlers(self.H5_SUBTYPE_ATTRIBUTE)
        try:
            while 1:
                try:
                    dict_group = h5_file[str(id)]
                except:
                    break
                list_of_dicts.append(h5_group_handlers.read_dictionary_from_group(dict_group))  # , type
                id += 1
            self._log_success_or_warn(None,  "List of Dictionaries")
        except Exception as e:
            self._log_success_or_warn(e, "List of Dictionaries")
        self._close_file(close_file)
        return list_of_dicts
