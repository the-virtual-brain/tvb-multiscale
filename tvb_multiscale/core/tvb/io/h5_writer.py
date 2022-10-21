# -*- coding: utf-8 -*-

from six import string_types
import os
import h5py
import inspect
import numpy

from tvb_multiscale.core.config import CONFIGURED, initialize_logger

from tvb.core.neocom import h5
from tvb.contrib.scripts.utils.log_error_utils import warning
from tvb.contrib.scripts.utils.data_structures_utils import is_numeric, ensure_list
from tvb.contrib.scripts.utils.file_utils import change_filename_or_overwrite
from tvb_multiscale.core.tvb.io.datatypes_h5 import REGISTRY


h5.REGISTRY = REGISTRY


class H5Writer(object):

    config = CONFIGURED
    logger = initialize_logger(__name__)
    H5_TYPE_ATTRIBUTE = "Type"
    H5_SUBTYPE_ATTRIBUTE = "Subtype"
    H5_VERSION_ATTRIBUTE = "Version"
    H5_DATE_ATTRIBUTE = "Last_update"
    force_overwrite = True
    write_mode = "a"

    def _open_file(self, name, path=None, h5_file=None):
        if h5_file is None:
            if self.write_mode == "w":
                path = change_filename_or_overwrite(path, self.force_overwrite)
            self.logger.info("Starting to write %s to: %s" % (name, path))
            h5_file = h5py.File(path, self.write_mode, libver='latest')
        return h5_file, path

    def _close_file(self, h5_file, close_file=True):
        if close_file:
            h5_file.close()

    def _log_success(self, name, path=None):
        if path is not None:
            self.logger.info("%s has been written to file: %s" % (name, path))

    def _determine_datasets_and_attributes(self, object, datasets_size=None):
        datasets_dict = {}
        metadata_dict = {}
        groups_keys = []

        try:
            if isinstance(object, dict):
                dict_object = object
            elif hasattr(object, "to_dict"):
                dict_object = object.to_dict()
            else:
                dict_object = vars(object)
            for key, value in dict_object.items():
                if isinstance(value, numpy.ndarray):
                    # if value.size == 1:
                    #     metadata_dict.update({key: value})
                    # else:
                    datasets_dict.update({key: value})
                    # if datasets_size is not None and value.size == datasets_size:
                    #     datasets_dict.update({key: value})
                    # else:
                    #     if datasets_size is None and value.size > 0:
                    #         datasets_dict.update({key: value})
                    #     else:
                    #         metadata_dict.update({key: value})
                # TODO: check how this works! Be carefull not to include lists and tuples if possible in tvb classes!
                elif isinstance(value, (list, tuple)):
                    warning("Writing %s %s to h5 file as a numpy array dataset !" % (value.__class__, key), self.logger)
                    datasets_dict.update({key: numpy.array(value)})
                else:
                    if is_numeric(value) or isinstance(value, str):
                        metadata_dict.update({key: value})
                    elif callable(value):
                        metadata_dict.update({key: inspect.getsource(value)})
                    elif value is None:
                        continue
                    else:
                        groups_keys.append(key)
        except Exception as e:
            msg = "Failed to decompose group object: " + str(object) + "!" + "\nThe error was\n%s" % str(e)
            try:
                self.logger.info(str(object.__dict__))
            except:
                msg += "\n It has no __dict__ attribute!"
            warning(msg, self.logger)

        return datasets_dict, metadata_dict, groups_keys

    def _write_dicts_at_location(self, datasets_dict, metadata_dict, location):
        for key, value in datasets_dict.items():
            try:
                try:
                    location.create_dataset(key, data=value)
                except:
                    location.create_dataset(key, data=numpy.str(value))
            except Exception as e:
                warning("Failed to write to %s dataset %s %s:\n%s !\nThe error was:\n%s" %
                        (str(location), value.__class__, key, str(value), str(e)), self.logger)

        for key, value in metadata_dict.items():
            try:
                location.attrs.create(key, value)
            except Exception as e:
                warning("Failed to write to %s attribute %s %s:\n%s !\nThe error was:\n%s" %
                        (str(location), value.__class__, key, str(value), str(e)), self.logger)
        return location

    def _prepare_object_for_group(self, group, object, h5_type_attribute="", nr_regions=None,
                                  regress_subgroups=True):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, h5_type_attribute)
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, object.__class__.__name__)
        datasets_dict, metadata_dict, subgroups = self._determine_datasets_and_attributes(object, nr_regions)
        # If empty return None
        if len(datasets_dict) == len(metadata_dict) == len(subgroups) == 0:
            if isinstance(group, h5py._hl.files.File):
                if regress_subgroups:
                    return group
                else:
                    return group, subgroups
            else:
                return None
        else:
            if len(datasets_dict) > 0 or len(metadata_dict) > 0:
                if isinstance(group, h5py._hl.files.File):
                    group = self._write_dicts_at_location(datasets_dict, metadata_dict, group)
                else:
                    self._write_dicts_at_location(datasets_dict, metadata_dict, group)
            # Continue recursively going deeper in the object
            if regress_subgroups:
                for subgroup in subgroups:
                    if isinstance(object, dict):
                        child_object = object.get(subgroup, None)
                    else:
                        child_object = getattr(object, subgroup, None)
                    if child_object is not None:
                        group.require_group(subgroup)
                        temp = self._prepare_object_for_group(group[subgroup], child_object,
                                                              h5_type_attribute, nr_regions)
                        # If empty delete it
                        if temp is None or (len(temp.keys()) == 0 and len(temp.attrs.keys()) == 0):
                            del group[subgroup]

                return group
            else:
                return group, subgroups

    def write_object(self, object, h5_type_attribute="", nr_regions=None,
                     path=None, h5_file=None, close_file=True):
        """
                :param object: object to write recursively in H5
                :param path: H5 path to be written
        """
        h5_file, path = self._open_file(object.__class__.__name__, path, h5_file)
        h5_file = self._prepare_object_for_group(h5_file, object, h5_type_attribute, nr_regions)
        self._close_file(h5_file, close_file)
        self._log_success(object.__class__.__name__, path)
        return h5_file, path

    def write_list_of_objects(self, list_of_objects, path=None, h5_file=None, close_file=True):
        h5_file, path = self._open_file("List of objects", path, h5_file)
        for idict, object in enumerate(list_of_objects):
            idict_str = str(idict)
            h5_file.create_group(idict_str)
            self.write_object(object, h5_file=h5_file[idict_str], close_file=False)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("List of objects"))
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_("list"))
        self._close_file(h5_file, close_file)
        self._log_success("List of objects", path)
        return h5_file, path

    def _convert_sequences_of_strings(self, sequence):
        new_sequence = []
        for val in ensure_list(sequence):
            if isinstance(val, string_types):
                new_sequence.append(numpy.string_(val))
            elif isinstance(val, (numpy.ndarray, tuple, list)):
                new_sequence.append(self._convert_sequences_of_strings(val))
            else:
                new_sequence.append(val)
        return numpy.array(new_sequence)

    def _write_dictionary_to_group(self, dictionary, group):
        group.attrs.create(self.H5_TYPE_ATTRIBUTE, "Dictionary")
        group.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, dictionary.__class__.__name__)
        for key, value in dictionary.items():
            try:
                if isinstance(value, (numpy.ndarray, list, tuple)) and len(value) > 0:
                    new_value = numpy.array(value)
                    if not numpy.issubdtype(value.dtype, numpy.number):
                        new_value = self._convert_sequences_of_strings(new_value)
                    group.create_dataset(key, data=new_value)
                else:
                    if callable(value):
                        group.attrs.create(key, inspect.getsource(value))
                    elif isinstance(value, dict):
                        group.create_group(key)
                        self._write_dictionary_to_group(value, group[key])
                    elif value is None:
                        continue
                    else:
                        group.attrs.create(key, numpy.string_(value))
            except Exception as e:
                self.logger.warning("Did not manage to write %s to h5 file %s !\nThe error was:\n%s"
                                    % (key, str(group), str(e)))

    def write_dictionary(self, dictionary, path=None, h5_file=None, close_file=True):
        """
        :param dictionary: dictionary/ies to write recursively in H5
        :param path: H5 path to be written
        Use this function only if you have to write dictionaries of data (scalars and arrays or lists of scalars,
        or of more such dictionaries recursively
        """
        h5_file, path = self._open_file("Dictionary", path, h5_file)
        self._write_dictionary_to_group(dictionary, h5_file)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Dictionary"))
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(dictionary.__class__.__name__))
        self._close_file(h5_file, close_file)
        self._log_success("Dictionary", path)
        return h5_file, path

    def write_list_of_dictionaries(self, list_of_dicts, path=None, h5_file=None, close_file=True):
        h5_file, path = self._open_file("List of dictionaries", path, h5_file)
        for idict, dictionary in enumerate(list_of_dicts):
            idict_str = str(idict)
            h5_file.create_group(idict_str)
            self._write_dictionary_to_group(dictionary, h5_file[idict_str])
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("List of dictionaries"))
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_("list"))
        self._close_file(h5_file, close_file)
        self._log_success("List of dictionaries", path)
        return h5_file, path

    def write_tvb_to_h5(self, datatype, path=None, recursive=True, force_overwrite=True):
        if path is None:
            path = self.config.out.FOLDER_RES
        if path.endswith("h5"):
            # It is a file path:
            dirpath = os.path.dirname(path)
            if os.path.isdir(dirpath):
                path = change_filename_or_overwrite(path, force_overwrite)
            else:
                os.mkdir(dirpath)
            h5.store(datatype, path, recursive)
        else:
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, datatype.title + ".h5")
            path = change_filename_or_overwrite(path, self.force_overwrite)
            h5.store(datatype, path, recursive)
        return path
