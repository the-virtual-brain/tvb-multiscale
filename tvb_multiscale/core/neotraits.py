# -*- coding: utf-8 -*-

import typing
from collections import OrderedDict

from tvb.basic.neotraits.api import HasTraits as HasTraitsTVB
from tvb.basic.neotraits.ex import TraitError
from tvb_multiscale.core.utils.data_structures_utils import trait_object_str, trait_object_repr_html, summary_info


class HasTraits(HasTraitsTVB):

    # The base __init__ and __str__ rely upon metadata gathered by MetaType
    # we could have injected these in MetaType, but we don't need meta powers
    # this is simpler to grok

    def __str__(self):
        return self.summary_info_to_string()

    def _repr_html_(self):
        return trait_object_repr_html(type(self).__name__, self.summary_info())

    def _add_to_info(self, info, aname, attr, recursive=0, details=False, **kwargs):
        if isinstance(attr, (HasTraits, dict, list, tuple)) and recursive > 0:
            try:
                if isinstance(attr, (dict, list, tuple)):
                    if isinstance(attr, (list, tuple)):
                        this_dct = self._info_list_to_dict(attr)
                    else:
                        this_dct = dict(attr)
                    info.update(self._info_dict(aname, this_dct, recursive=recursive-1, details=details, **kwargs))
                else:
                    if isinstance(attr, HasTraits):
                        info[aname] = "-" * 20
                        if details:
                            this_info = attr.info_details(recursive=recursive-1, **kwargs)
                        else:
                            this_info = attr.info(recursive=recursive-1)
                    else:  # isinstance(attr, HasTraitsTVB):
                        this_info = attr.summary_info()
                    for key, val in this_info.items():
                        lbl = "%s.%s" % (aname, key)
                        lbl = lbl.replace(".[", "[")
                        info[lbl] = val
            except Exception as e:
                print("Failed to serialize: ")
                print(aname)
                print(attr.__class__.__name__)
                print(Warning(e))
        else:
            info[aname] = attr
        return info

    def _info_dict(self, dname, d, recursive=0, details=False, **kwargs):
        info = OrderedDict()
        for key, val in d.items():
            info = self._add_to_info(info, "%s[%s]" % (dname, key), val, recursive, details, **kwargs)
        return info

    def _info_list_to_dict(self, lst):
        info = OrderedDict()
        for iV, val in enumerate(lst):
            info["%d" % iV] = val
        return info

    def _info(self, recursive=0, details=False, **kwargs):
        info = OrderedDict()
        if self.title:
            info['title'] = str(self.title)
        cls = type(self)
        info['Type'] = cls.__name__
        for aname in list(cls.declarative_attrs):
            try:
                attr = getattr(self, aname)
            except TraitError:
                attr = None
            info = self._add_to_info(info, aname, attr, recursive, details, **kwargs)
        return info

    def info(self, recursive=0):
        return self._info(recursive=recursive)

    def info_details(self, recursive=0, **kwargs):
        return self._info(recursive=recursive, details=True, **kwargs)

    def summary_info(self, recursive=0):
        # type: (int) -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return summary_info(self.info(recursive=recursive), to_string=False)

    def summary_info_details(self, recursive=0, **kwargs):
        # type: (int, dict) -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return summary_info(self.info_details(recursive=recursive, **kwargs), to_string=False)

    def summary_info_to_string(self, recursive=0):
        # type: (int) -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return trait_object_str(type(self).__name__, summary_info(self.info(recursive=recursive), to_string=True))

    def summary_info_details_to_string(self, recursive=0, **kwargs):
        # type: (int, dict) -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return trait_object_str(type(self).__name__,
                                summary_info(self.info_details(recursive=recursive, **kwargs), to_string=True))

    def print_summary_info(self, recursive=0):
        print(self.summary_info_to_string(recursive=recursive))

    def print_summary_info_details(self, recursive=0, **kwargs):
        print(self.summary_info_details_to_string(recursive=recursive, **kwargs))
