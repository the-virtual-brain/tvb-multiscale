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
        return trait_object_str(type(self).__name__, self.summary_info())

    def _repr_html_(self):
        return trait_object_repr_html(type(self).__name__, self.summary_info())

    def _info_dict(self, dname, d, recursive=0, details=False, **kwargs):
        info = OrderedDict()
        info[dname] = str(type(d))
        for key, val in d.items():
            if isinstance(val, HasTraits):
                info.update(val._info(recursive - 1, details, **kwargs))
            elif isinstance(val, HasTraitsTVB):
                info.update(val.summary_info())
            elif isinstance(val, dict):
                info.update(self._info_dict(key, val, recursive-1, details, **kwargs))
        return info

    def _info(self, recursive=0, details=False, **kwargs):
        info = OrderedDict()
        if self.title:
            info['title'] = str(self.title)
        cls = type(self)
        info['Type'] = cls.__name__
        for aname in self.declarative_attrs:
            try:
                attr = getattr(self, aname)
            except TraitError:
                attr = None
                info[aname] = 'unavailable'
            if attr is not None and recursive > 0:
                if isinstance(attr, HasTraits):
                    info.update(attr._info(recursive-1, details, **kwargs))
                elif isinstance(attr, HasTraitsTVB):
                    info.update(attr.summary_info())
                elif isinstance(attr, dict):
                    info.update(self._info_dict(aname, attr, recursive-1, details, **kwargs))
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
        return summary_info(self.info(recursive))

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
        return summary_info(self.info_details(recursive=recursive, details=True, **kwargs))
