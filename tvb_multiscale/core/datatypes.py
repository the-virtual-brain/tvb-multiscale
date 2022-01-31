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

    def info(self):
        ret = OrderedDict()
        cls = type(self)
        ret['Type'] = cls.__name__
        if self.title:
            ret['title'] = str(self.title)
        for aname in self.declarative_attrs:
            try:
                ret[aname] = getattr(self, aname)
            except TraitError:
                ret[aname] = 'unavailable'
        return ret

    def info_details(self, *kwargs):
        return self.info()

    def summary_info(self):
        # type: () -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return summary_info(self.info())

    def summary_info_details(self, *kwargs):
        # type: () -> typing.Dict[str, str]
        """
        A more structured __str__
        A 2 column table represented as a dict of str->str
        The default __str__ and html representations of this object are derived from
        this table.
        Override this method and return such a table filled with instance information
        that informs the user about your instance
        """
        return summary_info(self.info_details(*kwargs))
