# -*- coding: utf-8 -*-

import types
import typing
import inspect
from collections import OrderedDict

from tvb.basic.neotraits._attr import Attr as AttrTVB
from tvb.basic.neotraits.api import HasTraits as HasTraitsTVB
from tvb.basic.neotraits.ex import TraitError, TraitTypeError, TraitValueError
from tvb_multiscale.core.utils.data_structures_utils import trait_object_str, trait_object_repr_html, summary_info


class Attr(AttrTVB):

    def __init__(
            self, field_type, default=None, doc='', label='', required=True, final=False, choices=None
    ):
        # type: ((type, tuple), typing.Any, str, str, bool, bool, typing.Optional[tuple]) -> None
        """
        :param field_type: the python type of this attribute or a tuple of such possible types
        :param default: A shared default value. Behaves like class level attribute assignment.
                        Take care with mutable defaults.
        :param doc: Documentation for this field.
        :param label: A short description.
        :param required: required fields should not be None.
        :param final: Final fields can only be assigned once.
        :param choices: A tuple of the values that this field is allowed to take.
        """
        if not isinstance(field_type, tuple):
            field_type = tuple([field_type])
        super(Attr, self).__init__(field_type[0], default, doc, label, required, final, choices)
        self.field_type = field_type

    def __validate(self, value):
        """ check field_type and choices """
        if not isinstance(value, self.field_type) and not (
                inspect.isclass(self.default) and issubclass(value, self.field_type)):
            raise TraitTypeError("Attribute can't be set to an instance of {}".format(type(value)), attr=self)
        if self.choices is not None:
            if value not in self.choices and not (value is None and not self.required):
                raise TraitValueError("Value {!r} must be one of {}".format(value, self.choices), attr=self)

    def _post_bind_validate(self):
        # type: () -> None
        """
        Validates this instance of Attr.
        This is called just after field_name is set, by MetaType.
        We do checks here and not in init in order to give better error messages.
        Attr should be considered initialized only after this has run
        """
        if not isinstance(self.field_type, type):
            if isinstance(self.field_type, tuple):
                for it, typ in enumerate(self.field_type):
                    if not isinstance(typ, type):
                        msg = 'Every element of field_type must be type but the {!r}th one is {!r}.'.format(it+1, typ)
                        raise TraitTypeError(msg, attr=self)
            else:
                msg = 'Field_type must be a type or tuple of types not {!r}. Did you mean to declare a default?'.format(
                    self.field_type
                )
                raise TraitTypeError(msg, attr=self)

        skip_default_checks = self.default is None or isinstance(self.default, types.FunctionType)

        if not skip_default_checks:
            self.__validate(self.default)

        # heuristic check for mutability. might be costly. hasattr(__hash__) is fastest but less reliable
        try:
            hash(self.default)
        except TypeError:
            from tvb_multiscale.core.config import initialize_logger
            LOG = initialize_logger(__name__)
            LOG.warning('Field seems mutable and has a default value. '
                        'Consider using a lambda as a value factory \n   attribute {}'.format(self))
        # we do not check here if we have a value for a required field
        # it is too early for that, owner.__init__ has not run yet


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

    def __getattribute__(self, attr):
        return super(HasTraits, self).__getattribute__(attr)

    def __setattr__(self, attr, val):
        return super(HasTraits, self).__setattr__(attr, val)

    def __getstate__(self):
        return self.__dict__
    #
    # def __reduce__(self):
    #     return type(self), self.__getstate__()
    #
    # def __setstate__(self, d):
    #     for key, val in d.items():
    #         setattr(self, key, val)
