# -*- coding: utf-8 -*-
import json

import numpy
import pandas
from six import string_types

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.utils.data_structures_utils import is_iterable
from tvb_multiscale.tvb_nest.nest_models.server_client.node_collection import NodeCollection


def serializable(data):
    """Make data serializable for JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any

    Returns
    -------
    data_serialized : str, int, float, list, dict
        Data can be encoded to JSON
    """

    if isinstance(data, (numpy.ndarray, NodeCollection)):
        return data.tolist()
    if isinstance(data, SynapseCollection):
        # Get full information from SynapseCollection
        return serializable(data.todict())
    if isinstance(data, (list, tuple)):
        return [serializable(d) for d in data]
    if isinstance(data, dict):
        return dict([(key, serializable(value)) for key, value in data.items()])
    return data


def to_json(data, **kwargs):
    """Serialize data to JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any
    kwargs : keyword argument pairs
        Named arguments of parameters for `json.dumps` function.

    Returns
    -------
    data_json : str
        JSON format of the data
    """

    data_serialized = serializable(data)
    data_json = json.dumps(data_serialized, **kwargs)
    return data_json


class SynapseCollectionIterator(object):
    """
    Iterator class for SynapseCollection.
    Modifying the corresponding class for SynapseCollectionIterator of pynest.
    """

    def __init__(self, synapse_collection):
        self.nest_instance = synapse_collection.nest_instance
        self._iter = iter(dicts_of_lists_to_lists_of_dicts(synapse_collection.todict()))

    def __iter__(self):
        return self

    def __next__(self):
        return SynapseCollection(self.nest_instance, next(self._iter))


class SynapseCollection(object):
    """
    Class for SynapseCollection.
    Modifying the corresponding class for SynapseCollection of pynest.

    """

    nest_instance = None

    _attributes = ["source", "target", "weight", "delay", "receptor",
                   "synapse_model", "synapse_id", "port", "target_thread"]
    source = ()
    target = ()
    weight = ()
    delay = ()
    receptor = ()
    target_thread = ()
    port = ()
    synapse_id = ()
    synapse_model = ()

    def __init__(self, nest_instance, conns={}):
        self.nest_instance = nest_instance
        self.fromdict(conns)

    def fromdict(self, conns={}):
        for key, val in conns.items():
            if key not in self._attributes:
                self._attributes.append(key)
            setattr(self, key, tuple(ensure_list(val)))
        n = len(self.source)
        assert [len(getattr(self, attr)) == n for attr in self._attributes[1:]]

    def todict(self, keys=None):
        if keys is None:
            keys = self._attributes
        else:
            for key in keys:
                if key not in self._attributes:
                    raise ValueError("%s is not an attribute of %s!" % (key, self.__class__.__name__))
        output = {}
        for attr in keys:
            output[attr] = getattr(self, attr)
        return output

    def __getstate__(self):
        d = self.todict()
        d["nest_instance"] = self.nest_instance
        d["_attributes"] = self._attributes
        return d

    def __setstate__(self, d):
        super(SynapseCollection, self).__setattr__("nest_instance", d.pop("nest_instance", None))
        super(SynapseCollection, self).__setattr__("_attributes", d.pop("_attributes", self._attributes))
        self.fromdict(d)

    def __iter__(self):
        return SynapseCollectionIterator(self)

    def __len__(self):
        return len(self.source)

    def __eq__(self, other):
        if not isinstance(other, SynapseCollection):
            raise NotImplementedError()

        if self.__len__() != other.__len__():
            return False

        for attr in ['source', 'target', 'target_thread', 'synapse_id', 'port']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __neq__(self, other):
        if not isinstance(other, SynapseCollection):
            raise NotImplementedError()
        return not self == other

    def __getitem__(self, key):
        return self.__class__(self.nest_instance,
                              list_of_dicts_to_dict_of_tuples(list(numpy.array(self.todict())[key])))

    def __str__(self):
        """
        Printing a `SynapseCollection` returns something of the form:
            *--------*-------------*
            | source | 1, 1, 2, 2, |
            *--------*-------------*
            | target | 1, 2, 1, 2, |
            *--------*-------------*
        """
        srcs = self.source
        trgt = self.target

        if isinstance(srcs, int):
            srcs = [srcs]
        if isinstance(trgt, int):
            trgt = [trgt]

        # 35 is arbitrarily chosen.
        if len(srcs) < 35:
            source = '| source | ' + ''.join(str(e)+', ' for e in srcs) + '|'
            target = '| target | ' + ''.join(str(e)+', ' for e in trgt) + '|'
        else:
            source = ('| source | ' + ''.join(str(e)+', ' for e in srcs[:15]) +
                      '... ' + ''.join(str(e)+', ' for e in srcs[-15:]) + '|')
            target = ('| target | ' + ''.join(str(e)+', ' for e in trgt[:15]) +
                      '... ' + ''.join(str(e)+', ' for e in trgt[-15:]) + '|')

        borderline_s = '*--------*' + '-'*(len(source) - 12) + '-*'
        borderline_t = '*--------*' + '-'*(len(target) - 12) + '-*'
        borderline_m = max(borderline_s, borderline_t)

        result = (borderline_s + '\n' + source + '\n' + borderline_m + '\n' +
                  target + '\n' + borderline_t)
        return result

    def sources(self):
        """Returns iterator containing the source node IDs of the `SynapseCollection`."""
        if not isinstance(self.sources, (list, tuple)):
            self.sources = (self.sources,)
        return iter(self.sources)

    def targets(self):
        """Returns iterator containing the target node IDs of the `SynapseCollection`."""
        if not isinstance(self.targets, (list, tuple)):
            self.targets = (self.targets,)
        return iter(self.targets)

    def get(self, keys=None, output=''):
        """
        Return a parameter dictionary of the connections.

        If `keys` is a string, a list of values is returned, unless we have a
        single connection, in which case the single value is returned.
        `keys` may also be a list, in which case a dictionary with a list of
        values is returned.

        Parameters
        ----------
        keys : str or list, optional
            String or a list of strings naming model properties. get
            then returns a single value or a dictionary with lists of values
            belonging to the given `keys`.
        output : str, ['pandas','json'], optional
            If the returned data should be in a Pandas DataFrame or in a
            JSON serializable format.

        Returns
        -------
        dict:
            All parameters, or, if keys is a list of strings, a dictionary with
            lists of corresponding parameters
        type:
            If keys is a string, the corrsponding parameter(s) is returned


        Raises
        ------
        TypeError
            If input params are of the wrong form.
        KeyError
            If the specified parameter does not exist for the connections.
        """

        pandas_output = output == 'pandas'

        # Return empty tuple if we have no connections or if we have done a
        # nest.ResetKernel()
        if self.nest_instance:
            num_conn = self.nest_instance.GetKernelStatus('num_connections')
        else:
            num_conn = 0
        if self.__len__() == 0 or num_conn == 0:
            return ()

        if keys is None:
            keys = self._attributes
        elif isinstance(keys, string_types):
            keys = keys.split(" ")
        elif not is_iterable(keys):
            raise TypeError("keys should be either a string or an iterable")

        result = self.todict(keys)

        if pandas_output:
            index = (self.source if self.__len__() > 1 else
                     (self.source,))
            result = pandas.DataFrame(result, index=index)
        elif output == 'json':
            result = to_json(result)

        return result

    def set(self, params=None, **kwargs):
        """
        Set the parameters of the connections to `params`.

        NB! This is almost the same implementation as SetStatus

        If `kwargs` is given, it has to be names and values of an attribute as keyword argument pairs. The values
        can be single values or list of the same size as the `SynapseCollection`.

        Parameters
        ----------
        params : str or dict or list
            Dictionary of parameters or list of dictionaries of parameters of
            same length as the `SynapseCollection`.
        kwargs : keyword argument pairs
            Named arguments of parameters of the elements in the `SynapseCollection`.

        Raises
        ------
        TypeError
            If input params are of the wrong form.
        KeyError
            If the specified parameter does not exist for the connections.
        """
        if self.nest_instance:
            num_conn = self.nest_instance.GetKernelStatus('num_connections')
            if num_conn == 0:
                return
            self.nest_instance.set(self, params=params, **kwargs)
            self.fromdict(self.nest_instance.get())
