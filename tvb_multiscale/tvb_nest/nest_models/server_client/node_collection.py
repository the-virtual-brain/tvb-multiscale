# -*- coding: utf-8 -*-

import numpy

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class NodeCollectionIterator(object):
    """
    Iterator class for `NodeCollectionIterator`.
    Modifying the corresponding class for NodeCollectionIterator of pynest.

    Returns
    -------
    `NodeCollection`:
        Single node ID `NodeCollectionIterator` of respective iteration.
    """

    def __init__(self, nc):
        self._nc = nc
        self._increment = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._increment > len(self._nc) - 1:
            raise StopIteration

        val = self._nc[self._increment]
        self._increment += 1
        return val


class NodeCollection(object):
    """
    Class for `NodeCollection`.
    Modifying the corresponding class for NodeCollection of pynest.
    """

    nest_instance = None

    gids = ()

    def __init__(self, nest_instance, gids=()):
        self.nest_instance = nest_instance
        self.gids = tuple(numpy.unique(gids))

    def __getattr__(self, attr):
        if self.nest_instance:
            return self.get(attr)
        raise AttributeError("type object %s has no attribute %s" % (self.__class__.__name__, attr))

    def __setattr__(self, attr, value):
        if attr in ["nest_instance", "gids"]:
            super(NodeCollection, self).__setattr__(attr, value)
        elif self.nest_instance:
            return self.set({attr: value})

    def __getstate__(self):
        return {"gids": self.gids, "nest_instance": self.nest_instance}

    def __setstate__(self, d):
        super(NodeCollection, self).__setattr__("gids", d.get("gids", ()))
        super(NodeCollection, self).__setattr__("nest_instance", d.get("nest_instance", None))

    def __iter__(self):
        return NodeCollectionIterator(self)

    def __add__(self, other):
        if not isinstance(other, NodeCollection):
            raise NotImplementedError()

        return self.__class__(self.nest_instance, self.gids + other.gids)

    def __getitem__(self, key):
        return self.__class__(self.nest_instance, tuple(ensure_list(numpy.array(self.gids)[key])))

    def __contains__(self, node_id):
        return node_id in self.gids

    def __eq__(self, other):
        if not isinstance(other, NodeCollection):
            raise NotImplementedError('Cannot compare NodeCollection to {}'.format(type(other).__name__))

        if self.__len__() != other.__len__():
            return False

        return self.gids == other.gids

    def __neq__(self, other):
        if not isinstance(other, NodeCollection):
            raise NotImplementedError()

        return not self == other

    def __len__(self):
        return len(self.gids)

    def __str__(self):
        return "%s: %s" % (self.__class__.__name__, str(self.gids))

    def __repr__(self):
        return "%s:\nnest_instance=%s\ngids=%s" % (self.__class__.__name__, str(self.nest_instance), str(self.gids))

    def get(self, *params, **kwargs):
        return self.nest_instance.get(self, *params, **kwargs)

    def set(self, params=None, **kwargs):
        return self.nest_instance.set(self, params=params, **kwargs)

    def tolist(self):
        """
        Convert `NodeCollection` to list.
        """
        return list(self.gids)

    def index(self, node_id):
        """
        Find the index of a node ID in the `NodeCollection`.

        Parameters
        ----------
        node_id : int
            Global ID to be found.

        Raises
        ------
        ValueError
            If the node ID is not in the `NodeCollection`.
        """
        return list(self.gids).index(node_id)

    def __bool__(self):
        """Converts the NodeCollection to a bool. False if it is empty, True otherwise."""
        return len(self) > 0

    def __array__(self, dtype=None):
        """Convert the NodeCollection to a NumPy array."""
        return numpy.array(self.tolist(), dtype=dtype)
