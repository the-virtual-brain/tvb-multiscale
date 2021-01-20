# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Int, Attr, NArray, List

from tvb_multiscale.core.tvb.io.io import TVBIO


class TVBInterface(HasTraits):
    __metaclass__ = ABCMeta

    """TVBInterface abstract base class."""

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of TVB proxy nodes",
        doc="""Indices of TVB proxy nodes""",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc="""Indices of model's variables of interest (VOI) that""",
        required=True)

    voi_loc = np.array([])

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds.shape[0]

    @property
    def n_voi(self):
        return self.voi.shape[0]

    def configure(self):
        """Method to configure the interface"""
        super().configure()

    def set_local_indices(self, inds, simulator_inds):
        simulator_inds_list = list(simulator_inds)
        return np.array([simulator_inds_list.index(ind) for ind in inds])

    @abstractmethod
    def __call__(self):
        pass

class TVBInterfaces(HasTraits):
    """This class holds a list of TVB interfaces"""

    interfaces = List(of=TVBInterface)

    @property
    def number_of_interfaces(self):
        return len(self.interfaces)

    def _loop_get_from_interfaces(self, attr):
        out = []
        for interfaces in self.interfaces:
            out += list(getattr(interfaces, "attr"))
        return out

    @property
    def voi(self):
        return np.sort(self._loop_get_from_interfaces("voi"))

    @property
    def voi_unique(self):
        return np.unique(self._loop_get_from_interfaces("voi"))

    @property
    def proxy_inds(self):
        return np.sort(self._loop_get_from_interfaces("proxy_inds"))

    @property
    def proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("proxy_inds"))

    @property
    def n_vois(self):
        return self.voi_unique.shape[0]

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    def configure(self):
        """Method to configure the interfaces"""
        for interface in self.interfaces:
            interface.configure()
        super().configure()

    @abstractmethod
    def __call__(self):
        pass
