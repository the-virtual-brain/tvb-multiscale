# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.config import LINE
from tvb_multiscale.core.tvb.io.io import TVBIO


class TVBInterface(HasTraits):
    __metaclass__ = ABCMeta

    """TVBInterface abstract base class."""

    label = Attr(label="Interface label",
                 doc="""Interface label of the form TVB_state_variable -> or <- for output or input interfaces.""",
                 field_type=str,
                 required=True)

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

    tvbio = Attr(
        label="TVBIO",
        field_type=TVBIO,
        doc="""A TVBIO class instance to send/receive TVB data to/from the cosimulator.""",
        required=True
    )

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

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str(None)

    def print_str(self, sender_not_receiver_flag=None):
        if sender_not_receiver_flag is True:
            comm_str = "Sender"
            tvb_source_or_target = "Source"
        elif sender_not_receiver_flag is False:
            comm_str = "Receiver"
            tvb_source_or_target = "Target"
        else:
            comm_str = "TVB communicator"
            tvb_source_or_target = ""
        return "\n" + self.__repr__() + \
               "\nLabel: %s, %s: %s" \
               "TVB state variable indices: %s" \
               "\n%s TVB proxy nodes' indices: %s" % \
               (self.label, comm_str, self.tvbio.__class__.__name__, str(self.vois.tolist()),
                tvb_source_or_target, extract_integer_intervals(self.proxy_inds, print=True))


class TVBInterfaces(HasTraits):
    """This class holds a list of TVB interfaces"""

    interfaces = List(of=TVBInterface)

    @property
    def number_of_interfaces(self):
        return len(self.interfaces)

    def _loop_get_from_interfaces(self, attr):
        out = []
        for interfaces in self.interfaces:
            out += list(getattr(interfaces, attr))
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

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str(None)

    def print_str(self, sender_not_receiver):
        output = 2 * LINE + "%s\n\n" % self.__repr__()
        for ii, interface in enumerate(self.interfaces):
            output += "%d. %s" % (ii, interface.print_str(sender_not_receiver))
            output += LINE + "\n"
        return output
