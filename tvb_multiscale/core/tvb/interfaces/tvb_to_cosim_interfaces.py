# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Int, Attr, NArray, List

from tvb_multiscale.core.tvb.io.io import TVBSender


class TVBtoCosimInterface(HasTraits):

    """TVBtoCosimInterface base class holding a Monitor class instance."""

    monitor_id = Int(label="Monitor indice",
                     doc="Indice of monitor to get data from",
                     required=True,
                     default=0)

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of TVB proxy nodes",
        doc="""Indices of TVB proxy nodes""",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc="""Indices of model's variables of interest (VOI) that"
             "should couple to the other cosimulator.""",
        required=True)

    sender = Attr(
        label="TVBSender",
        field_type=TVBSender,
        doc="""A TVBSender class to send TVB data to the cosimulator.""",
        required=True
    )

    number_of_proxy_nodes = 0
    n_voi = 0

    def configure(self):
        """Method to configure the CosimMonitor of the interface
           and compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.sender.configure()
        self.number_of_proxy_nodes = self.proxy_inds.shape[0]
        self.n_voi = self.voi.shape[0]
        super(TVBtoCosimInterfaces).configure()

    def __call__(self, data):
        return self.sender([np.array([data[0][0], data[0][-1]]),        # time_steps[0], time_steps[-1]
                            data[1][:, self.voi, self.proxy_inds, :]])  # values


class TVBtoCosimInterfaces(HasTraits):

    """This class holds a list of TVB to cosimulator interfaces and sends data to them"""

    interfaces = List(of=TVBtoCosimInterface)

    number_of_interfaces = 0

    @property
    def voi(self):
        return [interfaces.voi for interfaces in self.interfaces]

    @property
    def proxy_inds(self):
        return [interfaces.proxy_inds for interfaces in self.interfaces]

    def configure(self):
        for interface in self.interfaces:
            interface.configure()
        self.number_of_interfaces = len(self.interfaces)
        super(TVBtoCosimInterfaces, self).configure()

    def __call__(self, data):
        for interface in self.interfaces:
            interface(data[interface.monitor_id])
