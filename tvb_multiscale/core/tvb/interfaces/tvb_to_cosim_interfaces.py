# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Int, Attr, NArray, List

from tvb_multiscale.core.tvb.interfaces.base import TVBInterface, TVBInterfaces
from tvb_multiscale.core.tvb.io.io import TVBSender


class TVBtoCosimInterface(TVBInterface):

    """Class to send data from TVB to the other cosimulator."""

    monitor_ind = Int(label="Monitor indice",
                      doc="Indice of monitor to get data from",
                      required=True,
                      default=0)

    sender = Attr(
        label="TVBSender",
        field_type=TVBSender,
        doc="""A TVBSender class to send TVB data to the cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the TVBtoCosimInterface interface"""
        self.sender.configure()
        super(TVBtoCosimInterface, self).configure()

    def set_local_voi_indices(self, monitor_voi):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor"""
        self.voi_loc = super(TVBtoCosimInterface, self).set_local_indices(self.voi, monitor_voi)

    def __call__(self, data):
        return self.sender([np.array([data[0][0], data[0][-1]]),            # time_steps[0], time_steps[-1]
                            data[1][:, self.voi_loc, self.proxy_inds, :]])  # values (voi_loc indices needed here,
                                                                            # specific to the attached monitor)


class TVBtoCosimInterfaces(TVBInterfaces):

    """This class holds a list of TVB to cosimulator interfaces and sends data to them"""

    interfaces = List(of=TVBtoCosimInterface)

    def set_local_voi_indices(self, cosim_monitors):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor,
           for each interface"""
        for interface in self.interfaces:
            interface.set_local_voi_indices(cosim_monitors[interface.monitor_ind])

    def __call__(self, data):
        for interface in self.interfaces:
            interface(data[interface.monitor_ind])
