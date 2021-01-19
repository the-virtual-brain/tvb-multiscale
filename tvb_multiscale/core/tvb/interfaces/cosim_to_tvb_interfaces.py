# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List

from tvb_multiscale.core.tvb.io.io import TVBReceiver


class CosimToTVBInterface(HasTraits):

    """Base class to get update TVB states from data from co-simulator"""

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of TVB proxy nodes",
        doc="""Indices of TVB proxy nodes""",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc=("Indices of model's variables of interest (VOI) that"
             "should be updated (i.e., overwriten) during cosimulation."),
        required=False)

    receiver = Attr(
        label="TVBReceiver",
        field_type=TVBReceiver,
        doc="""A TVBReceiver class to receive TVB data from the cosimulator.""",
        required=True
    )

    number_of_proxy_nodes = 0
    n_voi = 0

    def configure(self, simulator):
        """Method to configure CosimUpdate
           the variables of interest,
           and the number of proxy nodes"""
        self.receiver.configure()
        self.number_of_proxy_nodes = self.proxy_inds.shape[0]
        self.n_voi = self.voi.shape[0]
        super(CosimToTVBInterface, self).configure()

    def __call__(self):
        return self.receiver()


class CosimToTVBInterfaces(HasTraits):

    """This class holds a list of cosimulator to TVB interfaces and receives data from them"""

    interfaces = List(of=CosimToTVBInterface)

    number_of_interfaces = 0

    @property
    def vois(self):
        return [interface.voi for interface in self._interfaces]

    @property
    def proxy_inds(self):
        return [interface.proxy_inds for interface in self._interfaces]

    def configure(self, simulator):
        for interfaces in self.interfacess:
            interfaces.configure(simulator)
        self.number_of_interfaces = len(self.interfaces)
        super(CosimToTVBInterfaces, self).configure()

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates = np.empty(good_cosim_update_values_shape)
        for interface in self.interfaces:
            data = interface()  # [time_steps, values]
            cosim_updates[
                (data[0] % good_cosim_update_values_shape[0])[:, None, None, None],
                interface.vois[None, :, None, None],
                interface.proxy_inds[None, None, :, None],
                np.arange(good_cosim_update_values_shape[3])[None, None, None, :]] = data[1]
