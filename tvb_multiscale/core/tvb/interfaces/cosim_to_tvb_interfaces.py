# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List

from tvb_multiscale.core.tvb.io.io import TVBReceiver


class CosimToTVBInterface(TVBInterface):

    """Class to get update data for TVB states from co-simulator"""

    receiver = Attr(
        label="TVBReceiver",
        field_type=TVBReceiver,
        doc="""A TVBReceiver class to receive TVB data from the cosimulator.""",
        required=True
    )

    proxy_inds_loc = np.array([])

    def configure(self, simulator):
        """Method to configure CosimToTVBInterface interface"""
        self.receiver.configure()
        super(CosimToTVBInterface, self).configure()

    def set_local_voi_and_proxy_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSmulators' vois and proxy_inds"""
        self.voi_loc = super(CosimToTVBInterface, self).set_local_indices(self.voi, simulator_voi)
        self.proxy_inds_loc = super(CosimToTVBInterface, self).set_local_indices(self.proxy_inds, simulator_proxy_inds)

    def __call__(self):
        return self.receiver()


class CosimToTVBInterfaces(HasTraits):

    """This class holds a list of cosimulator to TVB interfaces and receives data from them"""

    interfaces = List(of=CosimToTVBInterface)

    def set_local_voi_and_proxy_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSmulators' vois and proxy_inds,
           for each interface"""
        for interface in self.interfaces:
            interface.set_local_voi_and_proxy_indices(simulator_voi, simulator_proxy_inds)

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates = np.empty(good_cosim_update_values_shape).astype(float)
        cosim_updates[:] = np.NAN
        for interface in self.interfaces:
            data = interface()  # [time_steps, values]
            cosim_updates[
                (data[0] % good_cosim_update_values_shape[0])[:, None, None, None],
                interface.voi_loc[None, :, None, None],        # indices specific to cosim_updates needed here
                interface.proxy_inds_loc[None, None, :, None], # indices specific to cosim_updates needed here
                np.arange(good_cosim_update_values_shape[3])[None, None, None, :]] = data[1]
