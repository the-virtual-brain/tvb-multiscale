# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Int, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.interfaces import \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces


class TVBInterface(HasTraits):
    __metaclass__ = ABCMeta

    """TVBInterface base class."""

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

    def _set_local_indices(self, inds, simulator_inds):
        simulator_inds_list = list(simulator_inds)
        return np.array([simulator_inds_list.index(ind) for ind in inds])

    def set_local_voi_indices(self, monitor_voi):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor or CosimHistory"""
        self.voi_loc = self._set_local_indices(self.voi, monitor_voi)

    @abstractmethod
    def set_local_indices(self, *args):
        pass

    def print_str(self, sender_not_receiver=None):
        if sender_not_receiver is True:
            tvb_source_or_target = "Sender "
        elif sender_not_receiver is False:
            tvb_source_or_target = "Receiver "
        else:
            tvb_source_or_target = ""
        return "\nTVB state variable indices: %s" \
               "\n%sTVB proxy nodes' indices: %s" % \
               (str(self.vois.tolist()),
                tvb_source_or_target, extract_integer_intervals(self.proxy_inds, print=True))


class TVBOutgoingInterface(TVBInterface):

    """SpikeNetOutgoingInterface base class."""

    monitor_ind = Int(label="Monitor indice",
                      doc="Indice of monitor to get data from",
                      required=True,
                      default=0)

    def set_local_indices(self, monitor_voi):
        self.set_local_voi_indices(monitor_voi)

    def __call__(self, data):
        return [np.array([data[self.monitor_ind][0][0],     # start time step
                          data[self.monitor_ind][0][-1]]),  # end time step
                # values (voi_loc indices needed here, specific to the attached monitor)
                data[self.monitor_ind][1][:, self.voi_loc, self.proxy_inds, :]]

    def print_str(self):
        super(TVBOutgoingInterface, self).print_str(self, sender_not_receiver=True)


class TVBIngoingInterface(TVBInterface):

    """SpikeNetIngoingInterface base class."""

    proxy_inds_loc = np.array([])

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        self.set_local_voi_indices(simulator_voi)
        self.proxy_inds_loc = self._set_local_indices(self.proxy_inds, simulator_proxy_inds)

    def print_str(self):
        super(TVBIngoingInterface, self).print_str(self, sender_not_receiver=False)


class TVBSenderInterface(SenderInterface, TVBOutgoingInterface):

    """SpikeNetSenderInterface class."""

    def __call__(self, data):
        return SenderInterface.__call__(self, TVBOutgoingInterface.__call__(self, data))

    def print_str(self):
        return SenderInterface.print_str(self) + TVBOutgoingInterface.print_str(self)


class TVBReceiverInterface(ReceiverInterface, TVBIngoingInterface):

    """SpikeNetReceiverInterface class."""

    def __call__(self):
        return ReceiverInterface.__call__(self)

    def print_str(self):
        return ReceiverInterface.print_str(self) + TVBIngoingInterface.print_str(self)


class TVBTransformerSenderInterface(TransformerSenderInterface, TVBOutgoingInterface):

    """SpikeNetTransformerSenderInterface class."""

    def __call__(self, data):
        return TVBTransformerSenderInterface.__call__(self, TVBOutgoingInterface.__call__(self, data))

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + TVBOutgoingInterface.print_str(self)


class TVBReceiverTransformerInterface(ReceiverTransformerInterface, TVBIngoingInterface):

    """SpikeNetReceiverTransformerInterface class."""

    def __call__(self):
        return ReceiverTransformerInterface.__call__(self)

    def print_str(self):
        return ReceiverTransformerInterface.print_str(self) + \
               TVBIngoingInterface.print_str(self)


class TVBInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """SpikeNetInterfaces abstract base class"""

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

    @abstractmethod
    def set_local_indices(self, *args):
        pass


class TVBOutgoingInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBOutgoingInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and sends data to them"""

    def set_local_indices(self, cosim_monitors):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(cosim_monitors[interface.monitor_ind])

    def __call__(self, data):
        for interface in self.interfaces:
            interface(data[interface.monitor_ind])


class TVBIngoingInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBIngoingInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them"""

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSmulators' vois and proxy_inds,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(simulator_voi, simulator_proxy_inds)

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates = np.empty(good_cosim_update_values_shape).astype(float)
        cosim_updates[:] = np.NAN
        for interface in self.interfaces:
            data = interface()  # [time_steps, values]
            # Convert start and end time step to a vector of integer time steps:
            data[0] = np.arange(data[0][0], data[0][1] + 1).astype("i")
            cosim_updates[
                (data[0] % good_cosim_update_values_shape[0])[:, None, None, None],
                interface.voi_loc[None, :, None, None],         # indices specific to cosim_updates needed here
                interface.proxy_inds_loc[None, None, :, None],  # indices specific to cosim_updates needed here
                np.arange(good_cosim_update_values_shape[3])[None, None, None, :]] = data[1]
