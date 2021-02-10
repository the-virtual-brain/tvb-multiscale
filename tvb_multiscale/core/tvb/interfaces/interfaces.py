# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, Int, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.io import SpikeNetInputDevice, SpikeNetEventsFromOutpuDevice
from tvb_multiscale.core.interfaces.interfaces import \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.spikeNet_interfaces import \
    SpikeNetIngoingInterface, SpikeNetOutgoingInterface, SpikeNetOutgoingInterfaces, SpikeNetIngoingInterfaces


class TVBInterface(HasTraits):
    __metaclass__ = ABCMeta

    """TVBInterface base class for interfaces sending/receivng data from/for TVB to/from a transformer of cosimulator"""

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of TVB proxy nodes",
        doc="""Indices of TVB proxy nodes""",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc="""Indices of model's variables of interest (VOI)""",
        required=True)

    voi_loc = np.array([])

    voi_labels = NArray(
        dtype=np.str,
        label="Cosimulation model state variables' labels",
        doc=""""Labels of model's variables of interest (VOI)""",
        required=True,
    )

    @property
    def label(self):
        return "%s: %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                extract_integer_intervals(self.proxy_inds))

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


class TVBOutputInterface(TVBInterface):

    """TVBOutputInterface base class for interfaces sending data from TVB to a transformer or cosimulator"""

    monitor_ind = Int(label="Monitor indice",
                      doc="Indice of monitor to get data from",
                      required=True,
                      default=0)

    @property
    def label(self):
        return "%s: %s (%s) ->" % (self.__class__.__name__, str(self.voi_labels),
                                   extract_integer_intervals(self.proxy_inds))

    def set_local_indices(self, monitor_voi):
        self.set_local_voi_indices(monitor_voi)

    def __call__(self, data):
        return [np.array([data[self.monitor_ind][0][0],     # start input_time step
                          data[self.monitor_ind][0][-1]]),  # end input_time step
                # values (voi_loc indices needed here, specific to the attached monitor)
                data[self.monitor_ind][1][:, self.voi_loc, self.proxy_inds, :]]

    def print_str(self):
        super(TVBOutputInterface, self).print_str(self, sender_not_receiver=True)


class TVBInputInterface(TVBInterface):

    """TVBInputInterface base class for interfaces receiving data for TVB from a transformer or cosimulator"""

    proxy_inds_loc = np.array([])

    @property
    def label(self):
        return "%s: %s (%s) <-" % (self.__class__.__name__, str(self.voi_labels),
                                   extract_integer_intervals(self.proxy_inds))

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        self.set_local_voi_indices(simulator_voi)
        self.proxy_inds_loc = self._set_local_indices(self.proxy_inds, simulator_proxy_inds)

    def print_str(self):
        super(TVBInputInterface, self).print_str(self, sender_not_receiver=False)


class TVBSenderInterface(SenderInterface, TVBOutputInterface):

    """TVBSenderInterface class to send data to a remote transformer or cosimulator.
    """

    def __call__(self, data):
        return SenderInterface.__call__(self, TVBOutputInterface.__call__(self, data))

    def print_str(self):
        return SenderInterface.print_str(self) + TVBOutputInterface.print_str(self)


class TVBReceiverInterface(ReceiverInterface, TVBInputInterface):

    """TVBReceiverInterface class to receive data for TVB from a remote transformer or cosimulator.
    """

    def __call__(self):
        return ReceiverInterface.__call__(self)

    def print_str(self):
        return ReceiverInterface.print_str(self) + TVBInputInterface.print_str(self)


class TVBTransformerSenderInterface(TransformerSenderInterface, TVBOutputInterface):

    """TVBTransformerSenderInterface class to get data from TVB, transform them locally,
       and, then, send them to a -potentially remote- cosimulator.
    """

    def __call__(self, data):
        return TVBTransformerSenderInterface.__call__(self, TVBOutputInterface.__call__(self, data))

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + TVBOutputInterface.print_str(self)


class TVBReceiverTransformerInterface(ReceiverTransformerInterface, TVBInputInterface):

    """TVBReceiverTransformerInterface class receive data from a -potentially remote- cosimulator,
       and, then, transform them and set them to TVB locally.
    """

    def __call__(self):
        return ReceiverTransformerInterface.__call__(self)

    def print_str(self):
        return ReceiverTransformerInterface.print_str(self) + \
               TVBInputInterface.print_str(self)


class TVBtoSpikeNetInterface(TVBTransformerSenderInterface, SpikeNetIngoingInterface):

    """TVBtoSpikeNetInterface class to get data from TVB, transform them,
       and finally set them to the Spiking Network cosimulator, all processes taking place in shared memmory.
    """

    communicator = Attr(
        label="Communicator directly to SpikeNet cosimulator",
        field_type=SpikeNetInputDevice,
        doc="""A SpikeNetInputDevice Communicator class instance 
              to send data to the Spiking Network co-simulator.""",
        required=True
    )

    @property
    def label(self):
        return "%s: %s (%s) -> %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                           extract_integer_intervals(self.proxy_inds),
                                           str(self.populations), extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        return TVBTransformerSenderInterface.print_str(self) + \
               SpikeNetIngoingInterface.print_str(self)


class SpikeNetToTVBInterface(TVBReceiverTransformerInterface, SpikeNetOutgoingInterface):

    """SpikeNetToTVBInterface class to get data the Spiking Network co-simulator, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    communicator = Attr(
        label="Communicator directly from SpikeNet cosimulator",
        field_type=SpikeNetEventsFromOutpuDevice,
        doc="""A SpikeNetEventsFromOutpuDevice Communicator class instance 
               to receive events' data from the Spiking Network co-simulator.""",
        required=True
    )

    @property
    def label(self):
        return "%s: %s (%s) <- %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                           extract_integer_intervals(self.proxy_inds),
                                           str(self.populations), extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        return TVBTransformerSenderInterface.print_str(self) + \
               SpikeNetOutgoingInterface.print_str(self)


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
    def voi_labels(self):
        return np.sort(self._loop_get_from_interfaces("voi_labels"))

    @property
    def voi_labels_unique(self):
        return np.unique(self._loop_get_from_interfaces("voi_labels"))

    @property
    def proxy_inds(self):
        return np.sort(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def n_vois(self):
        return self.voi_unique.shape[0]

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    @abstractmethod
    def set_local_indices(self, *args):
        pass


class TVBOutputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBOutputInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and sends data to them"""

    def set_local_indices(self, cosim_monitors):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(cosim_monitors[interface.monitor_ind])

    def __call__(self, data):
        for interface in self.interfaces:
            #                            start_time_step                 end_time_step
            interface([np.array([data[interface.monitor_ind][0][0], data[interface.monitor_ind][0][-1]]).astype("i"),
                       data[interface.monitor_ind][1][:, :, :, 0]])  # data values !!! assuming only 1 mode!!!


class TVBInputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBInputInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them"""

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and spiking_proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSimulators' vois and spiking_proxy_inds,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(simulator_voi, simulator_proxy_inds)

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates = np.empty(good_cosim_update_values_shape).astype(float)
        cosim_updates[:] = np.NAN
        for interface in self.interfaces:
            data = interface()  # [start_and_time_steps, values]
            # Convert start and end input_time step to a vector of integer input_time steps:
            time_steps = np.arange(data[0][0], data[0][1] + 1).astype("i")
            cosim_updates[
                (time_steps % good_cosim_update_values_shape[0])[:, None, None],
                interface.voi_loc[None, :, None],         # indices specific to cosim_updates needed here
                interface.proxy_inds_loc[None, None, :],  # indices specific to cosim_updates needed here
                0] = np.copy(data[1])                     # !!! assuming only 1 mode!!!


class TVBtoSpikeNetInterfaces(TVBOutputInterfaces, SpikeNetIngoingInterfaces):

    """TVBtoSpikeNetInterfaces"""

    pass


class SpikeNetToTVBInterfaces(TVBInputInterfaces, SpikeNetOutgoingInterfaces):

    """SpikeNetToTVBInterfaces"""

    pass
