# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.base import \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDevice, SpikeNetEventsFromOutpuDevice
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class SpikeNetInterface(HasTraits):

    """SpikeNetInterface abstract base class."""

    spiking_network = Attr(label="Spiking Network",
                           doc="""The instance of SpikingNetwork class""",
                           field_type=SpikingNetwork,
                           required=True)

    spiking_proxy_inds = NArray(
        dtype=np.int,
        label="Indices of Spiking Network proxy nodes",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
    )

    populations = NArray(
        dtype=str,
        label="Spiking Network populations",
        doc="""Spiking Network populations associated to the interface""",
        required=True)

    spikeNet_receiver_proxy = Attr(label="Spiking network receiver proxy",
                                   doc="""An instance of SpikeNetInputDevice 
                                          implementing a proxy node receiving inputs from the co-simulator 
                                          as an input to the spiking network""",
                                   field_type=SpikeNetInputDevice,
                                   required=True)

    spikeNet_sender_proxy = Attr(label="Spiking network sender proxy",
                                 doc="""An instance of SpikeNetEventsFromOutpuDevice 
                                        implementing a proxy node sending outputs from the spiking network
                                        to the co-simulator""",
                                 field_type=SpikeNetEventsFromOutpuDevice,
                                 required=True)

    @property
    def label(self):
        return "%s: %s (%s)" % (self.__class__.__name__, str(self.populations),
                                extract_integer_intervals(self.spiking_proxy_inds))

    @property
    def number_of_proxy_nodes(self):
        return self.spiking_proxy_inds.shape[0]

    @property
    def number_of_populations(self):
        return self.populations.shape[0]

    def configure(self):
        super(SpikeNetInterface, self).configure()
        self.spikeNet_receiver_proxy.configure()
        self.spikeNet_sender_proxy.configure()

    def print_str(self, sender_not_receiver=None):
        if sender_not_receiver is True:
            spikeNet_source_or_target = "Sender "
        elif sender_not_receiver is False:
            spikeNet_source_or_target = "Receiver "
        else:
            spikeNet_source_or_target = ""
        return "\nSpiking Network populations: %s" \
               "\n%Spiking Network proxy nodes' indices: %s" % \
               (str(self.populations.tolist()),
                spikeNet_source_or_target, extract_integer_intervals(self.spiking_proxy_inds, print=True))


class SpikeNetOutgoingInterface(SpikeNetInterface):

    """SpikeNetOutgoingInterface base class."""

    @property
    def label(self):
        return "%s: %s (%s) ->" % (self.__class__.__name__, str(self.populations),
                                   extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        super(SpikeNetOutgoingInterface, self).print_str(self, sender_not_receiver=True)

    def __call__(self):
        return self.spikeNet_sender_proxy()


class SpikeNetIngoingInterface(SpikeNetInterface):

    """SpikeNetIngoingInterface base class."""

    @property
    def label(self):
        return "%s: %s (%s)" % (self.__class__.__name__, str(self.populations),
                                        extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        super(SpikeNetIngoingInterface, self).print_str(self, sender_not_receiver=False)

    def __call__(self, data):
        return self.spikeNet_receiver_proxy(data)


class SpikeNetSenderInterface(SenderInterface, SpikeNetOutgoingInterface):

    """SpikeNetSenderInterface class."""

    def configure(self):
        SpikeNetOutgoingInterface.configure(self)
        SenderInterface.configure(self)

    def __call__(self):
        return SenderInterface.__call__(self, SpikeNetOutgoingInterface(self))

    def print_str(self):
        return SenderInterface.print_str(self) + SpikeNetOutgoingInterface.print_str(self)


class SpikeNetReceiverInterface(ReceiverInterface, SpikeNetIngoingInterface):

    """SpikeNetReceiverInterface class."""

    def configure(self):
        SpikeNetIngoingInterface.configure(self)
        ReceiverInterface.configure(self)

    def __call__(self):
        return SpikeNetIngoingInterface(self, ReceiverInterface.__call__(self))

    def print_str(self):
        return ReceiverInterface.print_str(self) + SpikeNetIngoingInterface.print_str(self)


class SpikeNetTransformerSenderInterface(TransformerSenderInterface, SpikeNetOutgoingInterface):

    """SpikeNetTransformerSenderInterface class."""

    def configure(self):
        SpikeNetOutgoingInterface.configure(self)
        TransformerSenderInterface.configure(self)

    def __call__(self, data):
        return SpikeNetTransformerSenderInterface.__call__(self, self.SpikeNetOutgoingInterface())

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + SpikeNetOutgoingInterface.print_str(self)


class SpikeNetReceiverTransformerInterface(ReceiverTransformerInterface, SpikeNetIngoingInterface):

    """SpikeNetReceiverTransformerInterface class."""

    def configure(self):
        SpikeNetIngoingInterface.configure(self)
        ReceiverTransformerInterface.configure(self)

    def __call__(self):
        return SpikeNetIngoingInterface(self, ReceiverTransformerInterface.__call__(self))

    def print_str(self):
        return ReceiverTransformerInterface.print_str(self) + SpikeNetIngoingInterface.print_str(self)


class SpikeNetInterfaces(HasTraits):

    """SpikeNetInterfaces abstract base class"""

    interfaces = List(of=SpikeNetInterface)

    @property
    def spiking_network(self):
        if len(self.interfaces):
            return self.interfaces[0].spiking_network
        else:
            return None

    @property
    def populations(self):
        return self._loop_get_from_interfaces("populations")

    @property
    def populations_unique(self):
        return np.unique(self._loop_get_from_interfaces("populations"))

    @property
    def spiking_proxy_inds(self):
        return self._loop_get_from_interfaces("spiking_proxy_inds")

    @property
    def spiking_proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def number_of_populations(self):
        return self.populations_unique.shape[0]

    @property
    def number_of_spiking_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    @property
    def spiking_proxy_receivers(self):
        return self._loop_get_from_interfaces("spiking_proxy_receiver")

    @property
    def spiking_proxy_senders(self):
        return self._loop_get_from_interfaces("spiking_proxy_sender")


class SpikeNetOutgoingInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetOutgoingInterfaces"""

    pass


class SpikeNetOutgoingRemoteInterfaces(SpikeNetOutgoingInterfaces):

    """SpikeNetOutgoingRemoteInterfaces"""

    def __call__(self):
        for interface in self.interfaces:
            interface()


class SpikeNetIngoingInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetIngoingInterfaces"""

    pass


class SpikeNetIngoingRemoteInterfaces(SpikeNetIngoingInterfaces):

    """SpikeNetIngoingRemoteInterfaces"""

    def __call__(self, *args):
        for interface in self.interfaces:
            interface(*args)
