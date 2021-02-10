# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.interfaces import \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces
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


class SpikeNetIngoingInterface(SpikeNetInterface):

    """SpikeNetIngoingInterface base class."""

    @property
    def label(self):
        return "%s: %s (%s) <-" % (self.__class__.__name__, str(self.populations),
                                   extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        super(SpikeNetIngoingInterface, self).print_str(self, sender_not_receiver=False)


class SpikeNetSenderInterface(SenderInterface, SpikeNetOutgoingInterface):

    """SpikeNetSenderInterface class."""

    def __call__(self, data):
        return SenderInterface.__call__(self, data)

    def print_str(self):
        return SenderInterface.print_str(self) + SpikeNetOutgoingInterface.print_str(self)


class SpikeNetReceiverInterface(ReceiverInterface, SpikeNetIngoingInterface):

    """SpikeNetReceiverInterface class."""

    def __call__(self):
        return ReceiverInterface.__call__(self)

    def print_str(self):
        return ReceiverInterface.print_str(self) + SpikeNetIngoingInterface.print_str(self)


class SpikeNetTransformerSenderInterface(TransformerSenderInterface, SpikeNetOutgoingInterface):

    """SpikeNetTransformerSenderInterface class."""

    def __call__(self, data):
        return SpikeNetTransformerSenderInterface.__call__(self, data)

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + SpikeNetOutgoingInterface.print_str(self)


class SpikeNetReceiverTransformerInterface(ReceiverTransformerInterface, SpikeNetIngoingInterface):

    """SpikeNetReceiverTransformerInterface class."""

    def __call__(self):
        return ReceiverTransformerInterface.__call__(self)

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
        return np.sort(self._loop_get_from_interfaces("populations"))

    @property
    def populations_unique(self):
        return np.unique(self._loop_get_from_interfaces("populations"))

    @property
    def proxy_inds(self):
        return np.sort(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def number_of_populations(self):
        return self.populations_unique.shape[0]

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]


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
