# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.interfaces.spikeNet_interfaces import SpikeNetInterfaces, \
    SpikeNetOutgoingInterface, SpikeNetIngoingInterface, \
    SenderInterface, ReceiverInterface, \
    TransformerSenderInterface, ReceiverTransformerInterface


class NESTInterface(HasTraits):

    proxy_gids = NArray(
        dtype=np.int,
        label="NEST proxy nodes gids",
        doc="""NEST proxy nodes gids""",
        required=True,
    )

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    def print_str(self):
        return "\nNEST proxy nodes' gids:\n%s" % extract_integer_intervals(self.proxy_gids, print=True)


class NESTOutgoingInterface(SpikeNetOutgoingInterface, NESTInterface):

    """NESTOutgoingInterface base class."""

    def print_str(self):
        SpikeNetOutgoingInterface.print_str(self) + NESTInterface.print_str(self)


class NESTIngoingInterface(SpikeNetIngoingInterface, NESTInterface):

    """NESTIngoingInterface base class."""

    def print_str(self):
        SpikeNetIngoingInterface.print_str(self) + NESTInterface.print_str(self)


class NESTSenderInterface(SenderInterface, NESTOutgoingInterface):

    """NESTSenderInterface class."""

    def __call__(self, data):
        return NESTSenderInterface.__call__(self, data)

    def print_str(self):
        return SenderInterface.print_str(self) + NESTIngoingInterface.print_str(self)


class NESTReceiverInterface(ReceiverInterface, NESTIngoingInterface):

    """NESTReceiverInterface class."""

    def __call__(self):
        return NESTReceiverInterface.__call__(self)

    def print_str(self):
        return ReceiverInterface.print_str(self) + NESTIngoingInterface.print_str(self)


class NESTTransformerSenderInterface(TransformerSenderInterface, NESTOutgoingInterface):

    """NESTTransformerSenderInterface class."""

    def __call__(self, data):
        return NESTTransformerSenderInterface.__call__(self, data)

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + NESTOutgoingInterface.print_str(self)


class NESTReceiverTransformerInterface(ReceiverTransformerInterface, NESTIngoingInterface):

    """NESTReceiverTransformerInterface class."""

    def __call__(self):
        return ReceiverTransformerInterface.__call__(self)

    def print_str(self):
        return ReceiverTransformerInterface.print_str(self) + NESTIngoingInterface.print_str(self)


class NESTInterfaces(HasTraits):

    """NESTInterfaces class"""

    @property
    def proxy_gids(self):
        return np.sort(self._loop_get_from_interfaces("proxy_gids"))

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]


class NESTOutgoingInterfaces(SpikeNetInterfaces, NESTInterfaces):

    """NESTOutgoingInterfaces"""

    pass


class NESTIngoingInterfaces(SpikeNetInterfaces, NESTInterfaces):
    __metaclass__ = ABCMeta

    """NESTIngoingInterfaces"""

    pass