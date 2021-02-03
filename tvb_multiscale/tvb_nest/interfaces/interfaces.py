# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.tvb.interfaces import \
    TVBtoSpikeNetInterface, SpikeNetToTVBInterface, TVBOutputInterfaces, TVBInputInterfaces
from tvb_multiscale.core.interfaces.spikeNet_interfaces import \
    SpikeNetInterfaces, SpikeNetOutgoingInterface, SpikeNetIngoingInterface

from tvb_multiscale.tvb_nest.interfaces.io import \
    NESTInputDeviceSetter, NESTEventsFromOutpuDevice, NESTEventsReaderFromRecorderFile


class NESTInterface(HasTraits):
    __metaclass__ = ABCMeta

    """NESTInterface base class for interfaces sending/receiving data from/to NEST."""

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
    @abstractmethod
    def nest_instance(self):
        pass

    def print_str(self):
        return "\nNEST proxy nodes' gids:\n%s" % extract_integer_intervals(self.proxy_gids, print=True)


class NESTOutputInterface(SpikeNetOutgoingInterface, NESTInterface):

    """NESTOutputInterface base class for interfaces sending data from NEST."""

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    def print_str(self):
        SpikeNetOutgoingInterface.print_str(self) + NESTInterface.print_str(self)


class NESTInputInterface(SpikeNetIngoingInterface, NESTInterface):

    """NESTInputInterface base class for interfaces receiving data to NEST."""

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance

    def print_str(self):
        SpikeNetIngoingInterface.print_str(self) + NESTInterface.print_str(self)


class TVBtoNESTInterface(TVBtoSpikeNetInterface, NESTOutputInterface):

    """TVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memmory.
    """

    communicator = Attr(
        label="Communicator directly to NEST",
        field_type=NESTInputDeviceSetter,
        doc="""A NESTInputDeviceSetter Communicator class instance to send data to NEST.""",
        required=True
    )

    def print_str(self):
        TVBtoSpikeNetInterface.print_str(self) + NESTOutputInterface.print_str(self)


class NESTtoTVBInterface(SpikeNetToTVBInterface, NESTInputInterface):

    """NESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    communicator = Attr(
        label="Communicator directly from NEST",
        field_type=(NESTEventsFromOutpuDevice, NESTEventsReaderFromRecorderFile),
        doc="""A NESTEventsFromOutpuDevice or NESTEventsReaderFromRecorderFile Communicator class instance 
               to receive events' data from NEST.""",
        required=True
    )

    def print_str(self):
        SpikeNetToTVBInterface.print_str(self) + NESTInputInterface.print_str(self)


class NESTInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """NESTInterfaces class holding a list of NESTInterface instances"""

    @property
    def proxy_gids(self):
        return np.sort(self._loop_get_from_interfaces("proxy_gids"))

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    @property
    @abstractmethod
    def nest_instance(self):
        pass


class NESTOutputInterfaces(SpikeNetInterfaces, NESTInterfaces):

    """NESTOutputInterfaces holding a list of NESTOutputInterface instances"""

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance


class NESTInputInterfaces(SpikeNetInterfaces, NESTInterfaces):

    """NESTInputInterfaces holding a list of NESTInputInterface instances"""

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance


class TVBtoNESTInterfaces(TVBOutputInterfaces, NESTInputInterfaces):

    """TVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    pass


class NESTtoTVBInterfaces(TVBInputInterfaces, NESTOutputInterfaces):
    """NESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    pass
