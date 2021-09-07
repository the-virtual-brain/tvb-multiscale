# -*- coding: utf-8 -*-

from tvb_multiscale.core.interfaces.tvb.ray_interfaces import \
    RayTVBtoSpikeNetInterface, RaySpikeNetToTVBInterface, RayTVBOutputInterfaces, RayTVBInputInterfaces
from tvb_multiscale.tvb_nest.interfaces.interfaces import \
    NESTOutputInterface, NESTInputInterface, NESTOutputInterfaces, NESTInputInterfaces

from tvb.basic.neotraits.api import List


class RayTVBtoNESTInterface(RayTVBtoSpikeNetInterface, NESTInputInterface):

    """TVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memmory.
    """

    # def __init__(self, spiking_network=None, **kwargs):
    #     if spiking_network:
    #         self.spiking_network = spiking_network
    #     super().__init__(**kwargs)

    def print_str(self):
        return RayTVBtoSpikeNetInterface.print_str(self) + NESTInputInterface.print_str(self)


class RayNESTtoTVBInterface(RaySpikeNetToTVBInterface, NESTOutputInterface):

    """NESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    # def __init__(self, spiking_network=None, **kwargs):
    #     if spiking_network:
    #         self.spiking_network = spiking_network
    #     super().__init__(**kwargs)

    def print_str(self):
        return RaySpikeNetToTVBInterface.print_str(self) + NESTOutputInterface.print_str(self)


class RayTVBtoNESTInterfaces(RayTVBOutputInterfaces, NESTInputInterfaces):

    """RayTVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    pass


class RayNESTtoTVBInterfaces(RayTVBInputInterfaces, NESTOutputInterfaces):
    """RayNESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    pass
