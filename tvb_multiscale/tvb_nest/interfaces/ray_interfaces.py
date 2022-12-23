# -*- coding: utf-8 -*-

from tvb_multiscale.core.interfaces.tvb.ray_interfaces import \
    RayTVBtoSpikeNetInterface, RaySpikeNetToTVBInterface, RayTVBOutputInterfaces, RayTVBInputInterfaces
from tvb_multiscale.tvb_nest.interfaces.interfaces import \
    NESTOutputInterface, NESTInputInterface, NESTOutputInterfaces, NESTInputInterfaces


class RayTVBtoNESTInterface(RayTVBtoSpikeNetInterface, NESTInputInterface):

    """TVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memmory.
    """

    pass


class RayNESTtoTVBInterface(RaySpikeNetToTVBInterface, NESTOutputInterface):

    """NESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    pass


class RayTVBtoNESTInterfaces(RayTVBOutputInterfaces, NESTInputInterfaces):

    """RayTVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    pass


class RayNESTtoTVBInterfaces(RayTVBInputInterfaces, NESTOutputInterfaces):
    """RayNESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    pass
