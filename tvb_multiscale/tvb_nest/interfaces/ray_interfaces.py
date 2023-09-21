# -*- coding: utf-8 -*-

from tvb_multiscale.core.interfaces.tvb.ray_interfaces import \
    RayTVBtoSpikeNetInterface, RaySpikeNetToTVBInterface, RayTVBOutputInterfaces, RayTVBInputInterfaces
from tvb_multiscale.tvb_nest.interfaces.interfaces import TVBtoNESTInterface, NESTtoTVBInterface, \
    NESTOutputInterface, NESTInputInterface, NESTOutputInterfaces, NESTInputInterfaces


class TVBtoNESTinRayInterface(TVBtoNESTInterface):

    pass


class NESTinRayToTVBInterface(NESTtoTVBInterface):

    pass


class RayTVBtoNESTInterface(RayTVBtoSpikeNetInterface, NESTInputInterface):

    """RayTVBtoNESTInterface class to get data from TVB, transform them,
       and finally set them to NEST, all processes taking place in shared memory.
    """

    pass


class RayNESTtoTVBInterface(RaySpikeNetToTVBInterface, NESTOutputInterface):

    """RayNESTtoTVBInterface class to get data from NEST, transform them,
       and finally set them to TVB, all processes taking place in shared memory.
    """

    pass


class RayTVBtoNESTInterfaces(RayTVBOutputInterfaces, NESTInputInterfaces):

    """RayTVBtoNESTInterfaces class holding a list of TVBtoNESTInterface instances"""

    pass


class RayNESTtoTVBInterfaces(RayTVBInputInterfaces, NESTOutputInterfaces):
    """RayNESTtoTVBInterfaces class holding a list of NESTtoTVBInterface instances"""

    pass
