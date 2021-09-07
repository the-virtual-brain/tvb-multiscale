# -*- coding: utf-8 -*-

from abc import ABCMeta

import ray

from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder


def RunLock(cls, time, ref_objs=[]):
    if len(ref_objs):
        ray.get(ref_objs)
    return cls.Run(time)


class RaySpikingNetworkBuilder(SpikingNetworkBuilder):

    __metaclass__ = ABCMeta

    pass


def create_ray_spiking_network_builder_type(spiking_network_builder=SpikingNetworkBuilder,
                                            ray_builder=RaySpikingNetworkBuilder):

    class ThisRaySpikingNetworkBuilder(spiking_network_builder, ray_builder):

        def build_spiking_network(self):
            return ray_builder.build_spiking_network(self)

    ThisRaySpikingNetworkBuilder.__name__ = "Ray%s" % spiking_network_builder.__name__

    return ThisRaySpikingNetworkBuilder
