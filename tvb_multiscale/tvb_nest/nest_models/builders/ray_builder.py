# -*- coding: utf-8 -*-

from types import MethodType

from tvb_multiscale.core.ray.client import create_ray_client
from tvb_multiscale.core.spiking_models.builders.ray_builder import \
    RunLock, RaySpikingNetworkBuilder, create_ray_spiking_network_builder_type
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder


class RayNESTNetworkBuilder(NESTNetworkBuilder, RaySpikingNetworkBuilder):

    def build_spiking_network(self):
        """A method to build the final NESTNetwork class based on the already created constituents."""
        ray_nest_network = \
            create_ray_client(NESTNetwork, non_blocking_methods=["Run", "Simulate"],
                              nest_instance=self.nest_instance,
                              brain_regions=self._spiking_brain,
                              output_devices=self._output_devices,
                              input_devices=self._input_devices,
                              config=self.config)
        setattr(ray_nest_network, "RunLock", MethodType(RunLock, ray_nest_network))
        return ray_nest_network


def create_ray_nest_network_builder_type(nest_network_builder=NESTNetworkBuilder,
                                         ray_builder=RayNESTNetworkBuilder):

    return create_ray_spiking_network_builder_type(spiking_network_builder=nest_network_builder,
                                                   ray_builder=ray_builder)
