# -*- coding: utf-8 -*-

from tvb_multiscale.core.ray.client import create_ray_client
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork


def create_ray_spiking_network_builder(spiking_network_builder):

    class RaySpikingNetworkBuilder(spiking_network_builder):

        def build_spiking_network(self):
            """A method to build the final NESTNetwork class based on the already created constituents."""
            return create_ray_client(NESTNetwork, non_blocking_methods=["Run", "Simulate"],
                                     nest_instance=self.nest_instance,
                                     brain_regions=self._spiking_brain,
                                     output_devices=self._output_devices,
                                     input_devices=self._input_devices,
                                     config=self.config)

    RaySpikingNetworkBuilder.__name__ = "Ray%s" % spiking_network_builder.__name__

    return RaySpikingNetworkBuilder
