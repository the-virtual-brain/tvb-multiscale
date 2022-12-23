# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.tvb.ray_builder import \
    RayTVBSpikeNetInterfaceBuilder, create_ray_TVB_spikeNet_interface_builder_type
from tvb_multiscale.tvb_nest.nest_models.ray import RayNESTNetwork
from tvb_multiscale.tvb_nest.interfaces.ray_interfaces import \
    RayTVBtoNESTInterface, RayNESTtoTVBInterface, RayTVBtoNESTInterfaces, RayNESTtoTVBInterfaces
from tvb_multiscale.tvb_nest.interfaces.builders import TVBNESTInterfaceBuilder


class RayTVBNESTInterfaceBuilder(TVBNESTInterfaceBuilder, RayTVBSpikeNetInterfaceBuilder):

    _output_interfaces_type = RayTVBtoNESTInterfaces
    _input_interfaces_type = RayNESTtoTVBInterfaces

    _output_interface_type = RayTVBtoNESTInterface
    _input_interface_type = RayNESTtoTVBInterface

    spiking_network = Attr(label="NEST Network Ray client",
                           doc="""A Ray client to an instance of NESTNetwork class""",
                           field_type=RayNESTNetwork,
                           required=True)

    def _configure_transformer_model(self, interface, interface_models, default_transformer_models, transformer_models):
        RayTVBSpikeNetInterfaceBuilder._configure_transformer_model(
            self, interface, interface_models, default_transformer_models, transformer_models)

    def build_transformer(self, model, **kwargs):
        return RayTVBSpikeNetInterfaceBuilder.build_transformer(self, model, **kwargs)


def create_ray_TVB_NEST_interface_builder_type(tvb_nest_interface_builder=TVBNESTInterfaceBuilder):
    return create_ray_TVB_spikeNet_interface_builder_type(tvb_spikeNet_interface_builder=tvb_nest_interface_builder,
                                                          ray_builder=RayTVBNESTInterfaceBuilder)
