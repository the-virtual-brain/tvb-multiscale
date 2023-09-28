# -*- coding: utf-8 -*-

from six import string_types

from tvb_multiscale.core.ray.client import create_ray_client, RayClient

from tvb_multiscale.core.interfaces.tvb.ray_interfaces import \
    RayTVBtoSpikeNetInterface, RaySpikeNetToTVBInterface, RayTVBtoSpikeNetInterfaces, RaySpikeNetToTVBInterfaces
from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder


def create_ray_transformer(transformer_type, *args, **kwargs):
    return create_ray_client(transformer_type,
                             ray_client_type=RayClient, non_blocking_methods="__call__", *args, **kwargs)


class RayTVBSpikeNetInterfaceBuilder(TVBSpikeNetInterfaceBuilder):

    _output_interfaces_type = RayTVBtoSpikeNetInterfaces
    _input_interfaces_type = RaySpikeNetToTVBInterfaces

    _output_interface_type = RayTVBtoSpikeNetInterface
    _input_interface_type = RaySpikeNetToTVBInterface

    default_to_ray_transformer_flag = False

    def __init__(self, **kwargs):
        self.default_to_ray_transformer_flag = False
        super(RayTVBSpikeNetInterfaceBuilder, self).__init__(**kwargs)

    def _configure_transformer_model(self, interface, interface_models, default_transformer_models, transformer_models):
        model = interface.get("transformer", interface.pop("transformer_model", None))
        if isinstance(model, RayClient):
            # If it is a Ray Transformer Client already built, do nothing more
            return
        else:
            parallel = False
            if isinstance(model, string_types) and model.find("Ray") == 0:
                # If the user has given the Transformer model with the prefix Ray,
                # remove the prefix and set the parallel flag to True
                interface["transformer"] = model.split("Ray")[-1]
                parallel = True
            elif self.default_to_ray_transformer_flag is True:
                parallel = True
            if parallel:
                d = interface.get("transformer_params", {})
                d["parallel"] = True
                interface["transformer_params"] = d
        # Now it is safe to call the corresponding parent class method:
        super(RayTVBSpikeNetInterfaceBuilder, self)._configure_transformer_model(
            interface, interface_models, default_transformer_models, transformer_models)

    def build_transformer(self, model, **kwargs):
        if kwargs.pop("parallel", False):
            return create_ray_transformer(model, **kwargs)
        else:
            return super(RayTVBSpikeNetInterfaceBuilder, self).build_transformer(model, **kwargs)


def create_ray_TVB_spikeNet_interface_builder_type(tvb_spikeNet_interface_builder=TVBSpikeNetInterfaceBuilder,
                                                   ray_builder=RayTVBSpikeNetInterfaceBuilder):

    class ThisRayTVBSpikeNetInterfaceBuilder(tvb_spikeNet_interface_builder, ray_builder):

        def _configure_transformer_model(self,
                                         interface, interface_models, default_transformer_models, transformer_models):
            return ray_builder._configure_transformer_model(
                self, interface, interface_models, default_transformer_models, transformer_models)

        def build_transformer(self, model, **kwargs):
            return ray_builder.build_transformer(self, model, **kwargs)

    ThisRayTVBSpikeNetInterfaceBuilder.__name__ = "Ray%s" % tvb_spikeNet_interface_builder.__name__

    return ThisRayTVBSpikeNetInterfaceBuilder
