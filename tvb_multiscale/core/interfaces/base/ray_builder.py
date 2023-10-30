# -*- coding: utf-8 -*-

import ray

from tvb.basic.neotraits._attr import List

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.ray.client import create_ray_client, RayClient


class RayInterbaseBuilder(HasTraits):

    def build_output_interface(self, interface, ii=0):
        interface = self._get_output_interface_arguments(interface, ii)
        return create_ray_client(self._output_interface_type,
                                 non_blocking_methods=["__call__"],
                                 **interface)

    def build_input_interface(self, interface, ii=0):
        return create_ray_client(self._input_interface_type,
                                 non_blocking_methods=["__call__"],
                                 **self._get_input_interface_arguments(interface, ii))


def rayfy_interfaces_class(interfaces_class):

    class RayInterfaces(interfaces_class):

        interfaces = List(of=RayClient)

    RayInterfaces.__name__ = "Ray%s" % interfaces_class.__name__

    return RayInterfaces


def create_rayfied_interface_builder_class(interface_builder_class):

    def _rayfy_interfaces_class(interface_builder_class, attr):
        interfaces_class = getattr(interface_builder_class, attr, None)
        if interfaces_class is not None:
            return rayfy_interfaces_class(interfaces_class)
        else:
            return None

    class RayInterfaceBuilder(interface_builder_class, RayInterbaseBuilder):

        _output_interfaces_type = _rayfy_interfaces_class(interface_builder_class, "_output_interfaces_type")
        _input_interfaces_type = _rayfy_interfaces_class(interface_builder_class, "_input_interfaces_type")

        def build_output_interface(self, interface, ii=0):
            return RayInterbaseBuilder.build_output_interface(self, interface, ii=ii)

        def build_input_interface(self, interface, ii=0):
            return RayInterbaseBuilder.build_input_interface(self, interface, ii=ii)

    RayInterfaceBuilder.__name__ = "Ray%s" % interface_builder_class.__name__

    return RayInterfaceBuilder


def create_rayfied_interface_builder(interface_builder_class, *args, **kwargs):

    return create_rayfied_interface_builder_class(interface_builder_class)(*args, **kwargs)
