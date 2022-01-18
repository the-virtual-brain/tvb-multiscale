# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, ABC
from logging import Logger
from six import string_types

import numpy as np

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import Attr, Float, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger


class InterfaceBuilder(HasTraits, ABC):
    __metaclass__ = ABCMeta

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of proxy nodes",
        doc="""Indices of proxy nodes""",
        required=True,
    )

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

    default_coupling_mode = Attr(
        label="Default coupling mode",
        field_type=str,
        doc="""Default coupling mode. Set to 'TVB' for large scale coupling to be computed in TVB 
               before being sent to a cosimulator. Default 'spikeNet', which entails that 
               large scale coupling for regions modeled outside TVB is handled by the cosimulator.""",
        required=True,
        default="spikeNet"
    )

    _output_interfaces = None
    _input_interfaces = None

    def _loop_to_get_from_interface_configs(self, interfaces, attr):
        output = []
        for interface in interfaces:
            output += ensure_list(interface[attr])
        return output

    def _loop_to_get_unique_from_interface_configs(self, interfaces, attr):
        return np.unique(self._loop_to_get_from_interface_configs(interfaces, attr))

    @property
    @abstractmethod
    def synchronization_time(self):
        pass

    @property
    @abstractmethod
    def synchronization_n_step(self):
        pass

    @property
    def out_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.output_interfaces, "proxy_inds")

    @property
    def in_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.input_interfaces, "proxy_inds")

    @property
    def number_of_out_proxy_nodes(self):
        return len(self.out_proxy_inds)

    @property
    def number_of_in_proxy_nodes(self):
        return len(self.in_proxy_inds)

    @property
    def number_of_output_interfaces(self):
        return len(self.output_interfaces)

    @property
    def number_of_input_interfaces(self):
        return len(self.output_interfaces)

    @staticmethod
    def _label_to_ind(these_labels, all_labels):
        all_labels = ensure_list(all_labels)
        if isinstance(these_labels, (list, tuple, np.ndarray)):
            return_array = True
        else:
            labels = ensure_list(these_labels)
            return_array = False
        inds = []
        for label in ensure_list(labels):
            inds.append(all_labels.index(label))
        if return_array:
            return np.array(inds)
        else:
            return inds[0]

    def _only_ind(self, value, labels):
        if isinstance(value, string_types):
            return self._label_to_ind(value, labels)
        else:
            return value

    def _only_inds(self, values, labels):
        inds = []
        for iV, value in enumerate(ensure_list(values)):
            inds.append(self._only_ind(value, labels))
        return inds

    def _only_inds_for_interfaces(self, interfaces, attr, labels):
        return self._only_inds(self._loop_to_get_unique_from_interface_configs(interfaces, attr), labels)

    @staticmethod
    def _assert_interfaces_component_config(interfaces_list, types_list, component):
        """This method will assert that all interfaces' components types are appropriate,
           i.e., included in the given types_list argument.
           It will also create any communicator or transformer classes' instances.
           Instead, even if the user enters by mistake an interface class instance with the keyword 'model',
           it will be converted to a type, since it is the job of the builder to generate this instance.
        """
        for interface in interfaces_list:
            if interface[component] in ensure_list(types_list):
                interface[component] = interface[component](**interface.get("%s_params" % component, {}))
            else:
                assert isinstance(interface[component], types_list)
        return interfaces_list

    def _assert_input_interfaces_component_config(self, types_list, component):
        self.input_interfaces = self.assert_interfaces_component_config(self.input_interfaces, types_list, component)

    def _assert_output_interfaces_component_config(self, types_list, component):
        self.output_interfaces = self.assert_interfaces_component_config(self.output_interfaces, types_list, component)

    def set_coupling_mode(self, interface):
        interface["coupling_mode"] = interface.get("coupling_mode", self.default_coupling_mode)
        return interface

    def is_tvb_coupling_interface(self, interface):
        return self.set_coupling_mode(interface)["coupling_mode"].upper() == "TVB"

    def _get_output_interface_arguments(self, interface):
        return interface

    def _get_input_interface_arguments(self, interface):
        return interface

    def build_output_interface(self, interface):
        return self._output_interface_type(**self._get_output_interface_arguments(interface))

    def build_input_interface(self, interface):
        return self._input_interface_type(**self._get_input_interface_arguments(interface))

    def build_interfaces(self):
        self._output_interfaces = []
        for interface in self.output_interfaces:
            self._output_interfaces.append(self.build_output_interface(interface))
        self._input_interfaces = []
        for interface in self.input_interfaces:
            self._input_interfaces.append(self.build_input_interface(interface))

    def build(self):
        self.build_interfaces()
        return self._output_interfaces, self._input_interfaces
