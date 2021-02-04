# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from six import string_types

import numpy as np

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import List
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class InterfaceBuilder(HasTraits):
    __metaclass__ = ABCMeta

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

    def _loop_to_get_from_interface_configs(self, interfaces, attr):
        return [interface[attr] for interface in interfaces]

    def _loop_to_get_unique_from_interface_configs(self, interfaces, attr):
        return np.unique(self._loop_to_get_from_interface_configs(interfaces, attr))

    @property
    def out_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.output_interfaces, "proxy")

    @property
    def in_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.input_interfaces, "proxy")

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
        if isinstance(these_labels, (list, tuple, np.ndarray)):
            return_array = True
        else:
            labels = ensure_list(these_labels)
            return_array = False
        inds = []
        for label in ensure_list(labels):
            inds.append(np.where(all_labels == label)[0][0])
        if return_array:
            return np.array(inds)
        else:
            return inds[0]

    @staticmethod
    def _only_inds(self, interfaces, attr, labels):
        inds = []
        for iP, value in enumerate(self._loop_to_get_unique_from_interface_configs(interfaces, attr)):
            if isinstance(value, string_types):
                inds[iP] = self._label_to_ind(value, labels)
        return inds

    @staticmethod
    def _assert_interfaces_component_config(interfaces_list, types_list, component):
        for interface in interfaces_list:
            if interface[component] in types_list:
                interface[component] = interface[attr](**interface.get("%s_params" % component, {}))
            else:
                assert isinstance(interface[component], types_list)
        return interfaces_list

    def _assert_input_interfaces_component_config(self, types_list, attr):
        self.input_interfaces = self.assert_interfaces_component_config(self.input_interfaces, types_list, component)

    def _assert_output_interfaces_component_config(self, types_list, attr):
        self.output_interfaces = self.assert_interfaces_component_config(self.output_interfaces, types_list, component)

    @abstractmethod
    def build(self):
        pass
