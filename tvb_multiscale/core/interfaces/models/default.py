# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Attr, Int

from tvb_multiscale.core.interfaces.base.builders import InterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBInterfaceBuilder, TVBSpikeNetInterfaceBuilder, TVBRemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetProxyNodesBuilder, SpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels


class DefaultInterfaceBuilder(InterfaceBuilder):
    __metaclass__ = ABCMeta

    model = Attr(
        label="Model",
        field_type=str,
        doc="""Name of interface model (string).""",
        required=True,
        default="RATE",
    )

    output_flag = Attr(label="Output interfaces flag",
                       doc="""Boolean flag that is true if default output interfaces,
                              should be built, even if the absence of user input.""",
                       field_type=bool,
                       default=True,
                       required=True)

    input_flag = Attr(label="Input interfaces flag",
                       doc="""Boolean flag that is true if default input interfaces,
                              should be built, even if the absence of user input.""",
                       field_type=bool,
                       default=True,
                       required=True)

    def _get_interfaces(self, interfaces, dim=0):
        while len(interfaces) <= dim:
            interfaces.append({})
        return interfaces[dim]

    def _get_output_interfaces(self, dim=0):
        return self._get_interfaces(self.output_interfaces, dim)

    def _get_input_interfaces(self, dim=0):
        return self._get_interfaces(self.input_interfaces, dim)

    @abstractmethod
    def default_output_config(self):
        pass

    @abstractmethod
    def default_input_config(self):
        pass

    def default_config(self):
        if self.output_flag and len(self.output_interfaces) == 0:
            self.default_output_config()
        if self.input_flag and len(self.input_interfaces) == 0:
            self.default_input_config()


class DefaultTVBInterfaceBuilder(TVBInterfaceBuilder, DefaultInterfaceBuilder):

    model = Attr(
        label="Model",
        field_type=str,
        doc="""Name of interface model (string).""",
        required=True,
        default=TVBtoSpikeNetModels.RATE.name,
        choices=tuple(TVBtoSpikeNetModels.__members__)
    )

    def _get_output_interfaces(self, dim=0):
        interface = DefaultInterfaceBuilder._get_output_interfaces(self, dim)
        interface["model"] = interface.get("model", self.model)
        return interface

    def default_output_config(self):
        self._get_output_interfaces()["voi"] = "R"

    def default_input_config(self):
        assert self.model in (TVBtoSpikeNetModels.RATE.name, TVBtoSpikeNetModels.SPIKES.name)
        self._get_input_interfaces()["voi"] = "R"


class DefaultTVBRemoteInterfaceBuilder(TVBRemoteInterfaceBuilder, DefaultTVBInterfaceBuilder):
    pass


class DefaultTVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["transformer_params"] = {"scale_factor": np.array([1.0]),
                                                   "number_of_neurons": np.array([self.N_E])}
            else:  # RATE
                interface["transformer_params"] = {"scale_factor": np.array([self.N_E])}


class DefaultSpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    def default_spikeNet_to_tvb_config(self, interfaces):
        for interface in interfaces:
            interface["transformer_params"] = {"scale_factor": np.array([1.0]) / self.N_E}


class DefaultSpikeNetProxyNodesBuilder(SpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="Number of regions", default=100,
              doc="""The number of regions represented in this Connectivity""")

    model = Attr(
        label="Model",
        field_type=str,
        doc="""Name of interface model (string).""",
        required=True,
        default=TVBtoSpikeNetModels.RATE.name,
        choices=tuple(TVBtoSpikeNetModels.__members__)
    )

    def default_spikeNet_to_tvb_config(self, interfaces):
        pass

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["proxy_params"] = {"number_of_neurons": self.N_E}


class DefaultTVBSpikeNetInterfaceBuilder(TVBSpikeNetInterfaceBuilder,
                                         DefaultTVBtoSpikeNetTransformerBuilder,
                                         DefaultSpikeNetToTVBTransformerBuilder,
                                         DefaultSpikeNetProxyNodesBuilder,
                                         DefaultTVBInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    def default_output_config(self):
        assert self.model in (TVBtoSpikeNetModels.RATE.name, TVBtoSpikeNetModels.SPIKES.name)
        DefaultTVBInterfaceBuilder.default_output_config(self)
        DefaultTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)
        self._get_output_interfaces()["populations"] = "E"
        DefaultSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        DefaultTVBInterfaceBuilder.default_input_config(self)
        DefaultSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
        self._get_input_interfaces()["populations"] = "E"
        DefaultSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
