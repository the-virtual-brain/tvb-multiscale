# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

import numpy as np

from tvb.basic.neotraits._attr import Int

from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels
from tvb_multiscale.core.interfaces.models.default import \
    DefaultSpikeNetProxyNodesBuilder, DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultTVBRemoteInterfaceBuilder, DefaultTVBTransfomerInterfaceBuilder, \
    DefaultTVBOutputTransformerInterfaceBuilder, DefaultTVBInputTransformerInterfaceBuilder, \
    DefaultSpikeNetRemoteInterfaceBuilder, DefaultSpikeNetTransformerInterfaceBuilder, \
    DefaultTVBInterfaceBuilder, DefaultSpikeNetInterfaceBuilder, DefaultInterfaceBuilder


class WilsonCowanTVBInterfaceBuilder(DefaultTVBInterfaceBuilder):

    def default_output_config(self):
        self._get_output_interfaces()["voi"] = "E"

    def default_input_config(self):
        assert self.model in (TVBtoSpikeNetModels.RATE.name, TVBtoSpikeNetModels.SPIKES.name)
        self._get_input_interfaces(0)["voi"] = "E"
        self._get_input_interfaces(1)["voi"] = "I"


class WilsonCowanTVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["transformer_params"] = \
                    {"scale_factor": np.array([100]),
                     "number_of_neurons": np.array([self.N_E])}
            else:  # RATE
                interface["transformer_params"] = {"scale_factor": 100 * np.array([self.N_E])}


class WilsonCowanSpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    N_I = Int(field_type=int, label="N_I", default=100, doc="""Number of inhibitory neurons""")

    def default_spikeNet_to_tvb_config(self, interfaces):
        for interface, N in zip(interfaces, [self.N_E, self.N_I]):
            interface["transformer_params"] = {"scale_factor": np.array([1e-4]) / N}


class WilsonCowanTVBRemoteInterfaceBuilder(WilsonCowanTVBInterfaceBuilder, DefaultTVBRemoteInterfaceBuilder):

    pass


class WilsonCowanTVBOutputTransformerInterfaceBuilder(WilsonCowanTVBInterfaceBuilder,
                                                      WilsonCowanTVBtoSpikeNetTransformerBuilder,
                                                      DefaultTVBOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanTVBInterfaceBuilder.default_output_config(self)
        WilsonCowanTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)


class WilsonCowanTVBInputTransformerInterfaceBuilder(WilsonCowanTVBInterfaceBuilder,
                                                     WilsonCowanSpikeNetToTVBTransformerBuilder,
                                                     DefaultTVBInputTransformerInterfaceBuilder):

    def default_input_config(self):
        WilsonCowanTVBInterfaceBuilder.default_input_config(self)
        WilsonCowanSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)


class WilsonCowanTVBTransfomerInterfaceBuilder(WilsonCowanTVBInterfaceBuilder,
                                               WilsonCowanTVBtoSpikeNetTransformerBuilder,
                                               WilsonCowanSpikeNetToTVBTransformerBuilder,
                                               DefaultTVBTransfomerInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanTVBInterfaceBuilder.default_output_config(self)
        WilsonCowanTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        WilsonCowanTVBInterfaceBuilder.default_input_config(self)
        WilsonCowanSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)


class WilsonCowanSpikeNetProxyNodesBuilder(DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanSpikeNetInterfaceBuilder(WilsonCowanSpikeNetProxyNodesBuilder, DefaultSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanSpikeNetRemoteInterfaceBuilder(WilsonCowanSpikeNetInterfaceBuilder,
                                                DefaultSpikeNetRemoteInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanSpikeNetTransformerInterfaceBuilder(WilsonCowanSpikeNetInterfaceBuilder,
                                                     WilsonCowanSpikeNetToTVBTransformerBuilder,
                                                     WilsonCowanTVBtoSpikeNetTransformerBuilder,
                                                     DefaultSpikeNetTransformerInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    def default_output_config(self):
        WilsonCowanSpikeNetInterfaceBuilder.default_input_config(self)
        WilsonCowanSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.output_interfaces)

    def default_input_config(self):
        WilsonCowanSpikeNetInterfaceBuilder.default_input_config(self)
        WilsonCowanTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.input_interfaces)


class WilsonCowanTVBSpikeNetInterfaceBuilder(WilsonCowanTVBInterfaceBuilder,
                                             WilsonCowanTVBtoSpikeNetTransformerBuilder,
                                             WilsonCowanSpikeNetToTVBTransformerBuilder,
                                             WilsonCowanSpikeNetProxyNodesBuilder,
                                             DefaultTVBSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    def default_output_config(self):
        WilsonCowanTVBInterfaceBuilder.default_output_config(self)
        WilsonCowanTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)
        self._get_output_interfaces()["populations"] = "E"
        WilsonCowanSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        WilsonCowanTVBInterfaceBuilder.default_input_config(self)
        WilsonCowanSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
        self._get_input_interfaces(0)["populations"] = "E"
        self._get_input_interfaces(1)["populations"] = "I"
        WilsonCowanSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
