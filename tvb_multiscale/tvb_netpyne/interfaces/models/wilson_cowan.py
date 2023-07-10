from abc import ABCMeta, ABC

import numpy as np

from tvb_multiscale.core.interfaces.models.wilson_cowan import \
    WilsonCowanTVBSpikeNetInterfaceBuilder, WilsonCowanSpikeNetProxyNodesBuilder
from tvb_multiscale.tvb_netpyne.interfaces.builders import NetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import receptor_by_source_region


class WilsonCowanNetpyneProxyNodesBuilder(NetpyneProxyNodesBuilder, WilsonCowanSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanTVBNetpyneInterfaceBuilder(WilsonCowanNetpyneProxyNodesBuilder, TVBNetpyneInterfaceBuilder,
                                            WilsonCowanTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_output_config(self)
        self.output_interfaces[0]["proxy_params"] = {"number_of_neurons": self.N_E}

        # overrides what's in WilsonCowanTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config()
        tuneable_factor = 100 # TODO: de-hardcode
        # due to the way Netpyne generates spikes, no scaling by population size is needed
        self.output_interfaces[0]["transformer_params"] = {"scale_factor": tuneable_factor * np.array([1.0])}

    def default_input_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_input_config(self)
