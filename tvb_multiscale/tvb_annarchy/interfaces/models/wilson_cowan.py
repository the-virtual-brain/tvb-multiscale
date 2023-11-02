# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.wilson_cowan import WilsonCowanTVBSpikeNetInterfaceBuilder, \
    WilsonCowanSpikeNetInterfaceBuilder, WilsonCowanSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_annarchy.interfaces.builders import \
    ANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder, TVBANNarchyInterfaceBuilder


class WilsonCowanANNarchyProxyNodesBuilder(ANNarchyProxyNodesBuilder, WilsonCowanSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class WilsonCowanANNarchyInterfaceBuilder(WilsonCowanANNarchyProxyNodesBuilder, ANNarchyInterfaceBuilder,
                                      WilsonCowanSpikeNetInterfaceBuilder):
    pass


class WilsonCowanTVBANNarchyInterfaceBuilder(WilsonCowanANNarchyProxyNodesBuilder,
                                             TVBANNarchyInterfaceBuilder,
                                             WilsonCowanTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        WilsonCowanTVBSpikeNetInterfaceBuilder.default_input_config(self)
