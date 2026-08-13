# -*- coding: utf-8 -*-

from logging import Logger
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.interfaces.tvb.builders import TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.spikeNet.builders import \
    SpikeNetProxyNodesBuilder, SpikeNetInterfaceBuilder, SpikeNetRemoteInterfaceBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_annarchy.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.interfaces.interfaces import \
    ANNarchyOutputInterface, ANNarchyInputInterface, \
    ANNarchySenderInterface, ANNarchyReceiverInterface, \
    TVBtoANNarchyInterface, ANNarchyToTVBInterface, \
    ANNarchyOutputInterfaces, ANNarchyInputInterfaces, \
    ANNarchySenderInterfaces, ANNarchyReceiverInterfaces, \
    TVBtoANNarchyInterfaces, ANNarchyToTVBInterfaces
from tvb_multiscale.tvb_annarchy.interfaces.io import \
    ANNarchySpikeEventMonitorSet, ANNarchySpikeMonitorSet, \
    ANNarchyMonitorSet, ANNarchyMonitorMeanSet, ANNarchyMonitorTotalSet, \
    ANNarchyTimedArraySet, ANNarchyTimedArrayToSpikesSet, \
    ANNarchyTimedPoissonPopulationSet, ANNarchyHomogeneousCorrelatedSpikeTrainsSet, \
    ANNarchySpikeSourceArraySet
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import create_device, connect_device


TVBtoANNarchyModels = TVBtoSpikeNetModels
ANNarchyToTVBModels = SpikeNetToTVBModels


class ANNarchyInputProxyModels(Enum):
    RATE = ANNarchyTimedPoissonPopulationSet
    RATE_TO_SPIKES = ANNarchyTimedPoissonPopulationSet
    RATE_TO_CORRELATED_SPIKES = ANNarchyHomogeneousCorrelatedSpikeTrainsSet
    SPIKES = ANNarchySpikeSourceArraySet
    PARROT_SPIKES = None
    CURRENT = ANNarchyTimedArraySet
    CURRENT_TO_SPIKES = ANNarchyTimedArrayToSpikesSet


class ANNarchyOutputProxyModels(Enum):
    SPIKES = ANNarchySpikeMonitorSet
    SPIKE_EVENTS = ANNarchySpikeEventMonitorSet
    POTENTIAL = ANNarchyMonitorSet
    POTENTIAL_MEAN = ANNarchyMonitorMeanSet
    POTENTIAL_TOTAL = ANNarchyMonitorTotalSet
    CURRENT = ANNarchyMonitorSet
    CURRENT_MEAN = ANNarchyMonitorMeanSet
    CURRENT_TOTAL = ANNarchyMonitorTotalSet


class DefaultTVBtoANNarchyProxyModels(Enum):
    RATE = ANNarchyInputProxyModels.RATE
    RATE_TO_SPIKES = ANNarchyInputProxyModels.RATE_TO_SPIKES
    RATE_TO_CORRELATED_SPIKES = ANNarchyInputProxyModels.RATE_TO_CORRELATED_SPIKES
    SPIKES = ANNarchyInputProxyModels.SPIKES
    PARROT_SPIKES = None
    CURRENT = ANNarchyInputProxyModels.CURRENT
    CURRENT_TO_SPIKES = ANNarchyInputProxyModels.CURRENT_TO_SPIKES


class DefaultANNarchytoTVBProxyModels(Enum):
    SPIKES = ANNarchyOutputProxyModels.SPIKES
    POTENTIAL = ANNarchyOutputProxyModels.POTENTIAL_MEAN
    CURRENT = ANNarchyOutputProxyModels.CURRENT_MEAN


class ANNarchyProxyNodesBuilder(SpikeNetProxyNodesBuilder):

    """ANNarchyProxyNodesBuilder class"""

    spiking_network = Attr(label="ANNarchy Network",
                           doc="""The instance of ANNarchyNetwork class""",
                           field_type=ANNarchyNetwork,
                           required=True)


    @property
    def annarchy_network(self):
        return self.spiking_network

    @property
    def spikeNet_min_delay(self):
        return self.spiking_network.annarchy_network.dt

    @property
    def annarchy_min_delay(self):
        return self.spikeNet_min_delay

    def _build_and_connect_devices(self, interface, **kwargs):
        return build_and_connect_devices(interface, create_device, connect_device,
                                         self.spiking_network.brain_regions,
                                         self.config, annarchy_network=self.annarchy_network.annarchy_network,
                                         input_proxies=self.spiking_network.input_proxies, **kwargs)

    def _build_and_connect_input_devices(self, interface, **kwargs):
        return self._build_and_connect_devices(interface, **kwargs)

    def _build_and_connect_output_devices(self, interface, **kwargs):
        interface["params"]["period"] = interface["params"].get("period", self.tvb_dt)
        return self._build_and_connect_devices(interface, **kwargs)

    def _default_receptor_type(self, source_node, target_node):
        return "exc"

    @property
    def _default_min_delay(self):
        return self.annarchy_min_delay


class ANNarchyInterfaceBuilder(ANNarchyProxyNodesBuilder, SpikeNetInterfaceBuilder):
    
    """ANNarchyInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoANNarchyModels
    _spikeNet_to_tvb_models = ANNarchyToTVBModels

    _default_tvb_to_spikeNet_proxy_models = DefaultTVBtoANNarchyProxyModels
    _default_spikeNet_to_tvb_proxy_models = DefaultANNarchytoTVBProxyModels

    _input_proxy_models = ANNarchyInputProxyModels
    _output_proxy_models = ANNarchyOutputProxyModels

    _output_interface_type = ANNarchyOutputInterface
    _input_interface_type = ANNarchyInputInterface

    _output_interfaces_type = ANNarchyOutputInterfaces
    _input_interfaces_type = ANNarchyInputInterfaces

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


class ANNarchyRemoteInterfaceBuilder(ANNarchyInterfaceBuilder, SpikeNetRemoteInterfaceBuilder):

    """ANNarchyRemoteInterfaceBuilder class"""

    _output_interface_type = ANNarchySenderInterface
    _input_interface_type = ANNarchyReceiverInterface

    _output_interfaces_type = ANNarchySenderInterfaces
    _input_interfaces_type = ANNarchyReceiverInterfaces

    def configure(self):
        SpikeNetRemoteInterfaceBuilder.configure(self)


class TVBANNarchyInterfaceBuilder(ANNarchyProxyNodesBuilder, TVBSpikeNetInterfaceBuilder):

    """TVBANNarchyInterfaceBuilder class"""

    _tvb_to_spikeNet_models = TVBtoANNarchyModels
    _spikeNet_to_TVB_models = ANNarchyToTVBModels

    _input_proxy_models = ANNarchyOutputProxyModels  # Input to SpikeNet is output of TVB
    _output_proxy_models = ANNarchyInputProxyModels  # Output of SpikeNet is input to TVB

    _output_interface_type = TVBtoANNarchyInterface
    _input_interface_type = ANNarchyToTVBInterface

    _output_interfaces_type = TVBtoANNarchyInterfaces
    _input_interfaces_type = ANNarchyToTVBInterfaces

    def configure(self):
        TVBSpikeNetInterfaceBuilder.configure(self)

    def _get_tvb_delays(self):
        spiking_dt = self.spiking_dt
        if spiking_dt is None:
            spiking_dt = 0.0
        return np.maximum(spiking_dt,
                          TVBSpikeNetInterfaceBuilder._get_tvb_delays(self) -
                          self.synchronization_time + self.tvb_dt)
