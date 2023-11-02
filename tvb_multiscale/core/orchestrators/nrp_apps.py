# -*- coding: utf-8 -*-

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorParallelBuilder
from tvb_multiscale.core.interfaces.models.default import \
    DefaultTVBSpikeNetInterfaceBuilder, DefaultTVBInterfaceBuilder
from tvb_multiscale.core.orchestrators.tvb_app import TVBParallelApp
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetParallelApp
from tvb_multiscale.core.orchestrators.transformer_app import \
    TransformerApp, TVBtoSpikeNetTransformerApp, SpikeNetToTVBTransformerApp


class NRPTVBApp(TVBParallelApp):

    """NRPTVBApp class"""

    _default_interface_builder = DefaultTVBInterfaceBuilder


class NRPSpikeNetApp(SpikeNetParallelApp):

    """NRPSpikeNetapp abstract base class"""

    pass


class NRPTransformerApp(TransformerApp):

    """NRPTransformerApp class"""

    pass


class NRPTVBtoSpikeNetTransformerApp(TVBtoSpikeNetTransformerApp):

    """NRPTVBtoSpikeNetTransformerApp class"""

    pass


class NRPSpikeNetToTVBTransformerApp(SpikeNetToTVBTransformerApp):

    """NRPSpikeNetToTVBTransformerApp class"""

    pass
