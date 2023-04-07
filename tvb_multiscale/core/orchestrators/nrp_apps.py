# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.tvb.cosimulator.cosimulator_nrp import CoSimulatorParallelNRP
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorNRPBuilder
from tvb_multiscale.core.orchestrators.tvb_app import TVBParallelApp
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetParallelApp
from tvb_multiscale.core.orchestrators.transformer_app import \
    TransformerApp, TVBtoSpikeNetTransformerApp, SpikeNetToTVBTransformerApp


class NRPApp(HasTraits):
    __metaclass__ = ABCMeta

    """NRPApp abstract base class"""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def init(self):
        self.start()
        self.build()

    def final(self):
        self.clean_up()
        self.stop()


class NRPTVBapp(TVBParallelApp, NRPApp):

    """NRPTVBApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorParallelBuilder",
        field_type=CoSimulatorNRPBuilder,
        doc="""Instance of TVB Parallel CoSimulator Builder class.""",
        required=False,
        default=CoSimulatorNRPBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulatorParallelNRP,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )


class NRPSpikeNetApp(SpikeNetParallelApp, NRPApp, ABC):
    __metaclass__ = ABCMeta

    """NRPSpikeNetapp abstract base class"""

    pass


class NRPTransformerApp(TransformerApp, NRPApp):

    """NRPTransformerApp class"""

    pass


class NRPTVBtoSpikeNetTransformerApp(TVBtoSpikeNetTransformerApp, NRPApp):

    """NRPTVBtoSpikeNetTransformerApp class"""

    pass


class NRPSpikeNetToTVBTransformerApp(SpikeNetToTVBTransformerApp, NRPApp):

    """NRPSpikeNetToTVBTransformerApp class"""

    pass
