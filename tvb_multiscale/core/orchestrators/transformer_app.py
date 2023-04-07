# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.interfaces.base.builders import RemoteTransformerBuilder, \
    TVBtoSpikeNetRemoteTransformerBuilder, SpikeNetToTVBRemoteTransformerBuilder
from tvb_multiscale.core.interfaces.base.interfaces import \
    TVBtoSpikeNetRemoteTransformerInterfaces, SpikeNetToTVBRemoteTransformerInterfaces


class TransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """TransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB<->spikeNet Remote Transformer Interfaces builder",
        field_type=RemoteTransformerBuilder,
        doc="""Instance of RemoteTransformerBuilder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Output (TVB->spikeNet) Remote Transformer interfaces",
        field_type=TVBtoSpikeNetRemoteTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBRemoteTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder = RemoteTransformerBuilder

    # def configure(self):
    #     super(TransformerApp, self).configure()

    # def configure_simulation(self):
    #     super(TransformerApp, self).configure()

    # def run(self, *args, **kwargs):
    #     self.configure()
    #     self.build()
    #     self.configure_simulation()
    #     self.simulate()


class TVBtoSpikeNetTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """TVBtoSpikeNetTransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB->spikeNet Remote Transformer Interfaces builder",
        field_type=TVBtoSpikeNetRemoteTransformerBuilder,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerBuilder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Output (TVB->spikeNet) Remote Transformer interfaces",
        field_type=TVBtoSpikeNetRemoteTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    input_interfaces = None

    _default_interface_builder = TVBtoSpikeNetRemoteTransformerBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            self.output_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def reset(self):
        self.output_interfaces = None
        self._interfaces_built = False


class SpikeNetToTVBTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetToTVBTransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB<-spikeNet Remote Transformer Interfaces builder",
        field_type=SpikeNetToTVBRemoteTransformerBuilder,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerBuilder class.""",
        required=False
    )

    input_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBRemoteTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    output_interfaces = None

    _default_interface_builder = SpikeNetToTVBRemoteTransformerBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            self.input_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def reset(self):
        self.input_interfaces = None
        self._interfaces_built = False
