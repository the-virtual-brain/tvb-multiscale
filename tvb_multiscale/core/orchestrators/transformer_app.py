# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TransformerInterfaceBuilder, RemoteTransformerBuilder, \
    TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder, \
    TVBtoSpikeNetRemoteTransformerBuilder, SpikeNetToTVBRemoteTransformerBuilder
from tvb_multiscale.core.interfaces.base.transformers.interfaces import \
    TVBtoSpikeNetTransformerInterfaces, SpikeNetToTVBTransformerInterfaces, \
    TVBtoSpikeNetRemoteTransformerInterfaces, SpikeNetToTVBRemoteTransformerInterfaces


class TransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """TransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB<->spikeNet Remote Transformer Interfaces builder",
        field_type=TransformerInterfaceBuilder,
        doc="""Instance of RemoteTransformerBuilder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Output (TVB->spikeNet) Transformer interfaces",
        field_type=TVBtoSpikeNetTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder = TransformerInterfaceBuilder

    # def configure(self):
    #     super(TransformerApp, self).configure()

    def build(self):
        self.build_interfaces()

    def run_for_synchronization_time(self, input_cosim_updates, output_cosim_updates):
        return self.output_interfaces(output_cosim_updates), self.input_interfaces(input_cosim_updates)


class TVBtoSpikeNetTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """TVBtoSpikeNetTransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB->spikeNet Transformer Interfaces builder",
        field_type=TVBtoSpikeNetTransformerBuilder,
        doc="""Instance of TVBtoSpikeNetTransformerBuilder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Output (TVB->spikeNet) Transformer interfaces",
        field_type=TVBtoSpikeNetTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetTransformerInterfaces.""",
        required=False
    )

    input_interfaces = None

    _default_interface_builder = TVBtoSpikeNetTransformerBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            self.output_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def run_for_synchronization_time(self, cosim_updates):
        return self.output_interfaces(cosim_updates)

    def reset(self):
        self.output_interfaces = None
        self._interfaces_built = False


class SpikeNetToTVBTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetToTVBTransformerApp abstract base class"""

    interfaces_builder = Attr(
        label="TVB<-spikeNet Transformer Interfaces builder",
        field_type=SpikeNetToTVBTransformerBuilder,
        doc="""Instance of TVBtoSpikeNetTransformerBuilder class.""",
        required=False
    )

    input_interfaces = Attr(
        label="Input (TVB<-spikeNet) Transformer interfaces",
        field_type=SpikeNetToTVBTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBTransformerInterfaces.""",
        required=False
    )

    output_interfaces = None

    _default_interface_builder = SpikeNetToTVBTransformerBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            self.input_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def run_for_synchronization_time(self, cosim_updates):
        return self.input_interfaces(cosim_updates)

    def reset(self):
        self.input_interfaces = None
        self._interfaces_built = False


class RemoteTransformerApp(TransformerApp, ABC):
    __metaclass__ = ABCMeta

    """RemoteTransformerApp abstract base class"""

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


class TVBtoSpikeNetRemoteTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """TVBtoSpikeNetRemoteTransformerApp abstract base class"""

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


class SpikeNetToTVBRemoteTransformerApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetToTVBRemoteTransformerApp abstract base class"""

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
