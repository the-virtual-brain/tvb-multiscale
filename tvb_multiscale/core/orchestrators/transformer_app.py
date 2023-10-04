# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TransformerInterfaceBuilder, RemoteTransformerInterfaceBuilder, \
    TVBtoSpikeNetTransformerInterfaceBuilder, SpikeNetToTVBTransformerInterfaceBuilder, \
    TVBtoSpikeNetRemoteTransformerInterfaceBuilder, SpikeNetToTVBRemoteTransformerInterfaceBuilder
from tvb_multiscale.core.interfaces.base.transformers.interfaces import \
    TVBtoSpikeNetTransformerInterfaces, SpikeNetToTVBTransformerInterfaces, \
    TVBtoSpikeNetRemoteTransformerInterfaces, SpikeNetToTVBRemoteTransformerInterfaces


class TransformerApp(NonTVBApp):

    """TransformerApp base class"""

    interfaces_builder = Attr(
        label="TVB<->spikeNet Remote Transformer Interfaces builder",
        field_type=TransformerInterfaceBuilder,
        doc="""Instance of RemoteTransformerInterfaceBuilder class.""",
        required=False
    )

    tvb_to_spikeNet_interfaces = Attr(
        label="TVB->spikeNet Transformer interfaces",
        field_type=TVBtoSpikeNetTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    spikeNet_to_tvb_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = TransformerInterfaceBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            super(TransformerApp, self).build_interfaces()
            self.tvb_to_spikeNet_interfaces, self.spikeNet_to_tvb_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True
            if self.verbosity:
                self._logprint(self.tvb_to_spikeNet_interfaces.summary_info_to_string(recursive=2))
                self._logprint(self.spikeNet_to_tvb_interfaces.summary_info_to_string(recursive=2))

    def configure_simulation(self):
        super(TransformerApp, self).configure_simulation()
        self.tvb_to_spikeNet_interfaces.configure()
        self.spikeNet_to_tvb_interfaces.configure()

    def run_for_synchronization_time(self, tvb_to_spikeNet_cosim_updates, spikeNet_to_tvb_cosim_updates):
        return self.tvb_to_spikeNet_interfaces(tvb_to_spikeNet_cosim_updates), \
               self.spikeNet_to_tvb_interfaces(spikeNet_to_tvb_cosim_updates)

    def _destroy(self):
        self.tvb_to_spikeNet_interfaces = None
        self.spikeNet_to_tvb_interfaces = None
        super(TransformerApp, self)._destroy()


class TVBtoSpikeNetTransformerApp(NonTVBApp):

    """TVBtoSpikeNetTransformerApp class"""

    interfaces_builder = Attr(
        label="TVB->spikeNet Transformer Interfaces builder",
        field_type=TVBtoSpikeNetTransformerInterfaceBuilder,
        doc="""Instance of TVBtoSpikeNetTransformerInterfaceBuilder class.""",
        required=False
    )

    tvb_to_spikeNet_interfaces = Attr(
        label="TVB->spikeNet Transformer interfaces",
        field_type=TVBtoSpikeNetTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = TVBtoSpikeNetTransformerInterfaceBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            super(TVBtoSpikeNetTransformerApp, self).build_interfaces()
            self.tvb_to_spikeNet_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True
            if self.verbosity:
                self._logprint(self.tvb_to_spikeNet_interfaces.summary_info_to_string(recursive=2))

    def configure_simulation(self):
        super(TVBtoSpikeNetTransformerApp, self).configure_simulation()
        self.tvb_to_spikeNet_interfaces.configure()

    def run_for_synchronization_time(self, cosim_updates):
        return self.tvb_to_spikeNet_interfaces(cosim_updates)


class SpikeNetToTVBTransformerApp(NonTVBApp):

    """SpikeNetToTVBTransformerApp class"""

    interfaces_builder = Attr(
        label="TVB<-spikeNet Transformer Interfaces builder",
        field_type=SpikeNetToTVBTransformerInterfaceBuilder,
        doc="""Instance of SpikeNetToTVBTransformerInterfaceBuilder class.""",
        required=False
    )

    spikeNet_to_tvb_interfaces = Attr(
        label="TVB<-spikeNet Transformer interfaces",
        field_type=SpikeNetToTVBTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = SpikeNetToTVBTransformerInterfaceBuilder

    def build_interfaces(self):
        if not self._interfaces_built:
            super(SpikeNetToTVBTransformerApp, self).build_interfaces()
            self.spikeNet_to_tvb_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True
            if self.verbosity:
                self._logprint(self.spikeNet_to_tvb_interfaces.summary_info_to_string(recursive=2))

    def configure_simulation(self):
        super(SpikeNetToTVBTransformerApp, self).configure_simulation()
        self.spikeNet_to_tvb_interfaces.configure()

    def run_for_synchronization_time(self, cosim_updates):
        return self.spikeNet_to_tvb_interfaces(cosim_updates)


class RemoteTransformerApp(TransformerApp):

    """RemoteTransformerApp class"""

    interfaces_builder = Attr(
        label="TVB<->spikeNet Remote Transformer Interfaces builder",
        field_type=RemoteTransformerInterfaceBuilder,
        doc="""Instance of RemoteTransformerInterfaceBuilder class.""",
        required=False
    )

    tvb_to_spikeNet_interfaces = Attr(
        label="Output (TVB->spikeNet) Remote Transformer interfaces",
        field_type=TVBtoSpikeNetRemoteTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    spikeNet_to_tvb_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBRemoteTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = RemoteTransformerInterfaceBuilder

    def run_for_synchronization_time(self):
        return self.tvb_to_spikeNet_interfaces(), self.spikeNet_to_tvb_interfaces()


class TVBtoSpikeNetRemoteTransformerApp(TVBtoSpikeNetTransformerApp):

    """TVBtoSpikeNetRemoteTransformerApp class"""

    interfaces_builder = Attr(
        label="TVB->spikeNet Remote Transformer Interfaces builder",
        field_type=TVBtoSpikeNetRemoteTransformerInterfaceBuilder,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaceBuilder class.""",
        required=False
    )

    tvb_to_spikeNet_interfaces = Attr(
        label="Output (TVB->spikeNet) Remote Transformer interfaces",
        field_type=TVBtoSpikeNetRemoteTransformerInterfaces,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = TVBtoSpikeNetRemoteTransformerInterfaceBuilder

    def run_for_synchronization_time(self):
        return self.tvb_to_spikeNet_interfaces()


class SpikeNetToTVBRemoteTransformerApp(SpikeNetToTVBTransformerApp):

    """SpikeNetToTVBRemoteTransformerApp class"""

    interfaces_builder = Attr(
        label="TVB<-spikeNet Remote Transformer Interfaces builder",
        field_type=SpikeNetToTVBRemoteTransformerInterfaceBuilder,
        doc="""Instance of TVBtoSpikeNetRemoteTransformerInterfaceBuilder class.""",
        required=False
    )

    spikeNet_to_tvb_interfaces = Attr(
        label="Input (TVB<-spikeNet) Remote Transformer interfaces",
        field_type=SpikeNetToTVBRemoteTransformerInterfaces,
        doc="""Instance of SpikeNetToTVBRemoteTransformerInterfaces.""",
        required=False
    )

    _default_interface_builder_type = SpikeNetToTVBRemoteTransformerInterfaceBuilder

    def run_for_synchronization_time(self):
        return self.spikeNet_to_tvb_interfaces()
