# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.orchestrators.spikeNet_app import \
    SpikeNetSerialApp, SpikeNetParallelApp, SpikeNetRemoteParallelApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator

from tvb_multiscale.tvb_annarchy.config import Config, CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyNetworkBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import load_annarchy
from tvb_multiscale.tvb_annarchy.interfaces.builders import \
    ANNarchyInterfaceBuilder, ANNarchyRemoteInterfaceBuilder, TVBANNarchyInterfaceBuilder


class ANNarchyApp(HasTraits):

    """ANNarchyApp class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    network_path = Attr(
        label="ANNarchy network path",
        field_type=str,
        doc="""Path to the compiled code of the ANNarchy network.""",
        required=False,
        default=""
    )

    spikeNet_builder = Attr(
        label="ANNarchy Network Builder",
        field_type=ANNarchyNetworkBuilder,
        doc="""Instance of ANNarchy Model Builder.""",
        required=False,
    )

    spiking_network = Attr(
        label="ANNarchy Network",
        field_type=ANNarchyNetwork,
        doc="""Instance of ANNarchyNetwork class.""",
        required=False
    )

    _spikeNet_builder_type = ANNarchyNetworkBuilder
    _spikeNet_type = ANNarchyNetwork

    @property
    def annarchy_instance(self):
        return self.spiking_cosimulator

    @property
    def annarchy_network(self):
        return self.spiking_network

    @property
    def annarchy_model_builder(self):
        return self.spikeNet_builder

    def start(self):
        self.spiking_cosimulator = load_annarchy(self.config)

    def configure(self, **kwargs):
        self.annarchy_instance.clear()  # This will restart ANNarchy!
        if self.spikeNet_builder is not None:
            self.spikeNet_builder.update_spiking_dt()
            self.spikeNet_builder.update_default_min_delay()
            kwargs["dt"] = self.spikeNet_builder.spiking_dt
        else:
            kwargs["dt"] = kwargs.pop("seed", self.config.SPIKING_DT)
        kwargs["seed"] = kwargs.pop("seed", self.config.ANNARCHY_SEED)
        kwargs["verbose"] = kwargs.pop("verbose", self.config.VERBOSE)
        self.annarchy_instance.setup(**kwargs)

    def configure_simulation(self):
        self.spiking_network.network_path = self.network_path

    def clean_up(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.annarchy_instance.clear()


class ANNarchySerialApp(ANNarchyApp, SpikeNetSerialApp):

    """ANNarchySerialApp class"""

    def start(self):
        SpikeNetSerialApp.start(self)
        ANNarchyApp.start(self)

    def configure(self):
        SpikeNetSerialApp.configure(self)
        ANNarchyApp.configure(self)

    def configure_simulation(self):
        SpikeNetSerialApp.configure_simulation(self)
        ANNarchyApp.configure_simulation(self)

    def reset(self):
        ANNarchyApp.reset(self)
        SpikeNetSerialApp.reset(self)

    def clean_up(self):
        ANNarchyApp.clean_up(self)
        SpikeNetSerialApp.clean_up(self)


class ANNarchyParallelApp(ANNarchyApp, SpikeNetParallelApp):

    """ANNarchyParallelApp class"""

    interfaces_builder = Attr(
        label="ANNarchy interfaces builder",
        field_type=ANNarchyInterfaceBuilder,
        doc="""Instance of ANNarchy Network interfaces' builder class.""",
        required=False,
    )

    _default_interface_builder_type = ANNarchyInterfaceBuilder

    def start(self):
        SpikeNetParallelApp.start(self)
        ANNarchyApp.start(self)

    def configure(self):
        SpikeNetParallelApp.configure(self)
        ANNarchyApp.configure(self)

    def build(self):
        SpikeNetParallelApp.build(self)

    def configure_simulation(self):
        SpikeNetParallelApp.configure_simulation(self)
        ANNarchyApp.configure_simulation(self)

    def reset(self):
        ANNarchyApp.reset(self)
        SpikeNetParallelApp.reset(self)

    def clean_up(self):
        ANNarchyApp.clean_up(self)
        SpikeNetParallelApp.clean_up(self)


class ANNarchyRemoteParallelApp(ANNarchyApp, SpikeNetRemoteParallelApp):

    """ANNarchyRemoteParallelApp class"""

    interfaces_builder = Attr(
        label="ANNarchy interfaces builder",
        field_type=ANNarchyRemoteInterfaceBuilder,
        doc="""Instance of ANNarchy Network interfaces' builder class.""",
        required=False,
    )

    _default_interface_builder_type = ANNarchyRemoteInterfaceBuilder

    def start(self):
        SpikeNetRemoteParallelApp.start(self)
        ANNarchyApp.start(self)

    def configure(self):
        SpikeNetRemoteParallelApp.configure(self)
        ANNarchyApp.configure(self)

    def build(self):
        SpikeNetRemoteParallelApp.build(self)

    def configure_simulation(self):
        SpikeNetRemoteParallelApp.configure_simulation(self)
        ANNarchyApp.configure_simulation(self)

    def reset(self):
        ANNarchyApp.reset(self)
        SpikeNetRemoteParallelApp.reset(self)

    def clean_up(self):
        ANNarchyApp.clean_up(self)
        SpikeNetRemoteParallelApp.clean_up(self)


class TVBSerialApp(TVBSerialAppBase):

    """TVBSerialApp class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    interfaces_builder = Attr(
        label="TVBANNarchyInterfaces builder",
        field_type=TVBANNarchyInterfaceBuilder,
        doc="""Instance of TVBANNarchyInterfaces' builder class.""",
        required=False,
    )

    spiking_network = Attr(
        label="ANNarchy Network",
        field_type=ANNarchyNetwork,
        doc="""Instance of ANNarchyNetwork class.""",
        required=False
    )

    _default_interface_builder_type = TVBANNarchyInterfaceBuilder


class TVBANNarchySerialOrchestrator(SerialOrchestrator):

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBSerialApp,
        doc="""Application for running TVB serially.""",
        required=True,
        default=TVBSerialApp()
    )

    spikeNet_app = Attr(
        label="ANNarchy Network app",
        field_type=ANNarchySerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=ANNarchySerialApp()
    )
