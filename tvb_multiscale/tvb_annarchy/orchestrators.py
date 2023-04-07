# -*- coding: utf-8 -*-

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetSerialApp, SpikeNetParallelApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator

from tvb_multiscale.tvb_annarchy.config import Config, CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.base import ANNarchyNetworkBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.default import DefaultExcIOBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import load_annarchy
from tvb_multiscale.tvb_annarchy.interfaces.interfaces import ANNarchyOutputInterfaces, ANNarchyInputInterfaces
from tvb_multiscale.tvb_annarchy.interfaces.builders import ANNarchyProxyNodesBuilder, TVBANNarchyInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models.default import \
    DefaultANNarchyRemoteInterfaceBuilder, DefaultTVBANNarchyInterfaceBuilder


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
        required=True,
        default=DefaultExcIOBuilder()
    )

    spiking_network = Attr(
        label="ANNarchy Network",
        field_type=ANNarchyNetwork,
        doc="""Instance of ANNarchyNetwork class.""",
        required=False
    )

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
        self.spikeNet_builder.annarchy_instance = self.spiking_cosimulator
        self.spikeNet_builder.update_spiking_dt()
        self.spikeNet_builder.update_default_min_delay()
        kwargs["dt"] = self.spikeNet_builder.spiking_dt
        kwargs["seed"] = kwargs.pop("seed", self.config.ANNARCHY_SEED)
        kwargs["verbose"] = kwargs.pop("verbose", self.config.VERBOSE)
        self.annarchy_instance.setup(**kwargs)

    def configure_simulation(self):
        self.spiking_network.network_path = self.network_path

    def simulate(self, simulation_length=None):
        if simulation_length is None:
            simulation_length = self.simulation_length
        self.spiking_network.Run(simulation_length)

    def clean_up(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.annarchy_instance.clear()


class ANNarchySerialApp(ANNarchyApp, SpikeNetSerialApp):

    """ANNarchySerialApp class"""

    def configure(self, **kwargs):
        SpikeNetSerialApp.configure(self)
        ANNarchyApp.configure(self)

    def configure_simulation(self):
        SpikeNetSerialApp.configure_simulation(self)
        ANNarchyApp.configure_simulation(self)

    def run(self, *args, **kwargs):
        self.configure(**kwargs)
        self.build()

    def reset(self):
        SpikeNetSerialApp.reset(self)
        ANNarchyApp.reset(self)


class ANNarchyParallelApp(ANNarchySerialApp, SpikeNetParallelApp):

    """ANNarchyParallelApp class"""

    interfaces_builder = Attr(
        label="ANNarchy interfaces builder",
        field_type=ANNarchyProxyNodesBuilder,
        doc="""Instance of ANNarchy Network interfaces' builder class.""",
        required=False,
        default=DefaultANNarchyRemoteInterfaceBuilder()
    )

    output_interfaces = Attr(
        label="ANNarchy Network output interfaces",
        field_type=ANNarchyOutputInterfaces,
        doc="""Instance of output ANNarchy Network interfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="ANNarchy Network input interfaces",
        field_type=ANNarchyInputInterfaces,
        doc="""Instance of input ANNarchy Network interfaces.""",
        required=False
    )

    _default_interface_builder = ANNarchyProxyNodesBuilder

    def build(self):
        SpikeNetParallelApp.build(self)

    def reset(self):
        ANNarchySerialApp.reset(self)
        SpikeNetParallelApp.reset(self)


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
        required=True,
        default=DefaultTVBANNarchyInterfaceBuilder()
    )

    spiking_network = Attr(
        label="ANNarchy Network",
        field_type=ANNarchyNetwork,
        doc="""Instance of ANNarchyNetwork class.""",
        required=False
    )

    _default_interface_builder = TVBANNarchyInterfaceBuilder


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

    def simulate(self):
        self.configure_simulation()
        self.tvb_app.simulate()
