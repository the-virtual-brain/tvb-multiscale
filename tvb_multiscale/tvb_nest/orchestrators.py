# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.orchestrators.spikeNet_app import \
    SpikeNetSerialApp, SpikeNetParallelApp, SpikeNetRemoteParallelApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel
from tvb_multiscale.tvb_nest.interfaces.interfaces import NESTReceiverInterface, NESTSenderInterface
from tvb_multiscale.tvb_nest.interfaces.builders import \
    NESTInterfaceBuilder, NESTRemoteInterfaceBuilder, TVBNESTInterfaceBuilder


class NESTApp(HasTraits):

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Config class instance.""",
        required=True,
        default=CONFIGURED
    )

    spikeNet_builder = Attr(
        label="NEST Network Builder",
        field_type=NESTNetworkBuilder,
        doc="""Instance of NEST Model Builder.""",
        required=False
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=NESTNetwork,
        doc="""Instance of NESTNetwork class.""",
        required=False
    )

    _spikeNet_builder_type = NESTNetworkBuilder
    _spikeNet_type = NESTNetwork

    @property
    def nest_instance(self):
        return self.spiking_cosimulator

    @property
    def nest_network(self):
        return self.spiking_network

    @property
    def nest_model_builder(self):
        return self.spikeNet_builder

    def start(self):
        self.spiking_cosimulator = load_nest(self.config)

    def configure(self):
        self.spiking_cosimulator = configure_nest_kernel(self._spiking_cosimulator, self.config)

    def configure_simulation(self):
        try:
            self.spiking_cosimulator.Prepare()
        except:
            pass

    def clean_up(self):
        # # Integrate NEST for one more NEST time step so that multimeters get the last time point
        # # unless you plan to continue simulation later
        # simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        self.spiking_cosimulator.Cleanup()

    def reset(self):
        self.spiking_cosimulator.ResetKernel()


class NESTSerialApp(NESTApp, SpikeNetSerialApp):

    """NESTSerialApp class"""

    def start(self):
        SpikeNetSerialApp.start(self)
        NESTApp.start(self)

    def configure(self):
        SpikeNetSerialApp.configure(self)
        NESTApp.configure(self)

    def configure_simulation(self):
        SpikeNetSerialApp.configure_simulation(self)
        NESTApp.configure_simulation(self)

    def reset(self):
        NESTApp.reset(self)
        SpikeNetSerialApp.reset(self)

    def clean_up(self):
        NESTApp.clean_up(self)
        SpikeNetSerialApp.clean_up(self)


class NESTParallelApp(NESTApp, SpikeNetParallelApp):

    """NESTParallelApp class"""

    interfaces_builder = Attr(
        label="NEST interfaces builder",
        field_type=NESTInterfaceBuilder,
        doc="""Instance of NEST Network interfaces' builder class.""",
        required=False,
    )

    _default_interface_builder_type = NESTInterfaceBuilder

    def start(self):
        SpikeNetParallelApp.start(self)
        NESTApp.start(self)

    def configure(self):
        SpikeNetParallelApp.configure(self)
        NESTApp.configure(self)

    def build(self):
        SpikeNetParallelApp.build(self)

    def configure_simulation(self):
        SpikeNetParallelApp.configure_simulation(self)
        NESTApp.configure_simulation(self)

    def reset(self):
        NESTApp.reset(self)
        SpikeNetParallelApp.reset(self)

    def clean_up(self):
        NESTApp.clean_up(self)
        SpikeNetParallelApp.clean_up(self)


class NESTRemoteParallelApp(NESTApp, SpikeNetRemoteParallelApp):

    """NESTRemoteParallelApp class"""

    interfaces_builder = Attr(
        label="NEST interfaces builder",
        field_type=NESTRemoteInterfaceBuilder,
        doc="""Instance of NEST Network interfaces' builder class.""",
        required=False,
    )

    _default_interface_builder_type = NESTRemoteInterfaceBuilder

    def start(self):
        SpikeNetRemoteParallelApp.start(self)
        NESTApp.start(self)

    def configure(self):
        SpikeNetRemoteParallelApp.configure(self)
        NESTApp.configure(self)

    def build(self):
        SpikeNetRemoteParallelApp.build(self)

    def configure_simulation(self):
        SpikeNetRemoteParallelApp.configure_simulation(self)
        NESTApp.configure_simulation(self)

    def reset(self):
        NESTApp.reset(self)
        SpikeNetRemoteParallelApp.reset(self)

    def clean_up(self):
        NESTApp.clean_up(self)
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
        label="TVBNESTInterfaces builder",
        field_type=TVBNESTInterfaceBuilder,
        doc="""Instance of TVBNESTInterfaces' builder class.""",
        required=False
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=NESTNetwork,
        doc="""Instance of NESTNetwork class.""",
        required=False
    )

    _default_interface_builder_type = TVBNESTInterfaceBuilder


class TVBNESTSerialOrchestrator(SerialOrchestrator):

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
        label="NEST Network app",
        field_type=NESTSerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=NESTSerialApp()
    )
