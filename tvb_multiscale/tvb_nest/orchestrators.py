# -*- coding: utf-8 -*-

from logging import Logger

from tvb.basic.neotraits._attr import Attr
from tvb.contrib.scripts.utils.file_utils import safe_makedirs

from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetSerialApp, SpikeNetParallelApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.models.default import DefaultExcIOBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest
from tvb_multiscale.tvb_nest.interfaces.interfaces import NESTOutputInterfaces, NESTInputInterfaces
from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, TVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.default import \
    DefaultNESTRemoteInterfaceBuilder, DefaultTVBNESTInterfaceBuilder


class NESTSerialApp(SpikeNetSerialApp):

    """NESTSerialApp class"""

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

    spikeNet_builder = Attr(
        label="NEST Network Builder",
        field_type=NESTNetworkBuilder,
        doc="""Instance of NEST Model Builder.""",
        required=True,
        default=DefaultExcIOBuilder()
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=NESTNetwork,
        doc="""Instance of NESTNetwork class.""",
        required=False
    )

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
        super(NESTSerialApp, self).configure()
        self._spiking_cosimulator.ResetKernel()  # This will restart NEST!
        self.spiking_cosimulator.set_verbosity(self.config.NEST_VERBOCITY)  # don't print all messages from NEST
        self.spikeNet_builder.nest_instance = self.spiking_cosimulator
        # Printing the time progress should only be used when the simulation is run on a local machine:
        #  kernel_config["print_time"] = self.nest_instance.Rank() == 0
        kernel_config = self.config.DEFAULT_NEST_KERNEL_CONFIG.copy()
        if "data_path" in kernel_config.keys():
            safe_makedirs(kernel_config["data_path"])  # Make sure this folder exists
        self.spiking_cosimulator.SetKernelStatus(kernel_config)

    def configure_simulation(self):
        super(NESTSerialApp, self).configure_simulation()
        try:
            self.spiking_cosimulator.Prepare()
        except:
            pass

    def simulate(self, simulation_length=None):
        if simulation_length is None:
            simulation_length = self.simulation_length
        self.spiking_cosimulator.Run(simulation_length)

    def run(self, *args, **kwargs):
        self.configure()
        self.build()

    def clean_up(self):
        # # Integrate NEST for one more NEST time step so that multimeters get the last time point
        # # unless you plan to continue simulation later
        # simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        self.spiking_cosimulator.Cleanup()

    def stop(self):
        pass

    def reset(self):
        super(NESTSerialApp, self).reset()
        self.spiking_cosimulator.ResetKernel()


class NESTParallelApp(NESTSerialApp, SpikeNetParallelApp):

    """NESTParallelApp class"""

    interfaces_builder = Attr(
        label="NEST interfaces builder",
        field_type=NESTProxyNodesBuilder,
        doc="""Instance of NEST Network interfaces' builder class.""",
        required=False,
        default=DefaultNESTRemoteInterfaceBuilder()
    )

    output_interfaces = Attr(
        label="NEST Network output interfaces",
        field_type=NESTOutputInterfaces,
        doc="""Instance of output NEST Network interfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="NEST Network input interfaces",
        field_type=NESTInputInterfaces,
        doc="""Instance of input NEST Network interfaces.""",
        required=False
    )

    _default_interface_builder = NESTProxyNodesBuilder

    def build(self):
        SpikeNetParallelApp.build(self)

    def reset(self):
        NESTSerialApp.reset(self)
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

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    interfaces_builder = Attr(
        label="TVBNESTInterfaces builder",
        field_type=TVBNESTInterfaceBuilder,
        doc="""Instance of TVBNESTInterfaces' builder class.""",
        required=True,
        default=DefaultTVBNESTInterfaceBuilder()
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=NESTNetwork,
        doc="""Instance of NESTNetwork class.""",
        required=False
    )

    _default_interface_builder = TVBNESTInterfaceBuilder


class TVBNESTSerialOrchestrator(SerialOrchestrator):

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
