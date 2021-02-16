# -*- coding: utf-8 -*-

from copy import deepcopy

from tvb.basic.neotraits._attr import Attr
from tvb.contrib.scripts.utils.file_utils import safe_makedirs

from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator

from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTModelBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest
from tvb_multiscale.tvb_nest.interfaces.builders import NESTInterfacesBuilder, TVBNESTInterfacesBuilder
from tvb_multiscale.tvb_nest.interfaces.interfaces import NESTOutputInterfaces, NESTInputInterfaces


class NESTApp(SpikeNetApp):

    """NESTApp base class"""

    spiking_model_builder = Attr(
        label="NEST Network Builder",
        field_type=NESTModelBuilder,
        doc="""Instance of NEST Model Builder.""",
        required=False
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
        super(NESTApp, self).configure()
        self.nest_instance.ResetKernel()  # This will restart NEST!
        self.nest_instance.set_verbosity(self.config.NEST_VERBOCITY)  # don't print all messages from NEST
        self.spikeNet_builder.nest_instance = self.spiking_cosimulator
        # Printing the time progress should only be used when the simulation is run on a local machine:
        #  kernel_config["print_time"] = self.nest_instance.Rank() == 0
        kernel_config = deepcopy(self.default_kernel_config)
        if "data_path" in kernel_config.keys():
            safe_makedirs(kernel_config["data_path"])  # Make sure this folder exists
        self.nest_instance.SetKernelStatus(kernel_config)

    def configure_simulation(self):
        self.spikeNet_builder.update_spiking_dt()
        self.spikeNet_builder.update_default_min_delay()
        self.nest_instance.SetKernelStatus({"resolution": self.spiking_dt})

    def run(self, *args, **kwargs):
        self.nest_network.Run(self.simulation_length, *args, **kwargs)

    def clean_up(self):
        # # Integrate NEST for one more NEST time step so that multimeters get the last time point
        # # unless you plan to continue simulation later
        # simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        self.nest_instance.Cleanup()

    def stop(self):
        pass


class NESTSerialApp(NESTApp):

    """NESTSerialApp class"""

    pass


class NESTParallelApp(SpikeNetApp):

    """NESTParallelApp class"""

    interfaces_builder = Attr(
        label="NEST interfaces builder",
        field_type=NESTInterfacesBuilder,
        doc="""Instance of NEST Network interfaces' builder class.""",
        required=False
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

    def build_interfaces(self):
        self.output_interfaces, self.input_interfaces = self.interfaces_builder.build()

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.interfaces_builder.spiking_network = self.spiking_network
        self.build_interfaces()


class TVBSerialApp(TVBSerialAppBase):

    """TVBSerialApp class"""

    interfaces_builder = Attr(
        label="TVBNESTInterfaces builder",
        field_type=TVBNESTInterfacesBuilder,
        doc="""Instance of TVBNESTInterfaces' builder class.""",
        required=False,
        defualt=TVBNESTInterfacesBuilder()
    )


class TVBNESTSerialOrchestrator(SerialOrchestrator):

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBSerialApp,
        doc="""Application for running TVB serially.""",
        required=False,
        default=TVBSerialApp()
    )

    spikeNet_app = Attr(
        label="NEST Network app",
        field_type=NESTSerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=NESTSerialApp()
    )
