# -*- coding: utf-8 -*-

from logging import Logger

import ray

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.ray.client import RayClient
from tvb_multiscale.core.orchestrators.ray_apps import \
    SpikeNetRayApp, TVBRayApp as TVBRayAppBase, RayOrchestrator

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.nest_ray import RayNESTServer, RayNESTClient
from tvb_multiscale.tvb_nest.nest_models.builders.ray_builder import RayNESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import configure_nest_kernel
from tvb_multiscale.tvb_nest.interfaces.ray_builder import RayTVBNESTInterfaceBuilder


class NESTRayApp(SpikeNetRayApp):

    """NESTRayApp class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Config class instance.""",
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
        label="Ray NEST Network Builder",
        field_type=RayNESTNetworkBuilder,
        doc="""Instance of NEST Model Builder.""",
        required=True,
        default=RayNESTNetworkBuilder()
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=RayClient,
        doc="""Instance of NESTNetwork RayClient class.""",
        required=False,
        default=None
    )

    spiking_cosimulator_server = Attr(
        label="NEST Server",
        field_type=ray.actor.ActorHandle,
        doc="""Handle to an instance of a server to a running NEST.""",
        required=False,
        default=None
    )

    spiking_cosimulator = Attr(
        label="NEST Client",
        field_type=RayNESTClient,
        doc="""Instance of a client to a running NEST server.""",
        required=False,
        default=None
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
        super(SpikeNetRayApp, self).start()
        RayNESTServerActor = ray.remote(RayNESTServer)
        try:
            self.spiking_cosimulator_server = \
                RayNESTServerActor.options(name="nest_server",
                                           num_cpus=self.config.DEFAULT_LOCAL_NUM_THREADS).remote(config=self.config)
        except:
            self.spiking_cosimulator_server = ray.get_actor("nest_server")
        self.spiking_cosimulator = RayNESTClient(self.spiking_cosimulator_server)

    def configure(self):
        super(NESTRayApp, self).configure()
        self.spiking_cosimulator = configure_nest_kernel(self._spiking_cosimulator, self.config)
        self.spikeNet_builder.nest_instance = self.spiking_cosimulator

    def configure_simulation(self):
        super(NESTRayApp, self).configure_simulation()
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

    def reset(self):
        super(NESTRayApp, self).reset()
        self.spiking_cosimulator.ResetKernel()

    def stop(self):
        ray.kill(self.spiking_cosimulator_server)
        self.spiking_cosimulator = None
        super(NESTRayApp, self).stop()


class TVBRayApp(TVBRayAppBase):

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
        label="RayTVBNESTInterfaceBuilder builder",
        field_type=RayTVBNESTInterfaceBuilder,
        doc="""Instance of RayTVBNESTInterfaces' builder class.""",
        required=True,
        default=RayTVBNESTInterfaceBuilder()
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=RayClient,
        doc="""Instance of NESTNetwork RayClient class.""",
        required=False,
        default=None
    )

    _default_interface_builder = RayTVBNESTInterfaceBuilder


class TVBNESTRayOrchestrator(RayOrchestrator):

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
        field_type=TVBRayApp,
        doc="""Application for running TVB with Ray.""",
        required=True,
        default=TVBRayApp()
    )

    spikeNet_app = Attr(
        label="Ray NEST Network app",
        field_type=NESTRayApp,
        doc="""Application for running a ray NEST Network (co)simulator.""",
        required=True,
        default=NESTRayApp()
    )
