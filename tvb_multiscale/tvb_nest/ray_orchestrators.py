# -*- coding: utf-8 -*-

from logging import Logger

import ray

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.ray.client import RayClient
from tvb_multiscale.core.orchestrators.ray_apps import \
    SpikeNetRayApp, TVBRayApp as TVBRayAppBase, RayOrchestrator

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.server_client.ray import RayNESTClient
from tvb_multiscale.tvb_nest.nest_models.server_client.ray import RayNESTServer
from tvb_multiscale.tvb_nest.interfaces.ray_builder import RayTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.orchestrators import NESTApp


class NESTRayApp(NESTApp, SpikeNetRayApp):

    """NESTRayApp class"""

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

    def start(self):
        SpikeNetRayApp.start(self)
        RayNESTServerActor = ray.remote(RayNESTServer)
        try:
            self.spiking_cosimulator_server = \
                RayNESTServerActor.options(name="nest_server",
                                           num_cpus=self.config.DEFAULT_LOCAL_NUM_THREADS).remote(config=self.config)
        except:
            self.spiking_cosimulator_server = ray.get_actor("nest_server")
        self.spiking_cosimulator = RayNESTClient(self.spiking_cosimulator_server)

    def configure(self):
        SpikeNetRayApp.configure(self)
        NESTApp.configure(self)

    def configure_simulation(self):
        SpikeNetRayApp.configure_simulation(self)
        NESTApp.configure_simulation(self)

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
        NESTApp.reset(self)
        SpikeNetRayApp.reset(self)

    def stop(self):
        NESTApp.stop(self)
        SpikeNetRayApp.stop(self)


class TVBRayApp(TVBRayAppBase):

    """TVBSerialApp class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
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
