# -*- coding: utf-8 -*-

import ray

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.ray.client import RayClient
from tvb_multiscale.core.tvb.cosimulator.cosimulator_ray import CoSimulatorParallelRay
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorRayBuilder
from tvb_multiscale.core.spiking_models.builders.ray_builder import RaySpikingNetworkBuilder
from tvb_multiscale.core.interfaces.tvb.ray_builder import RayTVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp


class SpikeNetRayApp(SpikeNetApp):

    spikeNet_builder = Attr(
        label="Ray Spiking Network Builder",
        field_type=RaySpikingNetworkBuilder,
        doc="""Instance of RaySpikingNetworkBuilder.""",
        required=False,
        default=None
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=RayClient,
        doc="""Instance of Spiking Network RayClient class.""",
        required=False,
        default=None
    )

    spiking_cosimulator_server = None

    def start(self):
        ray.init(ignore_reinit_error=True)
        super(SpikeNetRayApp, self).start()

    def stop(self):
        ray.kill(self.spiking_cosimulator_server)
        super(SpikeNetRayApp, self).stop()


class TVBRayApp(TVBSerialApp):

    cosimulator_builder = Attr(
        label="TVB CoSimulatorRayBuilder",
        field_type=CoSimulatorRayBuilder,
        doc="""Instance of TVB CoSimulatorRayBuilder class.""",
        required=False,
        default=CoSimulatorRayBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulatorParallelRay",
        field_type=CoSimulatorParallelRay,
        doc="""Instance of TVB CoSimulatorRay for Ray cosimulation.""",
        required=False,
        default=None
    )

    interfaces_builder = Attr(
        label="RayTVBSpikeNetInterfaceBuilder builder",
        field_type=RayTVBSpikeNetInterfaceBuilder,
        doc="""Instance of RayTVBSpikeNetInterfaceBuilder' builder class.""",
        required=False,
        default=None
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=RayClient,
        doc="""Instance of Spiking Network RayClient class.""",
        required=False,
        default=None
    )

    _cosimulator_builder_type = CoSimulatorRayBuilder
    _default_interface_builder = RayTVBSpikeNetInterfaceBuilder

    def start(self):
        ray.init(ignore_reinit_error=True)
        super(TVBRayApp, self).start()

    def stop(self):
        super(TVBRayApp, self).stop()


class RayOrchestrator(SerialOrchestrator):

    """SerialOrchestrator base class"""

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBRayApp,
        doc="""Application for running TVB with Ray interface to SpikingNetwork.""",
        required=False,
        default=TVBRayApp()
    )

    spikeNet_app = Attr(
        label="Spiking Network app",
        field_type=SpikeNetRayApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=None
    )

    def start(self):
        ray.init(ignore_reinit_error=True)
        self.tvb_app.start()
        self.spikeNet_app.start()

    def link_spikeNet_to_TVB_cosimulator(self):
        self.tvb_app.cosimulator.spiking_simulator = self.spikeNet_app.spiking_cosimulator

    def build_cosimulators(self):
        self.tvb_app.build_tvb_simulator()
        self.spikeNet_app.tvb_cosimulator_serialized = self.tvb_app.serialize_tvb_cosimulator()
        self.spikeNet_app.build_spiking_network()
        self.link_spikeNet_to_TVB_cosimulator()

    @property
    def populations_sizes(self):
        return self.spikeNet_app.populations_sizes

    def build_interfaces(self):
        self.tvb_app.spiking_network = self.spiking_network
        self.tvb_app.build_interfaces()

    def configure_simulation(self):
        self.tvb_app.configure_simulation()
        self.simulation_length = self.tvb_app.simulation_length
        self.synchronization_time = self.tvb_app.synchronization_time
        self.spikeNet_app.simulation_length = self.simulation_length
        self.spikeNet_app.synchronization_time = self.synchronization_time
        self.spikeNet_app.configure_simulation()

    def simulate(self):
        self.configure_simulation()
        self.tvb_app.simulate()

    def run(self):
        self.configure()
        self.build()
        self.simulate()

    def clean_up(self):
        self.tvb_app.clean_up()
        self.spikeNet_app.clean_up()

    def reset(self):
        self.tvb_app.reset()
        self.spikeNet_app.reset()

    def stop(self):
        self.tvb_app.stop()
        self.spikeNet_app.stop()
        if ray.is_initialized():
            ray.shutdown()
