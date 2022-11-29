# -*- coding: utf-8 -*-

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import Orchestrator
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetSerialApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp


class SerialOrchestrator(Orchestrator):

    """SerialOrchestrator base class"""

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBSerialApp,
        doc="""Application for running TVB serially.""",
        required=False,
        default=TVBSerialApp()
    )

    spikeNet_app = Attr(
        label="Spiking Network app",
        field_type=SpikeNetSerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=None
    )

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                  if the co-simulator nodes are modelled exclusively by the co-simulator, 
                                  i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    @property
    def tvb_cosimulator(self):
        return self.tvb_app.cosimulator

    @property
    def dt(self):
        return self.tvb_app.cosimulator.integrator.dt

    @property
    def spiking_network(self):
        return self.spikeNet_app.spiking_network

    def configure(self):
        super(Orchestrator, self).configure()
        self.tvb_app.setup_from_orchestrator(self)
        self.tvb_app.configure()
        self.spikeNet_app.setup_from_orchestrator(self)
        self.spikeNet_app.configure()

    def start(self):
        self.tvb_app.start()
        self.spikeNet_app.start()

    def link_spikeNet_to_TVB_cosimulator(self):
        self.tvb_app.cosimulator.simulate_spiking_simulator = self.spikeNet_app.simulate

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

    def stop(self):
        self.tvb_app.stop()
        self.spikeNet_app.stop()

    def clean_up(self):
        self.tvb_app.clean_up()
        self.spikeNet_app.clean_up()

    def reset(self):
        self.tvb_app.reset()
        self.spikeNet_app.reset()
