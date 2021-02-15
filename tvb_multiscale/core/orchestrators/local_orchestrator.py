# -*- coding: utf-8 -*-

import numpy as np
from tvb.basic.neotraits._attr import Attr, NArray, Float

from tvb_multiscale.core.orchestrators.base import Orchestrator
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBApp


class SerialOrchestrator(Orchestrator):

    """SerialOrchestrator base class"""

    tvb_app = Attr(
        label="TVB app",
        field_type=TVBApp,
        doc="""Application for running TVB.""",
        required=False
    )

    spikeNet_app = Attr(
        label="Spiking Network app",
        field_type=SpikeNetApp,
        doc="""Application for running a Spiking Network (co)simulator.""",
        required=False
    )

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                      if the co-simulator nodes are modelled exclusively by the co-simulator, 
                                      i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    transformer_app = Attr(
        label="Spiking Network app",
        field_type=SpikeNetApp,
        doc="""Application for running a Spiking Network (co)simulator.""",
        required=False
    )

    @property
    def tvb_cosimulator(self):
        return self.tvb_app.tvb_cosimulator

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

    def build_cosimulators(self):
        self.tvb_app.build()
        self.spikeNet_app.tvb_cosimulator_serialized = self.tvb_app.serialize_tvb_cosimulator()
        self.spikeNet_app.build()

    def build_interfaces(self):
        self.tvb_app.tvb_interfaces_builder.spiking_network = self.spiking_network
        self.tvb_app.build_interfaces()

    def build(self):
        self.build_cosimulators()
        self.build_interfaces()

    def configure_simulation(self):
        self.tvb_app.configure_simulation()
        self.simulation_length = self.tvb_app.simulation_length
        self.spikeNet_app.simulation_length = self.simulation_length
        self.spikeNet_app.configure_simulation()

    def run(self):
        self.build()
        self.configure_simulation()
        self.tvb_app.run()

    def stop(self):
        self.tvb_app.stop()
        self.spikeNet_app.stop()

    def clean_up(self):
        self.tvb_app.clean_up()
        self.spikeNet_app.clean_up()
