# -*- coding: utf-8 -*-

from abc import abstractmethod

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder


class SpikeNetApp(NonTVBApp):

    """SpikeNetApp abstract base class"""

    spikeNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingModelBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=False
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    spiking_cosimulator = None

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    def configure(self):
        super(SpikeNetApp, self).configure()
        if not self.spikeNet_builder:
            self.spikeNet_builder = SpikingModelBuilder(self.tvb_cosimulator_serialized, self.spiking_proxy_inds,
                                                        config=self.config, logger=self.logger)
        self.spikeNet_builder.configure()

    @abstractmethod
    def start(self):
        pass

    def build_spiking_network(self):
        self.spiking_network = self.spikeNet_builder.build_spiking_brain()

    def build(self):
        if not self.spikeNet_builder:
            self.configure()
        self.spiking_network = self.build_spiking_network()

    @abstractmethod
    def configure_simulation(self):
        pass

    def run(self, *args, **kwargs):
        self.spiking_network.Run(self.simulation_length, *args, **kwargs)

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass
