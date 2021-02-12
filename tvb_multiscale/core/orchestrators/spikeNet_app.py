# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Attr, Float, NArray

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.core.tvb.cosimulator import CoSimulator


class SpikeNetApp(NonTVBApp):

    """SpikeNetApp abstract base class"""

    spikNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingModelBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=False
    )

    spiking_network = Attr(
        label="TVB Simulator",
        field_type=CoSimulator,
        doc="""Instance of TVB (Co)Simulator.""",
        required=False
    )

    spiking_cosimulator = None

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    def configure(self):
        super(SpikeNetApp, self).configure()
        if not self.spikNet_builder:
            self.spikNet_builder = SpikingModelBuilder(self.tvb_cosimulator_serialized, self.spiking_proxy_inds,
                                                       config=self.config, logger=self.logger)
        self.spikNet_builder.configure()

    @abstractmethod
    def start(self):
        pass

    def build_spiking_network(self):
        self.spiking_network = self.spikNet_builder.build_spiking_brain()

    def build(self):
        if not self.spikNet_builder:
            self.configure()
        self.spiking_network = self.build_spiking_network()

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass