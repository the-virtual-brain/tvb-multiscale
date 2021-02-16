# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr, Int

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import SpikeNetOutputInterfaces, SpikeNetInputInterfaces


class SpikeNetApp(NonTVBApp):
    __metaclass__ = ABCMeta

    """SpikeNetApp abstract base class"""

    spiking_model_builder = Attr(
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

    population_order = Int(
        label="Spiking Network populations' order",
        doc="""Instance of Spiking Network class.""",
        required=True,
        default=100
    )

    spiking_cosimulator = None

    populations = []
    populations_sizes = []

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    def configure(self):
        super(SpikeNetApp, self).configure()
        if not self.spiking_model_builder:
            self.spiking_model_builder = SpikingModelBuilder(self.tvb_cosimulator_serialized, self.spiking_proxy_inds,
                                                             config=self.config, logger=self.logger)
            self.spiking_model_builder.population_order = self.population_order
        self.spiking_model_builder.configure()

    def build_spiking_network(self):
        self.spiking_network = self.spiking_model_builder.build_spiking_brain()

    def get_populations_and_sizes(self):
        self.populations = []
        self.populations_sizes = []
        for pop in self.spiking_model_builder.populations:
            self.populations.append(pop["label"])
            try:
                self.populations_sizes.append(int(np.round(pop["scale"] * self.spiking_model_builder.population_order)))
            except:
                pass
        return self.populations, self.populations_sizes

    def build(self):
        if not self.spiking_model_builder:
            self.configure()
        self.spiking_network = self.build_spiking_network()

    @abstractmethod
    def configure_simulation(self):
        pass

    def run(self, *args, **kwargs):
        self.spiking_network.Run(self.simulation_length, *args, **kwargs)


class SpikeNetSerialApp(SpikeNetApp):
    __metaclass__ = ABCMeta

    """SpikeNetSerialApp abstract base class"""

    pass


class SpikeNetParallelApp(SpikeNetApp):
    __metaclass__ = ABCMeta

    """SpikeNetParallelApp abstract base class"""

    interfaces_builder = Attr(
        label="Spiking Network interfaces builder",
        field_type=SpikeNetInterfaceBuilder,
        doc="""Instance of Spiking Network interfaces' builder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Spiking Network output interfaces",
        field_type=SpikeNetOutputInterfaces,
        doc="""Instance of output Spiking Network interfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="Spiking Network input interfaces",
        field_type=SpikeNetInputInterfaces,
        doc="""Instance of input Spiking Network interfaces.""",
        required=False
    )

    def build_interfaces(self):
        self.output_interfaces, self.input_interfaces = self.interfaces_builder.build()

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.interfaces_builder.spiking_network = self.spiking_network
        self.build_interfaces()
