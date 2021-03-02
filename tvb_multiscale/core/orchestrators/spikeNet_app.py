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

    tvb_cosimulator_serialized = Attr(
        label="Serialized TVB CoSimunlator",
        field_type=dict,
        doc="""Serialized TVB CoSimulator (dict).""",
        required=False
    )

    spiking_cosimulator = None

    populations = []
    populations_sizes = []

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    def get_serialized_tvb_cosimulator(self):
        return self.load_tvb_simulator_serialized()

    @property
    def _spiking_cosimulator(self):
        if self.spiking_simulator is None:
            self.start()
        return self.spiking_cosimulator

    @property
    def _serialized_tvb_cosimulator(self):
        if not isinstance(self.tvb_cosimulator_serialized, dict):
            self.get_serialized_tvb_cosimulator()
        return self.tvb_cosimulator_serialized

    def configure(self):
        super(SpikeNetApp, self).configure()
        self.spiking_model_builder.population_order = self.population_order
        self.spiking_model_builder.configure()

    def build_spiking_network(self):
        self.spiking_network = self.spiking_model_builder.build()

    def get_number_of_neurons_per_region_and_populations(self, reg_inds_or_lbls=None, pop_inds_or_lbls=None):
        return self._spiking_network.get_number_of_neurons_per_region_and_population(reg_inds_or_lbls, pop_inds_or_lbls)

    @property
    def number_of_neurons_per_region_and_population(self):
        return self.get_number_of_neurons_per_region_and_population()

    def build(self):
        self.build_spiking_network()

    @abstractmethod
    def configure_simulation(self):
        pass

    def run(self, *args, **kwargs):
        self.configure()
        self.build()
        self.simulate()

    def reset(self):
        self.spiking_network = None


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

    _default_interface_builder = SpikeNetInterfaceBuilder
    _interfaces_built = False

    def configure_interfaces_builder(self):
        self.interfaces_builder.spiking_network = self._spiking_network

    @property
    def _interfaces_builder(self):
        if not isinstance(self.interfaces_builder, self._default_interface_builder):
            self.interfaces_builder = self._default_interface_builder()
            self.configure_interfaces_builder()
        return self.interfaces_builder

    def build_interfaces(self):
        if not self._interfaces_built:
            self.output_interfaces, self.input_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.build_interfaces()

    def reset(self):
        self._interfaces_built = False
        super(SpikeNetParallelApp, self).reset()
