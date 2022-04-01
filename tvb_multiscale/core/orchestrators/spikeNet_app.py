# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta

import numpy as np

from tvb.basic.neotraits._attr import Attr, Int

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import SpikeNetOutputInterfaces, SpikeNetInputInterfaces


class SpikeNetApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetApp abstract base class"""

    spikeNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingNetworkBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=True
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    population_order = Int(
        label="Spiking Network populations' order",
        doc="""Size of spiking populations.""",
        required=False,
        default=None
    )

    tvb_cosimulator_serialized = Attr(
        label="Serialized TVB CoSimunlator",
        field_type=dict,
        doc="""Serialized TVB CoSimulator (dict).""",
        required=False,
        default={}
    )

    spiking_cosimulator = None

    populations = []
    populations_sizes = []

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    @property
    def _spiking_cosimulator(self):
        if self.spiking_cosimulator is None:
            self.start()
        return self.spiking_cosimulator

    @property
    def _spiking_network(self):
        if self.spiking_network is None:
            self.build_spiking_network()
        return self.spiking_network

    @property
    def _serialized_tvb_cosimulator(self):
        if len(self.tvb_cosimulator_serialized) == 0:
            self.load_serialized_tvb_cosimulator()
        return self.tvb_cosimulator_serialized

    @property
    def tvb_dt(self):
        return self._serialized_tvb_cosimulator.get("integrator.dt", self.config.DEFAULT_DT)

    @property
    def tvb_model(self):
        return self._serialized_tvb_cosimulator.get("model", "")

    @property
    def tvb_model_state_variables(self):
        return self._serialized_tvb_cosimulator.get("model.state_variables", [])

    @property
    def tvb_model_cvar(self):
        return self._serialized_tvb_cosimulator.get("model.cvar", [])

    @property
    def number_of_regions(self):
        return self._serialized_tvb_cosimulator.get("connectivity.number_of_regions", 0)

    @property
    def region_labels(self):
        return self._serialized_tvb_cosimulator.get("connectivity.region_labels", np.array([]))

    @property
    def tvb_coupling_a(self):
        return self._serialized_tvb_cosimulator.get("coupling.a", np.array([1.0 / 256]))

    @property
    def tvb_weights(self):
        return self._serialized_tvb_cosimulator.get("connectivity.weights", np.zeros((0, 0)))

    @property
    def tvb_delays(self):
        return self._serialized_tvb_cosimulator.get("connectivity.delays", self._tvb_dt * np.ones((0, 0)))

    def configure(self):
        super(SpikeNetApp, self).configure()
        self.spikeNet_builder.config = self.config
        self.spikeNet_builder.logger = self.logger
        if hasattr(self, "population_order"):
            self.spikeNet_builder.population_order = self.population_order
        else:
            self.population_order = self.spikeNet_builder.population_order
        self.spikeNet_builder.spiking_nodes_inds = self.spiking_proxy_inds

    def build_spiking_network(self):
        self.spikeNet_builder.tvb_serial_sim = self._serialized_tvb_cosimulator
        self.spikeNet_builder.configure()
        self.spiking_network = self.spikeNet_builder.build()

    @property
    def populations_sizes(self):
        return self.spiking_network.populations_sizes

    def build(self):
        self.build_spiking_network()

    def configure_simulation(self):
        self.spiking_network.configure()

    def run(self, *args, **kwargs):
        self.configure()
        self.build()
        self.configure_simulation()
        self.simulate()

    def reset(self):
        self.spiking_network = None

    def info(self, recursive=0):
        info = super(SpikeNetApp, self).info(recursive=recursive)
        info["spiking_cosimulator"] = self.spiking_cosimulator
        return info

    def info_details(self, recursive=0, **kwargs):
        info = super(SpikeNetApp, self).info_details(recursive=recursive, **kwargs)
        info["spiking_cosimulator"] = self.spiking_cosimulator
        return info


class SpikeNetSerialApp(SpikeNetApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetSerialApp abstract base class"""

    pass


class SpikeNetParallelApp(SpikeNetApp, ABC):
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
        self.interfaces_builder.config = self.config
        self.interfaces_builder.logger = self.logger
        if hasattr(self.interfaces_builder, "default_config"):
            self.interfaces_builder.default_config()
        self.interfaces_builder.configure()

    @property
    def _interfaces_builder(self):
        if not isinstance(self.interfaces_builder, self._default_interface_builder):
            self.interfaces_builder = self._default_interface_builder(config=self.config, logger=self.logger)
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
