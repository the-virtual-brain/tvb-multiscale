# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import App
from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.tvb.cosimulator_serialization import dump_serial_tvb_cosimulator, serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder, TVBSpikeNetInterfaceBuilder
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class TVBApp(App):

    """TVBApp base class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulator Builder",
        field_type=CoSimulatorBuilder,
        doc="""Instance of TVB CoSimulator Builder class.""",
        required=False,
        default=CoSimulatorBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulator,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )

    interfaces_builder = Attr(
        label="TVB Interfaces builder",
        field_type=TVBInterfaceBuilder,
        doc="""Instance of TVB interfaces' builder class.""",
        required=False
    )

    results = None

    _interfaces_built = False

    _default_interface_builder = TVBInterfaceBuilder

    def setup_from_orchestrator(self, orchestrator):
        super(TVBApp, self).setup_from_another_app(orchestrator)

    def start(self):
        from tvb.basic.profile import TvbProfile
        TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    @property
    def _cosimulator_builder(self):
        if not isinstance(self.cosimulator_builder, CoSimulatorBuilder):
            self.cosimulator_builder = CoSimulatorBuilder(config=self.config)
            self.cosimulator_builder.configure()
        return self.cosimulator_builder

    def configure(self):
        super(TVBApp, self).configure()
        self.cosimulator_builder = CoSimulatorBuilder(config=self.config)
        self.cosimulator_builder.configure()

    @property
    def _cosimulator(self):
        if not isinstance(self.cosimulator, CoSimulator):
            self.build_tvb_simulator()
        return self.cosimulator

    def configure_interfaces_builder(self):
        self.interfaces_builder.cosimulator = self._cosimulator
        self.interfaces_builder.exclusive_nodes = self.exclusive_nodes
        self.interfaces_builder.spiking_proxy_inds = self.spiking_proxy_inds
        self.interfaces_builder.configure()

    @property
    def _interfaces_builder(self):
        if not isinstance(self.interfaces_builder, self._default_interface_builder):
            self.interfaces_builder = self._default_interface_builder()
            self.configure_interfaces_builder()
        return self.interfaces_builder

    @property
    def tvb_dt(self):
        return self._cosimulator.integrator.dt

    @property
    def tvb_model(self):
        return self._cosimulator.model

    @property
    def tvb_model_state_variables(self):
        return self._cosimulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        return self._cosimulator.model.cvar

    @property
    def number_of_regions(self):
        return self._cosimulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        return self._cosimulator.connectivity.region_labels

    @property
    def tvb_coupling_a(self):
        return self._cosimulator.coupling.a

    @property
    def tvb_weights(self):
        return self._cosimulator.connectivity.weights

    @property
    def tvb_delays(self):
        return self._cosimulator.connectivity.delays

    @property
    def tvb_output_interfaces(self):
        return self._cosimulator.output_interfaces

    @property
    def tvb_input_interfaces(self):
        return self._cosimulator.input_interfaces

    def build_tvb_simulator(self):
        self.cosimulator = self._cosimulator_builder.build()

    def serialize_tvb_cosimulator(self):
        return serialize_tvb_cosimulator(self._cosimulator)

    def dumb_tvb_simulator_serialized(self, tvb_cosimulator_serialized=None, filepath=None):
        if not tvb_cosimulator_serialized:
            tvb_cosimulator_serialized = self.serialize_tvb_cosimulator()
        if not filepath:
            filepath = self.def_tvb_serial_path
        dump_serial_tvb_cosimulator(tvb_cosimulator_serialized, filepath)

    def build_interfaces(self):
        if not self._interfaces_built:
            self.cosimulator = self._interfaces_builder.build()
            self._interfaces_built = True

    def build(self):
        self.build_tvb_simulator()
        self.build_interfaces()

    def assert_simulation_length(self):
        self.simulation_length = np.ceil(self.simulation_length / self._cosimulator.synchronization_time) * \
                                 self._cosimulator.synchronization_time

    def configure_simulation(self):
        self._cosimulator.configure()
        self.assert_simulation_length()

    def simulate(self):
        self.results = self._cosimulator.run()

    def clean_up(self):
        pass

    def run(self):
        self.configure()
        self.build()
        self.configure_simulation()
        self.simulate()

    def reset(self):
        self.cosimulator = None
        self._interfaces_built = False

    def stop(self):
        pass


class TVBSerialApp(TVBApp):

    """TVBSerialApp class"""

    interfaces_builder = Attr(
        label="TVBSpikeNetInterfaces builder",
        field_type=TVBSpikeNetInterfaceBuilder,
        doc="""Instance of TVBSpikeNetInterfaces' builder class.""",
        required=False
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    _default_interface_builder = TVBSpikeNetInterfaceBuilder

    def configure_interfaces_builder(self):
        self.interfaces_builder.spiking_network = self.spiking_network
        super(TVBSerialApp, self).configure_interfaces_builder()

    def reset(self):
        super(TVBSerialApp, self).reset()
        self.spiking_network = None
