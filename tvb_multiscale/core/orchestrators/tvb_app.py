# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import App
from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.tvb.cosimulator_serialization import dump_serial_tvb_cosimulator, serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder, TVBSpikeNetInterfaceBuilder


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

    def setup_from_orchestrator(self, orchestrator):
        super(TVBApp, self).setup_from_another_app(orchestrator)

    @property
    def tvb_dt(self):
        return self.cosimulator.integrator.dt

    @property
    def tvb_model(self):
        return self.cosimulator.model

    @property
    def tvb_model_state_variables(self):
        return self.cosimulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        return self.cosimulator.model.cvar

    @property
    def number_of_regions(self):
        return self.cosimulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        return self.cosimulator.connectivity.region_labels

    @property
    def tvb_coupling_a(self):
        return self.cosimulator.coupling.a

    @property
    def tvb_weights(self):
        return self.cosimulator.connectivity.weights

    @property
    def tvb_delays(self):
        return self.cosimulator.connectivity.delays

    @property
    def tvb_output_interfaces(self):
        return self.cosimulator.output_interfaces

    @property
    def tvb_input_interfaces(self):
        return self.cosimulator.input_interfaces

    def configure(self):
        super(TVBApp, self).configure()
        if not self.cosimulator_builder:
            self.cosimulator_builder = CoSimulatorBuilder(config=self.config)
            self.cosimulator_builder.configure()
        if self.interfaces_builder and not isinstance(self.interfaces_builder, TVBInterfaceBuilder):
            self.interfaces_builder = self.interfaces_builder()

    def start(self):
        from tvb.basic.profile import TvbProfile
        TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    def build_tvb_simulator(self):
        self.cosimulator = self.cosimulator_builder.build()

    def dumb_tvb_simulator_serialized(self, tvb_cosimulator_serialized=None, filepath=None):
        if not tvb_cosimulator_serialized:
            tvb_cosimulator_serialized = self.serialize_tvb_cosimulator()
        if not filepath:
            filepath = self.def_tvb_serial_path
        dump_serial_tvb_cosimulator(tvb_cosimulator_serialized, filepath)

    def serialize_tvb_cosimulator(self):
        return serialize_tvb_cosimulator(self.cosimulator)

    def build_interfaces(self):
        self.interfaces_builder.configure()
        self.cosimulator = self.interfaces_builder.build()

    def build(self):
        self.build_tvb_simulator()
        self.interfaces_builder.cosimulator = self.cosimulator
        self.interfaces_builder.exclusive_nodes = self.exclusive_nodes
        self.interfaces_builder.spiking_proxy_inds = self.spiking_proxy_inds

    def assert_simulation_length(self):
        self.simulation_length = np.ceil(self.simulation_length / self.cosimulator.synchronization_time) * \
                                 self.cosimulator.synchronization_time

    def configure_simulation(self):
        self.cosimulator.configure()
        self.assert_simulation_length()

    def run(self):
        self.configure_simulation()
        self.results = self.tvb_app.cosimulator.run()
        if self.plotter and self.results:
            self.plot()

    def stop(self):
        pass

    def clean_up(self):
        pass



class TVBSerialApp(TVBApp):

    """TVBSerialApp class"""

    interfaces_builder = Attr(
        label="TVBSpikeNetInterfaces builder",
        field_type=TVBSpikeNetInterfaceBuilder,
        doc="""Instance of TVBSpikeNetInterfaces' builder class.""",
        required=False
    )