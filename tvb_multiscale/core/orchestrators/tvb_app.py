# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
from tvb.basic.neotraits._attr import Attr, NArray

from tvb_multiscale.core.orchestrators.base import App
from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.tvb.cosimulator_serialization import dump_serial_tvb_cosimulator, serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder


class TVBApp(App):
    __metaclass__ = ABCMeta

    """TVBApp base abstract class"""

    tvb_cosimulator_builder = Attr(
        label="TVB CoSimulator Builder",
        field_type=CoSimulatorBuilder,
        doc="""Instance of TVB CoSimulator Builder.""",
        required=False
    )

    tvb_cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulator,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )

    tvb_interfaces_builder = Attr(
        label="TVB Simulator",
        field_type=TVBInterfaceBuilder,
        doc="""Instance of TVB (Co)Simulator.""",
        required=False
    )

    def setup_from_orchestrator(self, orchestrator):
        super(TVBApp, self).setup_from_another_app(orchestrator)

    @property
    def tvb_dt(self):
        return self.tvb_cosimulator.integrator.dt

    @property
    def tvb_model(self):
        return self.tvb_cosimulator.model

    @property
    def tvb_model_state_variables(self):
        return self.tvb_cosimulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        return self.tvb_cosimulator.model.cvar

    @property
    def number_of_regions(self):
        return self.tvb_cosimulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        return self.tvb_cosimulator.connectivity.region_labels

    @property
    def tvb_coupling_a(self):
        return self.tvb_cosimulator.coupling.a

    @property
    def tvb_weights(self):
        return self.tvb_cosimulator.connectivity.weights

    @property
    def tvb_delays(self):
        return self.tvb_cosimulator.connectivity.delays

    def configure(self):
        super(TVBApp, self).configure()
        if not self.tvb_cosimulator_builder:
            self.tvb_cosimulator_builder = CoSimulatorBuilder(config=self.config)
            self.tvb_cosimulator_builder.configure()
        if self.tvb_interfaces_builder and not isinstance(self.tvb_interfaces_builder, TVBInterfaceBuilder):
            self.tvb_interfaces_builder = self.tvb_interfaces_builder()

    def start(self):
        from tvb.basic.profile import TvbProfile
        TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    def build_tvb_simulator(self):
        self.tvb_cosimulator = self.tvb_cosimulator_builder.build()

    def dumb_tvb_simulator_serialized(self, tvb_cosimulator_serialized=None, filepath=None):
        if not tvb_cosimulator_serialized:
            tvb_cosimulator_serialized = self.serialize_tvb_cosimulator()
        if not filepath:
            filepath = self.def_tvb_serial_path
        dump_serial_tvb_cosimulator(tvb_cosimulator_serialized, filepath)

    def serialize_tvb_cosimulator(self):
        return serialize_tvb_cosimulator(self.tvb_cosimulator)

    def build_interfaces(self):
        self.tvb_interfaces_builder.configure()
        self.tvb_cosimulator = self.tvb_interfaces_builder.build()

    def build(self):
        self.build_tvb_simulator()
        self.tvb_interfaces_builder.tvb_cosimulator = self.tvb_cosimulator
        self.tvb_interfaces_builder.exclusive_nodes = self.exclusive_nodes
        self.tvb_interfaces_builder.spiking_proxy_inds = self.spiking_proxy_inds

    @abstractmethod
    def plot(self):
        pass

    def run(self):
        self.tvb_cosimulator.simulation_length = self.compute_simulation_length(self.simulation_length)
        self.tvb_cosimulator.configure()
        self.tvb_app.tvb_cosimulator.run()
        if self.plotter:
            self.plot()

    def stop(self):
        pass

    def clean_up(self):
        pass