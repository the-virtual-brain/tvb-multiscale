# -*- coding: utf-8 -*-

import time

import numpy as np

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import CoSimulatorApp
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import \
    CoSimulatorBuilder, CoSimulatorParallelBuilder, CoSimulatorSerialBuilder
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBInterfaceBuilder, TVBSpikeNetInterfaceBuilder, TVBRemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.models.default import \
    DefaultTVBSpikeNetInterfaceBuilder, DefaultTVBInterfaceBuilder, DefaultTVBRemoteInterfaceBuilder
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class TVBApp(CoSimulatorApp):

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
        label="TVBInterfaces builder",
        field_type=TVBInterfaceBuilder,
        doc="""Instance of TVBInterfaces' builder class.""",
        required=False
    )

    results = None

    _cosimulator_builder_type = CoSimulatorBuilder
    _default_interface_builder = TVBInterfaceBuilder

    def start(self):
        from tvb.basic.profile import TvbProfile
        TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    @property
    def _cosimulator_builder(self):
        if not isinstance(self.cosimulator_builder, self._cosimulator_builder_type):
            self.cosimulator_builder = self._cosimulator_builder_type(config=self.config, logger=self.logger)
            self.cosimulator_builder.configure()
        return self.cosimulator_builder

    def configure(self):
        super(TVBApp, self).configure()
        if self.cosimulator is None:
            self.cosimulator_builder.config = self.config
            self.cosimulator_builder.logger = self.logger
            self.cosimulator_builder.configure()

    @property
    def _cosimulator(self):
        if not isinstance(self.cosimulator, CoSimulator):
            self.build_tvb_simulator()
        return self.cosimulator

    def configure_interfaces_builder(self):
        # Get default options from the App and the TVB CoSimulator:
        self._interfaces_builder.tvb_cosimulator = self._cosimulator
        super(TVBApp, self).configure_interfaces_builder()

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
            filepath = self.default_tvb_serial_cosim_path
        dump_pickled_dict(tvb_cosimulator_serialized, filepath)

    def build_interfaces(self):
        if self._interfaces_built is False:
            self.configure_interfaces_builder()
            self.cosimulator = self.interfaces_builder.build()
            self._interfaces_build = True

    def build(self):
        assert self._cosimulator
        super(TVBApp, self).build()

    def assert_simulation_length(self):
        if self._cosimulator.synchronization_time > 0:
            self.simulation_length = np.ceil(self.simulation_length / self.cosimulator.synchronization_time) * \
                                     self._cosimulator.synchronization_time
        self.cosimulator.simulation_length = self.simulation_length

    def configure_simulation(self):
        self.cosimulator.simulation_length = self.simulation_length
        self._cosimulator.configure()
        self.assert_simulation_length()
        self.synchronization_time = self.cosimulator.synchronization_time

    def simulate(self):
        self.results = self._cosimulator.run()

    def reset(self):
        self.cosimulator = None
        super(TVBApp, self).reset()

    def clean_up(self):
        pass

    def stop(self):
        pass


class TVBSerialApp(TVBApp):

    """TVBSerialApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorSerialBuilder",
        field_type=CoSimulatorSerialBuilder,
        doc="""Instance of TVB Serial CoSimulator Builder class.""",
        required=False,
        default=CoSimulatorSerialBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulatorSerial",
        field_type=CoSimulatorSerial,
        doc="""Instance of TVB CoSimulator for serial cosimulation.""",
        required=False
    )

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

    _cosimulator_builder_type = CoSimulatorSerialBuilder
    _default_interface_builder = DefaultTVBSpikeNetInterfaceBuilder

    def configure_interfaces_builder(self):
        self.interfaces_builder.spiking_network = self.spiking_network
        super(TVBSerialApp, self).configure_interfaces_builder()

    def build_interfaces(self):
        if not self._interfaces_built:
            self.output_interfaces, self.input_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def reset(self):
        super(TVBSerialApp, self).reset()
        self.spiking_network = None


class TVBParallelApp(TVBApp):

    """TVBParallelApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorParallelBuilder",
        field_type=CoSimulatorParallelBuilder,
        doc="""Instance of TVB Parallel CoSimulator Builder class.""",
        required=False,
        default=CoSimulatorParallelBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulatorParallel,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )

    interfaces_builder = Attr(
        label="TVBRemoteInterfaceBuilder builder",
        field_type=TVBInterfaceBuilder,
        doc="""Instance of TVBRemoteInterfaceBuilder' builder class.""",
        required=False,
        default=DefaultTVBInterfaceBuilder()
    )

    _cosimulator_builder_type = CoSimulatorParallelBuilder
    _default_interface_builder = DefaultTVBInterfaceBuilder

    _wall_time_start = None
    _ts = None
    _xs = None

    def configure_simulation(self):
        super(TVBParallelApp, self).configure_simulation()
        ts, xs = [], []
        for _ in self.cosimulator.monitors:
            ts.append([])
            xs.append([])
        self._wall_time_start = time.time()
        self._ts = ts
        self._xs = xs
        self.send_cosim_coupling()  # We need to send the TVB initial condition to the the Spiking CoSimulator

    def run_for_synchronization_time(self, cosim_updates):
        steps_performed = \
            self._run_for_synchronization_time(self._ts, self._xs, self._wall_time_start,
                                               cosim_updates=self.cosimulator.get_cosim_updates(cosim_updates))
        return self.cosimulator.send_cosim_coupling()

    def reset(self):
        super(TVBParallelApp, self).reset()
        self._ts = None
        self._xs = None
        self._wall_time_start = None


class TVBRemoteParallelApp(TVBParallelApp):

    """TVBRemoteParallelApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorParallelBuilder",
        field_type=CoSimulatorParallelBuilder,
        doc="""Instance of TVB Parallel CoSimulator Builder class.""",
        required=False,
        default=CoSimulatorParallelBuilder()
    )

    cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulatorParallel,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )

    interfaces_builder = Attr(
        label="TVBRemoteInterfaceBuilder builder",
        field_type=TVBRemoteInterfaceBuilder,
        doc="""Instance of TVBRemoteInterfaceBuilder' builder class.""",
        required=False,
        default=DefaultTVBRemoteInterfaceBuilder()
    )

    _cosimulator_builder_type = CoSimulatorParallelBuilder
    _default_interface_builder = DefaultTVBRemoteInterfaceBuilder

    def run_for_synchronization_time(self):
        """Convenience method to call the CoSimulator with **kwds and collect output data."""
        steps_performed = \
            self._run_for_synchronization_time(self._ts, self._xs, self._wall_time_start,
                                               cosim_updates=self.cosimulator.get_cosim_updates())
        return self.cosimulator.send_cosim_coupling()
