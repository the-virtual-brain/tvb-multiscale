# -*- coding: utf-8 -*-

import time
from types import FunctionType

import numpy as np

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.orchestrators.base import CoSimulatorApp
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel, CoSimulatorRemoteParallel
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import \
    CoSimulatorBuilder, CoSimulatorSerialBuilder, CoSimulatorParallelBuilder, CoSimulatorRemoteParallelBuilder
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
from tvb_multiscale.core.interfaces.tvb.builders import \
    TVBInterfaceBuilder, TVBSpikeNetInterfaceBuilder, TVBRemoteInterfaceBuilder
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class TVBApp(CoSimulatorApp):

    """TVBApp base class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulator Builder",
        field_type=CoSimulatorBuilder,
        doc="""Instance of TVB CoSimulator Builder class.""",
        required=False
    )

    cosimulator_builder_function = Attr(
        label="TVB CoSimulator Builder function",
        field_type=FunctionType,
        doc="""Instance of TVB CoSimulator Builder function.""",
        required=False,
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

    _results = None

    _cosimulator_builder_type = CoSimulatorBuilder
    _default_interface_builder_type = TVBInterfaceBuilder

    def __init__(self, **kwargs):
        self._results = None
        self.cosimulator = None
        self.cosimulator_builder = None
        self.cosimulator_builder_function = None
        super(TVBApp, self).__init__(**kwargs)

    def start(self):
        super(TVBApp, self).start()
        self._logprint("Setting TVB LIBRARY_PROFILE...")
        from tvb.basic.profile import TvbProfile
        TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    @property
    def _cosimulator_builder(self):
        if isinstance(self.cosimulator_builder, self._cosimulator_builder_type):
            return self.cosimulator_builder
        elif hasattr(self.cosimulator_builder_function, "__call__"):
            return self.cosimulator_builder_function
        else:
            raise Exception("Neither a TVB CoSimulator of type %s, "
                            "nor a TVB CoSimulator builder or function has been provided as input to TVBApp!"
                            % self._cosimulator_builder_type)

    def configure(self):
        super(TVBApp, self).configure()
        if self.cosimulator is None:
            if isinstance(self.cosimulator_builder, self._cosimulator_builder_type):
                self.cosimulator_builder.config = self.config
                self.cosimulator_builder.logger = self.logger
                self.cosimulator_builder.configure()

    @property
    def _cosimulator(self):
        try:
            assert isinstance(self.cosimulator, CoSimulator)
        except:
            self.build_tvb_simulator()
        return self.cosimulator

    @property
    def results(self):
        return self._results

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

    def serialize_tvb_cosimulator(self):
        return serialize_tvb_cosimulator(self._cosimulator)

    def dumb_tvb_simulator_serialized(self, tvb_cosimulator_serialized=None, filepath=None):
        if not tvb_cosimulator_serialized:
            tvb_cosimulator_serialized = self.serialize_tvb_cosimulator()
        if not filepath:
            filepath = self.default_tvb_serial_cosim_path
        dump_pickled_dict(tvb_cosimulator_serialized, filepath)

    def build_tvb_simulator(self):
        if isinstance(self._cosimulator_builder, self._cosimulator_builder_type):
            self._logprint("Building TVB CoSimulator with builder %s of App %s..."
                           % (self.cosimulator_builder.__class__.__name__, self.__class__.__name__))
            self.cosimulator = self.cosimulator_builder.build()
        else:
            self._logprint("Building TVB CoSimulator with builder script %s of App %s..."
                           % (self.cosimulator_builder_function.__name__, self.__class__.__name__))
            self.cosimulator = self.cosimulator_builder_function(self.config)
        if self.verbosity:
            self._logprint(str(self.cosimulator))
        self.cosimulator.configure()
        self.dumb_tvb_simulator_serialized()

    def configure_interfaces_builder(self):
        # Get default options from the App and the TVB CoSimulator:
        self._interfaces_builder.tvb_cosimulator = self._cosimulator
        super(TVBApp, self).configure_interfaces_builder()

    def build_interfaces(self):
        if self._interfaces_built is False:
            super(TVBApp, self).build_interfaces()
            self.cosimulator = self.interfaces_builder.build()
            self._interfaces_build = True
            if self.verbosity:
                self._logprint(self.cosimulator.output_interfaces.summary_info_to_string(recursive=self.verbosity+1))
                self._logprint(self.cosimulator.input_interfaces.summary_info_to_string(recursive=self.verbosity+1))

    def assert_simulation_length(self):
        if self._cosimulator.synchronization_time > 0:
            simulation_length = np.ceil(self.cosimulator.simulation_length / self.cosimulator.synchronization_time) * \
                                        self.cosimulator.synchronization_time
        self.cosimulator.simulation_length = simulation_length

    def configure_simulation(self, ):
        super(TVBApp, self).configure_simulation()
        self.cosimulator.PRINT_PROGRESSION_MESSAGE = self.verbosity
        self.cosimulator.configure()
        self.assert_simulation_length()
        self.dumb_tvb_simulator_serialized()
        if self.verbosity:
            self._logprint(str(self.cosimulator))

    def simulate(self, simulation_length=None):
        if simulation_length is not None:
            self.cosimulator.simulation_length = simulation_length
        simulation_length = self.cosimulator.simulation_length
        super(TVBApp, self).simulate(simulation_length)
        self._results = self._cosimulator.run()

    def plot(self, tvb_results=None, transient=0.0, spiking_nodes_ids=None,
             # populations=["E", "I"], populations_sizes=[],
             tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
             plotter=None, writer=None, **kwargs):
        super(TVBApp, self).plot()
        from examples.plot_write_results import plot_write_tvb_results
        if tvb_results is None:
            tvb_results = self.results[0]
            if tvb_results is None:
                return None
        if transient is None:
            transient = getattr(self.config, "TRANSIENT", 0.0)
        if spiking_nodes_ids is None:
            spiking_nodes_ids = getattr(self.config, "SPIKING_NODES_INDS", [])
        return plot_write_tvb_results(tvb_results, self._cosimulator,
                                      transient=transient, spiking_nodes_ids=spiking_nodes_ids,
                                      # populations=["E", "I"], populations_sizes=[],
                                      tvb_state_variable_type_label=tvb_state_variable_type_label,
                                      tvb_state_variables_labels=tvb_state_variables_labels,
                                      plotter=plotter, writer=writer, config=self.config, **kwargs)

    def _destroy(self):
        del self._results
        self._results = None
        self.cosimulator = None
        self.cosimulator_builder = None
        self.cosimulator_builder_function = None
        super(TVBApp, self)._destroy()


class TVBSerialApp(TVBApp):

    """TVBSerialApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorSerialBuilder",
        field_type=CoSimulatorSerialBuilder,
        doc="""Instance of TVB Serial CoSimulator Builder class.""",
        required=False,
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
    _default_interface_builder_type = TVBSpikeNetInterfaceBuilder

    def __init__(self, **kwargs):
        self.spiking_network = None
        super(TVBSerialApp, self).__init__(**kwargs)

    def configure_interfaces_builder(self):
        self._interfaces_builder.spiking_network = self.spiking_network
        super(TVBSerialApp, self).configure_interfaces_builder()

    def reset(self):
        super(TVBSerialApp, self).reset()
        self.spiking_network = None

    def _destroy(self):
        self.spiking_network = None
        super(TVBSerialApp, self)._destroy()


class TVBParallelApp(TVBApp):

    """TVBParallelApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorParallelBuilder",
        field_type=CoSimulatorParallelBuilder,
        doc="""Instance of TVB Parallel CoSimulator Builder class.""",
        required=False
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
        default=TVBInterfaceBuilder()
    )

    _cosimulator_builder_type = CoSimulatorParallelBuilder
    _default_interface_builder_type = TVBInterfaceBuilder

    _wall_time_start = None
    _ts = None
    _xs = None
    tvb_init_cosim_coupling = None

    def __init__(self, **kwargs):
        self._wall_time_start = None
        self._ts = None
        self._xs = None
        self.tvb_init_cosim_coupling = None
        super(TVBParallelApp, self).__init__(**kwargs)

    def get_tvb_init_cosim_coupling(self, relative_output_interfaces_time_steps=None):
        if relative_output_interfaces_time_steps is not None:
            relative_output_interfaces_time_steps1 = self.cosimulator.relative_output_interfaces_time_steps
            self.cosimulator.relative_output_interfaces_time_steps = relative_output_interfaces_time_steps
            relative_output_interfaces_time_steps = relative_output_interfaces_time_steps1
        self.tvb_init_cosim_coupling = self.cosimulator.send_cosim_coupling(True)
        if relative_output_interfaces_time_steps is not None:
            self.cosimulator.relative_output_interfaces_time_steps = relative_output_interfaces_time_steps
        return self.tvb_init_cosim_coupling

    def configure_simulation(self):
        super(TVBParallelApp, self).configure_simulation()
        ts, xs = [], []
        for _ in self.cosimulator.monitors:
            ts.append([])
            xs.append([])
        self._wall_time_start = time.time()
        self._ts = ts
        self._xs = xs
        if not self.cosimulator.n_tvb_steps_ran_since_last_synch:
            self.cosimulator.n_tvb_steps_ran_since_last_synch = int(self.cosimulator.synchronization_n_step)
        # # Send TVB's initial condition to spikeNet!:
        # return self.get_tvb_init_cosim_coupling()

    def run_for_synchronization_time(self, cosim_updates=None, cosimulation=True):
        return self.cosimulator.run_for_synchronization_time(
            self._ts, self._xs, self._wall_time_start, cosim_updates, cosimulation=cosimulation)

    def return_tvb_results(self):
        if self._ts is not None and self._xs is not None:
            for i in range(len(self._ts)):
                self._ts[i] = np.array(self._ts[i])
                self._xs[i] = np.array(self._xs[i])
            return list(zip(self._ts, self._xs))
        return None

    def plot(self, tvb_result=None, transient=0.0, spiking_nodes_ids=None,
             # populations=["E", "I"], populations_sizes=[],
             tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=None,
             plotter=None, writer=None, **kwargs):
        if tvb_result is None:
            tvb_result = self.return_tvb_results()[0]
        if tvb_state_variables_labels is None:
            tvb_state_variables_labels = self._cosimulator.monitors[0].variables_of_interest
            if tvb_state_variables_labels is None:
                tvb_state_variables_labels = self._cosimulator.model.variables_of_interest
        super(TVBParallelApp, self).plot(tvb_result, transient, spiking_nodes_ids,
                                         # populations=["E", "I"], populations_sizes=[],
                                         tvb_state_variable_type_label, tvb_state_variables_labels,
                                         plotter, writer, **kwargs)

    def _destroy(self):
        self._ts = None
        self._xs = None
        self._wall_time_start = None
        self.tvb_init_cosim_coupling = None
        super(TVBParallelApp, self)._destroy()

    def reset(self):
        super(TVBParallelApp, self).reset()
        self._ts = None
        self._xs = None
        self._wall_time_start = None
        self.tvb_init_cosim_coupling = None


class TVBRemoteParallelApp(TVBParallelApp):

    """TVBRemoteParallelApp class"""

    cosimulator_builder = Attr(
        label="TVB CoSimulatorParallelBuilder",
        field_type=CoSimulatorRemoteParallelBuilder,
        doc="""Instance of TVB Parallel CoSimulator Builder class.""",
        required=False
    )

    cosimulator = Attr(
        label="TVB CoSimulator",
        field_type=CoSimulatorRemoteParallel,
        doc="""Instance of TVB CoSimulator.""",
        required=False
    )

    interfaces_builder = Attr(
        label="TVBRemoteInterfaceBuilder builder",
        field_type=TVBRemoteInterfaceBuilder,
        doc="""Instance of TVBRemoteInterfaceBuilder' builder class.""",
        required=False,
        default=TVBRemoteInterfaceBuilder()
    )

    _cosimulator_builder_type = CoSimulatorRemoteParallelBuilder
    _default_interface_builder_type = TVBRemoteInterfaceBuilder

    def run_for_synchronization_time(self, cosimulation=True):
        return self.cosimulator.run_for_synchronization_time(self._ts, self._xs, self._wall_time_start, cosimulation)
