# -*- coding: utf-8 -*-

import os
from logging import Logger

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.utils.file_utils import load_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serial_tvb_simulator_to_connectivity
from tvb_multiscale.core.interfaces.base.builders import InterfaceBuilder


class App(HasTraits):

    """App abstract base class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=None
    )

    default_tvb_serial_cosim_path = Attr(
        label="TVB serialized CoSimulator path",
        field_type=str,
        doc="""File path of TVB serialized CoSimulator.""",
        required=True,
        default=""
    )

    _app_or_orchestrator = "App"
    _attrs_to_info = []

    def _logprint(self, msg):
        msg = "\n" + msg
        try:
            self.logger(msg)
        except:
            pass
        if self.verbosity:
            print(msg)

    def setup_from_another_app(self, app):
        self.config = app.config
        self.logger = app.logger
        self.default_tvb_serial_cosim_path = app.default_tvb_serial_cosim_path

    def configure_tvb_serial_cosim_path(self):
        if len(self.default_tvb_serial_cosim_path) == 0:
            self.default_tvb_serial_cosim_path = \
                getattr(self.config, "DEFAULT_TVB_SERIAL_COSIM_PATH",
                        os.path.join(self.config.FOLDER_CONFIG, "tvb_serial_cosimulator.pkl"))

    def configure(self):
        try:
            self.logger
        except:
            self.logger = initialize_logger(__name__, config=self.config)
        super(App, self).configure()
        self.configure_tvb_serial_cosim_path()

    @property
    def verbosity(self):
        return self.config.VERBOSITY

    def start(self):
        self._logprint("Starting %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def build(self):
        self._logprint("Building with %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def configure_simulation(self):
        self._logprint("Configuring simulation with %s %s..."
                       % (self._app_or_orchestrator, self.__class__.__name__))

    def init(self):
        self._logprint("Initializing %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))
        self.configure()
        self.build()
        self.configure_simulation()

    def simulate(self, simulation_length=None):
        if simulation_length is not None:
            sim_len_str = "for %gms " % simulation_length
        self._logprint("Simulating %swith %s %s..." %
                       (sim_len_str, self._app_or_orchestrator, self.__class__.__name__))

    def plot(self):
        self._logprint("Plotting results with %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def run(self):
        self._logprint("Running simulation with %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))
        self.init()

    def clean_up(self):
        self._logprint("Cleaning up %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def reset(self):
        self._logprint("Resetting %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def stop(self):
        self._logprint("Stopping %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))

    def _add_attrs_to_info(self, info):
        for attr in self._attrs_to_info:
            info[attr] = getattr(self, attr, None)
        return info

    def info(self, recursive=0):
        return self._add_attrs_to_info(super(App, self).info(recursive=recursive))

    def info_details(self, recursive=0, **kwargs):
        return self._add_attrs_to_info(super(App, self).info_details(recursive=recursive, **kwargs))


class AppWithInterfaces(App):

    """AppWithInterfaces abstract base class"""

    interfaces_builder = Attr(
        label="Interfaces builder",
        field_type=InterfaceBuilder,
        doc="""Instance of Interfaces' builder class.""",
        required=False
    )

    _default_interface_builder_type = InterfaceBuilder
    _interfaces_built = False

    def __init__(self, **kwargs):
        self._interfaces_built = False
        super(AppWithInterfaces, self).__init__(**kwargs)

    def configure_interfaces_builder(self):
        self._logprint("Configuring interfaces' builder %s of App %s..."
                       % (self._interfaces_builder.__class__.__name__, self.__class__.__name__))
        self._interfaces_builder.config = self.config
        self.interfaces_builder.logger = self.logger
        # Now further configure the interfaces builder...
        if hasattr(self.interfaces_builder, "default_config"):
            # ...either from its default configuration, if any:
            if self.verbosity:
                self._logprint("Configuring default interfaces...")
            self.interfaces_builder.default_config()
        elif os.path.isfile(self.interfaces_builder.interface_filepath):
            # ...or from loading configurations from file:
            if self.verbosity:
                self._logprint("Reading interfaces' configurations from files...")
            self.interfaces_builder.load_all_interfaces()
        # Run the configuration of the builder:
        self.interfaces_builder.configure()

    @property
    def _interfaces_builder(self):
        try:
            assert isinstance(self.interfaces_builder, self._default_interface_builder_type)
        except:
            self._logprint("Instantiating interfaces' builder %s of App %s..."
                           % (self._default_interface_builder_type.__name__, self.__class__.__name__))
            self.interfaces_builder = self._default_interface_builder_type(config=self.config, logger=self.logger)
        return self.interfaces_builder

    def build_interfaces(self):
        self._logprint("Building interfaces' with builder %s of App %s..."
                       % (self._default_interface_builder_type.__name__, self.__class__.__name__))
        self.configure_interfaces_builder()

    def build(self):
        super(AppWithInterfaces, self).build()
        self.build_interfaces()

    def reset(self):
        super(AppWithInterfaces, self).reset()
        self._interfaces_built = False

    def stop(self):
        super(AppWithInterfaces, self).stop()
        self._interfaces_built = False


class CoSimulatorApp(AppWithInterfaces):

    """CoSimulatorApp abstract base class"""

    def run(self, simulation_length=None):
        super(CoSimulatorApp, self).run()
        self.configure()
        self.build()
        self.simulate(simulation_length)


class NonTVBApp(CoSimulatorApp):

    """NonTVBApp abstract base class"""

    tvb_cosimulator_serialized = Attr(
        label="TVB Simulator serialized",
        field_type=dict,
        doc="""Dictionary of TVB (Co)Simulator serialization.""",
        required=False
    )

    def load_serialized_tvb_cosimulator(self, filepath=None):
        if not filepath:
            filepath = self.default_tvb_serial_cosim_path
        try:
            self.tvb_cosimulator_serialized = load_pickled_dict(filepath)
        except:
            # TODO: Decide whether to raise an exception here
            self.logger.warning("Failed to load serialized TVB CoSimulator from file!:\n%s" % filepath)
        return self.tvb_cosimulator_serialized

    @property
    def _serialized_tvb_cosimulator(self):
        try:
            assert isinstance(self.tvb_cosimulator_serialized, dict)
        except:
            self.tvb_cosimulator_serialized = self.load_serialized_tvb_cosimulator()
        return self.tvb_cosimulator_serialized

    @property
    def tvb_dt(self):
        return self._serialized_tvb_cosimulator["integrator.dt"]

    @property
    def tvb_model(self):
        return self._serialized_tvb_cosimulator["model"]

    @property
    def tvb_model_state_variables(self):
        return self._serialized_tvb_cosimulator["model.state_variables"]

    @property
    def tvb_model_cvar(self):
        return self._serialized_tvb_cosimulator["model.cvar"]

    @property
    def tvb_connectivity(self):
        return serial_tvb_simulator_to_connectivity(self._serialized_tvb_cosimulator)

    @property
    def number_of_regions(self):
        return self._serialized_tvb_cosimulator["connectivity.number_of_regions"]

    @property
    def region_labels(self):
        return self._serialized_tvb_cosimulator["connectivity.region_labels"]

    @property
    def tvb_coupling_a(self):
        return self._serialized_tvb_cosimulator["coupling.a"]

    @property
    def tvb_weights(self):
        return self._serialized_tvb_cosimulator["connectivity.weights"]

    @property
    def tvb_delays(self):
        return self._serialized_tvb_cosimulator["connectivity.delays"]

    @property
    def tvb_delays(self):
        return self._serialized_tvb_cosimulator["connectivity.delays"]

    @property
    def tvb_monitor_period(self):
        return self._serialized_tvb_cosimulator["monitor.period"]

    def configure_interfaces_builder(self):
        self._interfaces_builder.tvb_simulator_serialized = self._serialized_tvb_cosimulator
        super(NonTVBApp, self).configure_interfaces_builder()

    def configure_simulation(self):
        super(NonTVBApp, self).configure_simulation()


class Orchestrator(App):

    """Orchestrator abstract base class"""

    _app_or_orchestrator = "App"

    def build_interfaces(self):
        self._logprint("Building interfaces with Orchestrator %s..." % self.__class__.__name__)

    def build(self):
        super(Orchestrator, self).build()
        self.build_cosimulators()
        self.build_interfaces()

    def run(self, simulation_length=None):
        super(Orchestrator, self).run()
        self.configure()
        self.build()
        self.simulate()
