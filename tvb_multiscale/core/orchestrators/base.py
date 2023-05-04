# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta, abstractmethod
import os
from logging import Logger

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, NArray
from tvb.contrib.scripts.utils.log_error_utils import warning

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.file_utils import load_pickled_dict
from tvb_multiscale.core.interfaces.base.builders import InterfaceBuilder


class App(HasTraits):
    __metaclass__ = ABCMeta

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

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of proxy nodes",
        doc="""Indices of proxy nodes""",
        required=True
    )

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true if the co-simulator nodes are modelled exclusively
                                  by the co-simulator, i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    synchronization_time = Float(
        label="Synchronization time (ms)",
        default=0.0,
        required=True,
        doc="""Synchronization time (default in milliseconds).""")

    simulation_length = Float(
        label="Simulation Length (ms)",
        default=110.0,
        required=True,
        doc="""The length of a simulation (default in milliseconds). 
               It will be corrected by ceiling to a multiple of the cosimulators synchronization time.""")

    _attrs_to_info = []

    def setup_from_another_app(self, app):
        self.config = app.config
        self.logger = app.logger
        self.default_tvb_serial_cosim_path = app.default_tvb_serial_cosim_path
        self.proxy_inds = app.proxy_inds
        self.exclusive_nodes = app.exclusive_nodes
        self.simulation_length = app.simulation_length
        self.synchronization_time = app.synchronization_time

    def configure(self):
        if self.logger is None:
            self.logger = initialize_logger(__name__, config=self.config)
        super(App, self).configure()
        if len(self.default_tvb_serial_cosim_path) == 0:
            self.default_tvb_serial_cosim_path = \
                getattr(self.config, "DEFAULT_TVB_SERIAL_COSIM_PATH",
                        os.path.join(self.config.out.FOLDER_RES, "tvb_cosimulator_serialized.pkl"))

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def configure_simulation(self):
        pass

    def init(self):
        self.configure()
        self.build()
        self.configure_simulation()

    def run(self):
        self.init()
        self.simulate()

    @abstractmethod
    def clean_up(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def _add_attrs_to_info(self, info):
        for attr in self._attrs_to_info:
            info[attr] = getattr(self, attr, None)
        return info

    def info(self, recursive=0):
        return self._add_attrs_to_info(super(App, self).info(recursive=recursive))

    def info_details(self, recursive=0, **kwargs):
        return self._add_attrs_to_info(super(App, self).info_details(recursive=recursive, **kwargs))


class AppWithInterfaces(App, ABC):
    __metaclass__ = ABCMeta

    """AppWithInterfaces abstract base class"""

    interfaces_builder = Attr(
        label="Interfaces builder",
        field_type=InterfaceBuilder,
        doc="""Instance of Interfaces' builder class.""",
        required=False
    )

    _default_interface_builder = InterfaceBuilder
    _interfaces_built = False

    def configure_interfaces_builder(self):
        self.interfaces_builder.config = self.config
        self.interfaces_builder.logger = self.logger
        self.interfaces_builder.exclusive_nodes = self.exclusive_nodes
        self.interfaces_builder.proxy_inds = self.proxy_inds
        # Now further configure the interfaces builder...
        if hasattr(self.interfaces_builder, "default_config"):
            # ...either from its default configuration, if any:
            self.interfaces_builder.default_config()
        elif os.path.isfile(self.interfaces_builder.interface_filepath):
            # ...or from loading configurations from file:
            self.interfaces_builder.load_all_interfaces()
        # Run the configuration of the builder:
        self.interfaces_builder.configure()
        # Update proxy inds and exclusive nodes:
        self.proxy_inds = self.interfaces_builder.proxy_inds
        self.exclusive_nodes = self.interfaces_builder.exclusive_nodes

    @property
    def _interfaces_builder(self):
        if not isinstance(self.interfaces_builder, self._default_interface_builder):
            self.interfaces_builder = self._default_interface_builder(config=self.config, logger=self.logger)
        return self.interfaces_builder

    @abstractmethod
    def build_interfaces(self):
        pass

    def build(self):
        self.build_interfaces()

    def reset(self):
        self.output_interfaces = None
        self.input_interfaces = None
        self._interfaces_built = False


class CoSimulatorApp(AppWithInterfaces, ABC):
    __metaclass__ = ABCMeta

    """CoSimulatorApp abstract base class"""


class NonTVBApp(CoSimulatorApp, ABC):
    __metaclass__ = ABCMeta

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
            warning("Failed to load serialized TVB CoSimulator from file!:\n%s" % filepath)

    @property
    def _serialized_tvb_cosimulator(self):
        if len(self.tvb_cosimulator_serialized) == 0:
            self.load_serialized_tvb_cosimulator()
        return self.tvb_cosimulator_serialized

    @property
    def tvb_dt(self):
        return self.tvb_cosimulator_serialized["integrator.dt"]

    @property
    def tvb_model(self):
        return self.tvb_cosimulator_serialized["model"]

    @property
    def tvb_model_state_variables(self):
        return self.tvb_cosimulator_serialized["model.state_variables"]

    @property
    def tvb_model_cvar(self):
        return self.tvb_cosimulator_serialized["model.cvar"]

    @property
    def number_of_regions(self):
        return self.tvb_cosimulator_serialized["connectivity.number_of_regions"]

    @property
    def region_labels(self):
        return self.tvb_cosimulator_serialized["connectivity.region_labels"]

    @property
    def tvb_coupling_a(self):
        return self.tvb_cosimulator_serialized["coupling.a"]

    @property
    def tvb_weights(self):
        return self.tvb_cosimulator_serialized["connectivity.weights"]

    @property
    def tvb_delays(self):
        return self.tvb_cosimulator_serialized["connectivity.delays"]

    def build_interfaces(self):
        if not self._interfaces_built:
            self.output_interfaces, self.input_interfaces = self._interfaces_builder.build()
            self._interfaces_built = True

    def configure_simulation(self):
        self.output_interfaces.configure()
        self.input_interfaces.configure()


class Orchestrator(App, ABC):
    __metaclass__ = ABCMeta

    """Orchestrator abstract base class"""

    @abstractmethod
    def build_interfaces(self):
        pass

    def build(self):
        self.build_cosimulators()
        self.build_interfaces()
