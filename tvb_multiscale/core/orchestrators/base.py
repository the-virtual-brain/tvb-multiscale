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
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    default_tvb_serial_cosim_path = Attr(
        label="TVB serialized CoSimulator path",
        field_type=str,
        doc="""File path of TVB serialized CoSimulator.""",
        required=True,
        default=""
    )

    spiking_proxy_inds = NArray(
        dtype=np.int,
        label="Indices of Spiking Network proxy nodes",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
    )

    exclusive_nodes = Attr(label="Flag of exclusive nodes",
                           doc="""Boolean flag that is true 
                                      if the co-simulator nodes are modelled exclusively by the co-simulator, 
                                      i.e., they are not simulated by TVB""",
                           field_type=bool,
                           default=True,
                           required=True)

    simulation_length = Float(
        label="Simulation Length (ms)",
        default=110.0,
        required=True,
        doc="""The length of a simulation (default in milliseconds). 
               It will be corrected by ceiling to a multiple of the cosimulators synchronization time.""")

    synchronization_time = Float(
        label="Synchronization time (ms)",
        default=0.0,
        required=True,
        doc="""Synchronization time (default in milliseconds).""")

    def setup_from_another_app(self, app):
        self.config = app.config
        self.logger = app.logger
        self.spiking_proxy_inds = app.spiking_proxy_inds
        self.exclusive_nodes = app.exclusive_nodes
        self.simulation_length = app.simulation_length
        self.synchronization_time = app.synchronization_time

    def configure(self):
        super(App, self).configure()
        if len(self.default_tvb_serial_cosim_path) == 0:
            self.default_tvb_serial_cosim_path = \
                os.path.join(self.config.out.FOLDER_RES, "tvb_cosimulator_serialized.pkl")

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def configure_simulation(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class NonTVBApp(App, ABC):
    __metaclass__ = ABCMeta

    """NonTVBApp abstract base class"""

    tvb_cosimulator_serialized = Attr(
        label="TVB Simulator serialized",
        field_type=dict,
        doc="""Dictionary of TVB (Co)Simulator serialization.""",
        required=False
    )

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

    def load_serialized_tvb_cosimulator(self, filepath=None):
        if not filepath:
            filepath = self.default_tvb_serial_cosim_path
        try:
            self.tvb_cosimulator_serialized = load_pickled_dict(filepath)
        except:
            # TODO: Decide whether to raise an exception here
            warning("Failed to load serialized TVB CoSimulator from file!:\n%s" % filepath)


class Orchestrator(App, ABC):
    __metaclass__ = ABCMeta

    """Orchestrator abstract base class"""

    @abstractmethod
    def build_interfaces(self):
        pass

    def build(self):
        self.build_cosimulators()
        self.build_interfaces()
