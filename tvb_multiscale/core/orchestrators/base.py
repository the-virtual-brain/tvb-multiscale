# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import os

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, Float, NArray

from tvb_multiscale.core.config import Config, CONFIGURED
from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.tvb.io.h5_writer import H5Writer
from tvb_multiscale.core.tvb.io.h5_reader import H5Reader
from tvb_multiscale.core.tvb.cosimulator_serialization import \
    load_serial_tvb_cosimulator


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

    # plotter = Attr(
    #     label="Plotter",
    #     field_type=Plotter,
    #     doc="""Plotter class instance.""",
    #     required=False
    # )
    #
    # h5_writer = Attr(
    #     label="H5Writer",
    #     field_type=H5Writer,
    #     doc="""H5Writer class instance.""",
    #     required=False
    # )
    #
    # h5_reader = Attr(
    #     label="H5Reader",
    #     field_type=H5Reader,
    #     doc="""H5Reader class instance.""",
    #     required=False
    # )

    def_tvb_serial_path = Attr(
        label="H5Reader",
        field_type=H5Reader,
        doc="""H5Reader class instance.""",
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
        label="Simulation Length (ms, s, m, h)",
        default=110.0,
        required=True,
        doc="""The length of a simulation (default in milliseconds). 
                       It will be corrected by ceiling to a multiple of the cosimulators synchronization time.""")

    # transient = Float(
    #     label="Transient Length (ms, s, m, h)",
    #     default=10.0,
    #     required=True,
    #     doc="""The length of a simulation (default in milliseconds).""")

    def setup_from_another_app(self, app):
        self.config = app.config
        # self.plotter = app.plotter
        # self.h5_reader = app.h5_reader
        # self.h5_writer = app.h5_writer
        self.spiking_proxy_inds = app.spiking_proxy_inds
        self.exclusive_nodes = app.exclusive_nodes
        self.simulation_length = app.simulation_length
        # self.transient = app.transient

    def configure(self):
        if len(self.def_tvb_serial_path) == 0:
            self.def_tvb_serial_path = os.path.join(self.config.out.FOLDER_RES, "tvb_cosimulator_serialized.pkl")
        super(App, self).configure()

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
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass


class NonTVBApp(App):
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

    def load_tvb_simulator_serialized(self, filepath=None):
        if not filepath:
            filepath = self.def_tvb_serial_path
        self.tvb_cosimulator_serialized = load_serial_tvb_cosimulator(filepath)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass


class Orchestrator(App):
    __metaclass__ = ABCMeta

    """Orchestrator abstract base class"""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean_up(self):
        pass
