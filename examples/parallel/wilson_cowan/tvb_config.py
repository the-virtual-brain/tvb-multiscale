# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator

from examples.parallel.wilson_cowan.config import configure


# This would run on TVB process before creating any multiscale cosimulation interface connections.
# Users can adapt it to their use case.
def build_tvb_simulator(config=None, config_class=Config, cosimulator_class=CoSimulatorParallel):

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.integrators import HeunStochastic
    from tvb.simulator.monitors import Raw  # , Bold, EEG

    from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan

    if config is None:
        config = configure(config_class)

    # We choose all defaults in this example
    # -----------------------------------Wilson Cowan oscillatory regime--------------------------------
    model_params = {
        "r_e": np.array([0.0]),
        "r_i": np.array([0.0]),
        "k_e": np.array([1.0]),
        "k_i": np.array([1.0]),
        "tau_e": np.array([10.0]),
        "tau_i": np.array([10.0]),
        "c_ee": np.array([10.0]),
        "c_ei": np.array([6.0]),
        "c_ie": np.array([10.0]),
        "c_ii": np.array([1.0]),
        "alpha_e": np.array([1.2]),
        "alpha_i": np.array([2.0]),
        "a_e": np.array([1.0]),
        "a_i": np.array([1.0]),
        "b_e": np.array([0.0]),
        "b_i": np.array([0.0]),
        "c_e": np.array([1.0]),
        "c_i": np.array([1.0]),
        "theta_e": np.array([2.0]),
        "theta_i": np.array([3.5]),
        "P": np.array([0.5]),
        "Q": np.array([0.0])
    }

    # -----------------------------------Build cosimunlator manually--------------------------------

    simulator = cosimulator_class(min_idelay_sync_n_step_ratio=config.TVB_MIN_IDELAY_TO_SYNC_N_STEP_RATIO)

    simulator.model = WilsonCowan(**model_params)

    simulator.integrator = HeunStochastic()
    simulator.integrator.dt = config.DEFAULT_DT
    simulator.integrator.noise.nsig = np.array([config.DEFAULT_NSIG, config.DEFAULT_NSIG])  # 0.001

    # Load connectivity
    # config.DEFAULT_CONNECTIVITY_ZIP = "/home/docker/packages/tvb_data/tvb_data/mouse/allen_2mm/ConnectivityAllen2mm.zip"
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)

    # -------------- Pick a minimal brain of only the first n_regions regions: ----------------
    if config.N_REGIONS:
        n_regions = config.N_REGIONS
        connectivity.number_of_regions = n_regions
        connectivity.region_labels = connectivity.region_labels[:n_regions]
        connectivity.centres = connectivity.centres[:n_regions]
        connectivity.areas = connectivity.areas[:n_regions]
        connectivity.orientations = connectivity.orientations[:n_regions]
        connectivity.hemispheres = connectivity.hemispheres[:n_regions]
        connectivity.cortical = connectivity.cortical[:n_regions]
        connectivity.weights = connectivity.weights[:n_regions][:, :n_regions]
        connectivity.tract_lengths = connectivity.tract_lengths[:n_regions][:, :n_regions]
    # Remove diagonal self-connections:
    np.fill_diagonal(connectivity.weights, 0.0)
    # -----------------------------------------------------------------------------------------

    # Normalize connectivity weights
    connectivity.weights = connectivity.scaled_weights(mode="region")
    connectivity.weights /= np.percentile(connectivity.weights, 99)
    # connectivity.weights[connectivity.weights > 1.0] = 1.0

    # connectivity.tract_lengths = np.maximum(connectivity.speed * simulator.integrator.dt,
    #                                         connectivity.tract_lengths)

    connectivity.configure()

    simulator.connectivity = connectivity

    simulator.initial_conditions = np.zeros((1, 2, connectivity.number_of_regions, 1))

    mon_raw = Raw(period=1.0)  # ms
    simulator.monitors = (mon_raw,)

    simulator.simulation_length = config.SIMULATION_LENGTH
    simulator.configure()

    # # -----------------------------------Or use the CoSimulator builder--------------------------------
    # from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorNRPBuilder

    # simulator_builder = CoSimulatorNRPBuilder()
    # simulator_builder.config = config
    # simulator_builder.model = WilsonCowan()
    # simulator_builder.model_params = model_params
    # simulator_builder.initial_conditions = np.zeros((1, 1, 1, 1))

    # # simulator_builder.configure()
    # simulator_builder.print_summary_info_details(recursive=1)

    # simulator = simulator_builder.build()

    simulator._preconfigure_synchronization_time()

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    sim_serial = serialize_tvb_cosimulator(simulator)
    sim_serial_filepath = os.path.join(config.FOLDER_CONFIG, "tvb_serial_cosimulator.pkl")
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    simulator.configure()

    if config.VERBOSITY:
        simulator.print_summary_info_details(recursive=1)

    return simulator
