# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import time
import numpy as np
from collections import OrderedDict
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw
from tvb_nest.config import *
from tvb_nest.plot.plotter import Plotter
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.simulator_nest.models_builders.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.interfaces.builders.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_nest.config import Config


def prepare_launch_default_simulation():
    config = Config(output_base="outputs/")
    config.SAVE_FLAG = False
    config.SHOW_FLAG = False
    config.MATPLOTLIB_BACKEND = "Agg"
    plotter = Plotter(config)

    connectivity = Connectivity.from_file(os.path.join(Config.DEFAULT_SUBJECT_PATH, Config.DEFAULT_CONNECTIVITY_ZIP))
    connectivity.configure()

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.model = ReducedWongWangExcIOInhI()
    simulator.connectivity = connectivity
    simulator.integrator.dt = float(
        int(np.round(simulator.integrator.dt / config.nest.NEST_MIN_DT))) * config.nest.NEST_MIN_DT
    # Some extra monitors for neuroimaging measures:
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)  # mon_bold, mon_eeg

    # Select the regions for the fine scale modeling with NEST spiking networks
    number_of_regions = simulator.connectivity.region_labels.shape[0]
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    for rid in range(number_of_regions):
        if simulator.connectivity.region_labels[rid].find("hippo") > 0:
            nest_nodes_ids.append(rid)

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = RedWWExcIOInhIBuilder(simulator, nest_nodes_ids)
    nest_model_builder.populations_names = ["E", "I"]
    nest_model_builder.populations_scales = [1.0, 0.7]
    nest_model_builder.default_synapse["params"]["rule"] = "fixed_indegree"

    # Connection weights
    # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA
    w_ee = 150.0
    w_ei = -1000.0
    w_ie = 150.0
    w_ii = -1000.0

    # Within node connections' weights
    # TODO: take care of J_N units conversion from TVB to NEST!
    nest_model_builder.population_connectivity_synapses_weights = np.array(
        [[w_ee, w_ei],  # exc_i -> exc_i, inh_i -> exc_i
         [w_ie, w_ii]])  # exc_i -> inh_i, inh_i -> inh_i
    nest_model_builder.population_connectivity_synapses_delays = np.array(nest_model_builder.tvb_dt / 4)

    # Between node connections
    # Given that w_ee == w_ie = J_N, we need only one connection type
    nest_model_builder.node_connections = [{"src_population": "E", "trg_population": ["E", "I"],
                                            "model": nest_model_builder.default_synapse["model"],
                                            "params": nest_model_builder.default_synapse["params"],
                                            "weight": w_ee,  # weight scaling the TVB connectivity weight
                                            "delay": 0.0}]  # additional delay to the one of TVB connectivity

    connections = OrderedDict()
    connections["E"] = "E"
    connections["I"] = "I"
    nest_model_builder.output_devices = [{"model": "spike_detector",
                                          "props": config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
                                          "nodes": None, "connections": connections}]

    nest_network = nest_model_builder.build_nest_network()

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = RedWWexcIOinhIBuilder(simulator, nest_network, nest_nodes_ids)

    # NEST -> TVB:
    # For current transmission from TVB to NEST,
    # either choose a NEST dc_generator device:
    # tvb_nest_builder.tvb_to_nest_interfaces = [{"model": "dc_generator", "sign": 1,
    #      "connections": {"S_e": ["E", "I"]}}]
    # or modify directly the external current stimulus parameter:
    tvb_nest_builder.tvb_to_nest_interfaces = [{"model": "current", "parameter": "I_e", "sign": 1,
                                                "connections": {"S_e": ["E", "I"]}}]

    # NEST -> TVB:
    # Use S_e and S_i instead of r_e and r_i
    # for transmitting to the TVB state variables directly
    connections = OrderedDict()
    connections["r_e"] = "E"
    connections["r_i"] = "I"
    tvb_nest_builder.nest_to_tvb_interfaces = [{"model": "spike_detector", "params": {}, "connections": connections}]

    tvb_nest_model = tvb_nest_builder.build_interface()

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t = time.time()
    results = simulator.run(simulation_length=100.0)

    return connectivity.weights, connectivity.tract_lengths, results[0][1]


def test_connectivity_weights_shape():
    (weights, tract_lengths, results) = prepare_launch_default_simulation()
    assert weights.shape == (68, 68)


def test_connectivity_tract_lengths_shape():
    (weights, tract_lengths, results) = prepare_launch_default_simulation()
    assert tract_lengths.shape == (68, 68)


def test_results_shape():
    (weights, tract_lengths, results) = prepare_launch_default_simulation()
    assert not np.isinf(results.ravel()).all()
    assert not np.isnan(results.ravel()).all()
    assert results.shape == (1000, 4, 68, 1)
