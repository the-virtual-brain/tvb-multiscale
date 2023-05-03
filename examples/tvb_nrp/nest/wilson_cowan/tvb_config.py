# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.tvb_nest.config import Config
from tvb_multiscale.core.tvb.cosimulator.cosimulator_nrp import CoSimulatorParallelNRP as CoSimulator


# This would run on TVB engine before creating any multiscale cosimulation interface connections.
# Users can adapt it to their use case.
def build_tvb_simulator(config_class=Config):

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.integrators import HeunStochastic
    from tvb.simulator.monitors import Raw  # , Bold, EEG

    from examples.tvb_nrp.nest.wilson_cowan.config import configure
    from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan

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

    simulator = CoSimulator()

    simulator.model = WilsonCowan(**model_params)

    simulator.integrator = HeunStochastic()
    simulator.integrator.dt = 0.1
    simulator.integrator.noise.nsig = np.array([config.DEFAULT_NSIG, config.DEFAULT_NSIG])  # 0.001

    # Load connectivity
    # config.DEFAULT_CONNECTIVITY_ZIP = "/home/docker/packages/tvb_data/tvb_data/mouse/allen_2mm/ConnectivityAllen2mm.zip"
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)

    # -------------- Pick a minimal brain of only the first n_regions regions: ----------------
    n_regions = config.n_regions
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

    # simulator.print_summary_info_details(recursive=1)

    simulator._preconfigure_synchronization_time()

    simulator.configure()

    return simulator, config


# FRONTEND used for user configuration of interfaces.

# These is an example that could be modified by users:

def configure_TVB_interfaces(simulator=None, config=None, config_class=Config):

    if config is None:
        from examples.tvb_nrp.nest.wilson_cowan.config import configure
        config = configure(config_class)

    from tvb_multiscale.core.interfaces.tvb.builders import TVBInterfaceBuilder


    tvb_interface_builder = TVBInterfaceBuilder(config=config)

    # or setting a nonopinionated builder:

    if simulator is not None:
        tvb_interface_builder.tvb_cosimulator = simulator

    # This can be used to set default tranformer and proxy models:
    tvb_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->spikeNet interface
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / spikeNet device for each spiking region,
    # "1-to-1" TVB->spikeNet coupling.
    # If any other value, we need 1 "TVB proxy node" / spikeNet device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in spikeNet,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->spikeNet coupling.
    tvb_interface_builder.default_coupling_mode = "TVB"
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    tvb_interface_builder.exclusive_nodes = True

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_interface_builder.output_interfaces = \
        [{'voi': np.array(["E"]),  # TVB state variable to get data from
          # --------------- Arguments that can default if not given by the user:------------------------------
          'model': 'RATE',  # This can be used to set default tranformer and proxy models
          'coupling_mode': 'TVB',  # or "spikeNet", "spikeNet", etc
          'proxy_inds': config.spiking_nodes_inds  # TVB proxy region nodes' indices
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop, sv in zip(["E", "I"], ["E", "I"]):
        tvb_interface_builder.input_interfaces.append(
            {'voi': np.array([sv]),
             'proxy_inds': config.spiking_nodes_inds
             }
        )

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (TVB-> coupling) interfaces' configurations:\n")

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (TVB<- update) interfaces' configurations:\n")

    tvb_interface_builder.dump_all_interfaces()

    return tvb_interface_builder
