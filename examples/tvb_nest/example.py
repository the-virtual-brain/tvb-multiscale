# -*- coding: utf-8 -*-
import os
import time
from six import string_types

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.tvb_nest.orchestrators import TVBNESTSerialOrchestrator, NESTSerialApp, TVBSerialApp
from tvb_multiscale.core.plot.plotter import Plotter
from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan


def results_path_fun(nest_model_builder, tvb_to_nest_mode, nest_to_tvb, config=None):
    if config is None:
        if tvb_to_nest_mode is not None:
            tvb_nest_str = "_" + tvb_to_nest_mode
        else:
            tvb_nest_str = ""
        return os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0],
                            nest_model_builder.__name__.split("Builder")[0] +
                            tvb_nest_str +
                            np.where(nest_to_tvb, "_bidir", "").item()
                            )
    else:
        return config.out.FOLDER_RES


def main_example(tvb_sim_model, nest_model_builder, nest_nodes_ids,
                 model_params={}, nest_populations_order=100,
                 tvb_to_nest_mode="RATE", tvb_to_nest_connections={},
                 tvb_to_nest_proxy_params={}, tvb_to_nest_transformer_params={},
                 nest_to_tvb_connections={}, nest_to_tvb_proxy_params={}, nest_to_tvb_transformer_params={},
                 exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP,
                 simulation_length=110.0, transient=10.0,
                 config=None, plot_write=True):

    if config is None:
        config = \
            Config(output_base=results_path_fun(nest_model_builder, tvb_to_nest_mode, len(nest_to_tvb_connections) > 0))

    orchestrator = TVBNESTSerialOrchestrator(
        config=config,
        exclusive_nodes=exclusive_nodes,
        spiking_proxy_inds=nest_nodes_ids,
        simulation_length=simulation_length
    )
    orchestrator.start()

    plotter = Plotter(config)

    # -----------------------------------1. Models' and simulation configuration ---------------------------------------

    print("\n\nConfiguring...")
    tic = time.time()

    # -----------------------------------------a. Configure a TVB simulator builder ------------------------------------
    orchestrator.tvb_app.simulator_builder.model = tvb_sim_model
    orchestrator.tvb_app.simulator_builder.model_params = model_params
    orchestrator.tvb_app.simulator_builder.connectivity = connectivity

    # -----------------------------------------b. Configure the NEST network model builder------------------------------
    orchestrator.spikeNet_app.spiking_model_builder = nest_model_builder
    orchestrator.spikeNet_app.population_order = nest_populations_order

    # -----------------------------------------c. Configure the TVB-NEST interface model -------------------------------
    orchestrator.tvb_app.interfaces_builder.output_interfaces = []
    for voi, pop in tvb_to_nest_connections.items():
        orchestrator.tvb_app.interfaces_builder.output_interfaces.append(
            {"model": tvb_to_nest_mode, "voi": voi, "populations": pop,
             "transformer_params": tvb_to_nest_transformer_params, "proxy_params": tvb_to_nest_proxy_params}
    )
    orchestrator.tvb_app.interfaces_builder.input_interfaces = []
    for voi, pop in nest_to_tvb_connections.items():
        orchestrator.tvb_app.interfaces_builder.input_interfaces.append(
            {"voi": voi, "populations": pop,
             "transformer_params": nest_to_tvb_transformer_params, "proxy_params": nest_to_tvb_proxy_params}
    )

    # ----------------------------------------d. Run the orchestrator configuration ------------------------------------
    orchestrator.configure()
    print("\nDone! in %f min\n" % ((time.time() - tic) / 60))

    # # -----------------------------------3. Build models, simulators and interfaces-----------------------------------
    print("\n\nBuilding...")
    tic = time.time()
    orchestrator.build()
    print("\nBuilt in %f secs!\n" % (time.time() - t_start))

    # -------------------------------------4. Configure, Simulate and gather results------------------------------------
    print("\n\nSimulating...")
    tic = time.time()
    orchestrator.simulate()
    print("\nSimulated in %f secs!\n" % (time.time() - t_start))

    simulator = orchestrator.tvb_app.cosimulator
    results = orchestrator.tvb_app.results

    populations = orchestrator.get_number_of_neurons_per_region_and_population()

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(results, simulator,
                               populations=populations.coords["Population"], populations_sizes=populations[0].values,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plot_per_neuron=True, plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    orchestrator.clean_up()
    orchestrator.stop()

    return results, simulator


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id, label in enumerate(connectivity.region_labels):
        if label.find("hippo") > 0:
            nest_nodes_ids.append(id)

    tvb_sim_model = WilsonCowan

    model_params = {}

    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

    if isinstance(tvb_sim_model, WilsonCowan):

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

    main_example(tvb_sim_model, WilsonCowanBuilder, InterfaceWilsonCowanBuilder,
                 nest_nodes_ids,  nest_populations_order=100,
                 tvb_to_nest_mode="RATE", tvb_to_nest_connections={},
                 tvb_to_nest_proxy_params={}, tvb_to_nest_transformer_params={},
                 nest_to_tvb_connections={}, nest_to_tvb_proxy_params={}, nest_to_tvb_transformer_params={},
                 exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 variables_of_interest=None,
                 config=None, **model_params)
