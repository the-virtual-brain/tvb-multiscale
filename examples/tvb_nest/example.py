# -*- coding: utf-8 -*-
import os
import time
from six import string_types

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.orchestrators import TVBNESTSerialOrchestrator, NESTSerialApp, TVBSerialApp
from tvb_multiscale.tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.plot.plotter import Plotter
from examples.plot_write_results import plot_write_results
from tvb.datatypes.connectivity import Connectivity


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


def main_example(tvb_sim_model, nest_model_builder, nest_nodes_inds,
                 model_params={}, nest_populations_order=100,
                 tvb_to_nest_interfaces=[], nest_to_tvb_interfaces=[], exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 config=None, plot_write=True):

    if config is None:
        if len(tvb_to_nest_interfaces):
            tvb_to_nest_mode = tvb_to_nest_interfaces[0]["model"]
        else:
            tvb_to_nest_mode = None
        config = \
            Config(output_base=results_path_fun(nest_model_builder, tvb_to_nest_mode, len(nest_to_tvb_interfaces) > 0))

    logger = initialize_logger(__name__, config=config)

    orchestrator = TVBNESTSerialOrchestrator(
        config=config,
        logger=logger,
        exclusive_nodes=exclusive_nodes,
        spiking_proxy_inds=np.array(nest_nodes_inds),
        simulation_length=simulation_length
    )
    orchestrator.start()

    plotter = Plotter(config)

    # -----------------------------------1. Models' and simulation configuration ---------------------------------------

    print("\n\nConfiguring...")
    tic = time.time()

    # -----------------------------------------a. Configure a TVB simulator builder ------------------------------------
    orchestrator.tvb_app.cosimulator_builder.model = tvb_sim_model()
    orchestrator.tvb_app.cosimulator_builder.model_params = model_params
    orchestrator.tvb_app.cosimulator_builder.connectivity = connectivity
    orchestrator.tvb_app.cosimulator_builder.delays_flag = delays_flag

    # -----------------------------------------b. Configure the NEST network model builder------------------------------
    orchestrator.spikeNet_app.spikeNet_builder = nest_model_builder(config=config)
    orchestrator.spikeNet_app.population_order = nest_populations_order

    # -----------------------------------------c. Configure the TVB-NEST interface model -------------------------------
    orchestrator.tvb_app.interfaces_builder.output_interfaces = tvb_to_nest_interfaces
    orchestrator.tvb_app.interfaces_builder.input_interfaces = nest_to_tvb_interfaces

    # ----------------------------------------d. Run the orchestrator configuration ------------------------------------
    orchestrator.configure()
    print("\nDone! in %f min\n" % ((time.time() - tic) / 60))

    # # -----------------------------------3. Build models, simulators and interfaces-----------------------------------
    print("\n\nBuilding...")
    tic = time.time()
    orchestrator.build()
    print("\nBuilt in %f secs!\n" % (time.time() - tic))

    # -------------------------------------4. Configure, Simulate and gather results------------------------------------
    print("\n\nSimulating...")
    tic = time.time()
    orchestrator.simulate()
    print("\nSimulated in %f secs!\n" % (time.time() - tic))

    populations_sizes = orchestrator.populations_sizes

    simulator = orchestrator.tvb_cosimulator
    results = orchestrator.tvb_app.results

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        print("\n\nPlotting and/or writing results to files...")
        tic = time.time()
        try:
            plot_write_results(results, simulator, orchestrator.spiking_network,
                               populations=list(populations_sizes.keys()),
                               populations_sizes=list(populations_sizes.values()),
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plot_per_neuron=True, plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))
        print("\nFinished in %f secs!\n" % (time.time() - tic))

    orchestrator.clean_up()
    orchestrator.stop()

    return results, simulator


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_inds = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id, label in enumerate(connectivity.region_labels):
        if label.find("hippo") > 0:
            nest_nodes_inds.append(id)

    model_params = {}

    tvb_sim_model = WilsonCowan

    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

    if tvb_sim_model == WilsonCowan:

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
        tvb_to_nest_interfaces = [{"model": "RATE", "voi": "E", "populations": "E",
                                   "proxy_params": {"number_of_neurons": 100}}]
        nest_to_tvb_interfaces = [{"voi": "Ein", "populations": "E"},
                                  {"voi": "Iin", "populations": "I"}]

    main_example(tvb_sim_model, WilsonCowanBuilder, nest_nodes_inds,
                 model_params=model_params, nest_populations_order=100,
                 tvb_to_nest_interfaces=tvb_to_nest_interfaces, nest_to_tvb_interfaces=nest_to_tvb_interfaces,
                 exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 config=None)
