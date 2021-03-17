# -*- coding: utf-8 -*-

import os
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.orchestrators import TVBNESTSerialOrchestrator
from tvb_multiscale.tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.datatypes.connectivity import Connectivity

from examples.plot_write_results import plot_write_results


def results_path_fun(spikeNet_model_builder, tvb_to_spikeNet_mode, spikeNet_to_tvb, config=None):
    if config is None:
        if tvb_to_spikeNet_mode is not None:
            tvb_spikeNet_str = "_" + tvb_to_spikeNet_mode
        else:
            tvb_spikeNet_str = ""
        return os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0],
                            spikeNet_model_builder.__name__.split("Builder")[0] +
                            tvb_spikeNet_str +
                            np.where(spikeNet_to_tvb, "_bidir", "").item()
                            )
    else:
        return config.out.FOLDER_RES


def main_example(tvb_sim_model, spikeNet_model_builder, spiking_proxy_inds,
                 model_params={}, populations_order=100,
                 tvb_to_spikeNet_interfaces=[], spikeNet_to_tvb_interfaces=[], exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 config=None, plot_write=True):

    if config is None:
        if len(tvb_to_spikeNet_interfaces):
            tvb_to_spikeNet_mode = tvb_to_spikeNet_interfaces[0]["model"]
        else:
            tvb_to_spikeNet_mode = None
        config = \
            Config(output_base=results_path_fun(spikeNet_model_builder,
                                                tvb_to_spikeNet_mode, len(spikeNet_to_tvb_interfaces) > 0))

    logger = initialize_logger(__name__, config=config)

    orchestrator = TVBNESTSerialOrchestrator(
        config=config,
        logger=logger,
        exclusive_nodes=exclusive_nodes,
        spiking_proxy_inds=np.array(spiking_proxy_inds),
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
    if not isinstance(connectivity, Connectivity):
        connectivity = Connectivity.from_file(connectivity)
    orchestrator.tvb_app.cosimulator_builder.connectivity = connectivity
    orchestrator.tvb_app.cosimulator_builder.delays_flag = delays_flag

    # -----------------------------------------b. Configure the spiking network model builder------------------------------
    orchestrator.spikeNet_app.spikeNet_builder = spikeNet_model_builder(config=config)
    orchestrator.spikeNet_app.population_order = populations_order

    # -----------------------------------------c. Configure the TVB-SpikeNet interface model -------------------------------
    orchestrator.tvb_app.interfaces_builder.output_interfaces = tvb_to_spikeNet_interfaces
    orchestrator.tvb_app.interfaces_builder.input_interfaces = spikeNet_to_tvb_interfaces

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

    simulator = orchestrator.tvb_cosimulator
    results = orchestrator.tvb_app.results

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        print("\n\nPlotting and/or writing results to files...")
        tic = time.time()
        # try:
        plot_write_results(results, simulator,
                               orchestrator.spiking_network, orchestrator.spiking_proxy_inds,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plot_per_neuron=True, plotter=plotter, config=config)
        # except Exception as e:
        #     print("Error in plotting or writing to files!:\n%s" % str(e))
        print("\nFinished in %f secs!\n" % (time.time() - tic))

    orchestrator.clean_up()
    orchestrator.stop()

    return results, simulator


if __name__ == "__main__":

    # Select the regions for the fine scale modeling with spiking networks
    spiking_proxy_inds = [0, 1]  # the indices of fine scale regions modeled as spiking networks

    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

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
    tvb_to_spikeNet_interfaces = [{"model": "RATE", "voi": "E", "populations": "E",
                                   "transformer_params":
                                       {"scale_factor": np.array([10000.0])},
                                   "proxy_params": {"number_of_neurons": 1}}]
    spikeNet_to_tvb_interfaces = [{"voi": "E", "populations": "E",
                                   "transformer_params":
                                       {"scale_factor": np.array([1e-6])}},  # (dt(ms) * 1000)Hz*100(neurons)
                                  {"voi": "I", "populations": "I",
                                   "transformer_params":
                                       {"scale_factor": np.array([1e-6])}}]  # (dt(ms) * 1000)Hz*100(neurons)

    main_example(WilsonCowan, WilsonCowanBuilder, spiking_proxy_inds,
                 model_params=model_params, populations_order=100,
                 tvb_to_spikeNet_interfaces=tvb_to_spikeNet_interfaces,
                 spikeNet_to_tvb_interfaces=spikeNet_to_tvb_interfaces,
                 exclusive_nodes=True, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 config=None)
