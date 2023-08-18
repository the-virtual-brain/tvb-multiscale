# -*- coding: utf-8 -*-

import os
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
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
                            spikeNet_model_builder.__class__.__name__.split("Builder")[0] +
                            tvb_spikeNet_str +
                            np.where(spikeNet_to_tvb, "_bidir", "").item()
                            )
    else:
        return config.out.FOLDER_RES


def main_example(orchestrator_app, tvb_sim_model, model_params={},
                 spikeNet_model_builder=None, spiking_proxy_inds=[0, 1],
                 tvb_spikeNet_interface_builder=None,
                 tvb_to_spikeNet_interfaces=[], spikeNet_to_tvb_interfaces=[], exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=1100.0, transient=None, initial_conditions=np.array([0.0]),
                 config=None, plot_write=True, config_type=Config, logger_initializer=initialize_logger, **kwargs):

    if config is None:
        try:
            spikeNet_to_tvb = tvb_spikeNet_interface_builder.input_flag
        except:
            spikeNet_to_tvb = len(spikeNet_to_tvb_interfaces) > 0
        config = config_type(output_base=results_path_fun(spikeNet_model_builder,
                                                          tvb_spikeNet_interface_builder.model
                                                          if tvb_spikeNet_interface_builder is not None else "",
                                                          spikeNet_to_tvb))

    logger = logger_initializer(__name__, config=config)

    print("\n"+"-"*100 + "\nFind results in %s !" % config.out.FOLDER_RES + "\n" + "-"*100)

    spiking_proxy_inds = getattr(config, "SPIKING_NODES_INDS", spiking_proxy_inds)
    spiking_proxy_inds = np.array(spiking_proxy_inds)
    exclusive_nodes = getattr(config, "EXCLUSIVE_NODES", exclusive_nodes)
    simulation_length = getattr(config, "SIMULATION_LENGTH", simulation_length)
    config.SIMULATION_LENGTH = simulation_length  # needed for NetPyNE
    if transient is None:
        transient = getattr(config, "TRANSIENT", simulation_length/11)

    orchestrator = orchestrator_app(
        config=config,
        logger=logger,
    )
    orchestrator.start()

    plotter = Plotter(config)

    # -----------------------------------1. Models' and simulation configuration ---------------------------------------

    print("\n\nConfiguring...")
    tic = time.time()

    # -----------------------------------------a. Configure a TVB simulator builder ------------------------------------
    orchestrator.tvb_app.cosimulator_builder = kwargs.pop("cosimulator_builder", CoSimulatorSerialBuilder())
    orchestrator.tvb_app.cosimulator_builder.model = tvb_sim_model
    orchestrator.tvb_app.cosimulator_builder.model_params = model_params
    orchestrator.tvb_app.cosimulator_builder.simulation_length = simulation_length  # needed at this stage by NETPYNE!!!
    if not isinstance(connectivity, Connectivity):
        connectivity = Connectivity.from_file(connectivity)
    orchestrator.tvb_app.cosimulator_builder.connectivity = connectivity
    orchestrator.tvb_app.cosimulator_builder.delays_flag = delays_flag
    if initial_conditions is not None:
        orchestrator.tvb_app.cosimulator_builder.initial_conditions = initial_conditions

    # -----------------------------------------b. Set the spiking network model builder---------------------------
    orchestrator.spikeNet_app.spikeNet_builder = spikeNet_model_builder
    orchestrator.spikeNet_app.spikeNet_builder.spiking_nodes_inds = spiking_proxy_inds

    # -----------------------------------------c. Configure the TVB-SpikeNet interface model ---------------------------
    orchestrator.tvb_app.interfaces_builder = tvb_spikeNet_interface_builder
    orchestrator.tvb_app.interfaces_builder.proxy_inds = spiking_proxy_inds
    orchestrator.tvb_app.interfaces_builder.exclusive_nodes = exclusive_nodes
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

    print_summary = True

    # only applicable for NetPyNE parallel simulation with MPI: skip printing and plotting the results unless being on root MPI node:
    if hasattr(orchestrator.spikeNet_app.spiking_cosimulator, 'isRootNode') and \
        not orchestrator.spikeNet_app.spiking_cosimulator.isRootNode():
        print_summary = False
        plot_write = False

    if print_summary:
        print(orchestrator.print_summary_info_details(recursive=2, connectivity=False))

    # -------------------------------------4. Configure, Simulate and gather results------------------------------------
    print("\n\nSimulating...")
    tic = time.time()
    orchestrator.simulate(simulation_length)
    print("\nSimulated in %f secs!\n" % (time.time() - tic))

    simulator = orchestrator.tvb_cosimulator
    results = orchestrator.tvb_app.results

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        print("\n\nPlotting and/or writing results to files...")
        tic = time.time()
        # try:
        orchestrator.plot(spiking_nodes_ids=spiking_proxy_inds, transient=transient,
                          # populations=["E", "I"], populations_sizes=[],
                          tvb_state_variable_type_label="State Variable",
                          tvb_state_variables_labels=simulator.model.variables_of_interest,
                          plot_per_neuron=True, plotter=plotter)
        # plot_write_results(results, simulator,
        #                    orchestrator.spiking_network, spiking_proxy_inds,
        #                    transient=transient, tvb_state_variable_type_label="State Variables",
        #                    tvb_state_variables_labels=simulator.model.variables_of_interest,
        #                    plot_per_neuron=True, plotter=plotter, config=config)
        # except Exception as e:
        #     print("Error in plotting or writing to files!:\n%s" % str(e))
        print("\nFinished in %f secs!\n" % (time.time() - tic))

    orchestrator.clean_up()
    orchestrator.stop()

    return results, simulator


def default_example(spikeNet_model_builder, tvb_spikeNet_model_builder, orchestrator_app, **kwargs):

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

    model_params.update(kwargs.pop("model_params", {}))

    spikeNet_model_builder.population_order = kwargs.pop("population_order", 100)

    model = kwargs.pop("model", "RATE").upper()
    tvb_spikeNet_model_builder.model = model
    tvb_to_spikeNet_interfaces = []
    spikeNet_to_tvb_interfaces = []
    tvb_spikeNet_model_builder.N_E = spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.input_flag = kwargs.pop("input_flag", True)
    tvb_spikeNet_model_builder.output_flag = kwargs.pop("output_flag", True)

    # An example of a minimal configuration:
    # tvb_to_spikeNet_interfaces = [{"model": model, "voi": "R", "populations": "E"}]
    # spikeNet_to_tvb_interfaces = [{"voi": "R", "populations": "E"}]

    return main_example(orchestrator_app,
                        Linear(), model_params,
                        spikeNet_model_builder, kwargs.pop("spiking_proxy_inds", [0, 1]),
                        tvb_spikeNet_model_builder, tvb_to_spikeNet_interfaces, spikeNet_to_tvb_interfaces, **kwargs)
