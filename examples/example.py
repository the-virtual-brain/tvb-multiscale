# -*- coding: utf-8 -*-

import os
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter

from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity


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


def main_example(tvb_sim_model, model_params={},
                 spikeNet_model_builder=None, spiking_proxy_inds=[],
                 tvb_spikeNet_interface_builder=None,
                 tvb_to_spikeNet_mode="rate", spikeNet_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 config=None, plot_write=True, config_type=Config,  logger_initializer=initialize_logger):

    if config is None:
        config = \
            config_type(output_base=results_path_fun(spikeNet_model_builder, tvb_to_spikeNet_mode, spikeNet_to_tvb))

    logger = logger_initializer(__name__, config=config)

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = CoSimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model()
    if not isinstance(connectivity, Connectivity):
        simulator_builder.connectivity = Connectivity.from_file(connectivity)
    simulator_builder.delays_flag = delays_flag
    simulator = simulator_builder.build(**model_params)

    # ----2. Build the Spiking Network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)------

    print("\n\nBuilding Spiking Network...")
    tic = time.time()
    # Build a Spiking Network model with the corresponding builder
    # Using all default parameters for this example
    # spikeNet_model_builder = spikeNet_model_builder(simulator, spiking_proxy_inds, config=config, logger=logger)
    # Common order of neurons' number per population:
    spikeNet_model_builder.configure()
    spiking_network = spikeNet_model_builder.build(set_defaults=True)
    print("\nDone! in %f min\n" % ((time.time() - tic) / 60))
    print(spiking_network.print_str(connectivity=True))

    # -----------------------------------3. Build the TVB-SpikeNet interface model -------------------------------------

    print("\n\nBuilding TVB-Spiking Network interface...")
    tic = time.time()
    populations = []
    populations_sizes = []
    for pop in spikeNet_model_builder.populations:
        populations.append(pop["label"])
        populations_sizes.append(int(np.round(pop["scale"] * spikeNet_model_builder.population_order)))
    # Build a TVB-SpikeNet interface with all the appropriate connections between the TVB and SpikeNet modelled regions
    # Using all default parameters for this example
    tvb_spikeNet_interface_builder.simulator = simulator
    tvb_spikeNet_interface_builder.spiking_network = spiking_network
    tvb_spikeNet_interface_builder.spiking_proxy_inds = spiking_proxy_inds
    tvb_spikeNet_interface_builder.exclusive_nodes = exclusive_nodes
    tvb_spikeNet_interface_builder.populations_sizes = populations_sizes
    # tvb_spikeNet_interface_builder = tvb_spikeNet_interface_builder(simulator, spiking_network, spiking_proxy_inds,
    #                                                                 exclusive_nodes,
    #                                                                 populations_sizes=populations_sizes)
    tvb_spikeNet_model = tvb_spikeNet_interface_builder.build_interface(tvb_to_spikeNet_mode=tvb_to_spikeNet_mode,
                                                                        spikeNet_to_tvb=spikeNet_to_tvb)
    print("\nDone! in %f min\n" % ((time.time() - tic)/60))
    print(tvb_spikeNet_model.print_str(detailed_output=True, connectivity=False))

    # -----------------------------------4. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-spikeNet interface...
    simulator.configure(tvb_spikeNet_model)
    # ...and simulate!
    print("\n\nSimulating...")
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    try:
        # Only for NEST:
        # Integrate NEST one more NEST time step so that multimeters get the last time point
        # unless you plan to continue simulation later
        simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        simulator.tvb_spikeNet_interface.nest_instance.Cleanup()
    except:
        pass
    print("\nSimulated in %f secs!\n" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        print("\n\nPlotting and/or writing results to files...")
        tic = time.time()
        # try:
        plot_write_results(results, simulator, spiking_network, spiking_proxy_inds,
                           transient=transient, tvb_state_variable_type_label="State Variables",
                           tvb_state_variables_labels=simulator.model.variables_of_interest,
                           plot_per_neuron=True, plotter=plotter, config=config)
        # except Exception as e:
        #     print("Error in plotting or writing to files!:\n%s" % str(e))
        print("\nFinished in %f secs!\n" % (time.time() - tic))

    return results, simulator


def default_example(spikeNet_builder, tvb_spikeNet_builder, **kwargs):

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
    spikeNet_builder.spiking_nodes_inds = kwargs.pop("spiking_proxy_inds", [0, 1])
    spikeNet_builder.populations_order = kwargs.pop("populations_order", 100)

    return main_example(Linear, model_params,
                        spikeNet_builder, spikeNet_builder.spiking_nodes_inds,
                        tvb_spikeNet_builder,
                        **kwargs)
