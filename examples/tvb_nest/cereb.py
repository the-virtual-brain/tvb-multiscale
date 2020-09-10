# -*- coding: utf-8 -*-
import time
import os
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import CONFIGURED, Config
from examples.tvb_nest.example import results_path_fun
from examples.plot_write_results import plot_write_results
from tvb_multiscale.tvb_nest.nest_models.builders.models.cereb import CerebBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww_cereb import RedWWexcIOBuilder
from tvb_multiscale.core.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.linear_with_stimulus import Linear


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder, nest_nodes_ids, stim_node_id=42,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 transient=0.0, variables_of_interest=None,
                 config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(
                    output_base=results_path_fun(nest_model_builder, tvb_nest_builder, tvb_to_nest_mode, nest_to_tvb,
                                                 config))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = SimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.variables_of_interest = variables_of_interest
    simulator_builder.connectivity = connectivity
    simulator_builder.delays_flag = delays_flag
    simulator_builder.noise_strength = 0.001
    simulator = simulator_builder.build(**model_params)
    simulator.use_numba = False
    simulator.initial_conditions = np.zeros((2, 2, simulator.connectivity.number_of_regions, 1))
    simulator.model.I_o = simulator.model.I_o[0] * np.ones((simulator.connectivity.number_of_regions,))

    # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    print("Building NEST network...")
    tic = time.time()

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = \
        nest_model_builder(simulator, nest_nodes_ids,
                           os.path.join(os.getcwd().split("tvb_nest")[0],
                                        "tvb_nest", "data", "cerebellar_cortex_scaffold_dcn.hdf5"),
                          config=config, set_defaults=True)
    nest_model_builder.modules_to_install = ["cereb"]
    if tvb_nest_builder is not None:
        nest_model_builder.STIMULUS = False

    populations = []
    populations_sizes = []
    for pop in nest_model_builder.populations:
        populations.append(pop["label"])
        populations_sizes.append(int(np.round(pop["scale"] * nest_model_builder.population_order)))
    # Common order of neurons' number per population:
    nest_network = nest_model_builder.build_spiking_network()

    print(nest_network.print_str(connectivity=True))
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model ---------------------------------------
    if tvb_nest_builder is not None:
        print("Building TVB-NEST interface...")
        tic = time.time()
        # Build a TVB-NEST interface with all the appropriate connections between the
        # TVB and NEST modelled regions
        # Using all default parameters for this example
        tvb_nest_builder = tvb_nest_builder(simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                            populations_sizes=populations_sizes[0])
        tvb_nest_model = tvb_nest_builder.build_interface(tvb_to_nest_mode=tvb_to_nest_mode, nest_to_tvb=nest_to_tvb)
        print(tvb_nest_model.print_str(detailed_output=True, connectivity=False))
        print("Done! in %f min" % ((time.time() - tic)/60))

    # -----------------------------------4. Simulate and gather results-------------------------------------------------
    tvb_results = []
    t_start = time.time()
    if tvb_nest_builder is not None:
        print("Simulating TVB-NEST!...")
        # Configure the simulator with the TVB-NEST interface...
        simulator.configure(tvb_nest_model)
        # ...and simulate!
        print("...simulating brackground resting state...")
        results1 = simulator.run(simulation_length=nest_model_builder.STIM_START)
        print("...simulating stimulus activity...")
        simulator.model.I_o[stim_node_id] = 10.0  # 0.75
        results2 = simulator.run(simulation_length=nest_model_builder.STIM_END - nest_model_builder.STIM_START,
                                 configure_spiking_simulator=False)
        print("...simulating relaxation to resting state...")
        simulator.model.I_o[stim_node_id] = 0.0
        results3 = simulator.run(simulation_length=nest_model_builder.TOT_DURATION - nest_model_builder.STIM_END,
                                 configure_spiking_simulator=False)
        tvb_results = [[np.concatenate([results1[0][0], results2[0][0], results3[0][0]]),  # concat time
                       np.concatenate([results1[0][1], results2[0][1], results3[0][1]])]]  # concat data
        del results1, results2, results3
        # Integrate NEST one more NEST time step so that multimeters get the last time point
        # unless you plan to continue simulation later
        simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        simulator.tvb_spikeNet_interface.nest_instance.Cleanup()
    else:
        print("Simulating only NEST!...")
        nest_network.nest_instance.Prepare()
        nest_network.nest_instance.Run(nest_model_builder.TOT_DURATION +
                                       nest_network.nest_instance.GetKernelStatus("resolution"))
        nest_network.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(tvb_results, simulator,
                               spiking_network=nest_network, spiking_nodes_ids=nest_nodes_ids,
                               populations=populations, populations_sizes=populations_sizes,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    return tvb_results, simulator


if __name__ == "__main__":
    import os

    home_path = os.path.join(os.getcwd().split("tvb-multiscale")[0], "tvb-multiscale")
    DATA_PATH = os.path.join(home_path, "examples/tvb_nest/data")
    w = np.loadtxt(os.path.join(DATA_PATH, "mouse_cereb_sum_weights.txt"))
    # t = np.loadtxt(os.path.join(DATA_PATH, "tract_lengths_Count_plusCRBL.txt"))
    # forcing one time step delay for all connections:
    speed = 4.0
    t = speed * 0.1 * np.ones(w.shape)
    # brain_regions_path = os.path.join(DATA_PATH, "centres_brain_MNI.txt")
    # rl = np.loadtxt(brain_regions_path,dtype="str", usecols=(0,))
    with open(os.path.join(DATA_PATH, "mouse_cereb_regions_labels.txt"), "r") as text:
        rl = []
        for line in text:
            rl.append(line)
    rl = np.array(rl)
    # c = np.loadtxt(brain_regions_path, usecols=range(1,3))
    c = np.random.uniform((w.shape[0], 3))
    connectivity = Connectivity(region_labels=rl, weights=w, centres=c, tract_lengths=t)

    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []
    for i_region, reg_lbl in enumerate(connectivity.region_labels):
        if "cereb" in reg_lbl.lower():
            nest_nodes_ids.append(i_region)  # the indices of fine scale regions modeled with NEST
    if len(nest_nodes_ids) == 0:
        nest_nodes_ids = [connectivity.number_of_regions - 1]

    print(["%d. %s" % (nest_node_id, connectivity.region_labels[nest_node_id])
           for nest_node_id in nest_nodes_ids])

    tvb_model = Linear  # ReducedWongWangExcIOInhI

    model_params = {"I_o": np.array([0.0]), "G": np.array([16.0]),
                    "tau": np.array([10.0]), "tau_rin": np.array([10.0])}

    main_example(tvb_model, CerebBuilder, RedWWexcIOBuilder,  # ,
                 nest_nodes_ids, stim_node_id=42,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True, transient=50.0,
                 variables_of_interest=None, config=None, **model_params)
