# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import CONFIGURED, Config
from tvb_multiscale.tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.interfaces.builders.models.red_ww_basal_ganglia_izhikevich \
    import RedWWexcIOBuilder as BasalGangliaRedWWexcIOBuilder
from tvb_multiscale.core.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter
from examples.tvb_nest.example import results_path_fun
from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder,
                 nest_nodes_ids, nest_populations_order=100,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0, variables_of_interest=None,
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
    simulator = simulator_builder.build(**model_params)

    # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    print("Building NEST network...")
    tic = time.time()

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = nest_model_builder(simulator, nest_nodes_ids, config=config)
    nest_model_builder.population_order = nest_populations_order
    populations = []
    populations_sizes = []
    for pop in nest_model_builder.populations:
        populations.append(pop["label"])
        populations_sizes.append(int(np.round(pop["scale"] * nest_model_builder.population_order)))
    # Common order of neurons' number per population:
    nest_network = nest_model_builder.build_spiking_network()
    print(nest_network.print_str(connectivity=False))
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model -----------------------------------------

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

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    # Integrate NEST one more NEST time step so that multimeters get the last time point
    # unless you plan to continue simulation later
    simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
    # Clean-up NEST simulation
    simulator.tvb_spikeNet_interface.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(results, simulator, populations=populations, populations_sizes=populations_sizes,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    return results, simulator


if __name__ == "__main__":

    nest_nodes_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    import os
    home_path = os.path.join(os.getcwd().split("tvb-multiscale")[0], "tvb-multiscale")
    DATA_PATH = os.path.join(home_path, "examples/data")

    w = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/conn_denis_weights.txt"))

    c = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/aal_plus_BG_centers.txt"), usecols=range(1, 3))
    rl = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/aal_plus_BG_centers.txt"), dtype="str", usecols=(0,))
    t = np.loadtxt(os.path.join(DATA_PATH, "./basal_ganglia_conn/BGplusAAL_tract_lengths.txt"))

    # Remove BG -> Cortex connections
    w[[0, 1, 2, 3, 6, 7], :][:, 10:] = 0.0
    connectivity = Connectivity(region_labels=rl, weights=w, centres=c, tract_lengths=t)

    tvb_model = ReducedWongWangExcIO  # ReducedWongWangExcIOInhI

    model_params = {}

    main_example(tvb_model, BasalGangliaIzhikevichBuilder, BasalGangliaRedWWexcIOBuilder,
                 nest_nodes_ids,  nest_populations_order=200,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,  # "rate"
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=0.0,
                 variables_of_interest=None,
                 config=None, **model_params)
