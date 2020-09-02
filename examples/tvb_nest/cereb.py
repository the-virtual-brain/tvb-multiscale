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
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder,
                 nest_nodes_ids, nest_populations_order=1,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=300.0, transient=0.0, variables_of_interest=None,
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
    nest_model_builder = \
        nest_model_builder(simulator, nest_nodes_ids,
                           os.path.join(os.getcwd().split("tvb_nest")[0],
                                        "tvb_nest", "data", "cerebellar_cortex_scaffold.hdf5"),
                          config=config, set_defaults=True)
    nest_model_builder.modules_to_install = ["cereb"]
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
        tvb_nest_model = tvb_nest_builder.build_interface()
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
        tvb_results = simulator.run(simulation_length=simulation_length)
        # Integrate NEST one more NEST time step so that multimeters get the last time point
        # unless you plan to continue simulation later
        simulator.run_spiking_simulator(simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
        # Clean-up NEST simulation
        simulator.tvb_spikeNet_interface.nest_instance.Cleanup()
    else:
        print("Simulating only NEST!...")
        nest_network.nest_instance.Prepare()
        nest_network.nest_instance.Run(simulation_length)
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
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id, label in enumerate(connectivity.region_labels):
        if label.find("cereb") > 0:
            nest_nodes_ids.append(id)

    if len(nest_nodes_ids) == 0:
        nest_nodes_ids = [0]  # if the connectivity doesn't have cerebellum, just set a region for testing
    tvb_model = ReducedWongWangExcIO  # ReducedWongWangExcIOInhI

    model_params = {}

    main_example(tvb_model, CerebBuilder, RedWWexcIOBuilder,  # ,
                 nest_nodes_ids,  nest_populations_order=1,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=300.0, transient=0.0,
                 variables_of_interest=None,
                 config=None, **model_params)
