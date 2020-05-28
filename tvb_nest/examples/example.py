# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.builders.models.default_exc_io_inh_i import \
    DefaultExcIOInhIBuilder, DefaultExcIOInhIMultisynapseBuilder
from tvb_nest.nest_models.builders.models.ww_deco2014 import WWDeco2014Builder
from tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder, WilsonCownMultisynapseBuilder
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder, \
    WilsonCowanMultisynapseBuilder as InterfaceWilsonCowanMultisynapseBuilder
from tvb_multiscale.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.examples.plot_write_results import plot_write_results
from tvb_multiscale.plot.plotter import Plotter
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder, nest_nodes_ids, nest_populations_order=100,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=False,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=1100.0, transient=100.0, variables_of_interest=None,
                 config=CONFIGURED, **model_params):

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
    populations_sizes = [int(np.round(nest_model_builder.scale_e * nest_model_builder.population_order)),
                         int(np.round(nest_model_builder.scale_i * nest_model_builder.population_order))]
    # Common order of neurons' number per population:
    nest_network = nest_model_builder.build_spiking_network()

    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model -----------------------------------------

    print("Building TVB-NEST interface...")
    tic = time.time()
    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = tvb_nest_builder(simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                        N_E=populations_sizes[0], N_I=populations_sizes[1])
    tvb_nest_model = tvb_nest_builder.build_interface(tvb_to_nest_mode=tvb_to_nest_mode, nest_to_tvb=nest_to_tvb)
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
    plot_write_results(results, simulator, populations_sizes, transient, "State Variables",
                       simulator.model.variables_of_interest, plotter, config)

    return results, simulator


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id, label in enumerate(connectivity.region_labels):
        if label.find("hippo") > 0:
            nest_nodes_ids.append(id)

    tvb_model = ReducedWongWangExcIOInhI
    model_params = {}

    # # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
    # from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
    #
    # tvb_model = WilsonCowan
    #
    # model_params = {
    #     "r_e": np.array([0.0]),
    #     "r_i": np.array([0.0]),
    #     "k_e": np.array([1.0]),
    #     "k_i": np.array([1.0]),
    #     "tau_e": np.array([10.0]),
    #     "tau_i": np.array([10.0]),
    #     "c_ee": np.array([10.0]),
    #     "c_ei": np.array([6.0]),
    #     "c_ie": np.array([10.0]),
    #     "c_ii": np.array([1.0]),
    #     "alpha_e": np.array([1.2]),
    #     "alpha_i": np.array([2.0]),
    #     "a_e": np.array([1.0]),
    #     "a_i": np.array([1.0]),
    #     "b_e": np.array([0.0]),
    #     "b_i": np.array([0.0]),
    #     "c_e": np.array([1.0]),
    #     "c_i": np.array([1.0]),
    #     "theta_e": np.array([2.0]),
    #     "theta_i": np.array([3.5]),
    #     "P": np.array([0.5]),
    #     "Q": np.array([0.0])
    # }

    main_example(tvb_model, WWDeco2014Builder, ReducedWongWangExcIOInhI,
                 nest_nodes_ids,  nest_populations_order=100,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=1100.0, transient=100.0,
                 variables_of_interest=None,
                 config=CONFIGURED, **model_params)
