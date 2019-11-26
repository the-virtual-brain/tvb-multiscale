# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.examples.plot_results import plot_results
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.simulator_tvb.models.wilson_cowan_constraint import WilsonCowan
from tvb_nest.simulator_tvb.models.generic_2d_oscillator import Generic2dOscillator
from tvb_nest.simulator_nest.builders.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder
from tvb_nest.simulator_nest.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i \
    import RedWWexcIOinhIBuilder as InterfaceRedWWexcIOinhIBuilder
from tvb_nest.interfaces.builders.models.wilson_cowan import \
    WilsonCowanBuilder as InterfaceWilsonCowanBuilder
from tvb_nest.interfaces.builders.models.generic_2d_oscillator import \
    Generic2DOscillatorBuilder as InterfaceGeneric2DOscillatorBuilder
from tvb_nest.plot.plotter import Plotter
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder, nest_nodes_ids, nest_populations_order=100,
                 connectivity=None, connectivity_zip=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, simulation_length=100.0,
                 tvb_state_variable_type_label="Synaptic Gating Variable",
                 dt=0.1, noise_strength=0.1, exclusive_nodes=False, config=CONFIGURED):

    plotter = Plotter(config)

    # --------------------------------------1. Load TVB connectivity----------------------------------------------------
    if connectivity is None:
        connectivity = Connectivity.from_file(connectivity_zip)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)


    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.integrator.dt = dt
    simulator.integrator.noise.nsig = np.array([noise_strength])
    simulator.model = tvb_sim_model

    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # ------3. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    print("Building NEST network...")
    tic = time.time()

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = nest_model_builder(simulator, nest_nodes_ids, config=config)
    # Common order of neurons' number per population:
    nest_model_builder.populations_order = nest_populations_order
    nest_network = nest_model_builder.build_nest_network()

    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------4. Build the TVB-NEST interface model -----------------------------------------

    print("Building TVB-NEST interface...")
    tic = time.time()
    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = tvb_nest_builder(simulator, nest_network, nest_nodes_ids, exclusive_nodes)
    tvb_nest_model = tvb_nest_builder.build_interface()
    print("Done! in %f min" % ((time.time() - tic)/60))

    # -----------------------------------5. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    # Integrate NEST one more NEST time step so that multimeters get the last time point
    # unless you plan to continue simulation later
    simulator.simulate_nest(simulator.tvb_nest_interface.nest_instance.GetKernelStatus("resolution"))
    # Clean-up NEST simulation
    if simulator.simulate_nest == simulator.tvb_nest_interface.nest_instance.Run:
        simulator.tvb_nest_interface.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------6. Plot results--------------------------------------------------------

    plot_results(results, simulator, tvb_nest_model, tvb_state_variable_type_label,
                 simulator.model.variables_of_interest, plotter)

    return connectivity, results


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id in range(connectivity.region_labels.shape[0]):
        if connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)
    # -----------------------------Generic2dOscillator oscillatory regime-----------------------------------------------
    model = Generic2dOscillator()
    model.d = np.array([0.1])
    model.e = np.array([0.0])
    model.f = np.array([1.0])
    model.g = np.array([1.0])
    model.I = np.array([0.0])
    model.tau = np.array([1.0])
    model.alpha = np.array([1.0])
    model.beta = np.array([0.0])
    model.a = np.array([0.0])
    model.b = np.array([-1.0])
    model.c = np.array([0.0])
    # model.variables_of_interest = ["V", "W"]
    # # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
    # model = WilsonCowan()
    # model.tau_e = np.array([8.0])
    # model.tau_i = np.array([8.0])
    # model.c_ee = np.array([16.0])
    # model.c_ei = np.array([12.0])
    # model.c_ie = np.array([15.0])
    # model.c_ii = np.array([3.0])
    # model.a_e = np.array([1.3])
    # model.a_i = np.array([2.0])
    # model.b_e = np.array([4.0])
    # model.b_i = np.array([3.7])
    # model.P = np.array([0.0])
    # model.variables_of_interest = ["E", "I"]
    main_example(model, DefaultExcIOInhIBuilder, InterfaceGeneric2DOscillatorBuilder,
                 nest_nodes_ids, nest_populations_order=100, connectivity=connectivity, simulation_length=1000.0,
                 tvb_state_variable_type_label="State Variables",
                 exclusive_nodes=True, noise_strength=0.000001, config=CONFIGURED)
