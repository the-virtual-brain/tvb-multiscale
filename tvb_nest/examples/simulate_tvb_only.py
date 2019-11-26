# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.examples.plot_results import plot_results
from tvb_nest.config import CONFIGURED
from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb_nest.plot.plotter import Plotter
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def main_example(tvb_sim_model, connectivity_zip=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, dt=0.1,
                 simulation_length=100.0, tvb_state_variable_type_label="Synaptic Gating Variable",
                 tvb_state_variables_labels=["S_e", "S_i"],  config=CONFIGURED):

    plotter = Plotter(config)

    # --------------------------------------1. Load TVB connectivity----------------------------------------------------
    connectivity = Connectivity.from_file(connectivity_zip)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)


    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.integrator.dt = dt
    # simulator.integrator.noise.nsig = np.array([0.00001])
    simulator.model = tvb_sim_model

    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # -----------------------------------3. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    # simulator.configure(tvb_nest_interface=tvb_nest_model)
    simulator.configure()
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------6. Plot results--------------------------------------------------------

    plot_results(results, simulator, None, tvb_state_variable_type_label, tvb_state_variables_labels,
                 plotter)

    return connectivity, results


if __name__ == "__main__":

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
    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

    main_example(model, simulation_length=1000.0, dt=0.1,
                 tvb_state_variable_type_label="State Variables", tvb_state_variables_labels=["V", "W"],
                 config=CONFIGURED)
