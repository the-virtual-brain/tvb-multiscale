# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.examples.plot_results import plot_results
from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.plot.plotter import Plotter
from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def main_example(tvb_sim_model, connectivity_zip=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, dt=0.1, noise_strength=0.001,
                 simulation_length=100.0, config=CONFIGURED):

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
    simulator.integrator = HeunStochastic(dt=dt)
    simulator.integrator.noise.nsig = np.array(ensure_list(noise_strength))
    simulator.model = tvb_sim_model
    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # -----------------------------------3. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    # simulator.configure(tvb_spikeNet_interface=tvb_nest_model)
    simulator.configure()
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------6. Plot results--------------------------------------------------------

    plot_results(results, simulator, "State Variables", simulator.model.variables_of_interest, plotter)

    return connectivity, results


if __name__ == "__main__":

    # # -----------------------------Generic2dOscillator oscillatory regime-----------------------------------------------
    # model = Generic2dOscillator()
    # model.d = np.array([0.1])
    # model.e = np.array([0.0])
    # model.f = np.array([1.0])
    # model.g = np.array([1.0])
    # model.I = np.array([0.0])
    # model.tau = np.array([1.0])
    # model.alpha = np.array([1.0])
    # model.beta = np.array([0.0])
    # model.a = np.array([0.0])
    # model.b = np.array([-1.0])
    # model.c = np.array([0.0])
    # tvb_state_variables_labels=["V", "W"]
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
    # tvb_state_variables_labels = ["E", "I"]

    main_example(ReducedWongWangExcInh(), dt=0.1, noise_strength=0.001, simulation_length=100.0, config=CONFIGURED)
