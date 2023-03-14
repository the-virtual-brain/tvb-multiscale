# -*- coding: utf-8 -*-
import os
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import CONFIGURED, Config
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO
# from tvb_multiscale.core.tvb.cosimulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb_multiscale.core.plot.plotter import Plotter
from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity


def results_path_fun(tvb_sim_model, config=None):
    if config is None:
        return os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0], tvb_sim_model.__name__)
    else:
        return config.out.FOLDER_RES


def main_example(tvb_sim_model=WilsonCowan, connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP,
                 simulation_length=110.0, transient=10.0, config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(output_base=results_path_fun(tvb_sim_model, config))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = CoSimulatorSerialBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model()
    simulator_builder.model_params = model_params
    if not isinstance(connectivity, Connectivity):
        connectivity = Connectivity.from_file(connectivity)
    simulator_builder.connectivity = connectivity

    simulator = simulator_builder.build(**model_params)

    # if isinstance(simulator.model, SpikingWongWangExcIOInhI):
    #     from tvb_multiscale.tvb_elephant.spike_stimulus_builder import SpikeStimulusBuilder
    #     from tvb.simulator.integrators import Dop853Stochastic
    #     populations_sizes = [int(simulator.model.N_E[0]), int(simulator.model.N_I[0])]
    #     spiking_regions_inds = list(range(simulator.connectivity.number_of_regions))
    #     # Some code only for SpikingWongWangExcIOInhI
    #     simulator.integrator = Dop853Stochastic()
    #     simulator.integrator.dt = 0.025
    #     simulator.integrator.noise.nsig = np.array(simulator.model.nvar * [simulator.integrator.noise.nsig[0]])
    #     # No noise for t_ref and derived variables
    #     simulator.integrator.noise.nsig[6:] = 0.0
    #     time_length = int(np.ceil(simulation_length / simulator.integrator.dt)) + 1
    #     stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=2400.0, A=None,  # A=None for uncorrelated spikes
    #                                number_of_neurons=20, number_of_regions=len(spiking_regions_inds),
    #                                target_regions=spiking_regions_inds,
    #                                t_start=0.0, dt=simulator.integrator.dt, time_length=time_length, sparse=False)
    #     simulator.spike_stimulus = stb.build_compound_poisson_process()
    # else:
    #     populations_sizes = [1, 1]

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -----------------------------------2. Simulate and gather results-------------------------------------------------
    # Configure the simulator
    simulator.configure()
    # ...and simulate!
    print("\n\nSimulating...")
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!\n" % (time.time() - t_start))

    # -------------------------------------------3. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(results, simulator, transient=transient,
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               # populations_sizes=populations_sizes,
                               plotter=Plotter(config), config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    return results, simulator


if __name__ == "__main__":

    import sys

    if sys.argv[-1] == "1":
        main_example(Linear)
    elif sys.argv[-1] == "2":
        main_example(LinearReducedWongWangExcIO)
    elif sys.argv[-1] == "3":
        main_example(ReducedWongWangExcIO)
    elif sys.argv[-1] == "4":
        main_example(ReducedWongWangExcIOInhI)
    else:

        # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
        model_params = {
            "tau_e": np.array([8.0]),
            "tau_i": np.array([8.0]),
            "c_ee": np.array([16.0]),
            "c_ei": np.array([12.0]),
            "c_ie": np.array([15.0]),
            "c_ii": np.array([3.0]),
            "a_e": np.array([1.3]),
            "a_i": np.array([2.0]),
            "b_e": np.array([4.0]),
            "b_i": np.array([3.7]),
            "P": np.array([0.0])
        }


        # # ----------------------------------------SpikingWongWangExcIOInhI/MultiscaleWongWangExcIOInhI------------------------------------------------
        # model_params = {
        #     "N_E": np.array([16, ]),
        #     "N_I": np.array([4, ]),
        #     "w_IE": np.array([1.0, ]),
        #     "lamda": np.array([0.5, ]),
        #     "G": np.array([200.0, ])
        # }

        main_example(WilsonCowan, **model_params)
