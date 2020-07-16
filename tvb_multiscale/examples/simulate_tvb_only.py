# -*- coding: utf-8 -*-
import os
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.examples.plot_write_results import plot_write_results
from tvb_multiscale.config import CONFIGURED, Config
from tvb_multiscale.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.plot.plotter import Plotter

from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI


# TODO Fix SpikingWongWangExcIOInhI & MultiscaleWongWangExcIOInhI functionality


def main_example(tvb_sim_model=ReducedWongWangExcIOInhI, connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP,
                 simulation_length=110.0, transient=10.0, config=None, **model_params):

    if config is None:
        config = Config(output_base=os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0],
                                                 tvb_sim_model.__name__))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = SimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.connectivity = connectivity

    simulator = simulator_builder.build(**model_params)

    if isinstance(simulator.model, SpikingWongWangExcIOInhI):
        from tvb_elephant.spike_stimulus_builder import SpikeStimulusBuilder
        from tvb.simulator.integrators import Dop853Stochastic
        populations_sizes = [int(simulator.model.N_E[0]), int(simulator.model.N_I[0])]
        spiking_regions_inds = list(range(simulator.connectivity.number_of_regions))
        # Some code only for SpikingWongWangExcIOInhI
        simulator.integrator = Dop853Stochastic()
        simulator.integrator.dt = 0.025
        simulator.integrator.noise.nsig = np.array(simulator.model.nvar * [simulator.integrator.noise.nsig[0]])
        # No noise for t_ref and derived variables
        simulator.integrator.noise.nsig[6:] = 0.0
        time_length = int(np.ceil(simulation_length / simulator.integrator.dt)) + 1
        stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=2400.0, A=None,  # A=None for uncorrelated spikes
                                   number_of_neurons=20, number_of_regions=len(spiking_regions_inds),
                                   target_regions=spiking_regions_inds,
                                   t_start=0.0, dt=simulator.integrator.dt, time_length=time_length, sparse=False)
        simulator.spike_stimulus = stb.build_compound_poisson_process()
    else:
        populations_sizes = [1, 1]

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -----------------------------------2. Simulate and gather results-------------------------------------------------
    # Configure the simulator
    simulator.configure()
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------3. Plot results--------------------------------------------------------

    plot_write_results(results, simulator, transient=transient,
                       tvb_state_variables_labels=simulator.model.variables_of_interest,
                       populations_sizes=populations_sizes, plotter=Plotter(config), config=config)

    return simulator.connectivity, results


if __name__ == "__main__":

    model_params = {}

    # # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
    # model_params = {
    #     "tau_e": np.array([8.0]),
    #     "tau_i": np.array([8.0]),
    #     "c_ee": np.array([16.0]),
    #     "c_ei": np.array([12.0]),
    #     "c_ie": np.array([15.0]),
    #     "c_ii": np.array([3.0]),
    #     "a_e": np.array([1.3]),
    #     "a_i": np.array([2.0]),
    #     "b_e": np.array([4.0]),
    #     "b_i": np.array([3.7]),
    #     "P": np.array([0.0])
    # }


    # ----------------------------------------SpikingWongWangExcIOInhI/MultiscaleWongWangExcIOInhI------------------------------------------------
    model_params = {
        "N_E": np.array([16, ]),
        "N_I": np.array([4, ]),
        "w_IE": np.array([1.0, ]),
        "lamda": np.array([0.5, ]),
        "G": np.array([200.0, ])
    }

    main_example(SpikingWongWangExcIOInhI, simulation_length=100.0, **model_params)
