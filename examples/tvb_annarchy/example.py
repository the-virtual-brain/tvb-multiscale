# -*- coding: utf-8 -*-

import os
import time
from six import string_types

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, Config
from tvb_multiscale.tvb_annarchy.annarchy_models.models.wilson_cowan import WilsonCowanBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.basal_ganglia_izhikevich \
    import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_annarchy.interfaces.builders.models.wilson_cowan \
    import WilsonCowanBuilder as InterfaceWilsonCowanBuilder
from tvb_multiscale.tvb_annarchy.interfaces.builders.models.red_ww_basal_ganglia_izhikevich \
    import RedWWexcIOBuilder as BasalGangliaRedWWexcIOBuilder
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.plot.plotter import Plotter

from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity



def results_path_fun(anarchy_model_builder, tvb_anarchy_builder,
                     tvb_to_anarchy_mode="rate", anarchy_to_tvb=True, config=None):
    if config is None:
        if tvb_anarchy_builder is not None:
            tvb_anarchy_builder_str = "_" + tvb_anarchy_builder.__name__.split("Builder")[0] + \
                                   np.where(isinstance(tvb_to_anarchy_mode, string_types),
                                             "_" + str(tvb_to_anarchy_mode), "").item()
        else:
            tvb_anarchy_builder_str = ""
        return os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0],
                            anarchy_model_builder.__name__.split("Builder")[0] +
                            tvb_anarchy_builder_str +
                            np.where(anarchy_to_tvb, "_bidir", "").item()
                            )
    else:
        return config.out._out_base


def main_example(tvb_sim_model, annarchy_model_builder, tvb_annarchy_builder,
                 annarchy_nodes_ids, annarchy_populations_order=100,
                 tvb_to_annarchy_mode="rate", annarchy_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0, use_numba=True, variables_of_interest=None,
                 config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(
            output_base=results_path_fun(annarchy_model_builder, tvb_annarchy_builder,
                                         tvb_to_annarchy_mode, annarchy_to_tvb, config))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = CoSimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model()
    simulator_builder.variables_of_interest = variables_of_interest
    simulator_builder.connectivity = connectivity
    simulator_builder.delays_flag = delays_flag
    simulator_builder.use_numba = use_numba
    simulator = simulator_builder.build(**model_params)

    # ------2. Build the annarchy network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)---

    print("Building annarchy network...")
    tic = time.time()
    # Build a annarchy network model with the corresponding builder
    # Using all default parameters for this example
    annarchy_model_builder = annarchy_model_builder(simulator, annarchy_nodes_ids, config=config)
    annarchy_model_builder.population_order = annarchy_populations_order
    # Common order of neurons' number per population:
    annarchy_model_builder.configure()
    annarchy_network = annarchy_model_builder.build(set_defaults=True)
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-ANNarchy interface model -------------------------------------

    print("Building TVB-ANNarchy interface...")
    tic = time.time()
    populations = []
    populations_sizes = []
    for pop in annarchy_model_builder.populations:
        populations.append(pop["label"])
        populations_sizes.append(int(np.round(pop["scale"] * annarchy_model_builder.population_order)))
    # Build a TVB-ANNarchy interface with all the appropriate connections between the
    # TVB and ANNarchy modelled regions
    # Using all default parameters for this example
    tvb_annarchy_builder = tvb_annarchy_builder(simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes,
                                                populations_sizes=populations_sizes)
    tvb_annarchy_model = tvb_annarchy_builder.build_interface(tvb_to_annarchy_mode=tvb_to_annarchy_mode,
                                                              annarchy_to_tvb=annarchy_to_tvb)
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------4. Compile network ---------------------------------------------------------
    # Compile the ANNarchy network...
    tic_compile = time.time()
    annarchy_network.configure()
    print("Compiled! in %f min" % ((time.time() - tic_compile) / 60))
    print(annarchy_network.print_str(connectivity=True))
    print(tvb_annarchy_model.print_str(detailed_output=True, connectivity=False))

    # -----------------------------------5. Simulate and gather results-------------------------------------------------
    # ...and simulate!
    t_start = time.time()
    simulator.configure(tvb_annarchy_model)
    results = simulator.run(simulation_length=simulation_length,
                            configure_spiking_simulator=False)  # because compilation has already be done
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(results, simulator, annarchy_network,
                                   populations=populations, populations_sizes=populations_sizes,
                                   transient=transient, tvb_state_variable_type_label="State Variables",
                                   tvb_state_variables_labels=simulator.model.variables_of_interest,
                                   plot_per_neuron=False, plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    annarchy_network.annarchy_instance.clear()

    del annarchy_network.annarchy_instance

    return results, simulator


if __name__ == "__main__":

    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)

    tvb_model = WilsonCowan  # ReducedWongWangExcIO

    model_params = {}

    if tvb_model == WilsonCowan:
        annarchy_nodes_ids = [0, 1]
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
        annarchy_model_builder = WilsonCowanBuilder
        tvb_annarchy_interface_builder = InterfaceWilsonCowanBuilder

    elif tvb_model == ReducedWongWangExcIO:
        annarchy_nodes_ids = list(range(10))
        annarchy_model_builder = BasalGangliaIzhikevichBuilder
        tvb_annarchy_interface_builder = BasalGangliaRedWWexcIOBuilder

    main_example(tvb_model, annarchy_model_builder, tvb_annarchy_interface_builder,  #
                 annarchy_nodes_ids,  annarchy_populations_order=200,
                 tvb_to_annarchy_mode="rate", annarchy_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=0.0,
                 use_numba=True, variables_of_interest=None,
                 config=None, plot_write=True, **model_params)
