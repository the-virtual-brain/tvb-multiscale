# -*- coding: utf-8 -*-
import time
from six import string_types

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, Config
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.models.basal_ganglia_izhikevich \
    import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_annarchy.interfaces.builders.models.red_ww_basal_ganglia_izhikevich \
    import RedWWexcIOBuilder as BasalGangliaRedWWexcIOBuilder
from tvb_multiscale.core.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter
from examples.plot_write_results import plot_write_results

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


def results_path_fun(annarchy_model_builder, tvb_annarchy_builder, config=None):
    if config is None:
        if tvb_annarchy_builder is not None:
            tvb_annarchy_builder_str = "_" + tvb_annarchy_builder.__name__.split("Builder")[0]
        else:
            tvb_annarchy_builder_str = ""
        return os.path.join(CONFIGURED.out.FOLDER_RES.split("/res")[0],
                            annarchy_model_builder.__name__.split("Builder")[0] +
                            tvb_annarchy_builder_str
                            )
    else:
        return config.out.FOLDER_RES


def main_example(tvb_sim_model, annarchy_model_builder, tvb_annarchy_builder,
                 annarchy_nodes_ids, annarchy_populations_order=100,
                 tvb_to_annarchy_mode="rate", annarchy_to_tvb=True, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 simulation_length=110.0, transient=10.0, variables_of_interest=None,
                 config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(
                    output_base=results_path_fun(annarchy_model_builder, tvb_annarchy_builder, config))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = SimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.variables_of_interest = variables_of_interest
    simulator_builder.connectivity = connectivity
    simulator_builder.delays_flag = delays_flag
    simulator = simulator_builder.build(**model_params)

    # ------2. Build the annarchy network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    print("Building annarchy network...")
    tic = time.time()

    # Build a annarchy network model with the corresponding builder
    # Using all default parameters for this example
    annarchy_model_builder = annarchy_model_builder(simulator, annarchy_nodes_ids, config=config)
    annarchy_model_builder.population_order = annarchy_populations_order
    populations = []
    populations_sizes = []
    for pop in annarchy_model_builder.populations:
        populations.append(pop["label"])
        populations_sizes.append(int(np.round(pop["scale"] * annarchy_model_builder.population_order)))
    # Common order of neurons' number per population:
    annarchy_network = annarchy_model_builder.build_spiking_network()
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model -----------------------------------------

    print("Building TVB-NEST interface...")
    tic = time.time()
    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_annarchy_builder = tvb_annarchy_builder(simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes,
                                                populations_sizes=populations_sizes[0])
    tvb_annarchy_model = tvb_annarchy_builder.build_interface(tvb_to_annarchy_mode=tvb_to_annarchy_mode,
                                                              annarchy_to_tvb=annarchy_to_tvb)
    print(tvb_annarchy_model.print_str(detailed_output=True, connectivity=False))
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------4. Compile network ---------------------------------------------------------
    tic_compile = time.time()
    annarchy_network.configure()
    print("Compiled! in %f min" % ((time.time() - tic_compile) / 60))
    print(annarchy_network.print_str(connectivity=True))

    # -----------------------------------5. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_annarchy_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
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

    return results, simulator


if __name__ == "__main__":

    annarchy_nodes_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    import os
    home_path = os.path.join(os.getcwd().split("tvb-multiscale")[0], "tvb-multiscale")
    DATA_PATH = os.path.join(home_path, "examples/data/basal_ganglia_conn")
    wTVB = np.loadtxt(os.path.join(DATA_PATH, "conn_denis_weights.txt"))
    cTVB = np.loadtxt(os.path.join(DATA_PATH, "aal_plus_BG_centers.txt"), usecols=range(1, 3))
    rlTVB = np.loadtxt(os.path.join(DATA_PATH, "aal_plus_BG_centers.txt"), dtype="str", usecols=(0,))
    tlTVB = np.loadtxt(os.path.join(DATA_PATH, "BGplusAAL_tract_lengths.txt"))

    # # ????Remove the second Thalamus????:
    inds_Th = (rlTVB.tolist().index("Thalamus_L"), rlTVB.tolist().index("Thalamus_R"))
    print("Connections between Thalami removed!:\n", wTVB[[8, 9], :][:, inds_Th] / wTVB.max())
    wTVB = np.delete(wTVB, inds_Th, axis=0)
    wTVB = np.delete(wTVB, inds_Th, axis=1)
    tlTVB = np.delete(tlTVB, inds_Th, axis=0)
    tlTVB = np.delete(tlTVB, inds_Th, axis=1)
    rlTVB = np.delete(rlTVB, inds_Th, axis=0)
    cTVB = np.delete(cTVB, inds_Th, axis=0)

    number_of_regions = len(rlTVB)
    speed = 4.0
    min_tt = speed * 0.1
    sliceBG = [0, 1, 2, 3, 6, 7]
    sliceCortex = slice(10, number_of_regions)

    # Remove BG -> Cortex connections
    print("Removing BG -> Cortex connections with max:")
    print(wTVB[sliceBG, :][:, sliceCortex].max())
    wTVB[sliceBG, sliceCortex] = 0.0
    tlTVB[sliceBG, sliceCortex] = min_tt

    # Remove GPe/i <- Cortex connections
    sliceBG = [0, 1, 2, 3]
    print("Removing BG <- Cortex connections with max:")
    print(wTVB[sliceCortex, :][:, sliceBG].max())
    wTVB[sliceCortex, sliceBG] = 0.0
    tlTVB[sliceCortex, sliceBG] = min_tt

    connectivity = Connectivity(region_labels=rlTVB, weights=wTVB, centres=cTVB, tract_lengths=tlTVB)

    tvb_model = ReducedWongWangExcIO  # ReducedWongWangExcIOInhI

    model_params = {}

    main_example(tvb_model, BasalGangliaIzhikevichBuilder, BasalGangliaRedWWexcIOBuilder,
                 annarchy_nodes_ids,  annarchy_populations_order=200,
                 tvb_to_annarchy_mode="rate", annarchy_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=0.0,
                 variables_of_interest=None,
                 config=None, plot_write=True, **model_params)
