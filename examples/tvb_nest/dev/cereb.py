# -*- coding: utf-8 -*-

import time
import os
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import CONFIGURED, Config
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel
from tvb_multiscale.tvb_nest.nest_models.models.cereb import CerebBuilder

from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorBuilder
from tvb_multiscale.core.plot.plotter import Plotter

from tvb.datatypes.connectivity import Connectivity

from examples.example import results_path_fun
from examples.plot_write_results import plot_write_results


NEST_MODEL_BUILDERS = True


def main_example(tvb_sim_model, nest_model_builder,
                 tvb_nest_builder=None, tvb_to_nest_mode=None, nest_to_tvb=False, exclusive_nodes=True,
                 connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, delays_flag=True,
                 transient=0.0, config=None, plot_write=True, **model_params):

    if config is None:
        config = Config(output_base= results_path_fun(nest_model_builder, tvb_to_nest_mode, nest_to_tvb, config)
                                     + ("noNestBuilders" if NEST_MODEL_BUILDERS is False else ""))

    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = CoSimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.connectivity = connectivity
    simulator_builder.delays_flag = delays_flag
    simulator = simulator_builder.build(**model_params)

    # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    # Select the regions for the fine scale modeling with NEST spiking networks
    ordered_populations_labels = ['mossy_fibers', 'glomerulus', "granule_cell", "golgi_cell",
                                  'io_cell', "basket_cell", "stellate_cell", "purkinje_cell",
                                  'dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']

    region_labels = simulator.connectivity.region_labels.tolist()

    cereberal_cortex = "Ansiform lobule"
    inferior_olive = "Inferior olivary complex"
    cereberal_nucleus = "Interposed nucleus"

    hemisphere = "Right"
    cereberal_cortex = "%s %s" % (hemisphere, cereberal_cortex)
    inferior_olive = "%s %s" % (hemisphere, inferior_olive)
    cereberal_nucleus = "%s %s" % (hemisphere, cereberal_nucleus)
    pops_to_nodes_inds = dict(zip(ordered_populations_labels,
                                  [region_labels.index(cereberal_cortex)] * len(ordered_populations_labels)))
    pops_to_nodes_inds['io_cell'] = region_labels.index(inferior_olive)
    dcn_cells = ['dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']
    cereberal_nucleus_id = region_labels.index(cereberal_nucleus)
    for dcn in dcn_cells:
        pops_to_nodes_inds[dcn] = cereberal_nucleus_id
    nest_nodes_ids = np.unique(list(pops_to_nodes_inds.values())).astype("i")

    nodes_inds_to_nodes_labels = dict(zip(nest_nodes_ids, connectivity.region_labels[nest_nodes_ids]))

    print(nodes_inds_to_nodes_labels)
    print(pops_to_nodes_inds)

    connections_to_cereb = simulator.connectivity.weights[:, nest_nodes_ids[0]]
    sorted_connections_to_cereb = np.argsort(connections_to_cereb)[::-1]
    print("sorted_connections_to_cereb =\n")
    for conn_id in sorted_connections_to_cereb:
        print("\n%d. %s, w = %g" %
              (conn_id, simulator.connectivity.region_labels[conn_id], connections_to_cereb[conn_id]))
    connections_from_cereb = simulator.connectivity.weights[nest_nodes_ids[0], :]
    sorted_connections_from_cereb = np.argsort(connections_from_cereb)[::-1]
    print("sorted_connections_from_cereb =\n")
    for conn_id in sorted_connections_from_cereb:
        print("\n%d. %s, w = %g" %
              (conn_id, simulator.connectivity.region_labels[conn_id], connections_from_cereb[conn_id]))

    print("Building NEST network...")
    tic = time.time()
    nest_instance = load_nest(config=config)
    nest_instance = configure_nest_kernel(nest_instance, config)
    nest_model_builder.tvb_serial_sim = simulator
    nest_model_builder.nest_instance = nest_instance
    nest_model_builder.config = config
    nest_model_builder.pops_to_nodes_inds = pops_to_nodes_inds
    nest_model_builder.path_to_network_source_file = os.path.join(data_path, "300x_200z_DCN_IO.hdf5")
    if not NEST_MODEL_BUILDERS:
        nest_model_builder.nodes_inds_to_nodes_labels = nodes_inds_to_nodes_labels
    nest_model_builder.configure()
    nest_network = nest_model_builder.build()
    print("Done! in %f min" % ((time.time() - tic) / 60))

    # -----------------------------------3. Build the TVB-NEST interface model ---------------------------------------
    if tvb_nest_builder is not None:
        print("Building TVB-NEST interface...")
        raise NotImplementedError
        print("Done! in %f min" % ((time.time() - tic)/60))

    # -----------------------------------4. Simulate and gather results-------------------------------------------------
    tvb_results = []
    t_start = time.time()
    if tvb_nest_builder is not None:
        print("Simulating TVB-NEST!...")
        raise NotImplementedError
    else:
        print("Simulating only NEST!...")
        nest_network.nest_instance.Prepare()
        nest_network.nest_instance.Run(nest_model_builder.TOT_DURATION
                                       + nest_network.nest_instance.GetKernelStatus("resolution"))
        nest_network.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------5. Plot results--------------------------------------------------------
    if plot_write:
        try:
            plot_write_results(tvb_results, simulator,
                               spiking_network=nest_network, spiking_nodes_ids=nest_nodes_ids,
                               transient=transient, tvb_state_variable_type_label="State Variables",
                               tvb_state_variables_labels=simulator.model.variables_of_interest,
                               plot_per_neuron=False, plotter=plotter, config=config)
        except Exception as e:
            print("Error in plotting or writing to files!:\n%s" % str(e))

    return tvb_results, simulator


if __name__ == "__main__":

    import h5py
    work_path = os.getcwd()
    data_path = os.path.join(work_path.split("examples")[0], "rising_net", "data")
    tvb_conn_filepath = os.path.join(data_path, "Connectivity_res100_summ49regions_IOsplit.h5")
    f = h5py.File(tvb_conn_filepath)
    connectivity = Connectivity(weights=np.array(f["weights"][()]), tract_lengths=np.array(f["tract_lengths"][()]),
                                centres=np.array(f["centres"][()]),  # hemispheres=np.array(f["hemispheres"][()]),
                                region_labels=np.array(f["region_labels"][()]).astype("<U128"))
    f.close()

    tvb_model = Linear()
    model_params = {"I_o": np.array([0.3]), "G": 0.5 * np.array([256.0]), "gamma": np.array([-0.1])}

    main_example(tvb_model, CerebBuilder(), None,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                 connectivity=connectivity, delays_flag=True, transient=0.0, config=None, **model_params)
