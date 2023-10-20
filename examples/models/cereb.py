# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear

from examples.example import main_example

from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def cereb_example(spikeNet_model_builder, tvb_spikeNet_model_builder, orchestrator_app, **kwargs):

    import os
    import h5py
    work_path = os.getcwd()
    data_path = os.path.join(work_path.split("examples")[0], "rising_net", "data")
    WHOLE_BRAIN = False
    if WHOLE_BRAIN:
        BRAIN_CONN_FILE = "Connectivity_res100_596_regions.h5"
    else:
        BRAIN_CONN_FILE = "Connectivity_res100_summ49regions_IOsplit.h5"
    tvb_conn_filepath = os.path.join(data_path, BRAIN_CONN_FILE)
    f = h5py.File(tvb_conn_filepath)
    connectivity = Connectivity(weights=np.array(f["weights"][()]), tract_lengths=np.array(f["tract_lengths"][()]),
                                centres=np.array(f["centres"][()]),  # hemispheres=np.array(f["hemispheres"][()]),
                                region_labels=np.array(f["region_labels"][()]).astype("<U128"))
    f.close()

    # Select the regions for the fine scale modeling with NEST spiking networks
    ordered_populations_labels = ['mossy_fibers', 'glomerulus', "granule_cell", "golgi_cell",
                                  'io_cell', "basket_cell", "stellate_cell", "purkinje_cell",
                                  'dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']

    region_labels = connectivity.region_labels.tolist()

    cereberal_cortex = "Ansiform lobule"
    inferior_olive = "Inferior olivary complex"
    cereberal_nucleus = "Interposed nucleus"

    hemisphere = "Right"
    cereberal_cortex = "%s %s" % (hemisphere, cereberal_cortex)
    inferior_olive = "%s %s" % (hemisphere, inferior_olive)
    cereberal_nucleus = "%s %s" % (hemisphere, cereberal_nucleus)
    cc_id = region_labels.index(cereberal_cortex)
    pops_to_nodes_inds = dict(zip(ordered_populations_labels,
                                  [cc_id] * len(ordered_populations_labels)))
    io_id = region_labels.index(inferior_olive)
    pops_to_nodes_inds['io_cell'] = io_id
    dcn_cells = ['dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']
    cereberal_nucleus_id = region_labels.index(cereberal_nucleus)
    for dcn in dcn_cells:
        pops_to_nodes_inds[dcn] = cereberal_nucleus_id
    spiking_proxy_inds = np.unique(list(pops_to_nodes_inds.values())).astype("i")

    regions_inds_to_regions_labels = dict(zip(spiking_proxy_inds, connectivity.region_labels[spiking_proxy_inds]))

    print(regions_inds_to_regions_labels)
    print(pops_to_nodes_inds)

    connections_to_cereb = connectivity.weights[:, spiking_proxy_inds[0]]
    sorted_connections_to_cereb = np.argsort(connections_to_cereb)[::-1]
    print("sorted_connections_to_cereb =\n")
    for conn_id in sorted_connections_to_cereb:
        print("\n%d. %s, w = %g" %
              (conn_id, connectivity.region_labels[conn_id], connections_to_cereb[conn_id]))
    connections_from_cereb = connectivity.weights[spiking_proxy_inds[0], :]
    sorted_connections_from_cereb = np.argsort(connections_from_cereb)[::-1]
    print("sorted_connections_from_cereb =\n")
    for conn_id in sorted_connections_from_cereb:
        print("\n%d. %s, w = %g" %
              (conn_id, connectivity.region_labels[conn_id], connections_from_cereb[conn_id]))

    if WHOLE_BRAIN:
        G = 0.5
    else:
        G = 4.0
    model_params = {"I_o": np.array([1.0]), "G": np.array([G]), "tau": np.array([10.0]), "gamma": np.array([-1.0])}

    model_params.update(kwargs.pop("model_params", {}))

    initial_conditions = kwargs.pop("initial_conditions", np.array([0.0]))

    spikeNet_model_builder.population_order = kwargs.pop("population_order", 1)
    spikeNet_model_builder.spiking_nodes_inds = spiking_proxy_inds
    spikeNet_model_builder.pops_to_nodes_inds = pops_to_nodes_inds
    spikeNet_model_builder.regions_inds_to_regions_labels = regions_inds_to_regions_labels
    spikeNet_model_builder.BACKGROUND = True
    spikeNet_model_builder.STIMULUS = False

    model = kwargs.pop("model", "RATE").upper()
    tvb_spikeNet_model_builder.model = model
    tvb_spikeNet_model_builder.input_flag = kwargs.pop("input_flag", True)
    tvb_spikeNet_model_builder.output_flag = kwargs.pop("output_flag", True)
    tvb_spikeNet_model_builder.default_coupling_mode = "TVB"
    tvb_spikeNet_model_builder.N_mf *= spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.N_grc *= spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.N_io *= spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.N_dcgl *= spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.CC_proxy_inds = np.array(ensure_list(cc_id))
    tvb_spikeNet_model_builder.CN_proxy_inds = np.array(ensure_list(cereberal_nucleus_id))
    tvb_spikeNet_model_builder.IO_proxy_inds = np.array(ensure_list(io_id))
    tvb_to_spikeNet_interfaces = []
    spikeNet_to_tvb_interfaces = []
    tvb_spikeNet_model_builder.output_flag = False


    # An example of a configuration:
    # G = model_params.get("G", np.array([2.0]))[0].item()
    # coupling_a = model_params.pop("coupling_a", np.array([1.0 / 256]))[0].item()
    # global_coupling_scaling = G * coupling_a
    # tvb_to_spikeNet_transformer = kwargs.pop("tvb_to_spikeNet_transformer",
    #                                          kwargs.pop("tvb_to_spikeNet_transformer_model", None))
    # tvb_spikeNet_transformer_params = {"scale_factor": population_order*np.array([1.0])}
    # tvb_spikeNet_transformer_params.update(kwargs.pop("tvb_spikeNet_transformer_params", {}))
    #
    # tvb_to_spikeNet_proxy = kwargs.pop("tvb_to_spikeNet_proxy", kwargs.pop("tvb_to_spikeNet_proxy_model", None))
    # tvb_spikeNet_proxy_params = {"number_of_neurons": 1,
    #                              "weights": lambda source_node, target_node, tvb_weights:
    #                                 scale_tvb_weight(source_node, target_node, tvb_weights, global_coupling_scaling),
    #                              "receptor_type": lambda source_node, target_node:
    #                                 receptor_by_source_region(source_node, target_node, start=1)}
    # tvb_spikeNet_proxy_params.update(kwargs.pop("tvb_spikeNet_proxy_params", {}))
    #
    # tvb_to_spikeNet_interfaces = []
    # for ii, (trg_pop, nodes, _pop) in \
    #         enumerate(zip(["E",                                 ["IdSN", "IiSN"]],
    #                       [tvb_nest_model_builder.E_proxy_inds, tvb_nest_model_builder.Striatum_proxy_inds],
    #                       ["E",                                 "ISN"])):
    #     tvb_to_spikeNet_interfaces.append({"model": "RATE", "voi": "R", "populations": trg_pop,
    #                                        "transformer_params": tvb_spikeNet_transformer_params,
    #                                        "proxy_params": tvb_spikeNet_proxy_params,
    #                                        "spiking_proxy_inds": np.array(nodes)})
    #     if tvb_to_spikeNet_transformer:
    #         tvb_to_spikeNet_interfaces[ii]["transformer_model"] = tvb_to_spikeNet_transformer
    #     tvb_to_spikeNet_interfaces[ii]["transformer_params"].update(
    #         kwargs.pop("tvb_to_spikeNet_transformer_params_%s" % _pop, {}))
    #     if tvb_to_spikeNet_proxy:
    #         tvb_to_spikeNet_interfaces[ii]["proxy_model"] = tvb_to_spikeNet_proxy
    #     tvb_to_spikeNet_interfaces[ii]["proxy_params"].update(
    #         kwargs.pop("tvb_to_spikeNet_proxy_params_%s" % _pop, {}))
    #
    # spikeNet_to_tvb_transformer = kwargs.pop("spikeNet_to_tvb_transformer",
    #                                          kwargs.pop("spikeNet_to_tvb_transformer_model",
    #                                                     ElephantSpikesRateRedWongWangInh))
    # spikeNet_to_tvb_interfaces = []
    # for ii, (src_pop, nodes, _pop) in \
    #     enumerate(zip(["E", "I", ["IdSN", "IiSN"]],
    #                   [tvb_nest_model_builder.E_proxy_inds,
    #                    tvb_nest_model_builder.I_proxy_inds,
    #                    tvb_nest_model_builder.Striatum_proxy_inds],
    #                    ["E", "I", "ISN"])):
    #     spikeNet_to_tvb_interfaces.append(
    #         {"voi": ("S", "R"), "populations": src_pop,
    #          "transformer": spikeNet_to_tvb_transformer,
    #          "transformer_params": {"scale_factor": np.array([1.0]) / tvb_nest_model_builder.N_E,
    #                                 "integrator":
    #                                     HeunStochastic(dt=0.1,
    #                                                    noise=Additive(
    #                                                        nsig=np.array([[1e-3], [0.0]]))),
    #                                 "state": np.zeros((2, len(nodes))),
    #                                 "tau_s": model_params.get("tau_s",
    #                                                           np.array([100.0, ])),
    #                                 "tau_r": np.array([10.0, ]),
    #                                 "gamma": model_params.get("gamma",
    #                                                           np.array([0.641 / 1000, ]))},
    #          "proxy_inds": np.array(nodes)
    #          })
    #     spikeNet_to_tvb_interfaces[ii]["transformer_params"].update(
    #         kwargs.pop("spikeNet_to_tvb_transformer_params_%s" % _pop, {}))

    return main_example(orchestrator_app,
                        Linear(), model_params,
                        spikeNet_model_builder, spiking_proxy_inds,
                        tvb_spikeNet_model_builder, tvb_to_spikeNet_interfaces, spikeNet_to_tvb_interfaces,
                        connectivity=connectivity, initial_conditions=initial_conditions, **kwargs)
