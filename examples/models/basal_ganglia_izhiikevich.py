# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO

from examples.example import main_example

from tvb.datatypes.connectivity import Connectivity


def basal_ganglia_izhikevich_example(spikeNet_model_builder, tvb_spikeNet_model_builder, orchestrator_app, **kwargs):

    import os

    home_path = os.path.join(os.getcwd().split("tvb-multiscale")[0], "tvb-multiscale")
    DATA_PATH = os.path.join(home_path, "examples/data/basal_ganglia/conn")
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

    spiking_proxy_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    spikeNet_model_builder.population_order = kwargs.pop("population_order", 200)

    model_params = kwargs.pop("model_params", {})

    model = kwargs.pop("model", "RATE").upper()
    tvb_spikeNet_model_builder.model = model
    tvb_spikeNet_model_builder.input_flag = kwargs.pop("input_flag", True)
    tvb_spikeNet_model_builder.output_flag = kwargs.pop("output_flag", True)
    tvb_spikeNet_model_builder.N_E = spikeNet_model_builder.population_order
    tvb_spikeNet_model_builder.GPe_proxy_inds = np.array([0, 1])
    tvb_spikeNet_model_builder.GPi_proxy_inds = np.array([2, 3])
    tvb_spikeNet_model_builder.STN_proxy_inds = np.array([4, 5])
    tvb_spikeNet_model_builder.Striatum_proxy_inds = np.array([6, 7])
    tvb_spikeNet_model_builder.Thal_proxy_inds = np.array([8, 9])
    tvb_to_spikeNet_interfaces = []
    spikeNet_to_tvb_interfaces = []

    # An example of a configuration:
    # G = model_params.get("G", np.array([2.0]))[0].item()
    # coupling_a = model_params.pop("coupling_a", np.array([1.0 / 256]))[0].item()
    # global_coupling_scaling = G * coupling_a
    # tvb_to_spikeNet_transformer = kwargs.pop("tvb_to_spikeNet_transformer",
    #                                          kwargs.pop("tvb_to_spikeNet_transformer_model", None))
    # tvb_spikeNet_transformer_params = {"scale_factor": tvb_spikeNet_model_builder.N_E*np.array([1.0])}
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
                        LinearReducedWongWangExcIO(), model_params,
                        spikeNet_model_builder, spiking_proxy_inds,
                        tvb_spikeNet_model_builder, tvb_to_spikeNet_interfaces, spikeNet_to_tvb_interfaces,
                        connectivity=connectivity, **kwargs)
