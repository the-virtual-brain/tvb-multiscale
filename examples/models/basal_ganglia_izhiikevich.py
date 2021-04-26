# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.models.linear_reduced_wong_wang_exc_io import LinearReducedWongWangExcIO

from examples.example import main_example

from tvb.datatypes.connectivity import Connectivity


def basal_ganglia_izhikevich_example(spikeNet_model_builder, tvb_spikeNet_model_builder, **kwargs):

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

    spiking_proxy_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    populations_order = kwargs.pop("populations_order", 200)

    model_params = kwargs.pop("model_params", {})

    return main_example(LinearReducedWongWangExcIO, model_params,
                        spikeNet_model_builder, spiking_proxy_inds, populations_order,
                        tvb_spikeNet_model_builder,
                        connectivity=connectivity, **kwargs)
