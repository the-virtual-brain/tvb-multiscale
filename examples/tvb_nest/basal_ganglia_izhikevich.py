# -*- coding: utf-8 -*-

import os
import importlib

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.nest_models.builders.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.interfaces.models \
    import RedWWexcIOBuilder as BasalGangliaRedWWexcIOBuilder

example_core_path = os.path.dirname(__file__)
example_module = \
    importlib.util.spec_from_file_location(".example",
                                           os.path.join(example_core_path, "example.py"))
example = importlib.util.module_from_spec(example_module)
example_module.loader.exec_module(example)
main_example = example.main_example

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


if __name__ == "__main__":

    nest_nodes_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
                 nest_nodes_ids,  nest_populations_order=200,
                 tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,  # "rate"
                 connectivity=connectivity, delays_flag=True,
                 simulation_length=110.0, transient=10.0,
                 variables_of_interest=None,
                 config=None, **model_params)
