# -*- coding: utf-8 -*-
import os
import shutil

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb_multiscale.core.config import Config
from examples.simulate_tvb_only import main_example
from tvb.datatypes.connectivity import Connectivity
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion


def create_time_series_region_object():
    config = Config(output_base="outputs/")
    config.figures.SAVE_FLAG = False
    config.figures.SHOW_FLAG = False
    config.figures.MATPLOTLIB_BACKEND = "Agg"

    # Select the regions for the fine scale modeling with NEST spiking networks
    spiking_proxy_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    for id in range(connectivity.region_labels.shape[0]):
        if connectivity.region_labels[id].find("hippo") > 0:
            spiking_proxy_ids.append(id)

    results, simulator = main_example(WilsonCowan, connectivity=connectivity,
                                      simulation_length=10.0, transient=0.0,
                                      config=config, plot_write=False)
    time = results[0][0]
    source = results[0][1]

    source_ts = TimeSeriesRegion(
        data=source, time=time,
        connectivity=simulator.connectivity,
        labels_ordering=["Time", "Synaptic Gating Variable", "Region", "Modes"],
        labels_dimensions={"Synaptic Gating Variable": ["E", "I"],
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)

    return source_ts, connectivity


def test_time_series_region_object():
    tsr, conn = create_time_series_region_object()

    # Check the correctness of time_series_region object
    assert tsr.shape == (100, 2, conn.number_of_regions, 1)


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    test_time_series_region_object()
