# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb.datatypes.connectivity import Connectivity

from tvb_nest.config import Config
from tvb_nest.examples.example import main_example
from tvb_nest.nest_models.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i \
    import RedWWexcIOinhIBuilder as InterfaceRedWWexcIOinhIBuilder
from tvb_scripts.time_series.model import TimeSeriesRegion
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


def create_time_series_region_object():

    config = Config(output_base="outputs/")
    config.figures.SAVE_FLAG = False
    config.figures.SHOW_FLAG = False
    config.figures.MATPLOTLIB_BACKEND = "Agg"

    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    for id in range(connectivity.region_labels.shape[0]):
        if connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)

    results, simulator = \
        main_example(ReducedWongWangExcIOInhI(), RedWWExcIOInhIBuilder, InterfaceRedWWexcIOinhIBuilder,
                     nest_nodes_ids, nest_populations_order=100, connectivity=connectivity,
                     simulation_length=100.0, exclusive_nodes=True, config=config)
    time = results[0][0]
    source = results[0][1]

    source_ts = TimeSeriesRegion(
            data=source, time=time,
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "Synaptic Gating Variable", "Region", "Neurons"],
            labels_dimensions={"Synaptic Gating Variable": ["S_e", "S_i"],
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.integrator.dt)

    return source_ts


def test_time_series_region_object():
    tsr = create_time_series_region_object()

    # Check the correctness of time_series_region object
    assert tsr.shape == (1000, 4, 68, 1)
