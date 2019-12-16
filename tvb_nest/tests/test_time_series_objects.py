# -*- coding: utf-8 -*-

import numpy
from collections import OrderedDict
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_nest.simulator_nest.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_scripts.time_series.model import TimeSeriesRegion
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.config import CONFIGURED
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw


def create_time_series_region_object():
    config = CONFIGURED

    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()

    simulator = Simulator()
    simulator.model = ReducedWongWangExcIOInhI()

    def boundary_fun(state):
        state[state < 0] = 0.0
        state[state > 1] = 1.0
        return state

    simulator.boundary_fun = boundary_fun
    simulator.connectivity = connectivity
    simulator.integrator.dt = \
        float(int(numpy.round(simulator.integrator.dt /
                              config.NEST_MIN_DT))) * config.NEST_MIN_DT

    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    number_of_regions = simulator.connectivity.region_labels.shape[0]
    nest_nodes_ids = []

    for id in range(number_of_regions):
        if simulator.connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)

    nest_model_builder = \
        RedWWExcIOInhIBuilder(simulator, nest_nodes_ids, config=config)

    nest_network = nest_model_builder.build_nest_network()

    tvb_nest_builder = RedWWexcIOinhIBuilder(simulator, nest_network, nest_nodes_ids)

    tvb_nest_builder.tvb_to_nest_interfaces = \
        [{"model": "current", "parameter": "I_e", "sign": 1,
          "connections": {"S_e": ["E", "I"]}}]

    connections = OrderedDict()
    connections["R_e"] = "E"
    connections["R_i"] = "I"
    tvb_nest_builder.nest_to_tvb_interfaces = \
        [{"model": "spike_detector", "params": {}, "connections": connections}]

    tvb_nest_model = tvb_nest_builder.build_interface()

    simulator.configure(tvb_nest_interface=tvb_nest_model)
    results = simulator.run(simulation_length=100.0)
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

    # Check for existence of S_e attribute
    assert hasattr(tsr, 'S_e') is True

    # Check for shape after slice
    assert tsr.S_e.shape == (1000, 1, 68, 1)
