# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

from tvb.datatypes.connectivity import Connectivity

from tvb_nest import config as config_m
config_m.CONFIGURED = config_m.Config(output_base="outputs/")
config = config_m.CONFIGURED
config.figures.SAVE_FLAG = False
config.figures.SHOW_FLAG = False
config.figures.MATPLOTLIB_BACKEND = "Agg"

from tvb_nest.examples.example import main_example
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.simulator_nest.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i \
    import RedWWexcIOinhIBuilder as InterfaceRedWWexcIOinhIBuilder


# Select the regions for the fine scale modeling with NEST spiking networks
nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
# In this example, we model parahippocampal cortices (left and right) with NEST
connectivity = Connectivity.from_file(config_m.CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
for id in range(connectivity.region_labels.shape[0]):
    if connectivity.region_labels[id].find("hippo") > 0:
        nest_nodes_ids.append(id)

main_example(ReducedWongWangExcIOInhI(), RedWWExcIOInhIBuilder, InterfaceRedWWexcIOinhIBuilder,
             nest_nodes_ids, nest_populations_order=100, connectivity=connectivity, simulation_length=100.0,
             tvb_state_variable_type_label="Synaptic Gating Variable", tvb_state_variables_labels=["S_e", "S_i"],
             exclusive_nodes=True, config=config_m.CONFIGURED)
