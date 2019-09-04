# -*- coding: utf-8 -*-
import time
from collections import OrderedDict
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.interfaces.builders.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_nest.simulator_nest.models_builders.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.plot.plotter import Plotter
from tvb_scripts.timeseries.model import Timeseries
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw  # , Bold  # , EEG

if __name__ == "__main__":

    config = CONFIGURED
    plotter = Plotter(config)

    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)

    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)-----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.model = ReducedWongWangExcIOInhI()


    def boundary_fun(state):
        state[state < 0] = 0.0
        state[state > 1] = 1.0
        return state


    # Synaptic gating state variables S_e, S_i need to be in the interval [0, 1]
    simulator.boundary_fun = boundary_fun
    simulator.connectivity = connectivity
    # TODO: Try to make this part of the __init__ of the Simulator!
    simulator.integrator.dt = \
        float(int(np.round(simulator.integrator.dt / config.nest.NEST_MIN_DT))) * config.nest.NEST_MIN_DT
    # Some extra monitors for neuroimaging measures:
    mon_raw = Raw(period=simulator.integrator.dt)
    # mon_bold = Bold(period=2000.)
    # mon_eeg = EEG(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)  # mon_bold, mon_eeg

    # ------3. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)--------

    # Select the regions for the fine scale modeling with NEST spiking networks
    number_of_regions = simulator.connectivity.region_labels.shape[0]
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    for id in range(number_of_regions):
        if simulator.connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = \
        RedWWExcIOInhIBuilder(simulator, nest_nodes_ids, config=config)
    nest_model_builder.populations_order = 100

    # # ----Uncomment below to modify the builder by changing the default options:---
    #
    # # Common order of neurons' number per population:
    # nest_model_builder.populations_order = 100
    # # Spiking populations labels:
    # nest_model_builder.populations_names = ["E", "I"]
    # # Spiking populations scalings for the number of neurons
    # nest_model_builder.populations_scales = [1.0, 0.7]
    # # Some properties for the default synapse to be used:
    # nest_model_builder.default_synapse["params"]["rule"] = "fixed_indegree"
    #
    # # Connection weights between the distinct populations:
    # # Choosing the values resulting from J_N = 150 pA and J_i = 1000 pA [1]
    # w_ee = 150.0
    # w_ei = -1000.0
    # w_ie = 150.0
    # w_ii = -1000.0
    #
    # # Within region-node connections' weights
    # nest_model_builder.population_connectivity_synapses_weights = \
    #     np.array([[w_ee, w_ei],  # exc_i -> exc_i, inh_i -> exc_i
    #               [w_ie, w_ii]])  # exc_i -> inh_i, inh_i -> inh_i
    # nest_model_builder.population_connectivity_synapses_delays = \
    #     np.array(nest_model_builder.tvb_dt / 4)
    #
    # # Among/Between region-node connections
    # # Given that w_ee == w_ie = J_N,
    # # and that only the excitatory population of one region-node couples to
    # # both excitatory and inhibitory populations of another region-node,
    # # we need only one connection type
    # nest_model_builder.node_connections = \
    #     [{"src_population": "E", "trg_population": ["E", "I"],
    #       "model": nest_model_builder.default_synapse["model"],
    #       "params": nest_model_builder.default_synapse["params"],
    #       "weight": w_ee,  # weight scaling the TVB connectivity weight
    #       "delay": 0.0}]  # additional delay to the one of TVB connectivity
    #
    # # Creating spike_detector devices to be able to observe NEST spiking activity:
    # connections = OrderedDict({})
    # #          label <- target population
    # connections["E"] = "E"
    # connections["I"] = "I"
    # nest_model_builder.output_devices = \
    #     [{"model": "spike_detector",
    #       "props": config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
    #       "nodes": None, "connections": connections}]
    #
    # # -----------------------------------------------------------------------------

    nest_network = nest_model_builder.build_nest_network()

    # -----------------------------------4. Build the TVB-NEST interface model ------------------------------------------

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = \
        RedWWexcIOinhIBuilder(simulator, nest_network, nest_nodes_ids, config=config)

    # ------------Modifications to the default options of the builder---------------

    # NEST -> TVB:
    #
    # 1.1. For current transmission from TVB to NEST,
    # either choose a NEST dc_generator device:
    # tvb_nest_builder.tvb_to_nest_interfaces = \
    #    [{"model": "dc_generator", "sign": 1,
    # #                      TVB  <-  NEST
    #      "connections": {"S_e": ["E", "I"]}}]

    # 1.2. or modify directly the external current stimulus parameter:
    tvb_nest_builder.tvb_to_nest_interfaces = \
        [{"model": "current", "parameter": "I_e", "sign": 1,
          #                TVB  <-  NEST
          "connections": {"S_e": ["E", "I"]}}]

    # 2.1. For spike transmission from TVB to NEST:
    # tvb_nest_builder.tvb_to_nest_interfaces = \
    #    [{"model": "poisson_generator", "sign": 1,
    # #                      TVB  <-  NEST
    #      "connections": {"S_e": ["E", "I"]}}]

    # NEST -> TVB:
    # Use S_e and S_i instead of r_e and r_i
    # for transmitting to the TVB state variables directly
    connections = OrderedDict()
    #            TVB <- NEST
    connections["r_e"] = "E"
    connections["r_i"] = "I"
    tvb_nest_builder.nest_to_tvb_interfaces = \
        [{"model": "spike_detector", "params": {}, "connections": connections}]

    # -----------------------------------------------------------------------------

    tvb_nest_model = tvb_nest_builder.build_interface()

    # -----------------------------------5. Simulate and gather results------- ------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t = time.time()
    results = simulator.run(simulation_length=100.0)
    print("\nSimulated in %f secs!" % (time.time() - t))
    time = results[0][0]
    source = results[0][1]

    # -------------------------------------------6. Plot results---------------------------------------------------------

    # Plot spikes and mean field spike rates
    rates, max_rate, spike_detectors, time = \
        nest_network.compute_mean_spike_rates(spike_counts_kernel_width=simulator.integrator.dt,  # ms
                                              spike_counts_kernel_overlap=0.0, time=time)
    plotter.plot_spikes(spike_detectors, time, rates=rates, max_rate=max_rate,
                        title='Population spikes and mean spike rate')

    #   Remove ts_type="Region" this argument too for TVB TimeSeriesRegion
    source_ts = Timeseries(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=source, time=time,
        connectivity=simulator.connectivity,
        # region_mapping=head.cortical_region_mapping,
        # region_mapping_volume=head.region_volume_mapping,
        labels_ordering=["Time", "Synaptic Gating Variable", "Region", "Neurons"],
        labels_dimensions={"Synaptic Gating Variable": ["S_e", "S_i"],
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt, ts_type="Region")

    # Use this to skip completely the tvb-scripts plotter
    # tvb_plotter = TimeSeriesInteractive(time_series=source_ts._tvb)
    # tvb_plotter.configure()
    # tvb_plotter.show()

    # Plot timeseries
    # Add _tvb from the function names if source_ts is a TVB TimeSeriesRegion object
    plotter.plot_timeseries(source_ts)
    plotter.plot_raster(source_ts, title="Region Time Series Raster")
    # ...interactively as well
    plotter.plot_timeseries_interactive(source_ts)
