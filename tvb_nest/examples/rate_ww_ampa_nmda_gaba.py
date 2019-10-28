# -*- coding: utf-8 -*-
import time
from collections import OrderedDict
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.interfaces.builders.rate_ww_ampa_nmda_gaba import RateWWAMPANMDAGABABuilder as InterfaceRateWWAMPANMDAGABABuilder
from tvb_nest.simulator_nest.models_builders.rate_ww_ampa_nmda_gaba import RateWWAMPANMDAGABABuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.plot.plotter import Plotter
from tvb_scripts.time_series.model import TimeSeriesRegion
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


if __name__ == "__main__":

    config = CONFIGURED
    plotter = Plotter(config)

    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)

    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.model = ReducedWongWangExcIOInhI()

    simulator.connectivity = connectivity
    # Some extra monitors for neuroimaging measures:
    mon_raw = Raw(period=simulator.integrator.dt)
    # mon_bold = Bold(period=2000.)
    # mon_eeg = EEG(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)  # mon_bold, mon_eeg

    # ------3. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

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
        RateWWAMPANMDAGABABuilder(simulator, nest_nodes_ids, config=config, J_i=1.0)
    # Common order of neurons' number per population:
    nest_model_builder.populations_order = 100
    # # Connection weights between the distinct populations:
    # # Inhibition to excitation feedback inhibition
    # # that could result from Feedback Inhibition Control
    # # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
    nest_model_builder.J_i = 1.0

    # # ----Uncomment below to modify the builder by changing the default options:---
    #
    # nest_model_builder.populations_names = ["AMPA", "NMDA", "GABA"]
    # nest_model_builder.populations_models = ["tvb_rate_ampa_gaba_wongwang",
    #                            "tvb_rate_nmda_wongwang",
    #                            "tvb_rate_ampa_gaba_wongwang"]
    # for model in ["tvb_rate_ampa_gaba_wongwang",
    #               "tvb_rate_nmda_wongwang"]:
    #     nest_models = nest_model_builder.nest_instance.Models()
    #     if model not in nest_models:
    #         # If the model is not install into NEST already
    #         try:
    #             # Try to install it...
    #             nest_model_builder.nest_instance.Install("tvb_rate_wongwangmodule")
    #         except:
    #             # ...unless we need to first compile it:
    #             compile_modules("tvb_rate_wongwang", config=nest_model_builder.config)
    #             # and now install it...
    #             nest_model_builder.nest_instance.Install("tvb_rate_wongwangmodule")
    #
    # nest_model_builder.populations_params = [{}, {},  # AMPA and NMDA get the default parameters
    #                            {"tau_syn": 10.0}]  # decay synaptic time for GABA has to change
    #
    # # Spiking populations scalings for the number of neurons:
    # rcptr_ampa_gaba = nest_model_builder.nest_instance.GetDefaults('tvb_rate_ampa_gaba_wongwang')['receptor_types']
    # rcptr_nmda = nest_model_builder.nest_instance.GetDefaults('tvb_rate_ampa_gaba_wongwang')['receptor_types']
    # nest_model_builder.populations_scales = [1.0, 1.0, 0.7]
    # # Some properties for the default synapse to be used:
    # nest_model_builder.default_connection["model"] = "rate_connection"
    # nest_model_builder.default_connection["params"]["rule"] = "fixed_indegree"
    #
    # # Within region-node connections' weights
    # w_ee = 1.4
    # w_ei = -nest_model_builder.J_i
    # w_ie = 1.0
    # w_ii = -1.0
    # nest_model_builder.population_connectivity_synapses_weights = \
    #     np.array([[w_ee, w_ee, w_ei],  # AMPA->AMPA, NMDA->AMPA, GABA->AMPA
    #               [w_ee, w_ee, w_ei],  # AMPA->NMDA, NMDA->NMDA, GABA->NMDA
    #               [w_ie, w_ie, w_ii]]) # AMPA->GABA, NMDA->GABA, GABA->GABA
    # nest_model_builder.population_connectivity_synapses_delays = \
    #     np.array(nest_model_builder.tvb_dt / 4)
    # nest_model_builder.population_connectivity_synapses_receptor_types = \
    #     np.array([[rcptr_ampa_gaba["AMPA_REC"], rcptr_ampa_gaba["NMDA"], rcptr_ampa_gaba["GABA"]],
    #               [rcptr_nmda["AMPA_REC"], rcptr_nmda["NMDA"], rcptr_nmda["GABA"]],
    #               [rcptr_ampa_gaba["AMPA_REC"], rcptr_ampa_gaba["NMDA"], rcptr_ampa_gaba["GABA"]]])
    #
    # # Among/Between region-node connections
    # # Given that only the AMPA population of one region-node couples to
    # # all populations of another region-node,
    # # we need only one connection type
    # nest_model_builder.node_connections = \
    #     [{"src_population": "AMPA", "trg_population": ["AMPA", "GABA"],
    #       "model": "rate_connection",
    #       "params": nest_model_builder.default_connection["params"],
    #       "weight": 1.0,  # weight scaling the TVB connectivity weight
    #       "delay": 0.0,  # additional delay to the one of TVB connectivity
    #       "receptor_type": rcptr_ampa_gaba["AMPA_EXT"]},
    #      {"src_population": "AMPA", "trg_population": ["NMDA"],
    #       "model": "rate_connection",
    #       "params": nest_model_builder.default_connection["params"],
    #       "weight": 1.0,  # weight scaling the TVB connectivity weight
    #       "delay": 0.0,  # additional delay to the one of TVB connectivity
    #       "receptor_type": rcptr_nmda["AMPA_EXT"]}
    #      ]
    #
    # Creating  devices to be able to observe NEST activity:
    # output_devices = []
    # connections = OrderedDict({})
    # #          label <- target population
    # connections["AMPA"] = "AMPA"
    # connections["NMDA"] = "NMDA"
    # connections["GABA"] = "GABA"
    # params = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]
    # params['record_from'] = ["V_m", "S",
    #                          "s_AMPA_ext", "s_AMPA_rec", "s_NMDA", "s_GABA",
    #                          "I_AMPA_ext", "I_AMPA_rec", "I_NMDA", "I_GABA", "I_leak"]
    # output_devices.append({"model": "multimeter", "params": params,
    #                             "nodes": None, "connections": connections}),
    # connections = OrderedDict({})
    # connections["AMPA spikes"] = "AMPA"
    # connections["NMDA spikes"] = "NMDA"
    # connections["GABA spikes"] = "GABA"
    # params = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_multimeter"]
    # output_devices.append({"model": "spike_multimeter", "params": params,
    #                        "nodes": None, "connections": connections})
    #
    # # -----------------------------------------------------------------------------

    nest_network = nest_model_builder.build_nest_network()

    # -----------------------------------4. Build the TVB-NEST interface model -----------------------------------------

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = \
        InterfaceRateWWAMPANMDAGABABuilder(simulator, nest_network, nest_nodes_ids, config=config)

    # ------------Modifications to the default options of the builder---------------

    # NEST -> TVB:
    #
    # 1.1. For current transmission from TVB to NEST,
    # either choose a NEST dc_generator device:
    # tvb_nest_builder.tvb_to_nest_interfaces = \
    #    [{"model": "dc_generator", "sign": 1,
    # #                      TVB  ->  NEST
    #      "connections": {"S_e": ["AMPA", "NMDA", "GABA"]}}]

    # 1.2. or modify directly the external current stimulus parameter:
    tvb_nest_builder.tvb_to_nest_interfaces = \
        [{"model": "current", "parameter": "I_e", "sign": 1,
          #                TVB  ->  NEST
          "connections": {"S_e": ["AMPA", "NMDA", "GABA"]}}]

    # NEST -> TVB:
    # Use S_e and S_i instead of r_e and r_i
    # for transmitting to the TVB state variables directly
    connections = OrderedDict()
    #            TVB <- NEST
    connections["r_e"] = ["AMPA", "NMDA"]
    connections["r_i"] = "GABA"
    tvb_nest_builder.nest_to_tvb_interfaces = \
        [{"model": "spike_multimeter", "connections": connections, "params": {}}]

    # -----------------------------------------------------------------------------

    tvb_nest_model = tvb_nest_builder.build_interface()

    # -----------------------------------5. Simulate and gather results------- -----------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=100.0)
    print("\nSimulated in %f secs!" % (time.time() - t_start))
    t = results[0][0]
    source = results[0][1]

    # -------------------------------------------6. Plot results--------------------------------------------------------

    # Plot spikes and mean field spike rates
    rates, max_rate, spike_detectors, time = \
        nest_network.compute_mean_spike_rates(spike_counts_kernel_width=simulator.integrator.dt,  # ms
                                              spike_counts_kernel_overlap=0.0, time=time)
    plotter.plot_spikes(spike_detectors, time, rates=rates, max_rate=max_rate,
                        title='Population spikes and mean spike rate')

    #   Remove ts_type="Region" this argument too for TVB TimeSeriesRegion
    source_ts = TimeSeriesRegion(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=source, time=time,
        connectivity=simulator.connectivity,
        # region_mapping=head.cortical_region_mapping,
        # region_mapping_volume=head.region_volume_mapping,
        labels_ordering=["Time", "Synaptic Gating Variable", "Region", "Neurons"],
        labels_dimensions={"Synaptic Gating Variable": ["S_e", "S_i"],
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)


    # Plot time_series
    plotter.plot_timeseries(source_ts)
    plotter.plot_raster(source_ts, title="Region Time Series Raster")
    # ...interactively as well
    plotter.plot_timeseries_interactive(source_ts)
