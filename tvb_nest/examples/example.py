# -*- coding: utf-8 -*-
import time
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.simulator_nest.models_builders.red_rate_ww_exc_io_inh_i import RedRateWWExcIOInhIBuilder
from tvb_nest.simulator_nest.models_builders.rate_ww_ampa_nmda_gaba import RateWWAMPANMDAGABABuilder
from tvb_nest.simulator_nest.models_builders.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.simulator_nest.models_builders.rate_ww_ampa_nmda_gaba import RateWWAMPANMDAGABABuilder
from tvb_nest.simulator_nest.models_builders.ww_deco2014 import WWDeco2014Builder
from tvb_nest.interfaces.builders.red_rate_ww_exc_io_inh_i \
    import RedRateWWexcIOinhIBuilder as InterfaceRedRateWWexcIOinhIBuilder
from tvb_nest.interfaces.builders.rate_ww_ampa_nmda_gaba \
    import RateWWAMPANMDAGABABuilder as InterfaceRateWWAMPANMDAGABABuilder
from tvb_nest.interfaces.builders.red_ww_exc_io_inh_i \
    import RedWWexcIOinhI as InterfaceRedWWexcIOinhI
from tvb_nest.interfaces.builders.ww_ampa_nmda_gaba import \
    WWAMPANMDAGABABuilder as InterfaceWWAMPANMDAGABABuilder
from tvb_nest.interfaces.builders.ww_deco2014 import \
    WWDeco2014Builder as InterfaceWWDeco2014Builder
from tvb_nest.plot.plotter import Plotter
from tvb_scripts.time_series.model import TimeSeriesRegion
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def main_example(tvb_sim_model, nest_model_builder, tvb_nest_builder, nest_nodes_ids, nest_populations_order=100,
                 connectivity=None, connectivity_zip=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP, simulation_length=100.0,
                 tvb_state_variable_type_label="Synaptic Gating Variable", tvb_state_variables_labels=["S_e", "S_i"],
                 dt=0.1, config=CONFIGURED):

    plotter = Plotter(config)

    # --------------------------------------1. Load TVB connectivity----------------------------------------------------
    if connectivity is None:
        connectivity = Connectivity.from_file(connectivity_zip)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)


    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.integrator.dt = dt
    simulator.model = tvb_sim_model

    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # ------3. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder =  nest_model_builder(simulator, nest_nodes_ids, config=config)
    # Common order of neurons' number per population:
    nest_model_builder.populations_order = nest_populations_order
    nest_network = nest_model_builder.build_nest_network()

    # -----------------------------------4. Build the TVB-NEST interface model -----------------------------------------

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = tvb_nest_builder(simulator, nest_network, nest_nodes_ids, config=config)
    tvb_nest_model = tvb_nest_builder.build_interface()

    # -----------------------------------5. Simulate and gather results-------------------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!" % (time.time() - t_start))
    t = results[0][0]
    source = results[0][1]

    # -------------------------------------------6. Plot results--------------------------------------------------------

    #   Remove ts_type="Region" this argument too for TVB TimeSeriesRegion
    source_ts = TimeSeriesRegion(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=source, time=t,
        connectivity=simulator.connectivity,
        # region_mapping=head.cortical_region_mapping,
        # region_mapping_volume=head.region_volume_mapping,
        labels_ordering=["Time", tvb_state_variable_type_label, "Region", "Neurons"],
        labels_dimensions={tvb_state_variable_type_label: tvb_state_variables_labels,
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)

    # Plot time_series
    plotter.plot_timeseries(source_ts)
    plotter.plot_raster(source_ts, title="Region Time Series Raster")
    # ...interactively as well
    plotter.plot_timeseries_interactive(source_ts)

    # Plot NEST multimeter variables
    multimeter_mean_data = tvb_nest_model.get_mean_data_from_NEST_multimeter_to_TVBTimeSeries()
    if multimeter_mean_data is not None:
        plotter.plot_multimeter_timeseries(multimeter_mean_data, plot_per_variable=True,
                                           time_series_class=TimeSeriesRegion, time_series_args={},
                                           var_pop_join_str=" - ", default_population_label="population",
                                           title="NEST region time series")
        plotter.plot_multimeter_raster(multimeter_mean_data,
                                       plot_per_variable=True,
                                       time_series_class=TimeSeriesRegion, time_series_args={},
                                       var_pop_join_str=" - ", default_population_label="population",
                                       title="NEST region time series raster")

    # Plot spikes and mean field spike rates
    rates, max_rate, spike_detectors, spike_time = \
        tvb_nest_model.get_mean_spikes_rates_from_NEST_to_TVBTimeSeries(
            spike_counts_kernel_width=simulator.integrator.dt,  # ms
            spike_counts_kernel_overlap=0.0, time=t)
    if spike_detectors is not None:
        plotter.plot_spikes(spike_detectors, spike_time, rates=rates, max_rate=max_rate,
                            title='Population spikes and mean spike rate')


if __name__ == "__main__":
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    for id in range(connectivity.region_labels.shape[0]):
        if connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)
    main_example(ReducedWongWangExcIOInhI(), RedWWExcIOInhIBuilder, InterfaceRedWWexcIOinhI,
                 nest_nodes_ids, nest_populations_order=100, connectivity=connectivity, simulation_length=100.0,
                 tvb_state_variable_type_label="Synaptic Gating Variable", tvb_state_variables_labels=["S_e", "S_i"],
                 config=CONFIGURED)
