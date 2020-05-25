# -*- coding: utf-8 -*-

import os

import numpy as np
from xarray import DataArray

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.io.h5_writer import H5Writer
from tvb_multiscale.plot.plotter import Plotter

from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def plot_write_results(results, simulator, population_sizes, transient=0.0,
                       tvb_state_variable_type_label="", tvb_state_variables_labels=[],
                       plotter=None, config=CONFIGURED):

    if plotter is None:
        plotter = Plotter(config)

    plotter.plot_tvb_connectivity(simulator.connectivity)

    writer = H5Writer()

    # -------------------------------------------6. Plot results--------------------------------------------------------

    #   Remove ts_type="Region" this argument too for TVB TimeSeriesRegion
    source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=results[0][1], time=results[0][0],
        connectivity=simulator.connectivity,
        labels_ordering=["Time", tvb_state_variable_type_label, "Region", "Neurons"],
        labels_dimensions={tvb_state_variable_type_label: list(tvb_state_variables_labels),
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)

    if transient:
        source_ts = source_ts[transient:]
        exclude_times = [0.0, transient]
    else:
        exclude_times = []
    t = source_ts.time
    writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                    connectivity=source_ts.connectivity),
                           os.path.join(config.out.FOLDER_RES, source_ts.title)+".h5")

    # Plot time_series
    source_ts.plot_timeseries(plotter=plotter, per_variable=True, figsize=(10, 5))
    if source_ts.number_of_labels > 9:
        source_ts.plot_raster(plotter=plotter, per_variable=True, figsize=(10, 5))

    try:
        if simulator.tvb_spikeNet_interface is None:
            return

    except:
        return

    # Focus on the nodes modelled in NEST:
    try:
        n_spiking_nodes = len(simulator.tvb_spikeNet_interface.spiking_nodes_ids)
        source_ts_nest = source_ts[:, :, simulator.tvb_spikeNet_interface.spiking_nodes_ids]
        source_ts_nest.plot_timeseries(plotter=plotter, per_variable=True, figsize=(10, 5),
                                       figname="Spiking nodes TVB Time Series")
        if n_spiking_nodes > 3:
            source_ts_nest.plot_raster(plotter=plotter, per_variable=True, figsize=(10, 5),
                                       figname="Spiking nodes TVB Time Series Raster")
    except:
        n_spiking_nodes = 0
        pass

    # In all the following we assume that all populations are in the same (equal number of) regions,
    # whereas we average across individual neurons

    # Plot spikes and mean field spike rates
    rates, spike_detectors = \
        simulator.tvb_spikeNet_interface.get_mean_spikes_rates_to_TVBTimeSeries(
            spikes_kernel_width=1.0,  # ms
            spikes_kernel_overlap=0.5, time=t)
    if spike_detectors is not None and rates.size > 0:
        plotter.plot_spike_detectors(spike_detectors, rates=rates, title='Population spikes and mean spike rate')

    # Plot mean field NEST multimeter variables using TVB default TimeSeries and their plotters

    # Get them directly as mean field quantities...
    # multimeter_mean_data = \
    #     TimeSeriesXarray(simulator.tvb_spikeNet_interface.get_mean_data_from_multimeter_to_TVBTimeSeries(),
    #                      connectivity=simulator.connectivity)
    # if multimeter_mean_data is not None and multimeter_mean_data.size > 0:
    #     plotter.plot_multimeter_timeseries(multimeter_mean_data, plot_per_variable=True,
    #                                        time_series_class=TimeSeriesXarray, time_series_args={},
    #                                        var_pop_join_str=" - ", default_population_label="population",
    #                                        figname="NEST region time series")
    #     if n_spiking_nodes > 3:
    #         plotter.plot_multimeter_raster(multimeter_mean_data, plot_per_variable=True,
    #                                        time_series_class=TimeSeriesXarray, time_series_args={},
    #                                        var_pop_join_str=" - ", default_population_label="population",
    #                                        figname="NEST region time series raster")
    #

    # ...or get data per neuron and compute mean-field...
    spiking_network = simulator.tvb_spikeNet_interface.spiking_network

    # Spikes
    nest_spikes = spiking_network.get_spikes(mode="events", return_type="Series", exclude_times=exclude_times)
    writer.write_object(nest_spikes.to_dict(), path=os.path.join(config.out.FOLDER_RES,  "NEST_Spikes") + ".h5")

    plotter.plot_spike_events(nest_spikes)

    # Spikes' rates
    rates = []
    pop_labels = []
    duration = (t[-1] - t[0]) / 1000  # in sec
    for i_pop, (pop_label, pop_spikes) in enumerate(nest_spikes.iteritems()):
        pop_labels.append(pop_label)
        rates.append([])
        reg_labels = []
        for reg_label, reg_spikes in pop_spikes.iteritems():
            reg_labels.append(reg_label)
            # rates (spikes/sec) =
            #   total_number_of_spikes (int) / total_time_duration (sec) / total_number_of_neurons_in_pop (int)
            rates[-1].append(len(reg_spikes["times"]) / duration / population_sizes[i_pop])

    rates = DataArray(np.array(rates), dims=["Population", "Region"], name="NEST_spike_rates",
                      coords={"Population": pop_labels, "Region": reg_labels})
    print(rates)
    writer.write_object(rates.to_dict(), path=os.path.join(config.out.FOLDER_RES, rates.name) + ".h5")

    # An alternative plot:
    rates = \
        spiking_network.compute_spikes_rates(mode="per_neuron", populations_devices=None, regions=None,
                                             devices_dim_name="Population", name="Spikes rates from NEST network",
                                             spikes_kernel_width=1.0,  # spikes_kernel_n_intervals=10,
                                             spikes_kernel_overlap=0.5, min_spike_interval=None, time=t,
                                             spikes_kernel=None)[0]
    if rates.size > 0:
        rates.plot(x=rates.dims[0], y=rates.dims[3], row=rates.dims[2], col=rates.dims[1])
        plotter.base._save_figure(figure_name="Spike rates per neuron")

    # Time Series
    nest_ts = spiking_network.get_data_from_multimeter(mode="per_neuron")
    if nest_ts.size > 0:
        nest_ts = TimeSeriesXarray(nest_ts, connectivity=simulator.connectivity)
        if transient:
            nest_ts = nest_ts[transient:]
        if n_spiking_nodes <= 3:
            nest_ts.plot_map(y=nest_ts._data.dims[4], row=nest_ts._data.dims[2], col=nest_ts._data.dims[3],
                             per_variable=True,  cmap="jet", figsize=(20, 10), plotter=plotter)

        # Compute mean field
        ts = TimeSeriesXarray(nest_ts._data.mean(axis=-1), connectivity=nest_ts.connectivity,
                              title="Mean field spiking nodes time series")
        writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(ts._data, connectivity=ts.connectivity),
                               os.path.join(config.out.FOLDER_RES, ts.title)+".h5",
                               recursive=False)
        ts.plot_timeseries(plotter=plotter, per_variable=True)
        if n_spiking_nodes > 3:
            ts.plot_raster(plotter=plotter, per_variable=True, linestyle="--", alpha=0.5, linewidth=0.5)
