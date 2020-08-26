# -*- coding: utf-8 -*-

import os

import numpy as np
from xarray import DataArray

from tvb.basic.profile import TvbProfile

from tvb_utils.computations_utils import \
    tvb_mean_field_per_population, tvb_spikes_per_population, tvb_spike_rates_from_TVB_spike_ts, \
    tvb_spike_rates_from_mean_field_rates, compute_event_spike_rates_corrs, compute_tvb_spike_rate_corrs

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import CONFIGURED
try:
    from tvb_multiscale.core.io.h5_writer import H5Writer
except:
    H5Writer = None

from tvb_multiscale.core.plot.plotter import Plotter

from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI

from tvb.contrib.scripts.utils.data_structures_utils import concatenate_heterogeneous_DataArrays
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def plot_write_results(results, simulator, populations=["E", "I"], populations_sizes=[], transient=0.0,
                       tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
                       plotter=None, config=CONFIGURED):

    if plotter is None:
        plotter = Plotter(config)

    figsize = config.figures.DEFAULT_SIZE

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -------------------------------------------6. Plot results--------------------------------------------------------

    #   Remove ts_type="Region" this argument too for TVB TimeSeriesRegion
    t_with_transient = results[0][0]
    source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=results[0][1], time=t_with_transient,
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

    if H5Writer is not None:
        writer = H5Writer()
        writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                        connectivity=source_ts.connectivity),
                               os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
    else:
        writer = None

    if isinstance(simulator.model, SpikingWongWangExcIOInhI):
        mean_field = tvb_mean_field_per_population(source_ts, populations, populations_sizes)
        # Plot time_series
        mean_field.plot_timeseries(plotter_config=plotter.config, per_variable=True, figsize=figsize, add_legend=False)
        if mean_field.number_of_labels > 9:
            mean_field.plot_raster(plotter_config=plotter.config, per_variable=True, figsize=figsize, add_legend=False)

        tvb_spikes, tvb_rates = \
            plot_results_with_spikes_and_rates(source_ts, simulator, source_ts.time[-1] - source_ts.time[0],
                                               plotter, populations, populations_sizes)

        tvb_corrs = compute_tvb_spike_rate_corrs(tvb_rates)

        if writer is not None:
            writer.write_object(tvb_spikes.to_dict(), path=os.path.join(config.out.FOLDER_RES, "TVB_Spikes") + ".h5")
            writer.write_tvb_to_h5(tvb_rates, os.path.join(config.out.FOLDER_RES, tvb_rates.title) + ".h5",
                                   recursive=False)
            writer.write_object(tvb_corrs, path=os.path.join(config.out.FOLDER_RES, "TVB_corrs") + ".h5")
    else:
        # Plot time_series
        source_ts.plot_timeseries(plotter_config=plotter.config, per_variable=True, figsize=figsize, add_legend=False)
        if source_ts.number_of_labels > 9:
            source_ts.plot_raster(plotter_config=plotter.config, per_variable=True, figsize=figsize, add_legend=False)


    try:
        if simulator.tvb_spikeNet_interface is None:
            return

    except:
        return

    # Focus on the nodes modelled in NEST:
    try:
        n_spiking_nodes = len(simulator.tvb_spikeNet_interface.spiking_nodes_ids)
        source_ts_nest = source_ts[:, :, simulator.tvb_spikeNet_interface.spiking_nodes_ids]
        source_ts_nest.plot_timeseries(plotter_config=plotter.config, per_variable=True, figsize=figsize,
                                       figname="Spiking nodes TVB Time Series")
        if n_spiking_nodes > 3:
            source_ts_nest.plot_raster(plotter_config=plotter.config, per_variable=True, figsize=figsize,
                                       figname="Spiking nodes TVB Time Series Raster")
    except:
        n_spiking_nodes = 0
        pass

    # Plot spikes and mean field spike rates
    mf_rates_ts, spike_detectors = \
        simulator.tvb_spikeNet_interface.get_mean_spikes_rates_to_TimeSeries(
            spikes_kernel_width=1.0,  # ms
            spikes_kernel_overlap=0.5, time=t, connectivity=simulator.connectivity)
    if spike_detectors is not None and mf_rates_ts.size > 0:
        plotter.plot_spike_detectors(spike_detectors, rates=mf_rates_ts, title='Population spikes and mean spike rate')

    # Plot mean field NEST multimeter variables using TVB default TimeSeries and their plotters

    # Get them directly as mean field quantities...
    multimeter_mean_data = \
       simulator.tvb_spikeNet_interface.get_mean_data_from_multimeter_to_TimeSeries(connectivity=simulator.connectivity)
    if multimeter_mean_data is not None and multimeter_mean_data.size > 0:
        plotter.plot_multimeter_timeseries(multimeter_mean_data, plot_per_variable=True,
                                           time_series_class=TimeSeriesXarray, time_series_args={},
                                           var_pop_join_str=" - ", default_population_label="population",
                                           figname="NEST region time series")
        if n_spiking_nodes > 3:
            plotter.plot_multimeter_raster(multimeter_mean_data, plot_per_variable=True,
                                           time_series_class=TimeSeriesXarray, time_series_args={},
                                           var_pop_join_str=" - ", default_population_label="population",
                                           figname="NEST region time series raster")


    # ...or get data per neuron and compute mean-field...
    spiking_network = simulator.tvb_spikeNet_interface.spiking_network

    # Spikes
    nest_spikes = spiking_network.get_spikes(mode="events", return_type="Series", exclude_times=exclude_times)

    plotter.plot_spike_events(nest_spikes)

    # Spikes' rates
    rates, rates_ts, corrs = compute_event_spike_rates_corrs(nest_spikes, populations_sizes, simulator.connectivity,
                                                             time=t_with_transient, transient=transient,
                                                             monitor_period=simulator.monitors[0].period)
    print(rates)

    # An alternative plot of rates per neuron and time wise:
    rates_ts_per_neuron = \
        spiking_network.compute_spikes_rates(mode="per_neuron", populations_devices=None, regions=None,
                                             devices_dim_name="Population", name="Spikes rates from NEST network",
                                             spikes_kernel_width=1.0,  # spikes_kernel_n_intervals=10,
                                             spikes_kernel_overlap=0.5, min_spike_interval=None, time=t,
                                             spikes_kernel=None)[0]
    if not isinstance(rates_ts_per_neuron, DataArray):
        # assuming a pandas Series due to heterogeneity of populations in among brain regions:
        rates_ts_per_neuron = concatenate_heterogeneous_DataArrays(rates_ts_per_neuron, "Population",
                                                                   transpose_dims=["Time", "Population",
                                                                                   "Region", "Neuron"])

    if rates_ts_per_neuron.size > 0:
        rates_ts_per_neuron.plot(x=rates_ts_per_neuron.dims[0], y=rates_ts_per_neuron.dims[3],
                                 row=rates_ts_per_neuron.dims[2], col=rates_ts_per_neuron.dims[1])
        plotter.base._save_figure(figure_name="Spike rates per neuron")

    # Time Series
    nest_ts = spiking_network.get_data_from_multimeter(mode="per_neuron")
    if not isinstance(nest_ts, DataArray):
        # assuming a pandas Series due to heterogeneity of populations in among brain regions:
        nest_ts = concatenate_heterogeneous_DataArrays(nest_ts, "Population",
                                                       transpose_dims=["Time", "Variable",
                                                                       "Region", "Population", "Neuron"])
    if nest_ts.size > 0:
        nest_ts = TimeSeriesXarray(nest_ts, connectivity=simulator.connectivity)
        if transient:
            nest_ts = nest_ts[transient:]
        if n_spiking_nodes <= 3:
            nest_ts.plot_map(y=nest_ts._data.dims[4], row=nest_ts._data.dims[2], col=nest_ts._data.dims[3],
                             per_variable=True,  cmap="jet", figsize=figsize, plotter_config=plotter.config)

        # Compute mean field
        ts = TimeSeriesXarray(nest_ts._data.mean(axis=-1), connectivity=nest_ts.connectivity,
                              title="Mean field spiking nodes time series")
        ts.plot_timeseries(plotter_config=plotter.config, per_variable=True)
        if n_spiking_nodes > 3:
            ts.plot_raster(plotter_config=plotter.config, per_variable=True, linestyle="--", alpha=0.5, linewidth=0.5)

        # Write results to file:
        if writer is not None:
            writer.write_object(nest_spikes.to_dict(), path=os.path.join(config.out.FOLDER_RES, "NEST_Spikes") + ".h5")
            writer.write_object(rates.to_dict(), path=os.path.join(config.out.FOLDER_RES, rates.name) + ".h5")
            writer.write_object(corrs, path=os.path.join(config.out.FOLDER_RES, "NEST_corrs") + ".h5")
            writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(ts._data, connectivity=ts.connectivity),
                                   os.path.join(config.out.FOLDER_RES, ts.title) + ".h5",
                                   recursive=False)
            writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(rates_ts._data,
                                                                            connectivity=rates_ts.connectivity),
                                   os.path.join(config.out.FOLDER_RES, rates_ts.title) + ".h5",
                                   recursive=False)


def plot_results_with_spikes_and_rates(source_ts, simulator, simulation_length, plotter, populations, pop_sizes):

    spiking_regions_inds = np.arange(simulator.connectivity.number_of_regions)

    mean_field = tvb_mean_field_per_population(source_ts, populations, pop_sizes)
    spikes = tvb_spikes_per_population(
                            source_ts.get_state_variables(
                                "spikes").get_subspace_by_index(spiking_regions_inds),
                            populations, pop_sizes)

    if "rate" not in mean_field.labels_dimensions["State Variable"]:
        T = np.maximum(np.minimum(100.0, 1000 * simulation_length / 10), 10.0)
        std = T / 3
        rates = tvb_spike_rates_from_TVB_spike_ts(spikes, simulator.integrator.dt, pop_sizes, sampling_period=0.1,
                                                 window_time_length=100.0, kernel="gaussian", std=std)

    else:
        mean_field[:, "rate", spiking_regions_inds, :] /= (simulator.integrator.dt * 0.001)  # rate in Hz
        rates = tvb_spike_rates_from_mean_field_rates(mean_field, spiking_regions_inds)
    rates.title = "Region mean field spike rate time series"

    plotter.plot_spikes(spikes, rates=rates)

    from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesXarray

    mean_field_xr = mean_field.get_subspace(spiking_regions_inds)
    mean_field_xr.plot_timeseries(per_variable=True, plotter_config=plotter.config, figsize=(10, 5))
    rates_xr = TimeSeriesXarray(rates)
    rates_xr.plot_timeseries(plotter_config=plotter.config, figsize=(10, 5))

    for i_pop, spike in enumerate(spikes):
        spike_xr = TimeSeriesXarray(spike)
        spike_xr.plot(y=spike_xr._data.dims[3], row=spike_xr._data.dims[2],
                      robust=True, figsize=(20, 10), plotter_config=plotter.config)
    return spikes, rates