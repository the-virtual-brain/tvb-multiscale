# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb.basic.profile import TvbProfile

from tvb_multiscale.core.utils.computations_utils import \
    tvb_mean_field_per_population, tvb_spikes_per_population, tvb_spike_rates_from_TVB_spike_ts, \
    tvb_spike_rates_from_mean_field_rates, compute_tvb_spike_rate_corrs

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import CONFIGURED
try:
    from tvb_multiscale.core.io.h5_writer import H5Writer
except:
    H5Writer = None

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.utils.data_structures_utils import combine_DataArray_dims
from tvb_multiscale.tvb_elephant.spiking_network_analyser import SpikingNetworkAnalyser

from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI

from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def _initialize(config, plotter, writer):
    if plotter is None:
        plotter = Plotter(config)
    figsize = config.figures.DEFAULT_SIZE

    if writer is None and H5Writer is not None:
        writer = H5Writer()
    return plotter, figsize, writer


def plot_tvb_results_with_spikes_and_rates(source_ts, simulator, simulation_length, plotter, populations, pop_sizes):

    spiking_regions_inds = np.arange(simulator.connectivity.number_of_regions)

    mean_field = tvb_mean_field_per_population(source_ts, populations, pop_sizes)
    spikes = tvb_spikes_per_population(
                            source_ts.get_state_variables(
                                "spikes_train").get_subspace_by_index(spiking_regions_inds),
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


def plot_write_tvb_results(tvb_results, simulator, transient=0.0, spiking_nodes_ids=[],
                           populations=["E", "I"], populations_sizes=[],
                           tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
                           plotter=None, writer=None, config=CONFIGURED):

    plotter, figsize, writer = _initialize(config, plotter, writer)

    time_with_transient = tvb_results[0][0]
    source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=tvb_results[0][1], time=time_with_transient,
        connectivity=simulator.connectivity,
        labels_ordering=["Time", tvb_state_variable_type_label, "Region", "Neurons"],
        labels_dimensions={tvb_state_variable_type_label: list(tvb_state_variables_labels),
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)

    if transient:
        source_ts = source_ts[transient:]
    time = source_ts.time

    if writer is not None:
        writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                        connectivity=source_ts.connectivity),
                               os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")

    if isinstance(simulator.model, SpikingWongWangExcIOInhI):
        mean_field = tvb_mean_field_per_population(source_ts, populations, populations_sizes)
        # Plot time_series
        mean_field.plot_timeseries(plotter_config=plotter.config, per_variable=True,
                                   figsize=figsize, add_legend=False)
        if mean_field.number_of_labels > 9:
            mean_field.plot_raster(plotter_config=plotter.config, per_variable=True,
                                   figsize=figsize, add_legend=False)

        tvb_spikes, tvb_rates = \
            plot_tvb_results_with_spikes_and_rates(source_ts, simulator, source_ts.time[-1] - source_ts.time[0],
                                                   plotter, populations, populations_sizes)

        tvb_corrs = compute_tvb_spike_rate_corrs(tvb_rates)

        if writer is not None:
            writer.write_object(tvb_spikes.to_dict(),
                                path=os.path.join(config.out.FOLDER_RES, "TVB_Spikes") + ".h5")
            writer.write_tvb_to_h5(tvb_rates, os.path.join(config.out.FOLDER_RES, tvb_rates.title) + ".h5",
                                   recursive=False)
            writer.write_object(tvb_corrs, path=os.path.join(config.out.FOLDER_RES, "TVB_corrs") + ".h5")
    else:
        # Plot time_series
        source_ts.plot_timeseries(plotter_config=plotter.config, per_variable=True,
                                  figsize=figsize, add_legend=False)
        if source_ts.number_of_labels > 9:
            source_ts.plot_raster(plotter_config=plotter.config, per_variable=True,
                                  figsize=figsize, add_legend=False)

    n_spiking_nodes = len(spiking_nodes_ids)
    if n_spiking_nodes > 0:
        # Focus on the nodes modelled in the Spiking Network
        source_ts_nest = source_ts[:, :, spiking_nodes_ids]
        source_ts_nest.plot_timeseries(plotter_config=plotter.config, per_variable=True, figsize=figsize,
                                       figname="Spiking nodes TVB Time Series")
        if n_spiking_nodes > 3:
            source_ts_nest.plot_raster(plotter_config=plotter.config, per_variable=True, figsize=figsize,
                                       figname="Spiking nodes TVB Time Series Raster")
    return time, time_with_transient


def plot_correlations(corrs):
    from xarray import DataArray
    from matplotlib import pyplot
    data, dims, coords = combine_DataArray_dims(corrs, [(0, 2), (1, 3)], join_string=", ", return_array=False)
    DataArray(data, dims=dims, name=corrs.name). \
        plot(x=dims[0], y=dims[1], cmap="jet",
             xticks=np.arange(data.shape[0]), yticks=np.arange(data.shape[1]))
    ax = pyplot.gca()
    ax.set_xticklabels(coords[dims[0]], rotation=45, ha="right")
    ax.set_yticklabels(coords[dims[1]])
    ax.set_aspect(1./ax.get_data_ratio())
    pyplot.tight_layout()
    return pyplot.gcf(), ax


def plot_write_spiking_network_results(spiking_network, connectivity=None,
                                       time=None, transient=0.0, monitor_period=1.0,
                                       plot_per_neuron=False, plotter=None, writer=None, config=CONFIGURED):

    plotter, figsize, writer = _initialize(config, plotter, writer)

    if time is not None:
        start_time = time[0]
        end_time = time[-1]
    else:
        start_time = None
        end_time = None

    spikeNet_analyzer = SpikingNetworkAnalyser(spikeNet=spiking_network,
                                               start_time=start_time, end_time=end_time, period=monitor_period,
                                               output_type="TVB", return_data=True, force_homogeneous_results=True,
                                               connectivity=connectivity)

    # Spikes
    spikes_res = \
        spikeNet_analyzer.\
            compute_spikeNet_spikes_rates_and_correlations(
                                                   populations_devices=None, regions=None,
                                                   populations_sizes=[],
                                                   rates_methods=[spikeNet_analyzer.compute_mean_rate,
                                                                  spikeNet_analyzer.compute_mean_rate_time_series],
                                                   rates_kwargs=[{}],
                                                   rate_results_names=["Mean Populations' Spikes' Rates",
                                                                       "Mean Populations' Spikes' Rates' Time Series"],
                                                   corrs_methods=[spikeNet_analyzer.compute_spikes_correlation_coefficient],
                                                   corrs_kwargs=[{}],
                                                   corrs_results_names=["Populations' Spikes' Correlation Coefficient"],
                                                   bin_kwargs={},
                                                   data_method=spikeNet_analyzer.get_spikes_from_device,
                                                   data_kwargs={},
                                                   return_devices=False
            )

    if spikes_res is not None:
        # Plot spikes' rasters together with mean population's spikes' rates' time series
        plotter.plot_spike_events(spikes_res["spikes"], rates=spikes_res["mean_rate_time_series"], figsize=figsize)

        # Mean rates
        print(spikes_res["mean_rate"])

        # Correlations
        print(spikes_res["spikes_correlation_coefficient"])
        fig, ax = plot_correlations(spikes_res["spikes_correlation_coefficient"])
        plotter.base._save_figure(figure_name="Populations' Spikes' Correlation Coefficient")

        # Write results to file:
        if writer is not None:
            writer.write_object(spikes_res["spikes"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES, "Spikes") + ".h5")
            writer.write_object(spikes_res["mean_rate"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES,
                                                  spikes_res["mean_rate"].name) + ".h5")
            writer.write_object(spikes_res["spikes_correlation_coefficient"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES,
                                                  spikes_res["spikes_correlation_coefficient"].name) + ".h5")
            writer.write_tvb_to_h5(
                                   TimeSeriesRegion().from_xarray_DataArray(
                                       spikes_res["mean_rate_time_series"]._data,
                                       connectivity=spikes_res["mean_rate_time_series"].connectivity),
                                   os.path.join(config.out.FOLDER_RES,
                                                spikes_res["mean_rate_time_series"].title) + ".h5",
                                   recursive=False)
        del spikes_res

        if plot_per_neuron:
            spikeNet_analyzer.return_data = False
            rates_ts_per_neuron = \
                spikeNet_analyzer. \
                    compute_spikeNet_rates_time_series(populations_devices=None, regions=None, populations_sizes=[],
                                                       computations_kwargs={}, data_method=None, data_kwargs={},
                                                       return_spikes_trains=False, return_devices=False)
            if rates_ts_per_neuron is not None and rates_ts_per_neuron.size:
                # Regions in rows
                row = rates_ts_per_neuron.dims[2] if rates_ts_per_neuron.shape[2] > 1 else None
                if row is None:
                    # Populations in rows
                    row = rates_ts_per_neuron.dims[1] if rates_ts_per_neuron.shape[1] > 1 else None
                    col = None
                else:
                    # Populations in columns
                    col = rates_ts_per_neuron.dims[1] if rates_ts_per_neuron.shape[1] > 1 else None
                rates_ts_per_neuron.plot(y=rates_ts_per_neuron.dims[3], row=row, col=col)
                plotter.base._save_figure(figure_name="Spike rates per neuron")

            del rates_ts_per_neuron

    # Continuous time data
    if plot_per_neuron:
        spikeNet_analyzer.return_data = True
    else:
        spikeNet_analyzer.return_data = False
    spikeNet_ts = \
        spikeNet_analyzer. \
            compute_spikeNet_mean_field_time_series(populations_devices=None, regions=None,
                                                    variables=None, populations_sizes=[],
                                                    computations_kwargs={}, data_kwargs={}, return_devices=False)
    if spikeNet_ts is not None:
        if plot_per_neuron:
            mean_field_ts = spikeNet_ts["mean_field_time_series"]
            spikeNet_ts = spikeNet_ts["data_by_neuron"]
            # Regions in rows
            row = spikeNet_ts.dims[2] if spikeNet_ts.shape[2] > 1 else None
            if row is None:
                # Populations in rows
                row = spikeNet_ts.dims[3] if spikeNet_ts.shape[3] > 1 else None
                col = None
            else:
                # Populations in cols
                col = spikeNet_ts.dims[3] if spikeNet_ts.shape[3] > 1 else None
            for var in spikeNet_ts.coords[spikeNet_ts.dims[1]]:
                this_var_ts = spikeNet_ts.loc[:, var, :, :, :]
                this_var_ts.name = var.item()
                this_var_ts.plot(y=spikeNet_ts.dims[4], row=row, col=col, cmap="jet", figsize=figsize)
                plotter.base._save_figure(
                    figure_name="Spiking Network variables' time series per neuron: %s" % this_var_ts.name)
        else:
            mean_field_ts = spikeNet_ts
        del spikeNet_ts

        if mean_field_ts.size > 0:
            mean_field_ts.plot_timeseries(plotter_config=plotter.config, per_variable=True)
            if mean_field_ts.shape[2] > 3:
                mean_field_ts.plot_raster(plotter_config=plotter.config, per_variable=True,
                                          linestyle="--", alpha=0.5, linewidth=0.5)

        # Write results to file:
        if writer is not None:
            writer.write_tvb_to_h5(
                                   TimeSeriesRegion().from_xarray_DataArray(
                                       mean_field_ts._data,
                                       connectivity=mean_field_ts.connectivity),
                                   os.path.join(config.out.FOLDER_RES, mean_field_ts.title) + ".h5", recursive=False)
        del mean_field_ts


def plot_write_results(tvb_results, simulator, spiking_network=None, spiking_nodes_ids=[],
                       populations=["E", "I"], populations_sizes=[], transient=0.0,
                       tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
                       plot_per_neuron=False, plotter=None, config=CONFIGURED):
    import time as timeit

    plotter, figsize, writer = _initialize(config, plotter, None)

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -------------------------------------------6. Plot results--------------------------------------------------------

    try:
        tvb_spikeNet_interface = simulator.tvb_spikeNet_interface
        spiking_network = tvb_spikeNet_interface.spiking_network
        spiking_nodes_ids = tvb_spikeNet_interface.spiking_nodes_ids
    except:
        pass

    time_with_transient = None
    if len(tvb_results) > 0:
        print("Plotting (and optionally writing to files) TVB results!")
        tic = timeit.time()
        time, time_with_transient = \
            plot_write_tvb_results(tvb_results, simulator, transient, spiking_nodes_ids,
                                   populations, populations_sizes,
                                   tvb_state_variable_type_label, tvb_state_variables_labels,
                                   plotter, writer, config)
        print("Done! in %f min" % ((timeit.time() - tic) / 60))

    if spiking_network is not None and len(spiking_nodes_ids) > 0:
        print("Plotting results from the SpikeNet simulation!")
        tic = timeit.time()
        plot_write_spiking_network_results(spiking_network, simulator.connectivity,
                                           time_with_transient, transient, simulator.monitors[0].period,
                                           plot_per_neuron, plotter, writer, config)
        print("Done! in %f min" % ((timeit.time() - tic) / 60))
