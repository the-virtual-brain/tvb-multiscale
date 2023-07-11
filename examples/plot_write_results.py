# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb.basic.profile import TvbProfile

from tvb_multiscale.core.plot.correlations_plot import plot_correlations
from tvb_multiscale.core.utils.computations_utils import \
    tvb_mean_field_per_population, tvb_spikes_per_population, tvb_spike_rates_from_TVB_spike_ts, \
    tvb_spike_rates_from_mean_field_rates, compute_tvb_spike_rate_corrs

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import CONFIGURED
try:
    from tvb_multiscale.core.tvb.io import H5Writer
except:
    H5Writer = None

from tvb_multiscale.core.plot.plotter import Plotter
from tvb_multiscale.core.data_analysis.spiking_network_analyser import SpikingNetworkAnalyser
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict

# from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb.simulator.plot.base_plotter import pyplot

from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


MAX_VARS_IN_COLS = 3
MAX_REGIONS_IN_ROWS = 10
MIN_REGIONS_FOR_RASTER_PLOT = 9


def _initialize(config, plotter, writer):
    if plotter is None:
        plotter = Plotter(config)
    figsize = config.figures.DEFAULT_SIZE

    if writer is None and H5Writer is not None:
        writer = H5Writer()
    return plotter, figsize, writer


def write_RegionTimeSeriesXarray_to_h5(ts, writer, path, **kwargs):
    dummy_ts = TimeSeriesRegion(data=ts._data, connectivity=ts.connectivity)
    dummy_ts.configure()
    writer.write_tvb_to_h5(dummy_ts, path, **kwargs)
    del dummy_ts


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
    mean_field_xr.plot_timeseries(per_variable=mean_field_xr.shape[1] > MAX_VARS_IN_COLS,
                                  hue="Region" if mean_field_xr.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                  plotter_config=plotter.config, figsize=plotter.config.DEFAULT_SIZE)
    if mean_field_xr.shape[2] > MIN_REGIONS_FOR_RASTER_PLOT:
        mean_field_xr.plot_raster(per_variable=mean_field_xr.shape[1] > MAX_VARS_IN_COLS,
                                  plotter_config=plotter.config, figsize=plotter.config.DEFAULT_SIZE)

    rates_xr = TimeSeriesXarray(data=rates)
    rates_xr.configure()
    rates_xr.plot_timeseries(plotter_config=plotter.config,
                             hue="Region" if rates_xr.shape[2] > MAX_REGIONS_IN_ROWS else None,
                             figsize=plotter.config.DEFAULT_SIZE)
    if rates_xr.shape[2] > MIN_REGIONS_FOR_RASTER_PLOT:
        rates_xr.plot_raster(plotter_config=plotter.config, figsize=plotter.config.DEFAULT_SIZE)

    for i_pop, spike in enumerate(spikes):
        spike_xr = TimeSeriesXarray(spike)
        spike_xr.configure()
        spike_xr.plot(y=spike_xr._data.dims[3], row=spike_xr._data.dims[2],
                      robust=True, figsize=(20, 10), plotter_config=plotter.config)
    return spikes, rates


def plot_write_tvb_results(tvb_result, simulator, transient=0.0, spiking_nodes_ids=[],
                           # populations=["E", "I"], populations_sizes=[],
                           tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
                           plotter=None, writer=None, config=CONFIGURED, **kwargs):

    plotter, figsize, writer = _initialize(config, plotter, writer)

    time_with_transient = tvb_result[0]
    source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
        data=tvb_result[1], time=time_with_transient,
        connectivity=simulator.connectivity,
        labels_ordering=["Time", tvb_state_variable_type_label, "Region", "Neurons"],
        labels_dimensions={tvb_state_variable_type_label: list(tvb_state_variables_labels),
                           "Region": simulator.connectivity.region_labels.tolist()},
        sample_period=simulator.integrator.dt)
    source_ts.configure()

    if transient:
        source_ts = source_ts[transient:]
    time = source_ts.time

    if writer is not None:
        write_RegionTimeSeriesXarray_to_h5(source_ts, writer,
                                           os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
    else:
        dump_pickled_dict(source_ts._data.to_dict(),
                          os.path.join(config.out.FOLDER_RES, "source_ts.pkl"))
    # if isinstance(simulator.model, SpikingWongWangExcIOInhI):
    #     populations = kwargs.get("populations", ["E", "I"])
    #     populations_sizes = kwargs.get("populations_sizes", [])
    #     mean_field = tvb_mean_field_per_population(source_ts, populations, populations_sizes)
    #     # Plot time_series
    #     mean_field.plot_timeseries(plotter_config=plotter.config,
    #                                hue="Region" if mean_field.shape[2] > MAX_REGIONS_IN_ROWS else None,
    #                                per_variable=mean_field.shape[1] > MAX_VARS_IN_COLS,
    #                                figsize=figsize)
    #     if mean_field.shape[2] > MIN_REGIONS_FOR_RASTER_PLOT:
    #         # Plot raster
    #         mean_field.plot_raster(plotter_config=plotter.config,
    #                                per_variable=mean_field.shape[1] > MAX_VARS_IN_COLS,
    #                                figsize=figsize)
    #
    #     tvb_spikes, tvb_rates = \
    #         plot_tvb_results_with_spikes_and_rates(source_ts, simulator, source_ts.time[-1] - source_ts.time[0],
    #                                                plotter, populations, populations_sizes)
    #
    #     tvb_corrs = compute_tvb_spike_rate_corrs(tvb_rates)
    #
    #     if writer:
    #         writer.write_object(tvb_spikes.to_dict(),
    #                             path=os.path.join(config.out.FOLDER_RES, "TVB_Spikes") + ".h5")
    #         writer.write_tvb_to_h5(tvb_rates, os.path.join(config.out.FOLDER_RES, tvb_rates.title) + ".h5",
    #                                recursive=False)
    #         writer.write_object(tvb_corrs, path=os.path.join(config.out.FOLDER_RES, "TVB_corrs") + ".h5")
    # else:
    # Plot time_series
    source_ts.plot_timeseries(plotter_config=plotter.config,
                              hue="Region" if source_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                              row="Region" if source_ts.shape[2] <= MAX_REGIONS_IN_ROWS else None,
                              per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                              figsize=figsize)
    if source_ts.shape[2] > MIN_REGIONS_FOR_RASTER_PLOT:
        # Plot raster
        source_ts.plot_raster(plotter_config=plotter.config,
                              per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                              figsize=figsize)

    n_spiking_nodes = len(spiking_nodes_ids)
    if n_spiking_nodes > 0:
        # Focus on the nodes modelled in the Spiking Network
        source_ts_nest = source_ts[:, :, spiking_nodes_ids]
        source_ts_nest.plot_timeseries(plotter_config=plotter.config,
                                       hue="Region" if source_ts_nest.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                       row="Region" if source_ts_nest.shape[2] <= MAX_REGIONS_IN_ROWS else None,
                                       per_variable=source_ts_nest.shape[1] > MAX_VARS_IN_COLS,
                                       figsize=figsize, figname="Spiking nodes TVB Time Series")
        if n_spiking_nodes > MIN_REGIONS_FOR_RASTER_PLOT:
            source_ts_nest.plot_raster(plotter_config=plotter.config,
                                       per_variable=source_ts_nest.shape[1] > MAX_VARS_IN_COLS,
                                       figsize=figsize, figname="Spiking nodes TVB Time Series Raster")

    return time, time_with_transient


def print_spikes_mean_result(result):
    print("Mean spike rates:")
    try:
        for pop in result.coords["Population"]:
            for reg in result.coords["Region"]:
                if not np.isnan(result.loc[pop, reg]):
                    print("%s - %s: %g" % (pop.values.item().split("_spikes")[0], reg.values.item(),
                                           result.loc[pop, reg].values.item()))
    except:
        print(result)


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

    spikeNet_analyzer = SpikingNetworkAnalyser(elephant=True, pyspike=True, spikeNet=spiking_network,
                                               start_time=start_time, end_time=end_time,
                                               period=monitor_period, transient=transient,
                                               time_series_output_type="TVB", return_data=True,
                                               force_homogeneous_results=True, connectivity=connectivity)

    # Spikes rates and correlations per Population and Region
    spikes_res = \
        spikeNet_analyzer.\
            compute_spikeNet_spikes_rates_and_correlations(
                                               populations_devices=None, regions=None,
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
        plotter.plot_spike_events(spikes_res["spikes"],
                                  time_series=spikes_res["mean_rate_time_series"],
                                  mean_results=spikes_res["mean_rate"],
                                  spikes_markersize=0.5, spikes_alpha=0.5,
                                  n_y_ticks=3, n_time_ticks=6, show_time_axis=True,
                                  figsize=figsize)

        # Mean rates
        print_spikes_mean_result(spikes_res["mean_rate"])

        # Correlations
        print(spikes_res["spikes_correlation_coefficient"])
        plot_correlations(spikes_res["spikes_correlation_coefficient"], plotter)

        # Write results to file:
        if writer:
            writer.write_object(spikes_res["spikes"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES, "Spikes") + ".h5")
            writer.write_object(spikes_res["mean_rate"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES,
                                                  spikes_res["mean_rate"].name) + ".h5")
            writer.write_object(spikes_res["spikes_correlation_coefficient"].to_dict(),
                                path=os.path.join(config.out.FOLDER_RES,
                                                  spikes_res["spikes_correlation_coefficient"].name) + ".h5")
            write_RegionTimeSeriesXarray_to_h5(spikes_res["mean_rate_time_series"], writer,
                                               os.path.join(config.out.FOLDER_RES,
                                                            spikes_res["mean_rate_time_series"].title) + ".h5",
                                               recursive=False)
        else:
            dump_pickled_dict(spikes_res["spikes"].to_dict(),
                                os.path.join(config.out.FOLDER_RES, "Spikes") + ".pkl")
            dump_pickled_dict(spikes_res["mean_rate"].to_dict(),
                              os.path.join(config.out.FOLDER_RES,
                                           spikes_res["mean_rate"].name) + ".pkl")
            dump_pickled_dict(spikes_res["spikes_correlation_coefficient"].to_dict(),
                              os.path.join(config.out.FOLDER_RES,
                                           spikes_res["spikes_correlation_coefficient"].name) + ".pkl")
            dump_pickled_dict(spikes_res["mean_rate_time_series"]._data.to_dict(),
                              os.path.join(config.out.FOLDER_RES,
                                           spikes_res["mean_rate_time_series"].title) + ".pkl")


        spikes_sync = \
            spikeNet_analyzer.compute_spikeNet_synchronization(
                populations_devices=None, regions=None,
                comp_methods=[spikeNet_analyzer.compute_spikes_sync,
                              spikeNet_analyzer.compute_spikes_sync_time_series],
                computations_kwargs=[{}], data_kwargs={},
                return_spikes_trains=False, return_devices=False)

        if spikes_sync is not None:
            print_spikes_mean_result(spikes_sync["spikes_sync"])
            # Plot spikes' rasters together with mean population's spikes' rates' time series
            plotter.plot_spike_events(spikes_res["spikes"],
                                      time_series=spikes_sync["spikes_sync_time_series"],
                                      mean_results=spikes_sync["spikes_sync"],
                                      spikes_markersize=0.5, spikes_alpha=0.5,
                                      n_y_ticks=3, n_time_ticks=6, show_time_axis=True,
                                      figsize=figsize)
            if writer:
                writer.write_object(spikes_sync["spikes_sync"].to_dict(),
                                    path=os.path.join(config.out.FOLDER_RES,
                                                      spikes_sync["spikes_sync"].name) + ".h5")
                write_RegionTimeSeriesXarray_to_h5(
                    spikes_sync["spikes_sync_time_series"], writer,
                    os.path.join(config.out.FOLDER_RES,
                                 spikes_sync["spikes_sync_time_series"].title) + ".h5",
                    recursive=False)
            else:
                dump_pickled_dict(spikes_sync["spikes_sync"].to_dict(),
                                  os.path.join(config.out.FOLDER_RES,
                                               spikes_sync["spikes_sync"].name) + ".pkl")
                dump_pickled_dict(spikes_sync["spikes_sync_time_series"]._data.to_dict(),
                                  os.path.join(config.out.FOLDER_RES,
                                               spikes_sync["spikes_sync_time_series"].title) + ".pkl")
            del spikes_sync

        del spikes_res
        if plot_per_neuron:
            # Spikes' rates per neuron:
            spikeNet_analyzer.return_data = False
            rates_ts_per_neuron = \
                spikeNet_analyzer. \
                    compute_spikeNet_rates_time_series(populations_devices=None, regions=None,
                                                       computations_kwargs={}, data_kwargs={},
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
                pyplot.figure()
                rates_ts_per_neuron.plot(y=rates_ts_per_neuron.dims[3], row=row, col=col, cmap="jet")
                plotter.base._save_figure(figure_name="Spike rates per neuron")
                if not plotter.base.config.SHOW_FLAG:
                    pyplot.close()
            del rates_ts_per_neuron

    # Continuous time data
    if plot_per_neuron:
        spikeNet_analyzer.return_data = True
    else:
        spikeNet_analyzer.return_data = False
    # Continuous time variables' data of spiking neurons
    spikeNet_ts = \
        spikeNet_analyzer. \
            compute_spikeNet_mean_field_time_series(populations_devices=None, regions=None, variables=None,
                                                    computations_kwargs={}, data_kwargs={}, return_devices=False)
    if spikeNet_ts:
        if plot_per_neuron:
            mean_field_ts = spikeNet_ts["mean_field_time_series"]  # mean field
            spikeNet_ts = spikeNet_ts["data_by_neuron"]  # per neuron data
            if spikeNet_ts.size > 0:
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
                    if this_var_ts.size > 0:
                        this_var_ts.name = var.item()
                        pyplot.figure()
                        try:
                            this_var_ts.plot(y=spikeNet_ts.dims[4], row=row, col=col, cmap="jet", figsize=figsize)
                        except:
                            this_var_ts.plot(y=spikeNet_ts.dims[4], row=row, col=col, cmap="jet", figsize=figsize)
                        plotter.base._save_figure(
                            figure_name="Spiking Network variables' time series per neuron: %s" % this_var_ts.name)
                        if not plotter.base.config.SHOW_FLAG:
                            pyplot.close()
        else:
            mean_field_ts = spikeNet_ts
        del spikeNet_ts

        if mean_field_ts.size > 0:
            mean_field_ts.plot_timeseries(plotter_config=plotter.config,
                                          per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS,
                                          figsize=figsize)
            if mean_field_ts.shape[2] > MIN_REGIONS_FOR_RASTER_PLOT:
                mean_field_ts.plot_raster(plotter_config=plotter.config,
                                          per_variable=mean_field_ts.shape[1] > MAX_VARS_IN_COLS,
                                          linestyle="--", alpha=0.5, linewidth=0.5, figsize=figsize)

        # Write results to file:
        if writer:
            write_RegionTimeSeriesXarray_to_h5(mean_field_ts, writer,
                                               os.path.join(config.out.FOLDER_RES, mean_field_ts.title) + ".h5",
                                               recursive=False)
        else:
            dump_pickled_dict(mean_field_ts._data.to_dict(),
                              os.path.join(config.out.FOLDER_RES, mean_field_ts.title) + ".pkl")
        del mean_field_ts


def plot_write_results(tvb_results, simulator,
                       spiking_network=None, spiking_nodes_ids=[],
                       transient=0.0,  # populations=["E", "I"], populations_sizes=[],
                       tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
                       plot_per_neuron=False, plotter=None, config=CONFIGURED):
    import time as timeit

    plotter, figsize, writer = _initialize(config, plotter, None)

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -------------------------------------------6. Plot results--------------------------------------------------------

    time_with_transient = None
    if len(tvb_results) > 0:
        print("Plotting (and optionally writing to files) TVB results!")
        tic = timeit.time()
        time, time_with_transient = \
            plot_write_tvb_results(tvb_results[0], simulator, transient, spiking_nodes_ids,
                                   # populations=["E", "I"], populations_sizes=[],
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
