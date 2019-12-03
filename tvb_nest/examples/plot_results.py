from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.plot.plotter import Plotter
from tvb_scripts.time_series.model import TimeSeriesRegion


def plot_results(results, simulator, tvb_nest_model, tvb_state_variable_type_label="", tvb_state_variables_labels=[],
                 plotter=Plotter(CONFIGURED)):
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
    # # ...interactively as well
    # plotter.plot_timeseries_interactive(source_ts)

    if tvb_nest_model is None:
       return

    # In all the following we assume that all populations are in the same (equal number of) regions,
    # whereas we average across individual neurons

    # Plot NEST multimeter variables
    multimeter_mean_data = tvb_nest_model.get_mean_data_from_NEST_multimeter_to_TVBTimeSeries()
    if multimeter_mean_data is not None and multimeter_mean_data.size > 0:
        plotter.plot_multimeter_timeseries(multimeter_mean_data, plot_per_variable=True,
                                           time_series_class=TimeSeriesRegion, time_series_args={},
                                           var_pop_join_str=" - ", default_population_label="population",
                                           title="NEST region time series")
        plotter.plot_multimeter_raster(multimeter_mean_data, plot_per_variable=True,
                                       time_series_class=TimeSeriesRegion, time_series_args={},
                                       var_pop_join_str=" - ", default_population_label="population",
                                       title="NEST region time series raster")

    # Plot spikes and mean field spike rates
    rates, spike_detectors = \
        tvb_nest_model.get_mean_spikes_rates_from_NEST_to_TVBTimeSeries(
            spikes_kernel_width=simulator.integrator.dt,  # ms
            spikes_kernel_overlap=0.0, time=t)
    if spike_detectors is not None and rates.size > 0:
        plotter.plot_spikes(spike_detectors, rates=rates, title='Population spikes and mean spike rate')

    # ------------------------------------Testing code for xarray TimeSeries--------------------------------------------

    nest_network = tvb_nest_model.nest_network
    multimeter_mean_data = nest_network.get_mean_data_from_multimeter()

    if multimeter_mean_data.size > 0:
        from tvb_scripts.time_series.time_series_xarray import TimeSeries as TimeSeriesXarray

        ts = TimeSeriesXarray(multimeter_mean_data)
        # ts.plot(plotter=plotter, )
        ts.plot_timeseries(plotter=plotter)
        ts.plot_raster(plotter=plotter, linestyle="--", alpha=0.5, linewidth=0.5)
        # print(ts[0].shape)
        # print(ts[:, 0].shape)
        # print(ts[:, "V_m"].shape)
        # print(ts[:, :, 0].shape)
        # print(ts[:, ["V_m"], 1, :].shape)
        # ts[::, ["V_m"], 1, :] = TimeSeriesXarray(ts[:, ["V_m"], 1, :])
        # ts[::, ["V_m"], 1, :] = TimeSeriesXarray(ts[:, ["V_m"], 1, :].data, time=ts[:, ["V_m"], 1, :].time)

    # ------------------------------------Testing code for plotting xarray data-----------------------------------------

    rates = \
        nest_network.compute_spikes_rates(mode="per_neuron", population_devices=None, regions=None,
                                          devices_dim_name="Population", name="Spikes rates from NEST network",
                                          spikes_kernel_width=1.0, # spikes_kernel_n_intervals=10,
                                          spikes_kernel_overlap=0.5, min_spike_interval=None, time=t,
                                          spikes_kernel=None)[0]
    if rates.size > 0:
        rates.plot(x=rates.dims[0], y=rates.dims[3], row=rates.dims[2], col=rates.dims[1], robust=True)
        plotter.base._save_figure(figure_name="Spike rates per neuron")
