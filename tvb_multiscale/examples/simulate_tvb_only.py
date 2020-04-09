# -*- coding: utf-8 -*-
import time

import numpy as np
from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.tvb.simulator_builder import SimulatorBuilder
from tvb_multiscale.plot.plotter import Plotter
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.multiscale_wong_wang_exc_io_inh_i import MultiscaleWongWangExcIOInhI


def main_example(tvb_sim_model=ReducedWongWangExcIOInhI, connectivity=CONFIGURED.DEFAULT_CONNECTIVITY_ZIP,
                 simulation_length=100.0, config=CONFIGURED, **model_params):
    plotter = Plotter(config)

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
    simulator_builder = SimulatorBuilder()
    # Optionally modify the default configuration:
    simulator_builder.model = tvb_sim_model
    simulator_builder.connectivity = connectivity

    simulator = simulator_builder.build(**model_params)

    # Some code only for MultiscaleWongWangExcIOInhI
    # Select the regions for the fine scale modeling with NEST spiking networks
    spiking_regions_inds = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    for id, label in enumerate(simulator.connectivity.region_labels):
        if label.find("hippo") > 0:
            spiking_regions_inds.append(id)
    simulator.model._spiking_regions_inds = spiking_regions_inds
    # simulator.model._spiking_regions_inds = list(range(simulator.connectivity.number_of_regions))

    # Some code only for SpikingWongWangExcIOInhI & MultiscaleWongWangExcIOInhI
    simulator.integrator.noise.nsig = np.array(simulator.model.nvar * [simulator.integrator.noise.nsig[0]])
    simulator.integrator.noise.nsig[5:] = 0.0  # No noise for t_ref and derived variables

    plotter.plot_tvb_connectivity(simulator.connectivity)

    # -----------------------------------2. Simulate and gather results-------------------------------------------------
    # Configure the simulator
    simulator.configure()
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------3. Plot results--------------------------------------------------------
    # Gather the results
    t = results[0][0]
    source = results[0][1]

    # Put the results in a Timeseries instance
    from tvb_scripts.time_series.model import TimeSeriesRegion

    source_ts = TimeSeriesRegion(source, time=t,
                                 connectivity=simulator.connectivity,
                                 labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                                 labels_dimensions=
                                 {"State Variable": simulator.model.state_variables,
                                  "Region": simulator.connectivity.region_labels.tolist()},
                                 sample_period=simulator.integrator.dt)
    source_ts.configure()

    # # For timeseries plot:
    # for sv in source_ts.variables_labels:
    #     plotter.plot_timeseries(source_ts.get_state_variables(sv),
    #                             title="Region %s Time Series" % sv, figsize=config.figures.NOTEBOOK_SIZE)

    if len(spiking_regions_inds) > 0:
        plot_results_with_spikes_and_rates(source_ts, simulator, plotter, spiking_regions_inds,
                                           ["Exhitatory", "Inhibitory"],
                                           [simulator.model.N_E[0], simulator.model.number_of_modes])

    # plot_results(results, simulator, "State Variables", simulator.model.variables_of_interest, plotter)

    return simulator.connectivity, results


def mean_field_per_population(source_ts, populations, pop_sizes):
    from tvb_scripts.time_series.service import TimeSeriesService

    ts_service = TimeSeriesService()

    pop_sizes = [0] + pop_sizes
    mean_field = []
    for i_pop, (pop_name, pop_size) in enumerate(zip(populations, pop_sizes[1:])):
        mean_field.append(
            ts_service.mean_across_dimension(
                source_ts.get_modes_by_index(
                    np.arange(pop_sizes[i_pop], pop_size).astype("i")),
                3)
        )

        mean_field[-1].title = "Region mean field time series of %s population" % pop_name

    mean_field = ts_service.concatenate_modes(mean_field)
    try:
        del mean_field.labels_dimensions[mean_field.labels_ordering[3]]
    except:
        pass
    mean_field.labels_ordering[3] = "Population"
    mean_field.labels_dimensions["Population"] = np.array(populations)
    mean_field.title = "Region mean field time series"
    return mean_field


def spikes_per_population(source_spikes, populations, pop_sizes):
    pop_sizes = [0] + pop_sizes
    spikes = []
    for i_pop, (pop_name, pop_size) in enumerate(zip(populations, pop_sizes[1:])):

        spikes.append(
            source_spikes.get_modes_by_index(
                np.arange(pop_sizes[i_pop], pop_size).astype("i")))
        spikes[-1].title = "Region spikes' time series of %s population" % pop_name
        try:
            del spikes[-1].labels_dimensions[spikes[-1].labels_ordering[1]]
        except:
            pass
        spikes[-1].labels_ordering[1] = "Population"
        spikes[-1].labels_dimensions['Population'] = np.array([pop_name])
    return spikes


def spike_rates_from_spike_ts(spikes, dt):
    from tvb.simulator.plot.utils.data_structures_utils import ensure_list
    from tvb_scripts.service.time_series_service import TimeSeriesService

    ts_service = TimeSeriesService()

    rate = []
    for i_pop, spike_ts in enumerate(ensure_list(spikes)):
        rate.append(ts_service.sum_across_dimension(spike_ts, 3) / dt * 1000)
        try:
            del rate[-1].labels_dimensions[rate[-1].labels_ordering[3]]
        except:
            pass
        rate[-1].labels_ordering[3] = "Mode"
        rate[-1].title = "Spike rate"
    rate = ts_service.concatenate_variables(rate)
    rate.title = "Mean field population spike rates"
    return rate


def spike_rates_from_mean_field_rates(mean_field, spiking_regions_inds):
    rate = mean_field.get_state_variables("rate").get_subspace_by_index(spiking_regions_inds)
    try:
        del rate.labels_dimensions[rate.labels_ordering[1]]
        del rate.labels_dimensions[rate.labels_ordering[3]]
    except:
        pass
    rate = rate.swapaxes(1, 3)  # Swap "State Variable" with "Population"
    rate.labels_ordering[3] = "Mode"
    rate.title = "Mean field population spike rates"
    return rate


def plot_results_with_spikes_and_rates(source_ts, simulator, plotter, spiking_regions_inds,
                                       populations, pop_sizes):
    mean_field = mean_field_per_population(source_ts, populations, pop_sizes)

    spikes = spikes_per_population(
        source_ts.get_state_variables("spikes"). \
            get_subspace_by_index(spiking_regions_inds),
        populations, pop_sizes)

    if "rate" not in mean_field.labels_dimensions["State Variable"]:
        rate = spike_rates_from_spike_ts(spikes, simulator.integrator.dt)

    else:
        mean_field[:, "rate", spiking_regions_inds, :] /= (simulator.integrator.dt * 0.001)  # rate in Hz
        rate = spike_rates_from_mean_field_rates(mean_field, spiking_regions_inds)
    rate.title = "Region mean field spike rate time series"

    fig_spikes, axes_spikes = plotter.plot_spikes(spikes, rates=rate)

    from tvb_scripts.time_series.time_series_xarray import TimeSeries as TimeSeriesXarray

    mean_field_xr = TimeSeriesXarray(mean_field.get_subspace(spiking_regions_inds))
    mean_field_xr.plot_timeseries(per_variable=True, plotter=plotter, figsize=(10, 5))
    rate_xr = TimeSeriesXarray(rate)
    rate_xr.plot_timeseries(plotter=plotter, figsize=(10, 5))

    for i_pop, spike in enumerate(spikes):
        spike_xr = TimeSeriesXarray(spike)
        spike_xr.plot(y=spike_xr._data.dims[3], row=spike_xr._data.dims[2],
                      robust=True, figsize=(20, 10), plotter=plotter)
    return fig_spikes, axes_spikes


if __name__ == "__main__":
    # # -----------------------------Generic2dOscillator oscillatory regime-----------------------------------------------
    # model_params = {
    #     "d": np.array([0.1]),
    #     "e": np.array([0.0]),
    #     "f": np.array([1.0]),
    #     "g": np.array([1.0]),
    #     "I": np.array([0.0]),
    #     "tau": np.array([1.0]),
    #     "alpha": np.array([1.0]),
    #     "beta": np.array([0.0]),
    #     "a": np.array([0.0]),
    #     "b": np.array([-1.0]),
    #     "c": np.array([0.0])
    # }

    # # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------
    # model_params = {
    #     "tau_e": np.array([8.0]),
    #     "tau_i": np.array([8.0]),
    #     "c_ee": np.array([16.0]),
    #     "c_ei": np.array([12.0]),
    #     "c_ie": np.array([15.0]),
    #     "c_ii": np.array([3.0]),
    #     "a_e": np.array([1.3]),
    #     "a_i": np.array([2.0]),
    #     "b_e": np.array([4.0]),
    #     "b_i": np.array([3.7]),
    #     "P": np.array([0.0])
    # }

    # # ----------------------------------------SpikingWongWangExcIOInhI/MultiscaleWongWangExcIOInhI------------------------------------------------
    model_params = {
        "N_E": np.array([10, ]),
        "N_I": np.array([7, ]),
        "J_i": np.array([0.1, ]),
        "w_IE": np.array([0.1, ]),
        "lamda": np.array([0.5, ]),
        "G": np.array([20.0, ])  # G / N_E to normalize for the order of the spiking networks
    }

    main_example(MultiscaleWongWangExcIOInhI, simulation_length=100.0, config=CONFIGURED, **model_params)
