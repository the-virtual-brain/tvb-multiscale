# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from pandas import MultiIndex, Series
from scipy import signal
from scipy.stats import spearmanr
from six import string_types
from xarray import DataArray

from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion
from tvb.contrib.scripts.service.time_series_service import TimeSeriesService
from tvb.contrib.scripts.utils.data_structures_utils import concatenate_heterogeneous_DataArrays, flatten_list


def Pearson(x):
    return np.corrcoef(x.T)


def Spearman(x):
    return spearmanr(x)[0]


# ---------------------------------------TVB spikes time series tools---------------------------------------------------


def mean_fields_generator(source_ts, populations, pop_sizes):
    ts_service = TimeSeriesService()
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()

    for i_pop, (pop_name, pop_inds, pop_size) in enumerate(zip(populations, pop_cumsum[1:], pop_sizes)):
        mean_field = ts_service.sum_across_dimension(
            source_ts.get_modes_by_index(
                np.arange(pop_cumsum[i_pop], pop_inds).astype("i")),
            3)
        mean_field.data /= pop_size
        mean_field.title = "Region mean field time series of %s population" % pop_name
        yield mean_field


def tvb_mean_field_per_population(source_ts, populations, pop_sizes):
    ts_service = TimeSeriesService()
    mean_field = ts_service.concatenate_modes(mean_fields_generator(source_ts, populations, pop_sizes))
    labels_dimensions = mean_field.labels_dimensions
    try:
        del labels_dimensions[mean_field.labels_ordering[3]]
    except:
        pass
    mean_field.update_dimension_names("Population", 3)
    labels_dimensions["Population"] = np.array(populations)
    mean_field.labels_dimensions = labels_dimensions
    mean_field.title = "Region mean field time series"
    return mean_field


def tvb_spikes_per_population_generator(source_spikes, populations, pop_sizes):
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()
    for i_pop, (pop_name, pop_inds) in enumerate(zip(populations, pop_cumsum[1:])):
        spike = source_spikes.get_modes_by_index(np.arange(pop_cumsum[i_pop], pop_inds).astype("i"))
        spike.name = "Region spikes' time series of %s population" % pop_name
        labels_dimensions = spike.labels_dimensions
        try:
            del labels_dimensions[spike.labels_ordering[1]]
        except:
            pass
        spike.update_dimension_names("Population", 1)
        labels_dimensions["Population"] = np.array([pop_name])
        spike.labels_dimensions = labels_dimensions
        yield spike


def tvb_spikes_per_population(source_spikes, populations, pop_sizes):
    tvb_spikes = Series()
    for pop, spike in zip(populations,
                           tvb_spikes_per_population_generator(source_spikes, populations, pop_sizes)):
        tvb_spikes[pop] = spike
    return tvb_spikes


def tvb_spikes_per_population_from_generator(spikes_generator, populations):
    tvb_spikes = Series()
    for pop, spike in zip(populations, spikes_generator):
        tvb_spikes[pop] = spike
    return tvb_spikes


def compute_rates_generator(spikes_generator, pop_sizes, integrator_dt, kernel, norm_kernel):
    ts_service = TimeSeriesService()

    for (spike_ts, pop_size) in zip(spikes_generator, pop_sizes):
        this_rate = ts_service.sum_across_dimension(spike_ts, 3)
        this_rate.data = this_rate.data / integrator_dt * 1000 / pop_size
        if kernel is not None:
            for i_reg in range(this_rate.shape[2]):
                this_rate.data[:, 0, i_reg, 0] = \
                    signal.convolve(this_rate.data[:, 0, i_reg, 0].squeeze(), kernel, mode="same") / norm_kernel
        try:
            del this_rate.labels_dimensions[this_rate.labels_ordering[3]]
        except:
            pass
        this_rate.labels_ordering[3] = "Mode"
        this_rate.title = "Spike rate"
        yield this_rate


def tvb_spike_rates_from_TVB_spike_ts(spikes_generator, integrator_dt, pop_sizes, sampling_period=0.1,
                                      window_time_length=100.0, kernel="gaussian", **kwargs):
    # spikes_ts are assumed to have an amplitude of tvb_integrator_dt / tvb_monitor_dt
    if kernel is not None:
        if isinstance(kernel, string_types):
            nt = np.maximum(3, int(np.round(window_time_length / sampling_period)))
            kernel = getattr(signal, kernel)(nt, **kwargs)
        norm_kernel = np.sum(kernel)
    ts_service = TimeSeriesService()
    rates = compute_rates_generator(spikes_generator, pop_sizes, integrator_dt, kernel, norm_kernel)
    rates = ts_service.concatenate_variables(rates)
    rates.title = "Mean field population spike rates"
    return rates


def tvb_spike_rates_from_mean_field_rates(mean_field):
    rate = mean_field.get_state_variables("rate")
    try:
        del rate.labels_dimensions[rate.labels_ordering[1]]
        del rate.labels_dimensions[rate.labels_ordering[3]]
    except:
        pass
    rate = rate.swapaxes(1, 3)  # Swap "State Variable" with "Population"
    rate.labels_ordering[3] = "Mode"
    rate.title = "Mean field population spike rates"
    return rate


def tvb_TimeSeries_correlation(ts, corrfun=Pearson, force_dims=4):
    data = ts._data # Get the DataArray of TimeSeries
    if data.shape[-1] == 1:  # Get rid of the 4th dimension if it is only 1
        data = data.squeeze(axis=-1)
    dims = list(data.dims[1:])
    # Combine variables State Variable x Region x ...
    # and prepare the two dimensions (i, j) of the correlation matrix
    stacked_dims = "-".join(dims)
    names = []
    new_dims = []
    for d in ["i", "j"]:
        names.append([dim+"_"+d for dim in dims])
        new_dims.append(stacked_dims+"_"+d)
    # Stack data across combined variables
    data = data.stack(**{stacked_dims: tuple(data.dims[1:])})
    # Prepare output DataArray
    n = len(data.coords[stacked_dims])
    corrs = DataArray(np.empty((n, n), dtype="f"), dims=new_dims,
                      name="Mean field %s correlation" % corrfun.__name__,
                      coords={new_dims[0]: MultiIndex.from_tuples(data.coords[stacked_dims].values, names=names[0]),
                              new_dims[1]: MultiIndex.from_tuples(data.coords[stacked_dims].values, names=names[1])})
    try:
        # TODO: a better hack for when Spearman returns nan
        corrs.values = corrfun(data.values)  # Compute all combinations of correlations across Time
    except:
        corrs.values = corrfun(data.values) * np.ones(corrs.values.shape)
    corrs = corrs.unstack(new_dims)  # Unstack the combinations of State Variable x Region x ...
    new_dims = list(corrs.dims)
    corrs = corrs.transpose(*tuple(new_dims[0::2] + new_dims[1::2]))  # Put variables in front of regions
    if force_dims is not None:  # Compute the mean over possible 4th dimension ("Mode", or "Neuron")
        while len(corrs.dims) > force_dims:
            corrs = corrs.mean(axis=-1)
    return corrs


def compute_tvb_spike_rate_corrs(tvb_rates_ts, transient=0.0):
    if transient:
        tvb_rates_ts_steady_state = tvb_rates_ts[transient:]
    else:
        tvb_rates_ts_steady_state = tvb_rates_ts
    return {"Pearson": tvb_TimeSeries_correlation(tvb_rates_ts_steady_state, corrfun=Pearson, force_dims=4),
            "Spearman": tvb_TimeSeries_correlation(tvb_rates_ts_steady_state, corrfun=Spearman, force_dims=4)}


# --------------------------------------------Event spikes tools--------------------------------------------------------


def compute_time_from_spike_times(spikes, monitor_period):
    t_start = np.inf
    t_stop = -np.inf
    for i_pop, (pop_label, pop_spikes) in enumerate(spikes.iteritems()):
        for reg_label, reg_spikes in pop_spikes.iteritems():
            if len(reg_spikes["times"]) > 0:
                temp = np.min(reg_spikes["times"]).item()
                if temp < t_start:
                    t_start = temp
                temp = np.max(reg_spikes["times"]).item()
                if temp > t_stop:
                    t_stop = temp
    time = np.arange(t_start, t_stop+monitor_period, monitor_period)
    time = time[time <= t_stop]
    return time, t_start, t_stop


def compute_event_spike_rates(spikes, populations_size,
                              t_start_ms, t_stop_ms, computation_t_start, dt_ms):
    from quantities import ms
    from neo.core import SpikeTrain
    from elephant.statistics import time_histogram, instantaneous_rate, mean_firing_rate
    from elephant import kernels
    # Convert spikes' times to a SpikeTrain
    spike_train = SpikeTrain(spikes["times"] * ms, t_start=t_start_ms, t_stop=t_stop_ms)
    # Compute mean firing rate
    rate = float(mean_firing_rate(spike_train, t_start=computation_t_start, t_stop=t_stop_ms)) / populations_size
    try:
        # Try to use the default elephant kernel
        rate_ts = np.array(
                    instantaneous_rate(spike_train, sampling_period=dt_ms, t_start=t_start_ms, t_stop=t_stop_ms)) \
                  / populations_size
    except:
        # Set a safe sigma for the Gaussian kernel:
        sigma = np.minimum(1000.0, np.maximum(np.array(t_stop_ms - computation_t_start), 10)) * ms
        kernel = kernels.GaussianKernel(sigma=sigma)
        rate_ts = np.array(
                    instantaneous_rate(spike_train, kernel=kernel,
                                       sampling_period=dt_ms, t_start=t_start_ms, t_stop=t_stop_ms)) \
                  / populations_size
    spike_ts = np.array(time_histogram([spike_train], binsize=dt_ms,
                                       t_start=computation_t_start, t_stop=t_stop_ms)).squeeze()
    return spike_ts, spike_train, rate, rate_ts


def regions_spike_rate_generator(population_spikes, pop_label, populations_size,
                                 t_start_ms, t_stop_ms, computation_t_start, dt_ms):
    for reg_label, reg_spikes in population_spikes.iteritems():
        spike_ts, spike_train, rate, rate_ts = compute_event_spike_rates(reg_spikes, populations_size,
                                                                         t_start_ms, t_stop_ms,
                                                                         computation_t_start, dt_ms)
        yield reg_label, (pop_label, reg_label), spike_ts, spike_train, rate, rate_ts


def populations_spike_rate_generator(spikes, populations_sizes,
                                     t_start_ms, t_stop_ms, computation_t_start, dt_ms):
    for i_pop, (pop_label, pop_spikes) in enumerate(spikes.iteritems()):
        reg_labels = []
        pop_reg_labels = []
        pop_spikes_ts = []
        pop_spikes_trains = []
        pop_rates = []
        pop_rates_ts = []
        for reg_label, pop_reg_label, reg_spikes_ts, reg_spikes_trains, reg_rates, reg_rates_ts \
                in regions_spike_rate_generator(pop_spikes, pop_label, populations_sizes[i_pop],
                                                t_start_ms, t_stop_ms, computation_t_start, dt_ms):
            reg_labels.append(reg_label)
            pop_reg_labels.append(pop_reg_label)
            pop_spikes_ts.append(reg_spikes_ts)
            pop_spikes_trains.append(reg_spikes_trains)
            pop_rates.append(reg_rates)
            pop_rates_ts.append(reg_rates_ts)
        pop_rates = DataArray(data=np.array(pop_rates) * 1000,  # convert to sec from ms
                              dims=["Region"], coords={"Region": reg_labels})
        pop_rates_ts = DataArray(data=np.moveaxis(np.array(pop_rates_ts), 0, 1),
                                 dims=["Time", "Region", "Neurons"], coords={"Region": reg_labels})
        yield pop_label, pop_reg_labels, pop_spikes_ts, pop_spikes_trains, pop_rates, pop_rates_ts


def compute_populations_spikes_rates(spikes, populations_sizes, connectivity,
                                     t_start_ms, t_stop_ms, computation_t_start, dt_ms):
    pop_labels = []
    pop_reg_labels = []
    spikes_ts = []
    spikes_trains = []
    rates = Series()
    rates_ts = Series()
    for pop_label, i_pop_reg_labels, pop_spikes_ts, pop_spikes_trains, pop_rates, pop_rates_ts \
            in populations_spike_rate_generator(spikes, populations_sizes,
                                                t_start_ms, t_stop_ms, computation_t_start, dt_ms):
        pop_labels.append(pop_label)
        pop_reg_labels += i_pop_reg_labels
        spikes_ts += pop_spikes_ts
        spikes_trains += pop_spikes_trains
        rates[pop_label] = pop_rates
        rates_ts[pop_label] = pop_rates_ts

    # Concatenate (potentially heterogeneous) DataArrays of mean firing rates for each population:
    # converting to spikes/sec from spikes/ms:
    rates = concatenate_heterogeneous_DataArrays(rates, "Population",
                                                 name="Mean population spiking rates",
                                                 fill_value=np.nan, transpose_dims=["Population", "Region"])

    # This is the total vector of regions' labels:
    all_regions_lbls = [lb.values.item() for lb in dict(rates.coords)["Region"]]

    # Concatenate (potentially heterogeneous) DataArrays of windowed mean firing rates' time series for each population:
    rates_ts = concatenate_heterogeneous_DataArrays(rates_ts, "Population",
                                                    name="Mean population spiking rates time series", fill_value=np.nan,
                                                    transpose_dims=["Time", "Population", "Region", "Neurons"])
    # ...and put them to a TimeSeries object:
    t_start = float(t_start_ms)
    dt = float(dt_ms)
    time = np.arange(t_start, t_start+len(rates_ts)*dt, dt)
    rates_ts = TimeSeriesRegion(rates_ts,  # This is already in spikes/sec
                                time=time, connectivity=connectivity,
                                labels_ordering=["Time", "Population", "Region", "Neurons"],
                                labels_dimensions={"Population": pop_labels, "Region": all_regions_lbls},
                                sample_period=dt, title="Mean population spiking rates time series")

    return rates, rates_ts, spikes_ts, spikes_trains, pop_labels, all_regions_lbls, pop_reg_labels


def compute_event_spike_corrs(spikes_ts, spikes_trains, dt_ms, computation_t_start, t_stop_ms,
                              pop_labels, all_regions_lbls, pop_reg_labels,
                              dims=["Population", "Region"]):
    from elephant.conversion import BinnedSpikeTrain
    from elephant import spike_train_correlation

    def new_cross_dims_coors(dims, pop_labels, all_regions_lbls):
        from pandas import MultiIndex
        stacked_dims = "-".join(dims)
        names = []
        new_dims = []
        for d in ["i", "j"]:
            names.append([dim + "_" + d for dim in dims])
            new_dims.append(stacked_dims + "_" + d)
        new_coords = {new_dims[0]: MultiIndex.from_product([pop_labels, all_regions_lbls], names=names[0]),
                      new_dims[1]: MultiIndex.from_product([pop_labels, all_regions_lbls], names=names[1])}
        return new_dims, new_coords

    # Convert spike_train to a binned one for further use for correlations
    spikes_trains = BinnedSpikeTrain(spikes_trains, binsize=dt_ms, t_start=computation_t_start, t_stop=t_stop_ms)

    corrs = {}
    # Compute correlations of spike trains
    corrs["spike_train"] = spike_train_correlation.corrcoef(spikes_trains)
    del spikes_trains  # Free memory...
    # Compute correlations of spike time series
    spike_ts = np.array(spikes_ts).T
    corrs["Pearson"] = Pearson(spike_ts)
    corrs["Spearman"] = Spearman(spike_ts)
    del spike_ts  # Free memory...
    # Store all correlations together:
    new_dims, new_coords = new_cross_dims_coors(dims, pop_labels, all_regions_lbls)
    n_cross_dims = len(pop_labels) * len(all_regions_lbls)
    for corr, corr_name in zip(["spike_train", "Pearson", "Spearman"],
                               ["train", "Pearson", "Spearman"]):
        temp = DataArray(np.nan * np.ones((n_cross_dims, n_cross_dims)),
                         name="Mean population spike %s correlation" % corr_name,
                         dims=new_dims, coords=new_coords)
        temp.loc[pop_reg_labels, pop_reg_labels] = corrs[corr]
        corrs[corr] = temp
        corrs[corr] = corrs[corr].unstack(new_dims)
        temp_dims = list(corrs[corr].dims)
        # Put variables in front of regions:
        corrs[corr] = corrs[corr].transpose(*tuple(temp_dims[0::2] + temp_dims[1::2]))

    return corrs


def compute_event_spike_rates_corrs(spikes, populations_sizes, connectivity,
                                    time=None, monitor_period=0.1, transient=0.0):
    from quantities import ms

    # Configure time vector
    if time is not None:
        t_start = time[0]
        t_stop = time[-1]
    else:
        time, t_start, t_stop = compute_time_from_spike_times(spikes, monitor_period)
    dt = np.diff(time).mean()
    t_start_ms = t_start * ms
    t_stop_ms = (t_stop + dt) * ms
    dt_ms = dt * ms
    if transient:
        computation_t_start = transient * ms
    else:
        computation_t_start = t_start * ms

    # Loop for populations and regions to construct spike trains and compute rates:
    rates, rates_ts, spikes_ts, spikes_trains, pop_labels, all_regions_lbls, pop_reg_labels = \
        compute_populations_spikes_rates(spikes, populations_sizes, connectivity,
                                         t_start_ms, t_stop_ms, computation_t_start, dt_ms)

    corrs = compute_event_spike_corrs(spikes_ts, spikes_trains, dt_ms, computation_t_start, t_stop_ms,
                                      pop_labels, all_regions_lbls, pop_reg_labels,
                                      dims=list(rates.dims))
    del spikes_trains, spikes_ts  # Free memory

    return rates, rates_ts, corrs
