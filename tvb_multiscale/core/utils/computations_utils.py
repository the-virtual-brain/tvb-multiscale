# -*- coding: utf-8 -*-

import numpy as np
from pandas import MultiIndex, Series
from scipy import signal
from scipy.stats import spearmanr
from six import string_types
from xarray import DataArray

from tvb.contrib.scripts.service.time_series_service import TimeSeriesService


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
