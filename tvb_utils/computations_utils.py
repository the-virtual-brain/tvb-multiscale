import numpy as np
from pandas import MultiIndex
from scipy import signal
from scipy.stats import spearmanr
from six import string_types
from xarray import DataArray


def _prepare_mean_fields(source_ts, populations, pop_sizes):
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


def mean_field_per_population(source_ts, populations, pop_sizes):
    ts_service = TimeSeriesService()
    mean_field = _prepare_mean_fields(source_ts, populations, pop_sizes)

    mean_field = ts_service.concatenate_modes_generator(mean_field)
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


def spikes_per_population(source_spikes, populations, pop_sizes):
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()
    spikes = []
    for i_pop, (pop_name, pop_inds) in enumerate(zip(populations, pop_cumsum[1:])):
        spikes.append(
            source_spikes.get_modes_by_index(
                np.arange(pop_cumsum[i_pop], pop_inds).astype("i")))
        spikes[-1].title = "Region spikes' time series of %s population" % pop_name
        labels_dimensions = spikes[-1].labels_dimensions
        try:
            del labels_dimensions[spikes[-1].labels_ordering[1]]
        except:
            pass
        spikes[-1].update_dimension_names("Population", 1)
        labels_dimensions["Population"] = np.array([pop_name])
        spikes[-1].labels_dimensions = labels_dimensions
    return spikes


def spikes_per_population_generator(source_spikes, populations, pop_sizes):
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()
    for i_pop, (pop_name, pop_inds) in enumerate(zip(populations, pop_cumsum[1:])):
        spike = source_spikes.get_modes_by_index(np.arange(pop_cumsum[i_pop], pop_inds).astype("i"))
        spike.title = "Region spikes' time series of %s population" % pop_name
        labels_dimensions = spike.labels_dimensions
        try:
            del labels_dimensions[spike.labels_ordering[1]]
        except:
            pass
        spike.update_dimension_names("Population", 1)
        labels_dimensions["Population"] = np.array([pop_name])
        spike.labels_dimensions = labels_dimensions
        yield spike


def spike_rates_from_TVB_spike_ts(spikes, integrator_dt, pop_sizes, sampling_period=0.1,
                                  window_time_length=100.0, kernel="gaussian", **kwargs):
    # spikes_ts are assumed to have an amplitude of tvb_integrator_dt / tvb_monitor_dt
    spikes = ensure_list(spikes)
    if kernel is not None:
        if isinstance(kernel, string_types):
            nt = np.maximum(3, int(np.round(window_time_length / sampling_period)))
            kernel = getattr(signal, kernel)(nt, **kwargs)
        norm_kernel = np.sum(kernel)
    ts_service = TimeSeriesService()
    rate = []
    for i_pop, (spike_ts, pop_size) in enumerate(zip(spikes, pop_sizes)):
        this_rate = ts_service.sum_across_dimension(spike_ts, 3)
        this_rate.data = this_rate.data / integrator_dt * 1000 / pop_size
        if kernel is not None:
            for i_reg in range(this_rate.shape[2]):
                this_rate.data[:, 0, i_reg, 0] = \
                    signal.convolve(this_rate.data[:, 0, i_reg, 0].squeeze(), kernel, mode="same") / norm_kernel
        rate.append(this_rate)
        try:
            del rate[-1].labels_dimensions[rate[-1].labels_ordering[3]]
        except:
            pass
        rate[-1].labels_ordering[3] = "Mode"
        rate[-1].title = "Spike rate"
    rate = ts_service.concatenate_variables(rate)
    rate.title = "Mean field population spike rates"
    return rate


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


def spike_rates_from_TVB_spike_ts_generator(spikes_generator, integrator_dt, pop_sizes, sampling_period=0.1,
                                  window_time_length=100.0, kernel="gaussian", **kwargs):
    # spikes_ts are assumed to have an amplitude of tvb_integrator_dt / tvb_monitor_dt
    if kernel is not None:
        if isinstance(kernel, string_types):
            nt = np.maximum(3, int(np.round(window_time_length / sampling_period)))
            kernel = getattr(signal, kernel)(nt, **kwargs)
        norm_kernel = np.sum(kernel)
    ts_service = TimeSeriesService()
    rates = compute_rates_generator(spikes_generator, pop_sizes, integrator_dt, kernel, norm_kernel)
    rates = ts_service.concatenate_variables_generator(rates)
    rates.title = "Mean field population spike rates"
    return rates


def spike_rates_from_mean_field_rates(mean_field):
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


def Pearson(x):
    return np.corrcoef(x.T)


def Spearman(x):
    return spearmanr(x)[0]


def TimeSeries_correlation(ts, corrfun=Pearson, force_dims=4):
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


def get_spike_rates_corrs(nest_spikes, populations_sizes, connectivity,
                          time=None, monitor_period=0.1, transient=0.0):
    from pandas import MultiIndex
    from elephant.statistics import time_histogram, instantaneous_rate, mean_firing_rate
    from elephant.conversion import BinnedSpikeTrain
    from elephant import spike_train_correlation
    from neo.core import SpikeTrain
    from quantities import ms

    if time is not None:
        t_start = time[0]
        t_stop = time[-1]
    else:
        t_start = []
        t_stop = []
        for i_pop, (pop_label, pop_spikes) in enumerate(nest_spikes.iteritems()):
            for reg_label, reg_spikes in pop_spikes.iteritems():
                t_start.append(np.min(reg_spikes["times"]).item())
                t_stop.append(np.max(reg_spikes["times"]).item())
        t_start = np.min(t_start)
        t_stop = np.max(t_stop)
        time = np.arange(t_start, t_stop + monitor_period, monitor_period)
    dt = np.diff(time).mean()
    t_start_ms = t_start * ms
    t_stop_ms = (t_stop + dt) * ms
    dt_ms = dt * ms
    if transient:
        computation_t_start = transient * ms
    else:
        computation_t_start = t_start * ms
    corrs = {}
    rates_ts = []
    rates = []
    spike_ts = []
    spike_trains = []
    pop_labels = []
    for i_pop, (pop_label, pop_spikes) in enumerate(nest_spikes.iteritems()):
        pop_labels.append(pop_label)
        rates_ts.append([])
        rates.append([])
        reg_labels = []
        for reg_label, reg_spikes in pop_spikes.iteritems():
            reg_labels.append(reg_label)
            # rates (spikes/sec) =
            #   total_number_of_spikes (int) / total_time_duration (sec) / total_number_of_neurons_in_pop (int)
            these_spikes = [spike for spike in reg_spikes["times"] if spike >= t_start and spike <= t_stop]
            spike_trains.append(SpikeTrain(these_spikes * ms, t_start=t_start_ms, t_stop=t_stop_ms))
            spike_ts.append(np.array(time_histogram([spike_trains[-1]], binsize=dt_ms,
                                                    t_start=computation_t_start, t_stop=t_stop_ms)).squeeze())
            rates[-1].append(float(mean_firing_rate(spike_trains[-1],
                                                    t_start=computation_t_start, t_stop=t_stop_ms))
                             / populations_sizes[i_pop])
            rates_ts[-1].append(np.array(instantaneous_rate(spike_trains[-1], sampling_period=dt_ms,
                                                            t_start=t_start_ms, t_stop=t_stop_ms))
                                / populations_sizes[i_pop])
    del these_spikes
    del nest_spikes  # Free memory...
    binned_spikes = BinnedSpikeTrain(spike_trains, binsize=dt_ms,
                                     t_start=computation_t_start, t_stop=t_stop_ms)
    del spike_trains  # Free memory...
    corrs["spike_train"] = spike_train_correlation.corrcoef(binned_spikes)
    del binned_spikes  # Free memory...
    spike_ts = np.array(spike_ts).T
    corrs["Pearson"] = Pearson(spike_ts)
    corrs["Spearman"] = Spearman(spike_ts)
    del spike_ts  # Free memory...
    # converting to spikes/sec from spikes/ms:
    rates = DataArray(1000 * np.array(rates), name="Mean population spiking ratew",
                      dims=["Population", "Region"],
                      coords={"Population": pop_labels, "Region": reg_labels})
    rates_ts = TimeSeriesRegionX(np.moveaxis(np.array(rates_ts), 2, 0),  # This is already in spikes/sec
                                 time=time, connectivity=connectivity,
                                 labels_ordering=["Time", "Population", "Region", "Neurons"],
                                 labels_dimensions={"Population": pop_labels, "Region": reg_labels},
                                 sample_period=dt, title="Mean population spiking rates time series")
    dims = list(rates.dims)
    stacked_dims = "-".join(dims)
    names = []
    new_dims = []
    for d in ["i", "j"]:
        names.append([dim + "_" + d for dim in dims])
        new_dims.append(stacked_dims + "_" + d)
    for corr, corr_name in zip(["spike_train", "Pearson", "Spearman"],
                               ["train", "Pearson", "Spearman"]):
        corrs[corr] = DataArray(corrs[corr], name="Mean population spike %s correlation" % corr_name, dims=new_dims,
                                coords={new_dims[0]:
                                            MultiIndex.from_product([pop_labels, reg_labels], names=names[0]),
                                        new_dims[1]:
                                            MultiIndex.from_product([pop_labels, reg_labels], names=names[1])})
        corrs[corr] = corrs[corr].unstack(new_dims)
        temp_dims = list(corrs[corr].dims)
        # Put variables in front of regions:
        corrs[corr] = corrs[corr].transpose(*tuple(temp_dims[0::2] + temp_dims[1::2]))
    return rates, rates_ts, corrs