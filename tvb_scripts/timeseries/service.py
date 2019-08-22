# -*- coding: utf-8 -*-

from six import string_types
from collections import OrderedDict
from itertools import izip, cycle

import numpy as np
from scipy.signal import decimate, convolve, detrend, hilbert
from scipy.stats import zscore
from pylab import demean

from tvb_scripts.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_scripts.utils.data_structures_utils import isequal_string, ensure_list
from tvb_scripts.utils.computations_utils import select_greater_values_array_inds, \
    select_by_hierarchical_group_metric_clustering
from tvb_scripts.utils.analyzers_utils import abs_envelope, spectrogram_envelope, filter_data
from tvb_scripts.timeseries.model import LABELS_ORDERING


def decimate_signals(signals, time, decim_ratio):
    if decim_ratio > 1:
        signals = decimate(signals, decim_ratio, axis=0, zero_phase=True, ftype="fir")
        time = decimate(time, decim_ratio, zero_phase=True, ftype="fir")
        dt = np.mean(np.diff(time))
        (n_times, n_signals) = signals.shape
        return signals, time, dt, n_times


def cut_signals_tails(signals, time, cut_tails):
    signals = signals[cut_tails[0]:-cut_tails[-1]]
    time = time[cut_tails[0]:-cut_tails[-1]]
    (n_times, n_signals) = signals.shape
    return signals, time, n_times


NORMALIZATION_METHODS = ["zscore", "mean", "min", "max", "baseline", "baseline-amplitude", "baseline-std", "minmax"]


# TODO: Add a service to convert to 2D Time Series TVB instances


def normalize_signals(signals, normalization=None, axis=None, percent=None):
    # Following pylab demean:

    def matrix_subtract_along_axis(x, y, axis=0):
        "Return x minus y, where y corresponds to some statistic of x along the specified axis"
        if axis == 0 or axis is None or x.ndim <= 1:
            return x - y
        ind = [slice(None)] * x.ndim
        ind[axis] = np.newaxis
        return x - y[ind]

    def matrix_divide_along_axis(x, y, axis=0):
        "Return x divided by y, where y corresponds to some statistic of x along the specified axis"
        if axis == 0 or axis is None or x.ndim <= 1:
            return x / y
        ind = [slice(None)] * x.ndim
        ind[axis] = np.newaxis
        return x / y[ind]

    for norm, ax, prcnd in izip(ensure_list(normalization), cycle(ensure_list(axis)), cycle(ensure_list(percent))):
        if isinstance(norm, string_types):
            if isequal_string(norm, "zscore"):
                signals = zscore(signals, axis=ax)  # / 3.0
            elif isequal_string(norm, "baseline-std"):
                signals = normalize_signals(["baseline", "std"], axis=axis)
            elif norm.find("baseline") == 0 and norm.find("amplitude") >= 0:
                signals = normalize_signals(signals, ["baseline", norm.split("-")[1]], axis=axis, percent=percent)
            elif isequal_string(norm, "minmax"):
                signals = normalize_signals(signals, ["min", "max"], axis=axis)
            elif isequal_string(norm, "mean"):
                signals = demean(signals, axis=ax)
            elif isequal_string(norm, "baseline"):
                if prcnd is None:
                    prcnd = 1
                signals = matrix_subtract_along_axis(signals, np.percentile(signals, prcnd, axis=ax), axis=ax)
            elif isequal_string(norm, "min"):
                signals = matrix_subtract_along_axis(signals, np.min(signals, axis=ax), axis=ax)
            elif isequal_string(norm, "max"):
                signals = matrix_divide_along_axis(signals, np.max(signals, axis=ax), axis=ax)
            elif isequal_string(norm, "std"):
                signals = matrix_divide_along_axis(signals, signals.std(axis=ax), axis=ax)
            elif norm.find("amplitude") >= 0:
                if prcnd is None:
                    prcnd = [1, 99]
                amplitude = np.percentile(signals, prcnd[1], axis=ax) - np.percentile(signals, prcnd[0], axis=ax)
                this_ax = ax
                if isequal_string(norm.split("amplitude")[0], "max"):
                    amplitude = amplitude.max()
                    this_ax = None
                elif isequal_string(norm.split("amplitude")[0], "mean"):
                    amplitude = amplitude.mean()
                    this_ax = None
                signals = matrix_divide_along_axis(signals, amplitude, axis=this_ax)
            else:
                raise_value_error("Ignoring signals' normalization " + normalization +
                                  ",\nwhich is not one of the currently available " + str(NORMALIZATION_METHODS) + "!")
    return signals


class TimeseriesService(object):
    logger = initialize_logger(__name__)

    def __init__(self, logger=initialize_logger(__name__)):

        self.logger = logger

    def decimate(self, timeseries, decim_ratio, **kwargs):
        if decim_ratio > 1:
            return timeseries.duplicate(timeseries.data[0:timeseries.time_length:decim_ratio],
                                        sample_period=decim_ratio * timeseries.sample_period, **kwargs)
        else:
            return timeseries.duplicate()

    def decimate_by_filtering(self, timeseries, decim_ratio, **kwargs):
        if decim_ratio > 1:
            decim_data, decim_time, decim_dt, decim_n_times = decimate_signals(timeseries.squeezed,
                                                                               timeseries.time, decim_ratio)
            return timeseries.duplicate(decim_data, sample_period=decim_dt, **kwargs)
        else:
            return timeseries.duplicate(**kwargs)

    def convolve(self, timeseries, win_len=None, kernel=None, **kwargs):
        n_kernel_points = np.int(np.round(win_len))
        if kernel is None:
            kernel = np.ones((n_kernel_points, 1, 1, 1)) / n_kernel_points
        else:
            kernel = kernel * np.ones((n_kernel_points, 1, 1, 1))
        return timeseries.duplicate(convolve(timeseries.data, kernel, mode='same'), **kwargs)

    def hilbert_envelope(self, timeseries, **kwargs):
        return timeseries.duplicate(np.abs(hilbert(timeseries.data, axis=0)), **kwargs)

    def spectrogram_envelope(self, timeseries, lpf=None, hpf=None, nperseg=None, **kwargs):
        data, time = spectrogram_envelope(timeseries.squeezed, timeseries.sampling_frequency, lpf, hpf, nperseg)
        if len(timeseries.sample_period_unit) > 0 and timeseries.sample_period_unit[0] == "m":
            time *= 1000
        return timeseries.duplicate(data=data, start_time=timeseries.start_time + time[0],
                                    sample_period=np.diff(time).mean(), **kwargs)

    def abs_envelope(self, timeseries, **kwargs):
        return timeseries.duplicate(data=abs_envelope(timeseries.data), **kwargs)

    def detrend(self, timeseries, type='linear', **kwargs):
        return timeseries.duplicate(data=detrend(timeseries.data, axis=0, type=type), **kwargs)

    def normalize(self, timeseries, normalization=None, axis=None, percent=None, **kwargs):
        return timeseries.duplicate(data=normalize_signals(timeseries.data, normalization, axis, percent), **kwargs)

    def filter(self, timeseries, lowcut=None, highcut=None, mode='bandpass', order=3, **kwargs):
        return timeseries.duplicate(data=filter_data(timeseries.data, timeseries.sampling_frequency,
                                                     lowcut, highcut, mode, order), **kwargs)

    def log(self, timeseries, **kwargs):
        return timeseries.duplicate(data=np.log(timeseries.data), **kwargs)

    def exp(self, timeseries, **kwargs):
        return timeseries.duplicate(data=np.exp(timeseries.data), **kwargs)

    def abs(self, timeseries, **kwargs):
        return timeseries.duplicate(data=np.abs(timeseries.data), **kwargs)

    def power(self, timeseries):
        return np.sum(self.square(self.normalize(timeseries, "mean", axis=0)).squeezed, axis=0)

    def square(self, timeseries, **kwargs):
        return timeseries.duplicate(data=timeseries.data ** 2, **kwargs)

    def correlation(self, timeseries):
        return np.corrcoef(timeseries.squeezed.T)

    def concatenate_in_time(self, timeseries_list, labels=None):
        timeseries_list = ensure_list(timeseries_list)
        out_timeseries = timeseries_list[0]
        if labels is None:
            labels = out_timeseries.space_labels
        else:
            out_timeseries = out_timeseries.get_subspace_by_labels(labels)
        for id, timeseries in enumerate(timeseries_list[1:]):
            if np.float32(out_timeseries.sample_period) == np.float32(timeseries.sample_period):
                out_timeseries.data = np.concatenate([out_timeseries.data,
                                                      timeseries.get_subspace_by_labels(labels).data], axis=0)
            else:
                raise_value_error("Timeseries concatenation in time failed!\n"
                                  "Timeseries %d have a different time step (%s) than the ones before(%s)!" \
                                  % (id, str(np.float32(timeseries.sample_period)),
                                     str(np.float32(out_timeseries.sample_period))))
        return out_timeseries

    def select_by_metric(self, timeseries, metric, metric_th=None, metric_percentile=None, nvals=None):
        selection = np.unique(select_greater_values_array_inds(metric, metric_th, metric_percentile, nvals))
        return timeseries.get_subspace_by_index(selection), selection

    def select_by_power(self, timeseries, power=np.array([]), power_th=None):
        if len(power) != timeseries.number_of_labels:
            power = self.power(timeseries)
        return self.select_by_metric(timeseries, power, power_th)

    def select_by_hierarchical_group_metric_clustering(self, timeseries, distance, disconnectivity=np.array([]),
                                                       metric=None, n_groups=10, members_per_group=1):
        selection = np.unique(select_by_hierarchical_group_metric_clustering(distance, disconnectivity, metric,
                                                                             n_groups, members_per_group))
        return timeseries.get_subspace_by_index(selection), selection

    def select_by_correlation_power(self, timeseries, correlation=np.array([]), disconnectivity=np.array([]),
                                    power=np.array([]), n_groups=10, members_per_group=1):
        if correlation.shape[0] != timeseries.number_of_labels:
            correlation = self.correlation(timeseries)
        if len(power) != timeseries.number_of_labels:
            power = self.power(timeseries)
        return self.select_by_hierarchical_group_metric_clustering(timeseries, 1 - correlation,
                                                                   disconnectivity, power, n_groups, members_per_group)

    def select_by_gain_matrix_power(self, timeseries, gain_matrix=np.array([]),
                                    disconnectivity=np.array([]), power=np.array([]),
                                    n_groups=10, members_per_group=1):
        if len(power) != timeseries.number_of_labels:
            power = self.power(timeseries)
        return self.select_by_hierarchical_group_metric_clustering(timeseries, 1 - np.corrcoef(gain_matrix),
                                                                   disconnectivity, power, n_groups, members_per_group)

    def select_by_rois_proximity(self, timeseries, proximity, proximity_th=None, percentile=None, n_signals=None):
        initial_selection = range(timeseries.number_of_labels)
        selection = []
        for prox in proximity:
            selection += (
                np.array(initial_selection)[select_greater_values_array_inds(prox, proximity_th,
                                                                             percentile, n_signals)]).tolist()
        selection = np.unique(selection)
        return timeseries.get_subspace_by_index(selection), selection

    def select_by_rois(self, timeseries, rois, all_labels):
        for ir, roi in rois:
            if not (isinstance(roi, string_types)):
                rois[ir] = all_labels[roi]
        return timeseries.get_subspace_by_labels(rois), rois

    def compute_seeg(self, source_timeseries, sensors, sum_mode="lin", **kwargs):
        if np.all(sum_mode == "exp"):
            seeg_fun = lambda source, gain_matrix: compute_seeg_exp(source.squeezed, gain_matrix)
        else:
            seeg_fun = lambda source, gain_matrix: compute_seeg_lin(source.squeezed, gain_matrix)
        labels_ordering = LABELS_ORDERING
        labels_ordering[1] = "SEEG"
        labels_ordering[2] = "SEEG Sensor"
        kwargs.update({"ts_type": "SEEG", "labels_ordering": labels_ordering,
                       "start_time": source_timeseries.start_time,
                       "sample_period": source_timeseries.sample_period,
                       "sample_period_unit": source_timeseries.sample_period_unit})
        if isinstance(sensors, dict):
            seeg = OrderedDict()
            for sensor_name, sensor in sensors.items():
                kwargs.update({"labels_dimensions": {labels_ordering[2]: sensor.labels,
                                                     labels_ordering[1]: [sensor.name]}})
                seeg[sensor_name] = \
                    source_timeseries.__class__(seeg_fun(source_timeseries, sensor.gain_matrix), **kwargs)
            return seeg
        else:
            kwargs.update({"labels_dimensions": {labels_ordering[2]: sensors.labels,
                                                 labels_ordering[1]: [sensors.name]}})
            return source_timeseries.__class__(seeg_fun(source_timeseries, sensors.gain_matrix), **kwargs)


def compute_seeg_lin(source_timeseries, gain_matrix):
    return source_timeseries.dot(gain_matrix.T)


def compute_seeg_exp(source_timeseries, gain_matrix):
    return np.log(np.exp(source_timeseries).dot(gain_matrix.T))
