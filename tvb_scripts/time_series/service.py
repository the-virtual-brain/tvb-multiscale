# -*- coding: utf-8 -*-

from six import string_types
from collections import OrderedDict
from itertools import cycle

import numpy as np
from scipy.signal import decimate, convolve, detrend, hilbert
from scipy.stats import zscore
from pylab import demean

from tvb_scripts.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_scripts.utils.data_structures_utils import isequal_string, ensure_list
from tvb.simulator.plot.utils.computations_utils import select_greater_values_array_inds, \
    select_by_hierarchical_group_metric_clustering
from tvb.simulator.plot.utils.analyzers_utils import abs_envelope, spectrogram_envelope, filter_data
from tvb_scripts.time_series.model import TimeSeriesSEEG, LABELS_ORDERING


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

    for norm, ax, prcnd in zip(ensure_list(normalization), cycle(ensure_list(axis)), cycle(ensure_list(percent))):
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


class TimeSeriesService(object):
    logger = initialize_logger(__name__)

    def __init__(self, logger=initialize_logger(__name__)):

        self.logger = logger

    def decimate(self, time_series, decim_ratio, **kwargs):
        if decim_ratio > 1:
            return time_series.duplicate(data=time_series.data[0:time_series.time_length:decim_ratio],
                                         sample_period=float(decim_ratio*time_series.sample_period), **kwargs)
        else:
            return time_series.duplicate()

    def decimate_by_filtering(self, time_series, decim_ratio, **kwargs):
        if decim_ratio > 1:
            decim_data, decim_time, decim_dt, decim_n_times = decimate_signals(time_series.squeezed,
                                                                               time_series.time, decim_ratio)
            return time_series.duplicate(data=decim_data, sample_period=float(decim_dt), **kwargs)
        else:
            return time_series.duplicate(**kwargs)

    def convolve(self, time_series, win_len=None, kernel=None, **kwargs):
        n_kernel_points = np.int(np.round(win_len))
        if kernel is None:
            kernel = np.ones((n_kernel_points, 1, 1, 1)) / n_kernel_points
        else:
            kernel = kernel * np.ones((n_kernel_points, 1, 1, 1))
        return time_series.duplicate(data=convolve(time_series.data, kernel, mode='same'), **kwargs)

    def hilbert_envelope(self, time_series, **kwargs):
        return time_series.duplicate(data=np.abs(hilbert(time_series.data, axis=0)), **kwargs)

    def spectrogram_envelope(self, time_series, lpf=None, hpf=None, nperseg=None, **kwargs):
        data, time = spectrogram_envelope(time_series.squeezed, time_series.sample_rate, lpf, hpf, nperseg)
        if len(time_series.sample_period_unit) > 0 and time_series.sample_period_unit[0] == "m":
            time *= 1000
        return time_series.duplicate(data=data, start_time=time_series.start_time + time[0],
                                     sample_period=np.diff(time).mean(), **kwargs)

    def abs_envelope(self, time_series, **kwargs):
        return time_series.duplicate(data=abs_envelope(time_series.data), **kwargs)

    def detrend(self, time_series, type='linear', **kwargs):
        return time_series.duplicate(data=detrend(time_series.data, axis=0, type=type), **kwargs)

    def normalize(self, time_series, normalization=None, axis=None, percent=None, **kwargs):
        return time_series.duplicate(data=normalize_signals(time_series.data, normalization, axis, percent), **kwargs)

    def filter(self, time_series, lowcut=None, highcut=None, mode='bandpass', order=3, **kwargs):
        return time_series.duplicate(data=filter_data(time_series.data, time_series.sample_rate,
                                                     lowcut, highcut, mode, order), **kwargs)

    def log(self, time_series, **kwargs):
        return time_series.duplicate(data=np.log(time_series.data), **kwargs)

    def exp(self, time_series, **kwargs):
        return time_series.duplicate(data=np.exp(time_series.data), **kwargs)

    def abs(self, time_series, **kwargs):
        return time_series.duplicate(data=np.abs(time_series.data), **kwargs)

    def power(self, time_series):
        return np.sum(self.square(self.normalize(time_series, "mean", axis=0)).squeezed, axis=0)

    def square(self, time_series, **kwargs):
        return time_series.duplicate(data=time_series.data ** 2, **kwargs)

    def correlation(self, time_series):
        return np.corrcoef(time_series.squeezed.T)

    def _compile_select_fun(self, **kwargs):
        select_fun = []
        for dim, lbl in enumerate(["times", "variables", "labels",  "samples"]):
            index = ensure_list(kwargs.pop(lbl, []))
            if len(index) > 0:
                select_fun.append(lambda ts: getattr(ts, "get_subset")(index, dim))
        return select_fun

    def select(self, time_series, select_fun=None, **kwargs):
        if select_fun is None:
            select_fun = self._compile_select_fun(**kwargs)
        for fun in select_fun:
            time_series = fun(time_series)
        return time_series, select_fun

    def concatenate(self, time_series_list, dim, **kwargs):
        time_series_list = ensure_list(time_series_list)
        n_ts = len(time_series_list)
        if n_ts > 0:
            out_time_series, select_fun = self.select(time_series_list[0], **kwargs)
            if n_ts > 1:
                for id, time_series in enumerate(time_series_list[1:]):
                    if np.float32(out_time_series.sample_period) != np.float32(time_series.sample_period):
                        raise_value_error("Timeseries concatenation failed!\n"
                                          "Timeseries %d have a different time step (%s) \n "
                                          "than the concatenated ones (%s)!" %
                                          (id, str(np.float32(time_series.sample_period)),
                                           str(np.float32(out_time_series.sample_period))))
                    else:
                        time_series = self.select(time_series, select_fun)[0]
                        try:
                            out_time_series.set_data(np.concatenate([out_time_series.data, time_series.data], axis=dim))
                            if len(out_time_series.labels_dimensions[out_time_series.labels_ordering[dim]]) > 0:
                                dim_label = out_time_series.labels_ordering[dim]
                                out_time_series.labels_dimensions[dim_label] = \
                                    np.array(ensure_list(out_time_series.labels_dimensions[dim_label]) +
                                             ensure_list(time_series.labels_dimensions[dim_label]))
                        except:
                            raise_value_error("Timeseries concatenation failed!\n"
                                              "Timeseries %d have a shape (%s) and the concatenated ones (%s)!" %
                                              (id, str(out_time_series.shape), str(time_series.shape)))
                return out_time_series
            else:
                return out_time_series
        else:
            raise_value_error("Cannot concatenate empty list of TimeSeries!")

    def concatenate_in_time(self, time_series_list, **kwargs):
        return self.concatenate(time_series_list, 0, **kwargs)

    def concatenate_variables(self, time_series_list, **kwargs):
        return self.concatenate(time_series_list, 1, **kwargs)

    def concatenate_in_space(self, time_series_list, **kwargs):
        return self.concatenate(time_series_list, 2, **kwargs)

    def concatenate_samples(self, time_series_list, **kwargs):
        return self.concatenate(time_series_list, 3, **kwargs)

    def select_by_metric(self, time_series, metric, metric_th=None, metric_percentile=None, nvals=None):
        selection = np.unique(select_greater_values_array_inds(metric, metric_th, metric_percentile, nvals))
        return time_series.get_subspace_by_index(selection), selection

    def select_by_power(self, time_series, power=np.array([]), power_th=None):
        if len(power) != time_series.number_of_labels:
            power = self.power(time_series)
        return self.select_by_metric(time_series, power, power_th)

    def select_by_hierarchical_group_metric_clustering(self, time_series, distance, disconnectivity=np.array([]),
                                                       metric=None, n_groups=10, members_per_group=1):
        selection = np.unique(select_by_hierarchical_group_metric_clustering(distance, disconnectivity, metric,
                                                                             n_groups, members_per_group))
        return time_series.get_subspace_by_index(selection), selection

    def select_by_correlation_power(self, time_series, correlation=np.array([]), disconnectivity=np.array([]),
                                    power=np.array([]), n_groups=10, members_per_group=1):
        if correlation.shape[0] != time_series.number_of_labels:
            correlation = self.correlation(time_series)
        if len(power) != time_series.number_of_labels:
            power = self.power(time_series)
        return self.select_by_hierarchical_group_metric_clustering(time_series, 1 - correlation,
                                                                   disconnectivity, power, n_groups, members_per_group)

    def select_by_projection_power(self, time_series, projection=np.array([]),
                                   disconnectivity=np.array([]), power=np.array([]),
                                   n_groups=10, members_per_group=1):
        if len(power) != time_series.number_of_labels:
            power = self.power(time_series)
        return self.select_by_hierarchical_group_metric_clustering(time_series, 1 - np.corrcoef(projection),
                                                                   disconnectivity, power, n_groups, members_per_group)

    def select_by_rois_proximity(self, time_series, proximity, proximity_th=None, percentile=None, n_signals=None):
        initial_selection = range(time_series.number_of_labels)
        selection = []
        for prox in proximity:
            selection += (
                np.array(initial_selection)[select_greater_values_array_inds(prox, proximity_th,
                                                                             percentile, n_signals)]).tolist()
        selection = np.unique(selection)
        return time_series.get_subspace_by_index(selection), selection

    def select_by_rois(self, time_series, rois, all_labels):
        for ir, roi in rois:
            if not (isinstance(roi, string_types)):
                rois[ir] = all_labels[roi]
        return time_series.get_subspace_by_labels(rois), rois

    def compute_seeg(self, source_time_series, sensors, projection=None, sum_mode="lin", **kwargs):
        if np.all(sum_mode == "exp"):
            seeg_fun = lambda source, projection_data: self.compute_seeg_exp(source.squeezed, projection_data)
        else:
            seeg_fun = lambda source, projection_data: self.compute_seeg_lin(source.squeezed, projection_data)
        labels_ordering = LABELS_ORDERING
        labels_ordering[1] = "SEEG"
        labels_ordering[2] = "SEEG Sensor"
        kwargs.update({"labels_ordering": labels_ordering,
                       "start_time": source_time_series.start_time,
                       "sample_period": source_time_series.sample_period,
                       "sample_period_unit": source_time_series.sample_period_unit})
        if isinstance(sensors, dict):
            seeg = OrderedDict()
            for sensor, projection in sensors.items():
                kwargs.update({"labels_dimensions": {labels_ordering[2]: sensor.labels,
                                                     labels_ordering[1]: [sensor.name]},
                               "sensors": sensor})
                seeg[sensor.name] = \
                    source_time_series.__class__(
                        np.expand_dims(seeg_fun(source_time_series, projection.projection_data), 1), **kwargs)
            return seeg
        else:
            kwargs.update({"labels_dimensions": {labels_ordering[2]: sensors.labels,
                                                 labels_ordering[1]: [sensors.name]},
                           "sensors": sensors})
            return TimeSeriesSEEG(
                np.expand_dims(seeg_fun(source_time_series, projection.projection_data), 1), **kwargs)

    def compute_seeg_lin(self, source_time_series, projection_data):
        return source_time_series.dot(projection_data.T)

    def compute_seeg_exp(self, source_time_series, projection_data):
        return np.log(np.exp(source_time_series).dot(projection_data.T))
