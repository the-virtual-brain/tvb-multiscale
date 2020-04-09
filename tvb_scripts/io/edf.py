# coding=utf-8

import numpy as np
from tvb.simulator.plot.utils.data_structures_utils import ensure_string
from tvb.simulator.plot.utils.log_error_utils import initialize_logger

from tvb_scripts.datatypes.time_series import TimeSeries, TimeSeriesDimensions


def read_edf_with_mne(path, exclude_channels):
    from mne.io import read_raw_edf
    raw_data = read_raw_edf(path, preload=True, exclude=exclude_channels)
    channel_names = raw_data.ch_names
    data, times = raw_data[:, :]
    return data, times, channel_names


def read_edf_with_pyedflib(path, exclude_channels):
    import pyedflib
    f = pyedflib.EdfReader(path)
    channel_names = np.delete(f.getSignalLabels(), exclude_channels, axis=0)
    n = f.signals_in_file
    data = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        data[i, :] = f.readSignal(i)
    data = np.delete(data, exclude_channels, axis=0).T
    # Assuming uniform sample frequency:
    fs = np.mean(f.getSampleFrequencies())
    times = np.arange(data.shape[0]) / fs
    return data, times, channel_names


def read_edf(path, sensors, rois_selection=None, label_strip_fun=None, time_units="ms", exclude_channels=[]):
    logger = initialize_logger(__name__)

    logger.info("Reading empirical dataset from edf file...")
    try:
        data, times, channel_names = read_edf_with_mne(path, exclude_channels)
    except:
        logger.warn("Reading edf file with mne failed! Trying with pyEDFlib...")
        try:
            data, times, channel_names = read_edf_with_pyedflib(path, exclude_channels)
        except:
            logger.error("Failed to read edf file both with MNE and pyEDFlib!")

    if not callable(label_strip_fun):
        label_strip_fun = lambda label: label

    channel_names = [label_strip_fun(s) for s in channel_names]

    rois = []
    rois_inds = []
    rois_lbls = []
    if len(rois_selection) == 0:
        rois_selection = sensors.labels

    logger.info("Selecting target signals from dataset...")
    for sensor_ind, sensor_label in enumerate(sensors.labels):
        if sensor_label in rois_selection and sensor_label in channel_names:
            rois.append(channel_names.index(sensor_label))
            rois_inds.append(sensor_ind)
            rois_lbls.append(sensor_label)

    data = data[rois].T

    # Assuming that edf file time units is "sec"
    if ensure_string(time_units).find("ms") == 0:
        times = 1000 * times
    # sort_inds = np.argsort(rois_lbls)
    rois = np.array(rois)  # [sort_inds]
    rois_inds = np.array(rois_inds)  # [sort_inds]
    rois_lbls = np.array(rois_lbls)  # [sort_inds]
    # data = data[:, sort_inds]

    return data, times, rois, rois_inds, rois_lbls


def read_edf_to_Timeseries(path, sensors, rois_selection=None, label_strip_fun=None, time_unit="ms", **kwargs):
    data, times, rois, rois_inds, rois_lbls = \
        read_edf(path, sensors, rois_selection, label_strip_fun, time_unit)

    return TimeSeries(data, time=times, labels_dimensions={TimeSeriesDimensions.SPACE.value: rois_lbls},
                      sample_period=np.mean(np.diff(times)), sample_period_unit=time_unit, **kwargs)
