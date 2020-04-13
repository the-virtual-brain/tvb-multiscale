# coding=utf-8
# Some math tools
from itertools import product
from sklearn.cluster import AgglomerativeClustering

import numpy as np

from tvb_scripts.config import FiguresConfig, CalculusConfig
from tvb_scripts.utils.log_error_utils import initialize_logger, warning
from tvb_scripts.utils.data_structures_utils import is_integer


logger = initialize_logger(__name__)


def normalize_weights(weights, percentile=CalculusConfig.WEIGHTS_NORM_PERCENT, remove_diagonal=True, ceil=1.0):
    # Create the normalized connectivity weights:
    if len(weights) > 0:
        normalized_w = np.array(weights)
        if remove_diagonal:
            # Remove diagonal elements
            n_regions = normalized_w.shape[0]
            normalized_w *= (1.0 - np.eye(n_regions))
        # Normalize with the 95th percentile
        normalized_w = np.array(normalized_w / np.percentile(normalized_w, percentile))
        if ceil:
            if ceil is True:
                ceil = 1.0
            normalized_w[normalized_w > ceil] = ceil
        return normalized_w
    else:
        return np.array([])


def spikes_events_to_time_index(spike_time, time):
    if spike_time < time[0] or spike_time > time[-1]:
        warning("Spike time is outside the input time vector!")
    return np.argmin(np.abs(time-spike_time))


def compute_spikes_counts(spikes_times, time):
    spikes_counts = np.zeros(time.shape)
    for spike_time in spikes_times:
        spikes_counts[spikes_events_to_time_index(spike_time, time)] += 1
    return spikes_counts


def spikes_rate_convolution(spike, spikes_kernel):
    if (spike != 0).any():
        if len(spikes_kernel) > 1:
            return np.convolve(spike, spikes_kernel, mode="same")
        else:
            return spike * spikes_kernel
    else:
        return np.zeros(spike.shape)
