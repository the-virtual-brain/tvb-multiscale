# -*- coding: utf-8 -*-

import os
import numpy as np
from pandas import Series
from xarray import DataArray

import quantities as pq
from elephant.spike_train_generation import compound_poisson_process
from elephant.statistics import time_histogram

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_utils.utils import initialize_logger


class SpikeStimulusBuilder(object):

    targets = ["spike"]
    number_of_targets = 1

    rate = 1.0 * pq.Hz
    A = None

    number_of_regions = 1
    number_of_target_regions = 1
    number_of_neurons = 1

    t_start = 0.0 * pq.ms
    dt = 0.1 * pq.ms
    time_length = 1000
    _t_stop = 1.0 * pq.ms

    shift = None

    _sparse = None
    _shape = (1, 1, 1, 1)
    _size = 1

    _configured = False

    _log = None

    def __init__(self, targets=["spike"], rate=10.0, A=None,
                 number_of_regions=1, number_of_neurons=1, target_regions=None,
                 t_start=0.0, dt=0.1, time_length=1000,
                 shift=None, sparse=None, squeeze=False,
                 logger=None):
        self.targets = ensure_list(targets)
        self.numper_of_targets = len(self.targets)
        if self.numper_of_targets < 1:
            raise ValueError("No target parameters have been provided!")
        self.rate = rate
        self.A = A
        self.number_of_regions = number_of_regions
        self.number_of_neurons = number_of_neurons
        self.target_regions = target_regions
        self.t_start = t_start * pq.ms
        self.dt = dt * pq.ms
        self.time_length = time_length
        self.shift = shift
        self._sparse = sparse
        self._shape = None
        self._squeeze = squeeze
        self._log = logger

    def configure(self):
        if self._log is None:
            self._log = initialize_logger(__name__, os.path.dirname(__file__))

        if self.target_regions is not None:
            self.number_of_target_regions = len(self.target_regions)
            _max_target_region = np.max(self.target_regions)
            if _max_target_region >= self.number_of_regions:
                self._log.warning("The maximum target regions' index %d must be smaller than the number of regions given %d!\n"
                                  "Setting number_of_regions=number_of_target_regions+1=%d"
                                  % (_max_target_region, self.number_of_regions, _max_target_region+1))
                self.number_of_regions = _max_target_region + 1
        else:
            self.target_regions = np.arange(self.number_of_regions)
            self.number_of_target_regions = self.number_of_regions

        if self.shift is not None:
            if not isinstance(self.shift, pq.unitquantity.UnitTime):
                self.shift *= pq.ms

        self._shape = (self.number_of_targets, self.number_of_target_regions, self.number_of_neurons)

        self._size = np.prod(self._shape)

        self.rate *= np.ones(self._shape)
        self.rate = self.rate.flatten()

        self._shape = tuple([self.time_length] + list(self._shape))

        if self.A is None:
            self.A = np.array([0.0, 1.0] + (self._size - 1) * [0.0])
        else:
            self.A = ensure_list(self.A)
            self.A += [0.0] * (self._size - len(self.A))

        self._t_stop = self.t_start + self.time_length * self.dt

        if self._sparse is None:
            if self.dt/1000 * np.mean(self.rate) * self.number_of_target_regions / self.number_of_regions < 0.5:
                self._sparse = True
            else:
                self._sparse = False

        self._configured = True

    def build(self):
        # Builder for TVB shaped spike stimulus to be read
        # for each target TVB parameter (e.g., "spike"), for time index t, as:
        # stimulus["spike"][t].to_xarray().values
        # of shape (number_of_regions, number_of_modes_or_neurons), unless squeezed

        if not self._configured:
            self.configure()

        spike_trains = compound_poisson_process(self.rate*pq.Hz, self.A,
                                                t_stop=self._t_stop, t_start=self.t_start, shift=self.shift)

        spike_ts = []
        for spike_train in spike_trains:
            spike_ts.append(
                np.array(
                    time_histogram([spike_train], self.dt, t_start=self.t_start, t_stop=self._t_stop)))

        spike_ts = np.array(spike_ts).swapaxes(0, 1).reshape(self._shape)

        if self.number_of_target_regions != self.number_of_regions:
            new_spike_ts = np.zeros((self.time_length, self.number_of_targets,
                                     self.number_of_regions, self.number_of_neurons))
            new_spike_ts[:, :, self.target_regions, :] = spike_ts
            spike_ts = new_spike_ts
            del new_spike_ts

        if self._squeeze:
            spike_ts = spike_ts.squeeze()
            if spike_ts.ndim < 2:
                spike_ts = spike_ts.newaxis(1)

        spike_ts = DataArray(spike_ts)

        stimulus = Series()
        if self._sparse:
            for it, target in enumerate(self.targets):
                stimulus[target] = spike_ts[it].to_series().to_sparse(fill_value=0.0)
        else:
            for it, target in enumerate(self.targets):
                stimulus[target] = spike_ts[it].to_series()

        return stimulus


if __name__ == "__main__":

    stb = SpikeStimulusBuilder(targets=["spike"], rate=10.0, A=None,
                               number_of_regions=100, number_of_neurons=1, target_regions=np.arange(0, 100, 10),
                               t_start=0.0, dt=0.1, time_length=1000, shift=None, sparse=None, squeeze=False)

    stimulus = stb.build()
    print(stimulus)



