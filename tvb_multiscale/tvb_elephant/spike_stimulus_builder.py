# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
import numpy as np
from pandas import Series
from xarray import DataArray

import quantities as pq
from elephant.spike_train_generation import \
    compound_poisson_process, homogeneous_poisson_process, inhomogeneous_poisson_process
from elephant.statistics import time_histogram
from neo import AnalogSignal

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.utils.log_utils import initialize_logger


def _assert_shape(x, y):
    x = np.array(ensure_list(np.array(x)))  # make sure rate is an array
    while x.ndim < y.ndim:
        x = x[:, None]
    return x*y


class SpikeStimulusBuilder(object):

    targets = ["spike"]
    number_of_targets = 1

    rate = 1.0  # in Hz or spikes/sec by default
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

    _return = "Dict"
    _return_array = "Numpy"

    _configured = False

    _log = None

    def __init__(self, targets=["spike"], rate=10.0, A=None,
                 number_of_regions=1, number_of_neurons=1, target_regions=None,
                 t_start=0.0, dt=0.1, time_length=1000,
                 shift=None, refractory_period=None,
                 sparse=None, squeeze=False, return_type="Dict", return_array_type="Numpy",
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
        self.refractory_period = refractory_period
        self._sparse = sparse
        self._shape = None
        self._squeeze = squeeze
        self._return = return_type
        self._return_array = return_array_type
        self._log = logger
        if self._log is None:
            self._log = initialize_logger(__name__, os.path.dirname(__file__))

    def _configure_target_shape(self):
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
        self._shape = (self.number_of_targets, self.number_of_target_regions, self.number_of_neurons)
        self._size = int(np.prod(self._shape))
        if self._sparse is None:
            if self.dt / 1000 * np.mean(self.rate) * self.number_of_target_regions / self.number_of_regions < 0.5:
                self._sparse = True
            else:
                self._sparse = False

    def _configure_time(self):
        self._t_stop = self.t_start + self.time_length * self.dt
        self._time_shape = tuple([self.time_length] + list(self._shape))

    def _configure(self):
        self._configure_target_shape()
        self._configure_time()

    def build_output(self, spike_trains):
        spike_ts = []
        for spike_train in spike_trains:
            spike_ts.append(
                np.array(
                    time_histogram([spike_train], self.dt, t_start=self.t_start, t_stop=self._t_stop)))

        #           trains x time -> time x trains -> time x targets x regions x neurons
        spike_ts = np.array(spike_ts).swapaxes(0, 1).reshape(self._time_shape)
        del spike_trains
        if self.number_of_target_regions != self.number_of_regions:
            new_spike_ts = np.zeros((self.time_length, self.number_of_targets,
                                     self.number_of_regions, self.number_of_neurons))
            new_spike_ts[:, :, self.target_regions, :] = spike_ts
            spike_ts = new_spike_ts
            del new_spike_ts
        if self._squeeze:
            spike_ts = spike_ts.squeeze()
            if spike_ts.ndim < 2:
                # Assuming that only time axis exists for sure:
                spike_ts = spike_ts[:, np.newaxis]
        # targets x time x ...
        spike_ts = spike_ts.swapaxes(0, 1)
        if self._return == "Series":
            stimulus = Series(dtype='object')
        else:
            stimulus = OrderedDict()
        if self._sparse or self._return_array == "DataArray":
            spike_ts = DataArray(spike_ts, dims=["Target", "Time", "Region", "Neuron"])
        if self._sparse:
            for it, target in enumerate(self.targets):
                stimulus[target] = spike_ts[it].to_series().to_sparse(fill_value=0.0)
        else:
            if self._return_array == "DataArray":
                for it, target in enumerate(self.targets):
                    stimulus[target] = spike_ts[it]
            else:
                for it, target in enumerate(self.targets):
                    stimulus[target] = spike_ts[it]
        del spike_ts
        return stimulus

    def configure_compound_poisson_process(self):
        self._configure()
        if self.shift is not None:
            if not isinstance(self.shift, pq.unitquantity.UnitTime):
                self.shift *= pq.ms
        self.rate = _assert_shape(self.rate, np.ones(self._shape))
        self.rate = self.rate.flatten()
        if self.A is None:
            self.A = np.array([0.0, 1.0] + (self._size - 1) * [0.0])
        else:
            self.A = ensure_list(self.A)
            self.A += [0.0] * (self._size - len(self.A))
        self._configured = True

    def build_compound_poisson_process(self):
        # Builder for TVB shaped spike stimulus to be read
        # for each target TVB parameter (e.g., "spike"), for time index t, as:
        # stimulus["spike"][t].to_xarray().values
        # of shape (number_of_regions, number_of_modes_or_neurons), unless squeezed
        if not self._configured:
            self.configure_compound_poisson_process()
        return self.build_output(
                        compound_poisson_process(self.rate * pq.Hz, self.A,
                                                 t_stop=self._t_stop, t_start=self.t_start, shift=self.shift))

    def configure_homogeneous_poisson_process(self):
        self._configure()
        if self.refractory_period is not None:
            if not isinstance(self.refractory_period, pq.unitquantity.UnitTime):
                self.refractory_period *= pq.ms
        self.rate = _assert_shape(self.rate, np.ones(self._shape))
        self.rate = self.rate.flatten()
        self._configured = True

    def build_homogeneous_poisson_process(self, ):
        if not self._configured:
            self.configure_homogeneous_poisson_process()
        spike_trains = []
        for ii in range(self._size):
            spike_trains.append(
                homogeneous_poisson_process(self.rate * pq.Hz, t_stop=self._t_stop, t_start=self.t_start,
                                            refractory_period=self.refractory_period))
        return self.build_output(spike_trains)

    def configure_inhomogeneous_poisson_process(self):
        self._configure()
        if self.refractory_period is not None:
            if not isinstance(self.refractory_period, pq.unitquantity.UnitTime):
                self.refractory_period *= pq.ms
        self.rate = _assert_shape(self.rate, np.ones(self._time_shape))
        self.rate = np.reshape(self.rate, (self.time_length, self._size))
        self.rate = AnalogSignal(self.rate*pq.Hz, sampling_period=self.dt, t_start=self.t_start)
        self._configured = True

    def build_inhomogeneous_poisson_process(self):
        if not self._configured:
            self.configure_inhomogeneous_poisson_process()
        spike_trains = []
        for ii in range(self._size):
            spike_trains.append(
                inhomogeneous_poisson_process(self.rate[:, ii], refractory_period=self.refractory_period))
        return self.build_output(spike_trains)


if __name__ == "__main__":

    stb = SpikeStimulusBuilder(targets=["spike"], rate=10.0, A=None, refractory_period=None, shift=None,
                               number_of_regions=100, number_of_neurons=1, target_regions=np.arange(0, 100, 10),
                               t_start=0.0, dt=0.1, time_length=1000, sparse=None, squeeze=False)

    stimulus = stb.build_compound_poisson_process()
    # stimulus = stb.build_homogeneous_poisson_process()
    # stimulus = stb.build_inhomogeneous_poisson_process()
    print(stimulus)




