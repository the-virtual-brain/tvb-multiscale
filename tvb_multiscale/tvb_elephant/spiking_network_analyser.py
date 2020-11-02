# -*- coding: utf-8 -*-

from six import string_types
from copy import deepcopy

import numpy as np
from xarray import DataArray

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.data_analysis.spiking_network_analyser \
    import SpikingNetworkAnalyser as SpikingNetworkAnalyzerBase

from tvb.basic.neotraits.api import Attr, Float


LOG = initialize_logger(__name__)


class SpikingNetworkAnalyser(SpikingNetworkAnalyzerBase):

    """SpikingNetworkAnalyser
       - gets data from the output devices of a SpikingNetwork,
       - or reads them from files,
       - performs computations of mean field quantities, including spikes_train' rates,
       - and returns the results in the desired output type
         (numpy.array, xarray.DataArray, TVB TimeSeries, pandas.Series of xarray.DataArray)
       """

    spikes_kernel = Attr(field_type=str, default="auto", required=True)

    spikes_kernel_width = Float(
        label="Kernel/window length",
        default=None,
        required=False,
        doc="""Kernel or sliding window time length (ms)""")

    elephant_mean_firing_rate = Attr(field_type=bool, default=True, required=True,
                                     label="elephant mean firing rate flag",
                                     doc="""If elephant_mean_firing_rate is True (default), 
                                            the elephant mean_firing_rate function is used.
                                            Otherwise, the mean_firing_rate is computed by 
                                            dividing the total number of spikes_train by the total number of neurons,
                                            without using elephant.""")

    def compute_spikes_train(self, spikes):
        from neo.core import SpikeTrain
        from quantities import ms
        t_stop = self.period*ms
        if len(spikes):
            t_stop = np.max(spikes)
            if self.end_time:
                t_stop = np.maximum(t_stop, self.end_time)
        return SpikeTrain(spikes*ms, t_stop=t_stop)

    def assert_spike_train(self, spikes):
        from neo.core import SpikeTrain
        if isinstance(spikes, SpikeTrain):
            return spikes
        else:
            return self.compute_spikes_train(spikes)

    def compute_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        from quantities import ms
        from elephant.conversion import BinnedSpikeTrain
        for i_spike_train, spikes_train, in enumerate(spikes_trains):
            spikes_train = self.assert_spike_train(spikes_train)
        if binsize or num_bins:
            if binsize:
                binsize = binsize * ms
            return BinnedSpikeTrain(spikes_train, binsize=binsize, num_bins=num_bins)
        else:
            return BinnedSpikeTrain(spikes_train, binsize=self.period * ms)

    def assert_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        from elephant.conversion import BinnedSpikeTrain
        if isinstance(spikes_trains, BinnedSpikeTrain):
            return spikes_trains
        else:
            return self.compute_binned_spikes_trains(spikes_trains, binsize, num_bins)

    def compute_elephant_mean_firing_rate(self, spikes_times, **elephant_kwargs):
        from quantities import ms
        from elephant.statistics import mean_firing_rate
        spikes_train = self.assert_spike_train(spikes_times)
        t_start, t_stop = self.assert_start_end_times_from_data_time(spikes_train)
        elephant_kwargs["t_start"] = elephant_kwargs.get("t_start", t_start)*ms
        elephant_kwargs["t_stop"] = elephant_kwargs.get("t_stop", t_stop)*ms
        if len(spikes_train):
            return mean_firing_rate(spikes_train, **elephant_kwargs), spikes_train
        else:
            return 0.0, spikes_train

    def compute_rate(self, spikes, number_of_neurons=1, name="", duration=None, **kwargs):
        if self.elephant_mean_firing_rate:
            res_type = self._get_comput_res_type()
            spikes_times = self._get_spikes_times_from_spikes_events(spikes)
            result, spikes_train = \
                    self.compute_elephant_mean_firing_rate(spikes_times, **kwargs)
            return {res_type: DataArray([float(result)], name=name).squeeze(),
                    self.spikes_train_name: spikes_train}
        else:
            return super(SpikingNetworkAnalyser, self).compute_rate(spikes, number_of_neurons,
                                                                    name, duration, **kwargs)

    def _compute_delta_rate(self, time, spikes_times):
        result = np.zeros(time.shape)
        for spike_time in np.unique(spikes_times):
            result[int(np.floor((spike_time - time[0]) / self.period))] += \
                np.sum(spikes_times == spike_time) / self.period
        return result

    def compute_rate_time_series(self, spikes_train, number_of_neurons=1, name="", **elephant_kwargs):
        res_type = self._get_comput_res_type()
        t_start, t_stop = self.assert_start_end_times_from_spikes_times(spikes_train)
        time = np.arange(t_start, t_stop+self.period, self.period)
        spikes_train = self.assert_spike_train(spikes_train)
        if  len(spikes_train) > 0:
            from quantities import ms
            from elephant.statistics import instantaneous_rate
            kwargs = deepcopy(elephant_kwargs)
            spikes_kernel = kwargs.pop("kernel", str(self.spikes_kernel).lower())
            if spikes_kernel != "auto":
                from elephant import kernels
                from elephant.statistics import optimal_kernel_bandwidth
                if not isinstance(spikes_kernel, kernels.Kernel):
                    # If it is note a Kernel instance, it has to be a Kernel module...
                    if isinstance(spikes_kernel, string_types):
                        # ...or a Kernel class name (string)
                        spikes_kernel = getattr(kernels, spikes_kernel + "Kernel")
                    if self.spikes_kernel_width:
                        spikes_kernel_width = self.spikes_kernel_width*ms
                    else:
                        spikes_kernel_width = optimal_kernel_bandwidth(np.sort(spikes_train.__array__()))["optw"]*ms
                    spikes_kernel = spikes_kernel(spikes_kernel_width*ms)
            kwargs["t_start"] = elephant_kwargs.get("t_start", t_start)*ms
            kwargs["t_stop"] = (elephant_kwargs.get("t_stop", t_stop) + 0.1)*ms
            kwargs["kernel"] = spikes_kernel
            try:
                rates = instantaneous_rate(spikes_train, self.period * ms, **kwargs).flatten().__array__()
            except:
                LOG.warning("Failed to compute instantaneous rate with a sliding timing window!\n"
                            "Computing instantaneous rate without any smoothing...")
                rates = self._compute_delta_rate(time, spikes_train.__array__())
        else:
            rates = 0.0 * time
        return {res_type: DataArray(rates, dims=["Time"], coords={"Time": time}, name=name),
                self.spikes_train_name: spikes_train}

    def compute_spikes_correlation_coefficient(self, binned_spikes_trains, binsize=None, num_bins=None, **kwargs):
        binned_spikes_trains = self.assert_binned_spikes_trains(binned_spikes_trains, binsize, num_bins)
        from elephant.spike_train_correlation import correlation_coefficient
        return {self._get_comput_res_type(): correlation_coefficient(binned_spikes_trains, **kwargs),
                self.binned_spikes_trains_name: binned_spikes_trains}

    def compute_spikes_covariance(self, binned_spikes_trains, binsize=None, num_bins=None, **kwargs):
        binned_spikes_trains = self.assert_binned_spikes_trains(binned_spikes_trains, binsize, num_bins)
        from elephant.spike_train_correlation import covariance
        return {self._get_comput_res_type(): covariance(binned_spikes_trains, **kwargs),
                self.binned_spikes_trains_name: binned_spikes_trains}
