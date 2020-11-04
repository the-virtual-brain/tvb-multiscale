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
       (TODO: - or reads them from files,)
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
        """Method to compute a neo.core.SpikesTrain data type from an array of spikes" times.
           Arguments:
            - spikes: an array of spikes' times.
           Returns:
            - the neo.core.SpikeTrain.
        """
        from neo.core import SpikeTrain
        from quantities import ms
        t_stop = self.period*ms
        if len(spikes):
            t_stop = np.max(spikes)
            if self.end_time:
                t_stop = np.maximum(t_stop, self.end_time)
        return SpikeTrain(spikes*ms, t_stop=t_stop)

    def _assert_spike_train(self, spikes):
        """A method to assert that an argument is of neo.core.SpikesTrain data type.
           If not, a neo.core.SpikesTrain is computed from the input spikes' times array.
           Arguments:
            - spikes: a neo.core.SpikesTrain or an array of spikes' times.
           Returns:
            - the neo.core.SpikesTrain.
        """
        from neo.core import SpikeTrain
        if isinstance(spikes, SpikeTrain):
            return spikes
        else:
            return self.compute_spikes_train(self._get_spikes_times_from_spikes_events(spikes))

    def compute_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        """Method to compute a elephant.conversion.BinnedSpikeTrain data type
            from a sequence (array, list, tuple) of arrays of spikes" times.
           Arguments:
            - spikes_trains: a sequence (array, list, tuple) of arrays of spikes" times
            - binsize: the size (float, in ms) of the bin to be used. Default=None.
            - num_bins: the number (integer > 0) of bins to be used. Default=None.
            If none of binsize or num_bins if given, a bin size equal to the sampling period is used.
           Returns:
            - the elephant.conversion.BinnedSpikeTrain instance.
        """
        from quantities import ms
        from elephant.conversion import BinnedSpikeTrain
        for i_spike_train, spikes_train, in enumerate(spikes_trains):
            spikes_trains[i_spike_train] = self._assert_spike_train(spikes_train)
        if binsize or num_bins:
            if binsize:
                binsize = binsize * ms
            return BinnedSpikeTrain(spikes_trains, binsize=binsize, num_bins=num_bins)
        else:
            return BinnedSpikeTrain(spikes_trains, binsize=self.period * ms)

    def _assert_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        """Method to assert that the input is of a elephant.conversion.BinnedSpikeTrain data type.
            If not it is computed from an input sequence (array, list, tuple)
            of Spikes Trains or of spikes' times arrays.
           Arguments:
            - binned_spikes_trains: an elephant.conversion.BinnedSpikeTrain instance, or a
                                    sequence (array, list, tuple) of Spikes Trains or of spikes' times arrays
            - binsize: the size (float, in ms) of the bin to be used. Default=None.
            - num_bins: the number (integer > 0) of bins to be used. Default=None.
            If none of binsize or num_bins if given, a bin size equal to the sampling period is used.
           Returns:
            - the elephant.conversion.BinnedSpikeTrain instance.
        """
        from elephant.conversion import BinnedSpikeTrain
        if isinstance(spikes_trains, BinnedSpikeTrain):
            return spikes_trains
        else:
            return self.compute_binned_spikes_trains(spikes_trains, binsize, num_bins)

    def compute_elephant_mean_firing_rate(self, spikes_times, **elephant_kwargs):
        """A method to compute mean (across time) rate from an input of spikes' events or spikes' times
           using the elephant.statistics.mean_firing_rate method.
           Arguments:
            - spikes: a neo.core.SpikeTrain or
                      an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            - duration: Default=None, in which case it is computed by start_time and end_time
           Returns:
            - the mean rate (float)
            - the neo.core.SpikeTrain used for the computation
        """
        from quantities import ms
        from elephant.statistics import mean_firing_rate
        spikes_train = self._assert_spike_train(spikes_times)
        t_start, t_stop = self._assert_start_end_times_from_data_time(spikes_train)
        elephant_kwargs["t_start"] = elephant_kwargs.get("t_start", t_start)*ms
        elephant_kwargs["t_stop"] = elephant_kwargs.get("t_stop", t_stop)*ms
        if len(spikes_train):
            return mean_firing_rate(spikes_train, **elephant_kwargs), spikes_train
        else:
            return 0.0, spikes_train

    def compute_rate(self, spikes, number_of_neurons=1, duration=None, **kwargs):
        """A method to compute rate from an input of spikes' events or spikes' times.
            If the class' flag elephant_mean_firing_rate is True,
            the elephant.statistics.mean_firing_rate method is used.
           Arguments:
            - spikes: a neo.core.SpikeTrain or
                      an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            - duration: Default=None, in which case it is computed by start_time and end_time
           Returns:
            - a dictionary of the following key-value pair(s):
             "rate": xarray.DataArray(rate).squeeze()
             and if elephant_mean_firing_rate=True:
             "spikes_train": the neo.core.SpikeTrain used for the computation
        """
        if self.elephant_mean_firing_rate:
            res_type = self._get_comput_res_type()
            spikes_times = self._get_spikes_times_from_spikes_events(spikes)
            result, spikes_train = \
                    self.compute_elephant_mean_firing_rate(spikes_times, **kwargs)
            return {res_type: DataArray([float(result)]).squeeze(),
                    self.spikes_train_name: spikes_train}
        else:
            return super(SpikingNetworkAnalyser, self).compute_rate(spikes, number_of_neurons, duration, **kwargs)

    def _compute_delta_rate(self, time, spikes_times):
        """A method to compute instantaneous spiking rate when the kernel method fails
           because there are not enough distinct bins, as when there are not enough spikes,
           and/or all the spikes are concentrated in only one time bin.
           The spikes contribute an equal weight of 1 / sampling period,
           to the corresponding sampling period time bin.
           Arguments:
            - time: the array of the time vector
            - spikes_times: an array of spikes' times
            Returns:
             - an array of instantaneous rate time series
        """
        result = np.zeros(time.shape)
        for spike_time in np.unique(spikes_times):
            result[int(np.floor((spike_time - time[0]) / self.period))] += \
                np.sum(spikes_times == spike_time) / self.period
        return result

    def compute_rate_time_series(self, spikes_train, number_of_neurons=1, **elephant_kwargs):
        """A method to compute instantaneous spiking rate time series,
           using the elephant.statistics.instantaneous_rate method.
           Arguments:
            - spikes_train: a neo.core.SpikeTrain instance or
                            an array of spikes' times or
                            a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            **elephant_kwargs: keyword arguments for the elephant.statistics.instantaneous_rate method
           Returns:
            - a dictionary of the following key-value pair(s):
             "rate_time_series": a xarray.DataArray of dimensions (Time,)
             "spikes_train": the neo.core.SpikeTrain used for the computation
        """
        res_type = self._get_comput_res_type()
        # Assert/compute the computations' start time, end time...
        t_start, t_stop = self._assert_start_end_times_from_spikes_times(spikes_train)
        # ...the time vector
        time = np.arange(t_start, t_stop+self.period, self.period)
        # ...and that the input spikes are in a neo.core.SpikeTrain instance
        spikes_train = self._assert_spike_train(spikes_train)
        if len(spikes_train) > 0:
            from quantities import ms
            from elephant.statistics import instantaneous_rate
            kwargs = deepcopy(elephant_kwargs)
            # Prepare the kernel
            spikes_kernel = kwargs.pop("kernel", str(self.spikes_kernel).lower())
            if spikes_kernel != "auto":
                # If automatic computation of the spike kernel is not used...
                from elephant import kernels
                from elephant.statistics import optimal_kernel_bandwidth
                if not isinstance(spikes_kernel, kernels.Kernel):
                    # If it is note a Kernel instance, it has to be a Kernel module...
                    if isinstance(spikes_kernel, string_types):
                        # ...or a Kernel class name (string)
                        spikes_kernel = getattr(kernels, spikes_kernel + "Kernel")
                    # Set or compute in this case also the (optimal) kernel width sigma parameter.
                    if self.spikes_kernel_width:
                        spikes_kernel_width = self.spikes_kernel_width*ms
                    else:
                        spikes_kernel_width = optimal_kernel_bandwidth(np.sort(spikes_train.__array__()))["optw"]*ms
                    spikes_kernel = spikes_kernel(spikes_kernel_width*ms)
            kwargs["t_start"] = elephant_kwargs.get("t_start", t_start)*ms
            kwargs["t_stop"] = (elephant_kwargs.get("t_stop", t_stop) + 0.1)*ms
            kwargs["kernel"] = spikes_kernel
            try:
                # Call the elephant method for the actual computation
                rates = instantaneous_rate(spikes_train, self.period * ms, **kwargs).flatten().__array__()
            except:
                # If it fails, compute a delta spike instantaneous rate with any kernel smoothing:
                LOG.warning("Failed to compute instantaneous rate with a sliding timing window!\n"
                            "Computing instantaneous rate without any smoothing...")
                rates = self._compute_delta_rate(time, spikes_train.__array__())
        else:
            rates = 0.0 * time
        return {res_type: DataArray(rates, dims=["Time"], coords={"Time": time}),
                self.spikes_train_name: spikes_train}

    def compute_spikes_correlation_coefficient(self, binned_spikes_trains, binsize=None, num_bins=None, **kwargs):
        """A method to compute the correlation coefficients
           among the spikes' trains of a set of elephant.conversion.BinnedSpikeTrain,
           using the elephant.spike_train_correlation.correlation_coefficient method.
           - binned_spikes_trains: an elephant.conversion.BinnedSpikeTrain instance, or a
                                   sequence (array, list, tuple) of Spikes Trains or of spikes' times arrays
           - binsize: the size (float, in ms) of the bin to be used. Default=None.
            - num_bins: the number (integer > 0) of bins to be used. Default=None.
            If none of binsize or num_bins if given, a bin size equal to the sampling period is used.
           Returns:
            - a dictionary of the following key-value pair(s):
             "correlation_coefficient": correlation_coefficient array
             "binned_spikes_trains": the elephant.conversion.BinnedSpikeTrain instance used for the computation
        """
        binned_spikes_trains = self._assert_binned_spikes_trains(binned_spikes_trains, binsize, num_bins)
        from elephant.spike_train_correlation import correlation_coefficient
        return {self._get_comput_res_type(): correlation_coefficient(binned_spikes_trains, **kwargs),
                self.binned_spikes_trains_name: binned_spikes_trains}

    def compute_spikes_covariance(self, binned_spikes_trains, binsize=None, num_bins=None, **kwargs):
        """A method to compute the covariances
           among the spikes' trains of a set of elephant.conversion.BinnedSpikeTrain,
           using the lephant.spike_train_correlation.covariance method.
           - binned_spikes_trains: an elephant.conversion.BinnedSpikeTrain instance, or a
                                   sequence (array, list, tuple) of Spikes Trains or of spikes' times arrays
           - binsize: the size (float, in ms) of the bin to be used. Default=None.
            - num_bins: the number (integer > 0) of bins to be used. Default=None.
            If none of binsize or num_bins if given, a bin size equal to the sampling period is used.
           Returns:
            - a dictionary of the following key-value pair(s):
             "covariance": correlation_coefficient array
             "binned_spikes_trains": the elephant.conversion.BinnedSpikeTrain instance used for the computation
        """
        binned_spikes_trains = self._assert_binned_spikes_trains(binned_spikes_trains, binsize, num_bins)
        from elephant.spike_train_correlation import covariance
        return {self._get_comput_res_type(): covariance(binned_spikes_trains, **kwargs),
                self.binned_spikes_trains_name: binned_spikes_trains}
