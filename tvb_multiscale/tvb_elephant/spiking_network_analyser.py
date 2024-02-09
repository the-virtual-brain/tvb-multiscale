# -*- coding: utf-8 -*-

from six import string_types
from copy import deepcopy

import numpy as np
from xarray import DataArray

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.data_analysis.spiking_network_analyser_base import SpikingNetworkAnalyserBase

from tvb.basic.neotraits.api import Attr, Float


LOG = initialize_logger(__name__)


class SpikingNetworkAnalyser(SpikingNetworkAnalyserBase):

    """SpikingNetworkAnalyserBase
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

    _spikes_train_name = None

    _binned_spikes_trains_name = None

    def compute_spikes_train(self, spikes):
        """Method to compute a neo.core.SpikesTrain data type from an array of spikes" times.
           Arguments:
            - spikes: an array of spikes' times.
           Returns:
            - the neo.core.SpikeTrain.
        """
        from neo.core import SpikeTrain
        from quantities import ms
        if len(spikes):
            t_stop = np.max(spikes) + self._fmin_resolution
        elif self.end_time:
            t_stop = self.end_time
        else:
            t_stop = self.period
        return SpikeTrain(spikes*ms, t_stop=t_stop*ms)

    @property
    def spikes_train_name(self):
        """A method to set and return the name of the Spikes Train data type."""
        if not self._spikes_train_name:
            self._spikes_train_name = self._get_comput_res_type(self.compute_spikes_train)
        return self._spikes_train_name

    def _assert_spikes_train(self, spikes):
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
        from elephant.utils import get_common_start_stop_times
        for i_spike_train, spikes_train, in enumerate(spikes_trains):
            spikes_trains[i_spike_train] = self._assert_spikes_train(spikes_train)
        t_start, t_stop = get_common_start_stop_times(spikes_trains)
        if binsize is not None:
            binsize = float(binsize) * ms
            num_bins = None
        elif num_bins is None:
            binsize = self.period * ms
        return BinnedSpikeTrain(spikes_trains, bin_size=binsize, t_start=t_start, t_stop=t_stop)

    @property
    def binned_spikes_trains_name(self):
        """A method to set and return the name of the Binned Spikes Train data type."""
        if not self._binned_spikes_trains_name:
            self._binned_spikes_trains_name = self._get_comput_res_type(self.compute_binned_spikes_trains)
        return self._binned_spikes_trains_name

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

    def compute_elephant_mean_firing_rate(self, spikes_times, res_type, **elephant_kwargs):
        """A method to compute mean (across time) rate from an input of spikes' events or spikes' times
           using the elephant.statistics.mean_firing_rate method.
           Arguments:
            - spikes_times: a neo.core.SpikeTrain or
                            an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - res_type: the name of the result
            - **elephant_kwargs: kwargs to the elephant mean firing rate function
           Returns:
            - the mean rate (float)
            - the neo.core.SpikeTrain used for the computation
        """
        from quantities import ms
        from elephant.statistics import mean_firing_rate
        t_start, t_stop = self._assert_start_end_times_from_spikes_times(spikes_times)
        spikes_train = self._assert_spikes_train(spikes_times)
        elephant_kwargs["t_start"] = elephant_kwargs.get("t_start", t_start-self._fmin_resolution) * ms
        elephant_kwargs["t_stop"] = elephant_kwargs.get("t_stop", t_stop+self._fmin_resolution) * ms
        if len(spikes_train):
            result = mean_firing_rate(spikes_train, **elephant_kwargs)
        else:
            result = 0.0
        return {res_type: DataArray([1000 * float(result)]).squeeze(),
                self.spikes_train_name: spikes_train}

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
        res_type = self._get_comput_res_type()
        spikes_times = self._get_spikes_times_from_spikes_events(spikes)
        if self.elephant_mean_firing_rate:
            return self.compute_elephant_mean_firing_rate(spikes_times, res_type, **kwargs)
        else:
            return super(SpikingNetworkAnalyser, self)._compute_rate_base(spikes_times, res_type,
                                                                          number_of_neurons, duration, **kwargs)

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
        t_start, t_stop = self._assert_start_end_times_from_spikes_times(spikes_train)
        # ...the time vector
        time = np.arange(t_start, t_stop+self._fmin_resolution, self.period)
        # ...and that the input spikes are in a neo.core.SpikeTrain instance
        spikes_train = self._assert_spikes_train(spikes_train)
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
            kwargs["t_start"] = elephant_kwargs.get("t_start", t_start-self._fmin_resolution)
            kwargs["t_stop"] = elephant_kwargs.get("t_stop", t_stop+self._fmin_resolution)
            kwargs["kernel"] = spikes_kernel
            try:
                # Call the elephant method for the actual computation
                rates = instantaneous_rate(spikes_train, self.period * ms, **kwargs).flatten().__array__()
            except:
                # If it fails, compute a delta spike instantaneous rate with any kernel smoothing:
                # LOG.warning("Failed to compute instantaneous rate with a sliding timing window!\n"
                #             "Computing instantaneous rate without any smoothing...")
                rates = 1000 * self._compute_delta_rate(time, spikes_train.__array__(), t_start, t_stop)
        else:
            rates = 0.0 * time
        # TODO: A better solution for handling the last time point with elephant!
        try:
            return {res_type: DataArray(rates, dims=["Time"], coords={"Time": time}),
                    self.spikes_train_name: spikes_train}
        except Exception as e:
            LOG.warning("Failed with exception %s!\n"
                        "Length of time is %d and of rates is %d.\n"
                        "Removing last time point and retrying!" % (str(e), len(time), len(rates)))
            return {res_type: DataArray(rates, dims=["Time"], coords={"Time": time[:-1]}),
                    self.spikes_train_name: spikes_train}

    def compute_spikes_rates_by_neuron(self, spikes, number_of_neurons=1, rate_method=None, **kwargs):
        """A method to compute any type of spiking rate, but separately for each neuron.
           Arguments:
            - spikes: a Spike Train instance or
                      an array of spikes' times
                      or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            -rate_method: Default=None, in which case the instantaneous rate time series for each neuron are computed
            **kwargs: keyword arguments for the method that computes the rate
           Returns:
            - a dictionary of the following key-value pair:
             result_name: a xarray.DataArray of the result of dimensions (Neuron, ) or (Time, Neuron)
        """
        if rate_method is None:
            rate_method = self.compute_rate_time_series
        return super(SpikingNetworkAnalyser, self).compute_spikes_rates_by_neuron(spikes, number_of_neurons,
                                                                                  rate_method, **kwargs)

    def compute_mean_rate_time_series(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute populations' mean instantaneous spiking rate time series.
           After the rate time series of the whole population is computed,
           it is divided by the number of neurons.
           Arguments:
            - spikes: a Spike Train instance or
                      an array of spikes' times
                      or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            **kwargs: keyword arguments for the method that computes the rate
           Returns:
            - a dictionary of the following key-value pair(s):
            "rate_time_series": a xarray.DataArray of dimensions (Time,)
            "spikes_train": the Spike Train used for the computation
        """
        res_type = self._get_comput_res_type()
        res2_type = self._get_comput_res_type(self.compute_rate_time_series)
        results = self.compute_rate_time_series(spikes, **kwargs)
        results[res_type] = results[res2_type] / number_of_neurons
        del results[res2_type]
        return results

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
