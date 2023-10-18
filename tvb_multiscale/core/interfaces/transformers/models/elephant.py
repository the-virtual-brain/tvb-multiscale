# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from tvb.basic.neotraits._attr import Attr, Float, NArray

from tvb_multiscale.core.interfaces.transformers.models.base import RatesToSpikes, SpikesToRates, configure_transformer_with_ray


try:
    import ray
except:
    ray = None


class ElephantFunctions(Enum):
    from elephant.spike_train_generation import StationaryPoissonProcess, NonStationaryPoissonProcess
    from elephant.statistics import time_histogram
    from elephant.statistics import instantaneous_rate

    STATIONARY_POISSON_PROCESS = StationaryPoissonProcess
    NONSTATIONARY_POISSON_PROCESS = NonStationaryPoissonProcess
    TIME_HISTOGRAM = time_histogram
    INSTANTANEOUS_RATE = instantaneous_rate


def rate_analog_signal(rates, sampling_period, t_start, t_stop, rate_unit, analog_signal_class=None):
    if analog_signal_class:
        from neo import AnalogSignal
        analog_signal_class = AnalogSignal
    if isinstance(rates, analog_signal_class):
        return rates
    return analog_signal_class(rates*rate_unit, sampling_period=sampling_period, t_start=t_start, t_stop=t_stop)


class RatesToSpikesElephant(RatesToSpikes):

    """
        RatesToSpikesElephant Transformer base class, using elephant software.
    """

    from quantities import Quantity, ms, s, sec, second, Hz, MHz
    from neo import AnalogSignal

    _analog_signal_class = AnalogSignal

    time_unit = Attr(label="Time unit",
                     doc="""Time unit instance of quantities.Quantity class. 
                                Choices = (ms (default), s, sec, second)""",
                     field_type=Quantity,
                     required=True,
                     default=ms,
                     choices=(ms, s, sec, second))

    rate_unit = Attr(label="Rate unit",
                     doc="""Rate unit instance of quantities.Quantity class. 
                                Choices = (Hz (default), MHz)""",
                     field_type=Quantity,
                     required=True,
                     default=Hz,
                     choices=(Hz, MHz))

    def configure(self):
        super(RatesToSpikesElephant, self).configure()
        self.sampling_period = self.dt * self.time_unit

    def _rates_analog_signal(self, rates):
        return rate_analog_signal(rates, self.sampling_period, self._t_start, self._t_stop, self.rate_unit,
                                  analog_signal_class=self._analog_signal_class)


def compute_for_n_spiketrains(rates, number_of_spiketrains, refractory_period,
                              sampling_period, rate_unit, t_start, t_stop, as_array=True, analog_signal_class=None,
                              stationary_poisson_process=None, non_stationary_poisson_process=None):
    rates = np.maximum(rates, 1e-6)  # avoid rates equal to zeros
    spiketrains = list()
    for _ in range(number_of_spiketrains):
        spiketrains.append(np.array(list()))
    if len(rates) > 1:
        this_rates = rate_analog_signal(rates, sampling_period, t_start, t_stop, rate_unit, analog_signal_class)
        if non_stationary_poisson_process is None:
            non_stationary_poisson_process = \
                ElephantFunctions.NONSTATIONARY_POISSON_PROCESS.value(this_rates, refractory_period=refractory_period)
        else:
            non_stationary_poisson_process.rate_signal = this_rates
        spiketrains = non_stationary_poisson_process.generate_n_spiketrains(number_of_spiketrains, as_array=as_array)
    else:
        this_rates = rates * rate_unit
        if stationary_poisson_process is None:
            stationary_poisson_process = \
                ElephantFunctions.STATIONARY_POISSON_PROCESS.value(this_rates, t_start=t_start, t_stop=t_stop,
                                                                   refractory_period=refractory_period)
        else:
            stationary_poisson_process.this_rates = this_rates
            stationary_poisson_process.t_start = t_start
            stationary_poisson_process.t_stop = t_stop
            stationary_poisson_process.refractory_period = refractory_period
        spiketrains = stationary_poisson_process.generate_n_spiketrains(number_of_spiketrains, as_array=as_array)
    return spiketrains


if ray is not None:

    @ray.remote
    def ray_compute_for_n_spiketrains(rates, number_of_spiketrains, refractory_period, sampling_period,
                                      rate_unit, t_start, t_stop, as_array=True, analog_signal_class=None):
        return compute_for_n_spiketrains(rates, number_of_spiketrains, refractory_period,
                                         sampling_period, rate_unit, t_start, t_stop,
                                         as_array=as_array, analog_signal_class=analog_signal_class)


class RatesToSpikesElephantPoisson(RatesToSpikesElephant):
    """
        RatesToSpikesElephantPoisson Transformer class,
        using elephant functions NonStationaryPoissonProcess and StationaryPoissonProcess,
        depending on whether rate varies with time or not.
        This class can be used to produce independent spike trains per proxy node.
    """

    def __init__(self, **kwargs):
        super(RatesToSpikesElephantPoisson, self).__init__(**kwargs)
        self._stationary_poisson_process = None
        self._non_stationary_poisson_process = None
        self.__compute = None

    def _compute_spiketrains(self, rates, proxy_count):
        """Method for the computation of rates data transformation to independent spike trains,
           using elephant (Non)StationaryPoissonProcess functions."""
        return compute_for_n_spiketrains(rates, self._number_of_neurons[proxy_count],
                                         self.refractory_period, self.sampling_period,
                                         self.rate_unit, self._t_start, self._t_stop,
                                         as_array=self.as_array, analog_signal_class=self._analog_signal_class,
                                         stationary_poisson_process=self._stationary_poisson_process,
                                         non_stationary_poisson_process=self._non_stationary_poisson_process)

    def _compute_ray(self, rates):
        object_refs = []
        for iP, proxy_rate in enumerate(rates):
            object_refs.append(
                ray_compute_for_n_spiketrains.remote(proxy_rate, self._number_of_neurons[iP],
                                                     self.refractory_period, self.sampling_period,
                                                     self.rate_unit, self._t_start, self._t_stop,
                                                     as_array=self.as_array,
                                                     analog_signal_class=self._analog_signal_class))
        return list(ray.get(object_refs))


class RatesToSpikesElephantPoissonInteraction(RatesToSpikesElephantPoisson):

    """
        RatesToSpikesElephantPoissonInteraction Transformer base class,
        using elephant functions NonStationaryPoissonProcess and StationaryPoissonProcess,
        depending on whether rate varies with time or not.
        This class can be used to produce interacting spike trains per proxy node.
        The algorithm implemented is based on
        Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter.
        “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.”
        Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702.
        We took it from https://github.com/multiscale-cosim/TVB-NEST
    """

    correlation_factor = NArray(
                            label="Correlation factor",
                            doc="Correlation factor per proxy, array of floats in the interval (0, 1], "
                                "default = 1.0 / number_of_neurons.",
                            required=True,
                            default=np.array([0.0]).astype('f')
                        )

    @property
    def _correlation_factor(self):
        return self._assert_size("correlation_factor")

    def configure(self):
        super(RatesToSpikesElephantPoissonInteraction, self).configure()
        correlation_factor = self._correlation_factor.copy()
        inds = np.where(correlation_factor <= 0.0)
        correlation_factor[inds] = 1.0 / self._number_of_neurons[inds]
        self.correlation_factor = correlation_factor.copy()

    def _compute_shared_spiketrain(self, rates, number_of_spiketrains, correlation_factor):
        raise NotImplementedError

    def _compute_interaction_spiketrains(self, shared_spiketrain, number_of_spiketrains, correlation_factor, *args):
        raise NotImplementedError

    def _compute_spiketrains(self, rates, proxy_count):
        """Method for the computation of rates data transformation to interacting spike trains,
           using (Non)StationaryPoissonProcess functions.
        """
        n_spiketrains = self._number_of_neurons[proxy_count]
        correlation_factor = self._correlation_factor[proxy_count]
        shared_spiketrain = self._compute_shared_spiketrain(rates, n_spiketrains, correlation_factor)
        if correlation_factor == 1.0:
            return shared_spiketrain * n_spiketrains
        else:
            return self._compute_interaction_spiketrains(shared_spiketrain, n_spiketrains, correlation_factor, rates)


def compute_shared_spiketrain_single_interaction(rates, number_of_spiketrains, correlation_factor,
                                                 refractory_period, sampling_period, rate_unit, t_start, t_stop,
                                                 as_array=True, analog_signal_class=None,
                                                 stationary_poisson_process=None, non_stationary_poisson_process=None):
    return list(compute_for_n_spiketrains(rates * correlation_factor, 1, refractory_period,
                                          sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
                                          stationary_poisson_process, non_stationary_poisson_process)[0])


def compute_single_interaction_spiketrains(rates, shared_spiketrain, number_of_spiketrains, correlation_factor,
                                           refractory_period, sampling_period, rate_unit, t_start, t_stop,
                                           as_array=True, analog_signal_class=None,
                                           stationary_poisson_process=None, non_stationary_poisson_process=None):
    spiketrains = compute_for_n_spiketrains(rates * (1-correlation_factor), number_of_spiketrains, refractory_period,
                                            sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
                                            stationary_poisson_process, non_stationary_poisson_process)
    for iSP, spiketrain in enumerate(spiketrains):
        spiketrains[iSP] = np.sort(list(spiketrain) + shared_spiketrain)
    return spiketrains


def compute_for_n_spiketrains_single_interaction(rates, number_of_spiketrains, correlation_factor,
                                                 refractory_period, sampling_period, rate_unit, t_start, t_stop,
                                                 as_array=True, analog_signal_class=None,
                                                 stationary_poisson_process=None, non_stationary_poisson_process=None):
    shared_spiketrain = compute_shared_spiketrain_single_interaction(
        rates, number_of_spiketrains, correlation_factor,
        refractory_period, sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
        stationary_poisson_process, non_stationary_poisson_process)
    if correlation_factor == 1.0:
        return shared_spiketrain * n_spiketrains
    else:
        return compute_single_interaction_spiketrains(
            rates, shared_spiketrain, number_of_spiketrains, correlation_factor,
            refractory_period, sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
            stationary_poisson_process, non_stationary_poisson_process)


if ray is not None:

    @ray.remote
    def ray_compute_for_n_spiketrains_single_interaction(rates, number_of_spiketrains, correlation_factor,
                                                         refractory_period, sampling_period,
                                                         rate_unit, t_start, t_stop,
                                                         as_array=True, analog_signal_class=None):
        return compute_for_n_spiketrains_single_interaction(
            rates, number_of_spiketrains, correlation_factor,
            refractory_period, sampling_period, rate_unit, t_start, t_stop,
            as_array=as_array, analog_signal_class=analog_signal_class)


class RatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonInteraction):
    """
        RatesToSpikesElephantPoissonSingleInteraction Transformer class,
        using elephant functions NonStationaryPoissonProcess and StationaryPoissonProcess,
        depending on whether rate varies with time or not.
        This class can be used to produce interacting spike trains per proxy node with single interaction
        The single interaction algorithm implemented is based on
        Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter.
        “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.”
        Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702.
        We took it from https://github.com/multiscale-cosim/TVB-NEST-0
    """

    def __init__(self, **kwargs):
        super(RatesToSpikesElephantPoissonSingleInteraction, self).__init__(**kwargs)
        self.__compute = None

    def _compute_shared_spiketrain(self, rates, number_of_spiketrains, correlation_factor):
        return compute_shared_spiketrain_single_interaction(
            rates, number_of_spiketrains, correlation_factor,
            self.refractory_period, self.sampling_period, self.rate_unit, self._t_start, self._t_stop,
            as_array=self.as_array, analog_signal_class=self._analog_signal_class,
            stationary_poisson_process=self._stationary_poisson_process,
            non_stationary_poisson_process=self._non_stationary_poisson_process)

    def _compute_interaction_spiketrains(self, shared_spiketrain, number_of_spiketrains, correlation_factor, rates):
        return compute_single_interaction_spiketrains(
            rates, shared_spiketrain, number_of_spiketrains, correlation_factor,
            self.refractory_period, self.sampling_period, self.rate_unit, self._t_start, self._t_stop,
            as_array=self.as_array, analog_signal_class=self._analog_signal_class,
            stationary_poisson_process=self._stationary_poisson_process,
            non_stationary_poisson_process=self._non_stationary_poisson_process)

    def _compute_ray(self, rates):
        object_refs = []
        for iP, proxy_rate in enumerate(rates):
            object_refs.append(ray_compute_for_n_spiketrains_single_interaction.remote(
                                   proxy_rate, self._number_of_neurons[iP], self._correlation_factor[iP],
                                   self.refractory_period, self.sampling_period,
                                   self.rate_unit, self._t_start, self._t_stop,
                                   as_array=self.as_array, analog_signal_class=self._analog_signal_class))
        return list(ray.get(object_refs))


def compute_shared_spiketrain_multiple_interaction(rates, number_of_spiketrains, correlation_factor,
                                                   refractory_period, sampling_period, rate_unit, t_start, t_stop,
                                                   as_array=True, analog_signal_class=None,
                                                   stationary_poisson_process=None,
                                                   non_stationary_poisson_process=None):
    return list(compute_for_n_spiketrains(rates * correlation_factor * number_of_spiketrains, 1, refractory_period,
                                          sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
                                          stationary_poisson_process, non_stationary_poisson_process)[0])


def compute_multiple_interaction_spiketrains(shared_spiketrain, number_of_spiketrains, correlation_factor):
    select = np.random.binomial(n=1, p=correlation_factor, size=(number_of_spiketrains, len(shared_spiketrain)))
    spiketrains = []
    for spiketrain_mask in np.repeat([shared_spiketrain], number_of_spiketrains, axis=0) * select:
        spiketrains.append(np.sort(spiketrain_mask[np.where(spiketrain_mask != 0)]))
    return spiketrains


def compute_for_n_spiketrains_multiple_interaction(rates, number_of_spiketrains, correlation_factor,
                                                 refractory_period, sampling_period, rate_unit, t_start, t_stop,
                                                 as_array=True, analog_signal_class=None,
                                                 stationary_poisson_process=None, non_stationary_poisson_process=None):
    shared_spiketrain = compute_shared_spiketrain_multiple_interaction(
        rates, number_of_spiketrains, correlation_factor,
        refractory_period, sampling_period, rate_unit, t_start, t_stop, as_array, analog_signal_class,
        stationary_poisson_process, non_stationary_poisson_process)
    if correlation_factor == 1.0:
        return shared_spiketrain * n_spiketrains
    else:
        return compute_multiple_interaction_spiketrains(shared_spiketrain, number_of_spiketrains, correlation_factor)


if ray is not None:

    @ray.remote
    def ray_compute_for_n_spiketrains_multiple_interaction(rates, number_of_spiketrains, correlation_factor,
                                                           refractory_period, sampling_period,
                                                           rate_unit, t_start, t_stop,
                                                           as_array=True, analog_signal_class=None):
        return compute_for_n_spiketrains_multiple_interaction(
            rates, number_of_spiketrains, correlation_factor,
            refractory_period, sampling_period, rate_unit, t_start, t_stop,
            as_array=as_array, analog_signal_class=analog_signal_class)


class RatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonInteraction):
    """
        RatesToSpikesElephantPoissonSingleInteraction Transformer class,
        using elephant functions NonStationaryPoissonProcess and StationaryPoissonProcess,
        depending on whether rate varies with time or not.
        This class can be used to produce interacting spike trains per proxy node with multiple interaction.
        The multiple interaction algorithm implemented is based on
        Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter.
        “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.”
        Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702.
        We took it from https://github.com/multiscale-cosim/TVB-NEST
    """

    def __init__(self, **kwargs):
        super(RatesToSpikesElephantPoissonMultipleInteraction, self).__init__(**kwargs)
        self.__compute = None

    def _compute_shared_spiketrain(self, rates, number_of_spiketrains, correlation_factor):
        return compute_shared_spiketrain_multiple_interaction(
            rates, number_of_spiketrains, correlation_factor,
            self.refractory_period, self.sampling_period, self.rate_unit, self._t_start, self._t_stop,
            as_array=self.as_array, analog_signal_class=self._analog_signal_class,
            stationary_poisson_process=self._stationary_poisson_process,
            non_stationary_poisson_process=self._non_stationary_poisson_process)

    def _compute_interaction_spiketrains(self, shared_spiketrain, number_of_spiketrains, correlation_factor, *args):
        return compute_multiple_interaction_spiketrains(shared_spiketrain, number_of_spiketrains, correlation_factor)

    def _compute_ray(self, rates):
        object_refs = []
        for iP, proxy_rate in enumerate(rates):
            object_refs.append(ray_compute_for_n_spiketrains_multiple_interaction.remote(
                                    proxy_rate, self._number_of_neurons[iP], self._correlation_factor[iP],
                                    self.refractory_period, self.sampling_period,
                                    self.rate_unit, self._t_start, self._t_stop,
                                    as_array=self.as_array, analog_signal_class=self._analog_signal_class))
        return list(ray.get(object_refs))


def spiketrain(spikes, time_unit, t_start, t_stop, spike_train_class=None):
    if spike_train_class is None:
        from neo import SpikeTrain
        spike_train_class = SpikeTrain
    return spike_train_class(spikes.astype(np.float64) * time_unit, t_start=t_start, t_stop=t_stop)


class SpikesToRatesElephant(SpikesToRates):

    """
        RateToSpikes Transformer base class using elephant software
    """

    from quantities import Quantity, ms, s, sec, second, Hz, MHz
    from neo import SpikeTrain

    _spike_train_class = SpikeTrain

    time_unit = Attr(label="Time unit",
                     doc="""Time unit instance of quantities.Quantity class. 
                            Choices = (ms (default), s, sec, second)""",
                     field_type=Quantity,
                     required=True,
                     default=ms,
                     choices=(ms, s, sec, second))

    rate_unit = Attr(label="Rate unit",
                     doc="""Rate unit instance of quantities.Quantity class. 
                            Choices = (Hz (default), MHz)""",
                     field_type=Quantity,
                     required=True,
                     default=Hz,
                     choices=(Hz, MHz))

    def configure(self):
        super(SpikesToRatesElephant, self).configure()
        self.bin_size = self.dt * self.time_unit

    @property
    def _t_start(self):
        return (self.dt * (self.input_time[0] - 1) + self.time_shift - np.finfo(np.float32).resolution) * self.ms

    @property
    def _t_stop(self):
        return (self.dt * self.input_time[-1] + self.time_shift + np.finfo(np.float32).resolution) * self.ms

    def spiketrain(self, spikes):
        return spiketrain(spikes, self.time_unit, self._t_start, self._t_stop, self._spike_train_class)


def compute_rates_ElephantSpikesHistogram(spikes, bin_size, time_unit, t_start, t_stop, spike_train_class=None):
    return ElephantFunctions.TIME_HISTOGRAM([spiketrain(spikes, time_unit, t_start, t_stop,
                                                        spike_train_class=spike_train_class)],
                                            bin_size, output="counts")


if ray is not None:

    @ray.remote
    def ray_compute_rates_ElephantSpikesHistogram(spikes, scale_factor, translation_factor,
                                                  bin_size, time_unit, t_start, t_stop, spike_train_class=None):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return scale_factor \
                * np.array(
                    compute_rates_ElephantSpikesHistogram(
                        spikes, bin_size, time_unit, t_start, t_stop, spike_train_class).flatten()) \
                + translation_factor


class ElephantSpikesHistogram(SpikesToRatesElephant):

    """
        ElephantSpikesHistogram Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains.
    """

    def __init__(self, **kwargs):
        super(ElephantSpikesHistogram, self).__init__(**kwargs)
        self.__compute = None

    def _compute_rates(self, spikes):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return np.array(
                compute_rates_ElephantSpikesHistogram(spikes,  self.bin_size,
                                                      self.time_unit, self._t_start, self._t_stop,
                                                      spike_train_class=self._spike_train_class).flatten())

    def _compute_ray(self, input_buffer):
        object_refs = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(input_buffer, self._scale_factor, self._translation_factor):
            # At this point we assume that input_buffer has shape (proxy,)
            object_refs.append(ray_compute_rates_ElephantSpikesHistogram.remote(
                                    proxy_buffer, scale_factor, translation_factor,
                                    self.bin_size, self.time_unit, self._t_start, self._t_stop,
                                    spike_train_class=self._spike_train_class))
        return list(ray.get(object_refs))


def compute_rates_ElephantSpikesHistogramRate(spikes, bin_size,
                                              rate_unit, time_unit, t_start, t_stop, spike_train_class=None):
    return (compute_rates_ElephantSpikesHistogram(spikes, bin_size, time_unit, t_start, t_stop,
                                                  spike_train_class=spike_train_class)
            / bin_size).rescale(rate_unit)


if ray is not None:

    @ray.remote
    def ray_compute_rates_ElephantSpikesHistogramRate(spikes, scale_factor, translation_factor,
                                                      bin_size, rate_unit, time_unit, t_start, t_stop,
                                                      spike_train_class=None):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return scale_factor \
               * np.array(
                    compute_rates_ElephantSpikesHistogramRate(
                        spikes, bin_size, rate_unit, time_unit, t_start, t_stop,
                        spike_train_class=spike_train_class).flatten()) \
                       + translation_factor


class ElephantSpikesHistogramRate(ElephantSpikesHistogram):

    """
        ElephantSpikesHistogramRate Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains,
        and then dividing with the bin width.
    """

    def __init__(self, **kwargs):
        super(ElephantSpikesHistogramRate, self).__init__(**kwargs)
        self.__compute = None

    def _compute_rates(self, spikes):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return np.array(
            compute_rates_ElephantSpikesHistogramRate(spikes, self.bin_size,
                                                      self.rate_unit, self.time_unit, self._t_start, self._t_stop,
                                                      spike_train_class=self._spike_train_class).flatten())

    def _compute_ray(self, input_buffer):
        object_refs = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(input_buffer, self._scale_factor, self._translation_factor):
            # At this point we assume that input_buffer has shape (proxy,)
            object_refs.append(ray_compute_rates_ElephantSpikesHistogramRate.remote(
                                    proxy_buffer, scale_factor, translation_factor,
                                    self.bin_size, self.rate_unit, self.time_unit, self._t_start, self._t_stop,
                                    spike_train_class=self._spike_train_class))
        return list(ray.get(object_refs))


def compute_ElephantSpikesRate(spikes,  kernel, bin_size,
                               rate_unit, time_unit, t_start, t_stop, spike_train_class=None):
    spikestrain = spiketrain(spikes, time_unit, t_start, t_stop, spike_train_class=spike_train_class)
    if kernel != "auto" or spiketrain.size > 2:
        return np.array(
            ElephantFunctions.INSTANTANEOUS_RATE(
                spikestrain, bin_size, kernel).rescale(rate_unit))
    else:
        # If we have less than 3 spikes or kernel="auto", we revert to time_histogram computation
        return np.array((ElephantFunctions.TIME_HISTOGRAM([spikestrain], bin_size , output="counts"))
                        / bin_size).rescale(rate_unit)


if ray is not None:

    @ray.remote
    def ray_compute_ElephantSpikesRate(spikes, scale_factor, translation_factor,
                                       kernel, bin_size, rate_unit, time_unit, t_start, t_stop, spike_train_class=None):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return scale_factor \
                * np.array(
                    compute_ElephantSpikesRate(
                        spikes, kernel, bin_size,  rate_unit, time_unit, t_start, t_stop,
                        spike_train_class=spike_train_class).flatten()) \
               + translation_factor


class ElephantSpikesRate(ElephantSpikesHistogramRate):
    """
        ElephantSpikesRate Transformer class using the function instantaneous_rate of elephant software.
        The algorithm is based on convolution of spike trains with a kernel.
    """

    from elephant.kernels import Kernel, GaussianKernel

    _default_kernel_class = GaussianKernel
    _kernel_class = Kernel

    # kernel = Attr(label="Convolution kernel",
    #               doc="""Convolution kernel, either "auto" or an elephants.kernels.Kernel.
    #                      Default: GaussianKernel(sigma=dt*ms)""",
    #               field_type=Kernel,  # TODO: find a way to use here (str, Kernel)
    #               required=False,
    #               default=None)

    kernel = None

    def __init__(self, **kwargs):
        super(ElephantSpikesRate, self).__init__(**kwargs)
        self.kernel = None
        self.__compute = None

    def configure(self):
        super(ElephantSpikesRate, self).configure()
        # This is a temporary hack to go around the above problem with TVB traits' system:
        if self.kernel is None:
            self.kernel = self._default_kernel_class(self.dt * self.time_unit)
        assert self.kernel == "auto" or isinstance(self.kernel, self._kernel_class)
        self.output_type = "rate"

    def _compute_rates(self, spikes):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.instantaneous_rate function."""
        return np.array(
            compute_ElephantSpikesRate(spikes, self.kernel, self.bin_size, self.rate_unit,
                                       self.time_unit, self._t_start, self._t_stop,
                                       spike_train_class=self._spike_train_class).flatten())

    def _compute_ray(self, input_buffer):
        object_refs = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(input_buffer, self._scale_factor, self._translation_factor):
            # At this point we assume that input_buffer has shape (proxy,)
            object_refs.append(ray_compute_ElephantSpikesRate.remote(
                                    proxy_buffer, scale_factor, translation_factor,
                                    self.kernel, self.bin_size, self.rate_unit,
                                    self.time_unit, self._t_start, self._t_stop,
                                    spike_train_class=self._spike_train_class))
        return list(ray.get(object_refs))

    def info(self, recursive=0):
        info = super(ElephantSpikesRate, self).info(recursive=recursive)
        info['kernel'] = self.kernel
        return info


class ElephantRatesToSpikesTransformers(Enum):
    ELEPHANT_POISSON = RatesToSpikesElephantPoisson
    ELEPHANT_POISSON_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    ELEPHANT_POISSON_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction


class ElephantSpikesToRatesTransformers(Enum):
    ELEPHANT_HISTOGRAM = ElephantSpikesHistogram
    ELEPHANT_HISTOGRAM_RATE = ElephantSpikesHistogramRate
    ELEPHANT_RATE = ElephantSpikesRate
