# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, ABC
from enum import Enum

import numpy as np

from tvb.basic.neotraits._attr import Attr, Float, NArray

from tvb_multiscale.core.interfaces.transformers.models.base import RatesToSpikes, SpikesToRates


class ElephantFunctions(Enum):
    from elephant.spike_train_generation import homogeneous_poisson_process, inhomogeneous_poisson_process
    from elephant.statistics import time_histogram
    from elephant.statistics import instantaneous_rate

    HOMOGENEOUS_POISSON_PROCESS = homogeneous_poisson_process
    INHOMOGENEOUS_POISSON_PROCESS = inhomogeneous_poisson_process
    TIME_HISTOGRAM = time_histogram
    INSTANTANEOUS_RATE = instantaneous_rate


class RatesToSpikesElephant(RatesToSpikes, ABC):
    __metaclass__ = ABCMeta

    """
        RatesToSpikesElephant Transformer abstract base class, using elephant software.
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

    def _rates_analog_signal(self, rates):
        return self._analog_signal_class(rates*self.rate_unit, sampling_period=self.dt*self.time_unit,
                                         t_start=self._t_start, t_stop=self._t_stop)


class RatesToSpikesElephantPoisson(RatesToSpikesElephant):
    """
        RatesToSpikesElephantPoisson Transformer class,
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
        depending on whether rate varies with time or not.
        This class can be used to produce independent spike trains per proxy node.
    """

    refractory_period = Float(label="Refractory period",
                              doc="The time period after one spike no other spike is emitted. "
                                  "pq.Quantity scalar with dimension time. Default: None.",
                              required=False,
                              default=None)

    @staticmethod
    def _spike_gen_fun_h(*args, **kwargs):
        return ElephantFunctions.HOMOGENEOUS_POISSON_PROCESS(*args, **kwargs)

    @staticmethod
    def _spike_gen_fun_inh(*args, **kwargs):
        return ElephantFunctions.INHOMOGENEOUS_POISSON_PROCESS(*args, **kwargs)

    def _compute_for_n_spiketrains(self, rates, number_of_spiketrains):
        spiketrains = []
        if len(rates) > 1:
            this_rates = self._rates_analog_signal(rates)
            for iS in range(number_of_spiketrains):
                spiketrains.append(
                    self._spike_gen_fun_inh(
                        this_rates, as_array=True, refractory_period=self.refractory_period))
        else:
            this_rates = rates * self.rate_unit
            for iS in range(number_of_spiketrains):
                spiketrains.append(
                    self._spike_gen_fun_inh(this_rates,
                                            t_start=self._t_start, t_stop=self._t_stop,
                                            as_array=True, refractory_period=self.refractory_period))
        return spiketrains

    def _compute_spiketrains(self, rates, proxy_count):
        """Method for the computation of rates data transformation to independent spike trains,
           using elephant (in)homogeneous_poisson_process functions."""
        return self._compute_for_n_spiketrains(rates, self._number_of_neurons[proxy_count])


class RatesToSpikesElephantPoissonInteraction(RatesToSpikesElephantPoisson):
    __metaclass__ = ABCMeta

    """
        RatesToSpikesElephantPoissonInteraction Transformer abstract class, 
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
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

    @abstractmethod
    def _compute_shared_spiketrain(self, rates, n_spiketrains, correlation_factor):
        pass

    @abstractmethod
    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, *args):
        pass

    def _compute_spiketrains(self, rates, proxy_count):
        """Method for the computation of rates data transformation to interacting spike trains,
           using (in)homogeneous_poisson_process functions.
        """
        n_spiketrains = self._number_of_neurons[proxy_count]
        correlation_factor = self._correlation_factor[proxy_count]
        shared_spiketrain, rates = self._compute_shared_spiketrain(rates, n_spiketrains, correlation_factor)
        if correlation_factor == 1.0:
            return shared_spiketrain * n_spiketrains
        else:
            return self._compute_interaction_spiketrains(shared_spiketrain, n_spiketrains, correlation_factor, rates)


class RatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonInteraction):
    """
        RatesToSpikesElephantPoissonSingleInteraction Transformer class,
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
        depending on whether rate varies with time or not.
        This class can be used to produce interacting spike trains per proxy node with single interaction
        The single interaction algorithm implemented is based on
        Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter.
        “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.”
        Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702.
        We took it from https://github.com/multiscale-cosim/TVB-NEST-0
    """

    def _compute_shared_spiketrain(self, rates, n_spiketrains, correlation_factor):
        return super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                            rates * correlation_factor, 1)[0], \
               rates

    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, rates):
        spiketrains = \
            super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                        rates * (1 - correlation_factor), n_spiketrains)
        for iSP, spiketrain in enumerate(spiketrains):
            spiketrains[iSP] = np.sort(spiketrain + shared_spiketrain)
        return spiketrains


class RatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonInteraction):
    """
        RatesToSpikesElephantPoissonSingleInteraction Transformer class,
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
        depending on whether rate varies with time or not.
        This class can be used to produce interacting spike trains per proxy node with multiple interaction.
        The multiple interaction algorithm implemented is based on
        Kuhn, Alexandre, Ad Aertsen, and Stefan Rotter.
        “Higher-Order Statistics of Input Ensembles and the Response of Simple Model Neurons.”
        Neural Computation 15, no. 1 (January 2003): 67–101. https://doi.org/10.1162/089976603321043702.
        DOI: 10.1162/089976603321043702.
        We took it from https://github.com/multiscale-cosim/TVB-NEST
    """

    def _compute_shared_spiketrain(self, rates, n_spiketrains, correlation_factor):
        rates = np.maximum(rates * n_spiketrains, 1e-12)  # avoid rates equal to zeros
        return super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                            rates * correlation_factor, 1)[0], \
               rates

    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, *args):
        select = np.random.binomial(n=1, p=correlation_factor, size=(n_spiketrains, len(shared_spiketrain)))
        spiketrains = []
        for spiketrain_mask in np.repeat([shared_spiketrain], n_spiketrains, axis=0)*select:
            spiketrains.append(np.sort(spiketrain_mask[np.where(spiketrain_mask != 0)]))
        return spiketrains


class SpikesToRatesElephant(SpikesToRates, ABC):
    __metaclass__ = ABCMeta

    """
        RateToSpikes Transformer abstract base class using elephant software
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

    @property
    def _t_start(self):
        return (self.dt * (self.input_time[0] - 1) + self.time_shift - np.finfo(np.float32).resolution) * self.ms

    @property
    def _t_stop(self):
        return (self.dt * self.input_time[-1] + self.time_shift + np.finfo(np.float32).resolution) * self.ms

    def _spiketrain(self, spikes):
        return self._spike_train_class(spikes * self.time_unit, t_start=self._t_start, t_stop=self._t_stop)


class ElephantSpikesHistogram(SpikesToRatesElephant):

    """
        ElephantSpikesHistogram Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains.
    """

    @staticmethod
    def _time_hist_fun(*args, **kwargs):
        return ElephantFunctions.TIME_HISTOGRAM(*args, **kwargs)

    def _compute_fun(self, spiketrains):
        return self._time_hist_fun(spiketrains, self.dt * self.time_unit, output="counts").flatten()

    def _compute_rates(self, spikes, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return np.array(self._compute_fun(self._spiketrain(spikes)))


class ElephantSpikesHistogramRate(ElephantSpikesHistogram):

    """
        ElephantSpikesHistogramRate Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains,
        and then dividing with the bin width.
    """

    def _compute_fun(self, spiketrains):
        return (ElephantSpikesHistogram._compute_fun(self, spiketrains) /
                (self.dt * self.time_unit)).rescale(self.rate_unit).flatten()


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
        self.kernel = None
        super(ElephantSpikesRate, self).__init__(**kwargs)

    @staticmethod
    def _rate_fun(*args, **kwargs):
        return ElephantFunctions.INSTANTANEOUS_RATE(*args, **kwargs)

    def configure(self):
        # This is a temporary hack to go around the above problem with TVB traits' system:
        if self.kernel is None:
            self.kernel = self._default_kernel_class(self.dt*self.time_unit)
        assert self.kernel == "auto" or isinstance(self.kernel, self._kernel_class)
        self.output_type = "rate"
        super(ElephantSpikesRate, self).configure()

    def _compute_rates(self, spikes, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.instantaneous_rate function."""
        spikes = spikes.astype(np.float64)
        spiketrain = self._spiketrain(spikes)
        if self.kernel != "auto" or spiketrain.size > 2:
            data = np.array(
                self._rate_fun(spiketrain,
                               self.dt*self.time_unit, self.kernel, *args, **kwargs).rescale(self.rate_unit)).flatten()
            return data
        else:
            # If we have less than 3 spikes amd kernel="auto", we revert to time_histogram computation
            return np.array(ElephantSpikesHistogramRate._compute_fun(spiketrain).flatten())

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