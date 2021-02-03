# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, Float, NArray


class Base(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract Base Transformer class comprising:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - an abstract method for the computations applied 
              upon the input buffer data for the output buffer data to result.
    """

    input_buffer = NArray(
        label="Input buffer",
        doc="""Buffer of incoming data.""",
        required=True,
        default=np.array([])
    )

    output_buffer = NArray(
        label="Output buffer",
        doc="""Buffer of outgoing data.""",
        required=True,
        default=np.array([])
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    input_time = NArray(
        label="Input time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the input buffer.""",
        required=True,
        default=np.array([])
    )

    output_time = NArray(
        label="Output time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the output bufer.""",
        required=True,
        default=np.array([])
    )

    def compute_time(self):
        self.output_time = np.copy(self.input_time)

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        pass

    def __call__(self):
        self.compute_time()
        self.compute()

    def configure(self):
        if self.receiver:
            self.receiver.configure()
        if self.sender:
            self.sender.configure()
        super(Base, self).configure()

    def print_str(self):
        output = "\n%s, dt = %g" % (self.__repr__(), self.dt)
        if self.receiver:
            output += "\nReceiver: %s" % str(self.receiver)
        if self.sender:
            output += "\nSender: %s" % str(self.sender)


# A few basic examples:


class Elementary(Base):
    """
        Elementary Transformer just copies the input to the output without any computation.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a method to copy the input buffer data to the output buffer.
    """

    def compute(self):
        """Method that just copies input buffer data to the output buffer"""
        self.output_buffer = np.copy(self.input_buffer)


class Scale(Base):
    """
        Scale Transformer scales the input with a scale factor in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a scale factor numpy.array,
            - a method to multiply the input buffer data by the scale factor for the output buffer data to result.
    """

    scale_factor = NArray(
        label="Scale factor",
        doc="""Array to scale input buffer.""",
        required=True,
        default=np.array([1.0])
    )

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = self.scale_factor * self.input_buffer


class DotProduct(Base):
    """
        DotProduct Transformer computes the dot product of the input with a scale factor
        in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a dot factor numpy.array,
            - a method to perform the left dot product upon the input buffer for the output buffer data to result.
    """

    dot_factor = NArray(
        label="Dot factor",
        doc="""Array to perform the left dot product upon the input buffer.""",
        required=True,
        default=np.array([1.0])
    )

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = np.dot(self.dot_factor * self.input_buffer)


class RatesToSpikes(Scale):
    __metaclass__ = ABCMeta

    """
        RatesToSpikes Transformer abstract base class
    """

    n_spiketrains = NArray(
        label="Number of spiketrains",
        doc="""Number of spiketrains to generate for each proxy node.""",
        required=True,
        default=np.array([1]).astype('i')
    )

    def configure(self):
        super(RatesToSpikesElephantPoissonInteraction, self).configure()

    def _assert_size(self, attr):
        value = getattr(self, attr).flatten()
        if self.input_buffer.shape[0] != value.shape[0]:
            if value.shape[0] == 1:
                value = np.repeat(value, self.input_buffer.shape[0])
            else:
                raise ValueError("%s (=%s) is neither of length 1 "
                                 "nor of length equal to the proxy dimension (1st) of the input buffer (=%d)"
                                 % (attr, str(value), self.input_buffer.shape[0]))
            setattr(self, attr, value)
        return value

    def _assert_n_spiketrains_size(self):
        return self._assert_size("n_spiketrains")

    def _assert_parameters_sizes(self):
        self._assert_n_spiketrains_size()

    @abstractmethod
    def _compute(self, rates, proxy_count, *args, **kwargs):
        """Abstract method for the computation of rates data transformation to spike trains."""
        pass

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        self._assert_parameters_sizes()
        self.output_buffer = []
        for iP, proxy_buffer in enumerate(self.scale_factor*self.input_buffer):
            self.output_buffer.append(self._compute(proxy_buffer, iP, *args, **kwargs))


class RatesToSpikesElephant(RatesToSpikes):
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

    @property
    def _t_start(self):
        return self.input_time[0] * self.ms

    @property
    def _t_stop(self):
        return self.input_time[-1] * self.ms

    def _rates_analog_signal(self, rates):
        return self._analog_signal_class(rates*self.rate_unit, sampling_period=self.dt*self.time_unit,
                                         t_start=self._t_start, t_stop=self._t_stop)
    @abstractmethod
    def _compute(self, rates, proxy_count, *args, **kwargs):
        """Abstract method for the computation of rates data transformation to spike trains, using elephant software."""
        pass


class RatesToSpikesElephantPoisson(RatesToSpikesElephant):
    """
        RatesToSpikesElephantPoisson Transformer class,
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
        depending on whether rate varies with time or not.
        This class can be used to produce independent spike trains per proxy node.
    """

    from elephant.spike_train_generation import homogeneous_poisson_process, inhomogeneous_poisson_process

    _spike_gen_fun_h = homogeneous_poisson_process
    _spike_gen_fun_inh = inhomogeneous_poisson_process

    refractory_period = Float(label="Refractory period",
                              doc="The time period after one spike no other spike is emitted. "
                                  "pq.Quantity scalar with dimension time. Default: None.",
                              required=False,
                              default=None)

    def _compute_for_n_spiketrains(self, rates, n_spiketrains):
        spiketrains = []
        if len(rates) > 1:
            this_rates = self._rates_analog_signal(rates)
            for _ in range(n_spiketrains):
                spiketrains.append(
                    self._spike_gen_fun_inh(this_rates, as_array=True, refractory_period=self.refractory_period))
        else:
            this_rates = rates * self.rate_unit
            for _ in range(n_spiketrains):
                spiketrains.append(self._spike_gen_fun_inh(this_rates, t_start=self._t_start, t_stop=self._t_stop,
                                                           as_array=True, refractory_period=self.refractory_period))
        return spiketrains

    def _compute(self, rates, proxy_count):
        """Method for the computation of rates data transformation to independent spike trains,
           using elephant (in)homogeneous_poisson_process functions."""
        return self._compute_for_n_spiketrains(rates, self.n_spiketrains[proxy_count])


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
                                "default = 1.0 / n_spiketrains.",
                            required=True,
                            default=np.array([]).astype('f')
                        )

    def _assert_correlation_factor_size(self):
        return self._assert_size("correlation_factor")

    def _assert_parameters_sizes(self):
        self._assert_n_spiketrains_size()
        self._assert_correlation_factor_size()

    def configure(self):
        super(RatesToSpikesElephantPoissonInteraction, self).configure()
        if self.correlation_factor.size == 0:
            self.correlation_factor = 1.0 / self.n_spiketrains
        else:
            assert np.all(0.0 < self.correlation_factor <= 1.0)

    def _compute_shared_spiketrain(self, rates, n_spiketrains, correlation_factor):
        rates = np.maximum(rates * n_spiketrains, 1e-12)  # avoid rates equal to zeros
        return super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                            rates * correlation_factor, 1)[0], \
               rates

    @abstractmethod
    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, *args):
        pass

    def _compute(self, rates, proxy_count):
        """Method for the computation of rates data transformation to interacting spike trains,
           using (in)homogeneous_poisson_process functions.
        """
        n_spiketrains = self.n_spiketrains[proxy_count]
        correlation_factor = self.correlation_factor[proxy_count]
        shared_spiketrain, rates = self._compute_shared_spiketrain(rates, n_spiketrains, correlation_factor)
        if self.correlation_factor[proxy_count] == 1.0:
            return np.array(shared_spiketrain.tolist() * n_spiketrains)
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
        We took it from https://github.com/multiscale-cosim/TVB-NEST
    """

    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, rates):
        spiketrains = \
            super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                        rates * (1 - correlation_factor), n_spiketrains)
        for iSP, spiketrain in enumerate(spiketrains):
            spiketrains[iSP] = np.around(np.sort(np.concatenate([spiketrain, shared_spiketrain]), decimals=1))
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

    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, *args):
        select = np.random.binomial(n=1, p=correlation_factor, size=(n_spiketrains, shared_spiketrain.shape[0]))
        spiketrains = []
        for spiketrain_mask in np.repeat([shared_spiketrain], n_spiketrains, axis=0)*select:
            spiketrains.append(spiketrain_mask[np.where(spiketrain_mask != 0)])
        return spiketrains


class SpikesToRates(Scale):
    __metaclass__ = ABCMeta

    """
        RateToSpikes Transformer abstract base class
    """

    @abstractmethod
    def _compute(self, spikes, *args, **kwargs):
        """Abstract method for the computation of spike trains data transformation
           to instantaneous mean spiking rates."""
        pass

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        self.output_buffer = []
        for proxy_buffer in self.input_buffer:  # At this point we assume that input_buffer has shape (proxy,)
            self.output_buffer.append(
                self._compute(proxy_buffer, *args, **kwargs))
        return self.scale_factor * np.array(self.output_buffer)


class SpikesToRatesElephant(SpikesToRates):
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
        return self.input_time[0] * self.ms

    @property
    def _t_stop(self):
        return self.input_time[-1] * self.ms

    def _spiketrain(self, spikes):
        return self._spike_train_class(spikes * self.time_unit, t_start=self._t_start, t_stop=self._t_stop)

    @abstractmethod
    def _compute(self, spikes, *args, **kwargs):
        """Abstract method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant software."""
        pass


class SpikesToRatesElephantHistogram(SpikesToRatesElephant):

    """
        SpikesToRatesElephantHistogram Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains.
    """

    from elephant.statistics import time_histogram

    _time_hist_fun = time_histogram

    output_type = Attr(label="Output type",
                       doc="""Output type with choices ('counts' (default), 'mean', 'rate')""",
                       field_type=str,
                       required=True,
                       default="counts",
                       choices=("counts", "mean", "rate"))

    def _compute_rate(self, spikes):
        return np.array(self._compute_time_histogram(spikes).rescale(self.rate_unit))

    def _compute_counts(self, spikes):
        return np.array(self._compute_time_histogram(spikes))

    def _compute_time_histogram(self, spiketrains):
        return self._time_hist_fun(spiketrains, self.dt, output=self.output_type)

    _compute_fun = _compute_counts

    def configure(self):
        super(SpikesToRatesElephant, self).configure()
        if self.output_type == "rate":
            self._compute_fun = self._compute_rate
        else:
            self._compute_fun = self._compute_counts

    def _compute(self, spikes):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        return self._compute_fun(self._spiketrain(spikes))


class SpikesToRatesElephantRate(SpikesToRatesElephantHistogram):
    """
        SpikesToRatesElephantRate Transformer class using the function instantaneous_rate of elephant software.
        The algorithm is based on convolution of spike trains with a kernel.
    """

    from elephant.statistics import instantaneous_rate
    from elephant.kernels import Kernel, GaussianKernel

    _rate_fun = instantaneous_rate

    _default_kernel_class = GaussianKernel

    kernel = Attr(label="Convolution kernel",
                  doc="""Convolution kernel, either "auto" or an elephants.kernels.Kernel.
                         Default: GaussianKernel(sigma=dt*ms)""",
                  field_type=(str, Kernel),
                  required=True,
                  default=None)

    def configure(self):
        if self.kernel is None:
            self.kernel = self._default_kernel_class(self.dt*self.time_unit)
        self.output_type = "rate"
        super(SpikesToRatesElephantRate, self).configure()

    def _compute(self, spikes, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.instantaneous_rate function."""
        spiketrain = self._spiketrain(spikes)
        if self.kernel != "auto" or spiketrain.size > 2:
            return np.array(self._rate_fun(spiketrain,  self.dt,
                                           kernel=self.kernel, *args, **kwargs).rescale(self.rate_unit))
        else:
            # If we have less than 3 spikes amd kernel="auto", we revert to time_histogram computation
            return SpikesToRatesElephantHistogram._compute_rate(spiketrain)


class RatesToSpikesTransformers(Enum):
    ELEPHANT_POISSON = RatesToSpikesElephantPoisson
    ELEPHANT_POISSON_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    ELEPHANT_POISSON_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction


class SpikesToRatesTransformers(Enum):
    ELEPHANT_HISTOGRAM = SpikesToRatesElephantHistogram
    ELEPHANT_RATE = SpikesToRatesElephantRate


class Transformers(Enum):
    ELEMENTARY = Elementary
    SCALE = Scale
    DOT_PRODUCT = DotProduct
    ELEPHANT_POISSON = RatesToSpikesElephantPoisson
    ELEPHANT_POISSON_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    ELEPHANT_POISSON_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction
    ELEPHANT_HISTOGRAM = SpikesToRatesElephantHistogram
    ELEPHANT_RATE = SpikesToRatesElephantRate
