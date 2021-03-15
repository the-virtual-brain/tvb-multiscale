# -*- coding: utf-8 -*-

from copy import deepcopy
from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, Float, NArray, List

from tvb_multiscale.core.utils.data_structures_utils import combine_enums


class Transformer(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract Transformer base class comprising:
            - an input buffer data,
            - an output buffer data,
            - an abstract method for the computations applied 
              upon the input buffer data for the output buffer data to result.
    """

    input_buffer = []

    output_buffer = []

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    input_time = NArray(
        dtype=np.int,
        label="Input time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the input buffer.""",
        required=True,
        default=np.array([]).astype("i")
    )

    output_time = NArray(
        dtype=np.int,
        label="Output time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the output bufer.""",
        required=True,
        default=np.array([]).astype("i")
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

    def _assert_size(self, attr, buffer="input", dim=0):
        value = getattr(self, attr)
        input_buffer_size = len(getattr(self, "%s_buffer" % buffer))
        if input_buffer_size != 0:
            if input_buffer_size != value.shape[dim]:
                if value.shape[dim] == 1:
                    value = np.repeat(value, input_buffer_size, axis=dim)
                else:
                    raise ValueError("%s (=%s) is neither of length 1 "
                                     "nor of length equal to the proxy dimension (1st) of the input buffer (=%d)"
                                     % (attr, str(value), input_buffer_size))
                setattr(self, attr, value)
        return value

    def print_str(self):
        output = "\n%s, dt = %g" % (self.__repr__(), self.dt)
        if self.receiver:
            output += "\nReceiver: %s" % str(self.receiver)
        if self.sender:
            output += "\nSender: %s" % str(self.sender)


# A few basic examples:


class Elementary(Transformer):
    """
        Elementary Transformer just copies the input to the output without any computation.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a method to copy the input buffer data to the output buffer.
    """

    def compute(self):
        """Method that just copies input buffer data to the output buffer"""
        self.output_buffer = deepcopy(self.input_buffer)


class Scale(Transformer):
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

    @property
    def _scale_factor(self):
        return self._assert_size("scale_factor")

    def configure(self):
        super(Scale, self).configure()
        self._scale_factor

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        if isinstance(self.input_buffer, np.ndarray):
            self.output_buffer = self.scale_factor * self.input_buffer
        else:
            self.output_buffer = []
            for scale_factor, input_buffer in zip(self._scale_factor, self.input_buffer):
                self.output_buffer.append(scale_factor * input_buffer)


class ScaleRate(Scale):

    """ScaleRate class that just scales mean field rates to spiking rates,
       including any unit conversions and conversions from mean field to total rates"""

    pass


class ScaleCurrent(Scale):
    """ScaleCurrent class that just scales mean field currents to spiking network ScaleCurrent,
       including any unit conversions and conversions from mean field to total rates"""

    pass


class DotProduct(Transformer):
    """
        DotProduct Transformer computes the dot product of the input with a scale factor
        in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a dot factor numpy.array,
            - a method to perform the left dot product upon the input buffer for the output buffer data to result.
    """

    input_buffer = NArray(
        label="Input buffer",
        doc="""Array to store temporarily the data to be transformed.""",
        required=True,
        default=np.array([])
    )

    output_buffer = NArray(
        label="Output buffer",
        doc="""Array to store temporarily the transformed data.""",
        required=True,
        default=np.array([])
    )

    dot_factor = NArray(
        label="Dot factor",
        doc="""Array to perform the left dot product upon the input buffer.""",
        required=True,
        default=np.array([[1.0]])
    )

    @property
    def _dot_factor(self):
        return self._assert_size("dot_factor", dim=1)

    def configure(self):
        super(DotProduct, self).configure()
        self._dot_factor

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = np.dot(self._dot_factor * self.input_buffer)


class ElephantFunctions(Enum):
    from elephant.spike_train_generation import homogeneous_poisson_process, inhomogeneous_poisson_process
    from elephant.statistics import time_histogram
    from elephant.statistics import instantaneous_rate

    HOMOGENEOUS_POISSON_PROCESS = homogeneous_poisson_process
    INHOMOGENEOUS_POISSON_PROCESS = inhomogeneous_poisson_process
    TIME_HISTOGRAM = time_histogram
    INSTANTANEOUS_RATE = instantaneous_rate


class RatesToSpikes(Scale):
    __metaclass__ = ABCMeta

    """
        RatesToSpikes Transformer abstract base class
    """

    input_buffer = NArray(
        label="Input buffer",
        doc="""Array to store temporarily the data to be transformed.""",
        required=True,
        default=np.array([])
    )

    output_buffer = List(
        of=list,
        doc="""List of spiketrains (lists) storing temporarily the generated spikes.""",
        default=(())
    )

    number_of_neurons = NArray(
        dtype="i",
        label="Number of neurons",
        doc="""Number of neuronal spiketrains to generate for each proxy node.""",
        required=True,
        default=np.array([1]).astype('i')
    )

    def configure(self):
        super(RatesToSpikes, self).configure()
        self._number_of_neurons

    @property
    def _number_of_neurons(self):
        return self._assert_size("number_of_neurons")

    @abstractmethod
    def _compute(self, rates, proxy_count, *args, **kwargs):
        """Abstract method for the computation of rates data transformation to spike trains."""
        pass

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        self.output_buffer = []
        for iP, (scale_factor, proxy_buffer) in enumerate(zip(self._scale_factor, self.input_buffer)):
            self.output_buffer.append(self._compute(scale_factor*proxy_buffer, iP, *args, **kwargs))


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
        return self.dt * self.input_time[0] * self.ms

    @property
    def _t_stop(self):
        return self.dt * self.input_time[-1] * self.ms

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
                        this_rates, as_array=True, refractory_period=self.refractory_period).tolist())
        else:
            this_rates = rates * self.rate_unit
            for iS in range(number_of_spiketrains):
                spiketrains.append(
                    self._spike_gen_fun_inh(this_rates,
                                            t_start=self._t_start, t_stop=self._t_stop,
                                            as_array=True, refractory_period=self.refractory_period).tolist())
        return spiketrains

    def _compute(self, rates, proxy_count):
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
                            default=np.array([]).astype('f')
                        )

    @property
    def _correlation_factor(self):
        return self._assert_size("correlation_factor")

    def configure(self):
        super(RatesToSpikesElephantPoissonInteraction, self).configure()
        if self.correlation_factor.size == 0:
            self.correlation_factor = 1.0 / self.number_of_neurons
        else:
            assert np.all(0.0 < self.correlation_factor <= 1.0)
        self._correlation_factor

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

    def _compute_interaction_spiketrains(self, shared_spiketrain, n_spiketrains, correlation_factor, rates):
        spiketrains = \
            super(RatesToSpikesElephantPoissonInteraction, self)._compute_for_n_spiketrains(
                                                                        rates * (1 - correlation_factor), n_spiketrains)
        for iSP, spiketrain in enumerate(spiketrains):
            spiketrains[iSP] = np.sort(spiketrain + shared_spiketrain).tolist()
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
        select = np.random.binomial(n=1, p=correlation_factor, size=(n_spiketrains, len(shared_spiketrain)))
        spiketrains = []
        for spiketrain_mask in np.repeat([shared_spiketrain], n_spiketrains, axis=0)*select:
            spiketrains.append(np.sort(spiketrain_mask[np.where(spiketrain_mask != 0)]).tolist())
        return spiketrains


class SpikesToRates(Scale):
    __metaclass__ = ABCMeta

    """
        RateToSpikes Transformer abstract base class
    """

    input_buffer = List(
        of=list,
        doc="""List of spiketrains (lists) storing temporarily the spikes to be transformed into rates.""",
        default=(())
    )

    output_buffer = NArray(
        label="Output buffer",
        doc="""Array to store temporarily the output rate data.""",
        required=True,
        default=np.array([])
    )

    @abstractmethod
    def _compute(self, spikes, *args, **kwargs):
        """Abstract method for the computation of spike trains data transformation
           to instantaneous mean spiking rates."""
        pass

    @property
    def _scale_factor(self):
        return self._assert_size("scale_factor", "output")

    def configure(self):
        super(SpikesToRates, self).configure()

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        output_buffer = []
        for scale_factor, proxy_buffer in zip(self._scale_factor, self.input_buffer):  # At this point we assume that input_buffer has shape (proxy,)
            output_buffer.append(scale_factor * self._compute(proxy_buffer, *args, **kwargs))
        self.output_buffer = np.array(output_buffer)
        return self.output_buffer


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
        return (self.dt * self.input_time[0] - np.finfo(np.float32).resolution) * self.ms

    @property
    def _t_stop(self):
        return (np.finfo(np.float32).resolution + self.dt * (self.input_time[-1] + 1)) * self.ms

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

    output_type = Attr(label="Output type",
                       doc="""Output type with choices ('counts' (default), 'mean', 'rate')""",
                       field_type=str,
                       required=True,
                       default="counts",
                       choices=("counts", "mean", "rate"))

    @staticmethod
    def _time_hist_fun(*args, **kwargs):
        return ElephantFunctions.TIME_HISTOGRAM(*args, **kwargs)

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

    @staticmethod
    def _rate_fun(*args, **kwargs):
        return ElephantFunctions.INSTANTANEOUS_RATE(*args, **kwargs)

    def configure(self):
        # This is a temporary hack to go around the above proble with TVB traits' system:
        if self.kernel is None:
            self.kernel = self._default_kernel_class(self.dt*self.time_unit)
        assert self.kernel == "auto" or isinstance(self.kernel, self._kernel_class)
        self.output_type = "rate"
        super(SpikesToRatesElephantRate, self).configure()

    def _compute(self, spikes, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.instantaneous_rate function."""
        spiketrain = self._spiketrain(spikes)
        if self.kernel != "auto" or spiketrain.size > 2:
            data = np.array(
                self._rate_fun(spiketrain,
                               self.dt*self.time_unit, self.kernel, *args, **kwargs).rescale(self.rate_unit)).flatten()
            return data
        else:
            # If we have less than 3 spikes amd kernel="auto", we revert to time_histogram computation
            return SpikesToRatesElephantHistogram._compute_rate(spiketrain).flatten()


class BasicTransformers(Enum):
    ELEMENTARY = Elementary
    SCALE = Scale
    DOT_PRODUCT = DotProduct


class RatesToSpikesTransformers(Enum):
    ELEPHANT_POISSON = RatesToSpikesElephantPoisson
    ELEPHANT_POISSON_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    ELEPHANT_POISSON_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction


class SpikesToRatesTransformers(Enum):
    ELEPHANT_HISTOGRAM = SpikesToRatesElephantHistogram
    ELEPHANT_RATE = SpikesToRatesElephantRate


Transformers = combine_enums("Transformers", BasicTransformers, RatesToSpikesTransformers, SpikesToRatesTransformers)


TVBRatesToSpikesTransformers = RatesToSpikesTransformers


class TVBOutputScaleTransformers(Enum):
    RATE = ScaleRate
    CURRENT = ScaleCurrent


TVBOutputTransformers = combine_enums("TVBOutputTransformers", TVBRatesToSpikesTransformers, TVBOutputScaleTransformers)


class TVBSpikesToRatesTransformers(Enum):
    ELEPHANT_HISTOGRAM = SpikesToRatesElephantHistogram
    ELEPHANT_RATE = SpikesToRatesElephantRate


TVBInputTransformers = TVBSpikesToRatesTransformers
TVBTransformers = combine_enums("TVBTransformers", TVBOutputTransformers, TVBInputTransformers)