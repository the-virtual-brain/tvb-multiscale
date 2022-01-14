# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import Int, NArray, Range

from tvb_multiscale.core.interfaces.base.transformers.models.base import LinearRate, RatesToSpikes, SpikesToRates
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    RatesToSpikesElephantPoisson, RatesToSpikesElephantPoissonSingleInteraction, \
    RatesToSpikesElephantPoissonMultipleInteraction, \
    ElephantSpikesHistogram, ElephantSpikesHistogramRate, ElephantSpikesRate


class JansenRitSigmoidal(HasTraits):

    cmin = NArray(
        label=":math:`c_{min}`",
        default=np.array([0.0, ]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Minimum of the sigmoid function", )

    cmax = NArray(
        label=":math:`c_{max}`",
        default=np.array([2.0 * 0.0025, ]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Maximum of the sigmoid function", )

    midpoint = NArray(
        label="midpoint",
        default=np.array([6.0, ]),
        domain=Range(lo=-1000.0, hi=1000.0, step=10.0),
        doc="Midpoint of the linear portion of the sigmoid", )

    r = NArray(
        label=r":math:`r`",
        default=np.array([1.0, ]),
        domain=Range(lo=0.01, hi=1000.0, step=10.0),
        doc="the steepness of the sigmoidal transformation", )

    ind_Ein = Int(default=1,
                  field_type=int,
                  label="Excitatory interneurons Index",
                  doc="""Index of excitatory interneurons coupling state variable""")

    ind_Iin = Int(default=2,
                  field_type=int,
                  label="Inhibitory interneurons Index",
                  doc="""Index of inhibitory interneurons coupling state variable""")

    def configure(self):
        super(JansenRitSigmoidal, self).configure()

    def _compute(self, input_buffer):
        return self.cmax / (1.0 + np.exp(self.r * (self.midpoint - (input_buffer[:, self.ind_Ein] -
                                                                    input_buffer[:, self.ind_Iin]))))

    def print_str(self):
        out = ""
        for p in ["cmin", "cmax", "midpoint", "r", "ind_Ein", "ind_Iin"]:
            out += "\n     - %s = %s" % (p, str(getattr(self, p)))
        return out


class JansenRitSigmoidalLinearRate(LinearRate, JansenRitSigmoidal):

    def configure(self):
        LinearRate.configure(self)
        JansenRitSigmoidal.configure(self)

    def compute(self):
        """Method that just scales and translates the input buffer data to compute the output buffer data."""
        if isinstance(self.input_buffer, np.ndarray):
            self.output_buffer = \
                self.scale_factor * JansenRitSigmoidal._compute(self, self.input_buffer) + self.translation_factor
        else:
            self.output_buffer = []
            for scale_factor, translation_factor, input_buffer in \
                    zip(self.input_buffer, self._scale_factor, self._translation_factor):
                self.output_buffer.append(
                    scale_factor * JansenRitSigmoidal._compute(self, input_buffer) + translation_factor)

    def print_str(self):
        return LinearRate.print_str(self) + \
               "\n" + JansenRitSigmoidal.print_str(self)


class JansenRitSigmoidalRatesToSpikes(RatesToSpikes, JansenRitSigmoidal, ABC):
    __metaclass__ = ABCMeta

    def configure(self):
        RatesToSpikes.configure(self)
        JansenRitSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        self.output_buffer = []
        for iP, (proxy_buffer, scale_factor, translation_factor) in \
                enumerate(zip(self.input_buffer, self._scale_factor, self._translation_factor)):
            self.output_buffer.append(
                self._compute(scale_factor * JansenRitSigmoidal._compute(self, proxy_buffer) + translation_factor,
                              iP, *args, **kwargs))


class JansenRitSigmoidalRatesToSpikesElephantPoisson(RatesToSpikesElephantPoisson,
                                                     JansenRitSigmoidalRatesToSpikes,
                                                     JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoisson.configure(self)
        JansenRitSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitSigmoidalRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoisson.print_str(self) + \
               JansenRitSigmoidal.print_str(self)


class JansenRitSigmoidalRatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonSingleInteraction,
                                                                      JansenRitSigmoidalRatesToSpikes,
                                                                      JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoissonSingleInteraction.configure(self)
        JansenRitSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitSigmoidalRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoissonSingleInteraction.print_str(self) + \
               JansenRitSigmoidal.print_str(self)


class JansenRitSigmoidalRatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonMultipleInteraction,
                                                                        JansenRitSigmoidalRatesToSpikes,
                                                                        JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoissonMultipleInteraction.configure(self)
        JansenRitSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitSigmoidalRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoissonMultipleInteraction.print_str(self) + \
               JansenRitSigmoidal.print_str(self)


class JansenRitInverseSigmoidal(HasTraits):

    Rmax = NArray(
        label=r":math:`\R_max`",
        default=np.array([12., ]),
        doc="""[Hz]. Maximum rate.""")

    Rmin = NArray(
        label=r":math:`\R_max`",
        default=np.array([0.1, ]),
        doc="""[Hz]. Maximum rate.""")

    v0 = NArray(
        label=r":math:`\v_0`",
        default=np.array([5.52, ]),
        doc="""[mV] Firing threshold (PSP) for which a 50% firing rate is achieved.
                In other words, it is the value of the average membrane potential
                corresponding to the inflection point of the sigmoid.
                The usual value for this parameter is around 6.0.""")

    r = NArray(
        label=r":math:`\v_0`",
        default=np.array([0.56, ]),
        doc="""[mV^-1] Steepness of the sigmoidal transformation .""")

    def _compute(self, input_buffer):
        return self.v0 - np.log(2*self.Rmax / np.minimum(np.maximum(input_buffer,
                                                                    self.Rmin),
                                                         self.Rmax) - 1)/self.r

    def print_str(self):
        out = ""
        for p in ['Rmin', 'Rmax', 'v0', 'r']:
            out += "\n     - %s = %s" % (p, str(getattr(self, p)))
        return out


class SpikesToRatesJansenRitInverseSigmoidal(SpikesToRates, JansenRitInverseSigmoidal, ABC):
    __metaclass__ = ABCMeta

    def configure(self):
        SpikesToRates.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        output_buffer = []
        for proxy_buffer, scale_factor, translation_factor in \
                zip(self.input_buffer, self._scale_factor, self._translation_factor):
            output_buffer.append(
                JansenRitInverseSigmoidal._compute(self, scale_factor * self._compute(proxy_buffer, *args, **kwargs)
                                                         + translation_factor)
            )
        self.output_buffer = np.array(output_buffer)
        return self.output_buffer


class ElephantSpikesHistogramJansenRitInverseSigmoidal(ElephantSpikesHistogram,
                                                       SpikesToRatesJansenRitInverseSigmoidal,
                                                       JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesHistogram.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        SpikesToRatesJansenRitInverseSigmoidal.compute(self, *args, **kwargs)

    def print_str(self):
        return ElephantSpikesHistogram.print_str(self) + \
               JansenRitInverseSigmoidal.print_str(self)


class ElephantSpikesHistogramRateJansenRitInverseSigmoidal(ElephantSpikesHistogramRate,
                                                           SpikesToRatesJansenRitInverseSigmoidal,
                                                           JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        SpikesToRatesJansenRitInverseSigmoidal.compute(self, *args, **kwargs)

    def print_str(self):
        return ElephantSpikesHistogramRate.print_str(self) + \
               JansenRitInverseSigmoidal.print_str(self)


class ElephantSpikesRateJansenRitInverseSigmoidal(ElephantSpikesRate,
                                                  SpikesToRatesJansenRitInverseSigmoidal,
                                                  JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesRate.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        SpikesToRatesJansenRitInverseSigmoidal.compute(self, *args, **kwargs)

    def print_str(self):
        return ElephantSpikesRate.print_str(self) + \
               JansenRitInverseSigmoidal.print_str(self)


class DefaultTVBtoSpikeNetTransformersJansenRitCoupling(Enum):
    RATE = JansenRitSigmoidalLinearRate
    SPIKES = JansenRitSigmoidalRatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = JansenRitSigmoidalRatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = JansenRitSigmoidalRatesToSpikesElephantPoissonMultipleInteraction


class DefaultSpikeNetToTVBTransformersJansenRitInverseSigmoidal(Enum):
    SPIKES = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
    SPIKES_TO_RATE = ElephantSpikesRateJansenRitInverseSigmoidal
    SPIKES_TO_HIST = ElephantSpikesHistogramJansenRitInverseSigmoidal
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
