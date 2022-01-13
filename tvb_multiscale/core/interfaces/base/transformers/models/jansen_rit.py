# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import Attr, NArray

from tvb_multiscale.core.tvb.cosimulator.models.jansen_rit import JansenRit, SigmoidalJansenRit
from tvb_multiscale.core.interfaces.base.transformers.models.base import \
    ScaleCurrent, ScaleRate, RatesToSpikes, SpikesToRates
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    RatesToSpikesElephantPoisson, RatesToSpikesElephantPoissonSingleInteraction, \
    RatesToSpikesElephantPoissonMultipleInteraction, \
    ElephantSpikesHistogram, ElephantSpikesHistogramRate, ElephantSpikesRate


class JansenRitCoupling(HasTraits):

    coupling = Attr(
        field_type=SigmoidalJansenRit,
        label="Long-range coupling function",
        default=SigmoidalJansenRit(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
                   between regions by the ``Long-range connectivity`` before it enters the local
                   dynamic equations of the Model. Its primary purpose is to 'rescale' the
                   incoming activity to a level appropriate to Model.""")

    def configure(self):
        super(JansenRitCoupling, self).configure()
        self.coupling.configure()

    def _compute(self, input_buffer):
        return self.coupling(input_buffer).squeeze()

    def print_str(self):
        out = ""
        for p in self.coupling._own_declarative_attrs:
            out += "\n     - SigmoidalJansenRit coupling.%s = %s" % (p, str(getattr(self.coupling, p)))
        return out


class JansenRitCouplingScaleRate(ScaleRate, JansenRitCoupling):

    def configure(self):
        ScaleRate.configure(self)
        JansenRitCoupling.configure(self)

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = self.scale_factor * JansenRitCoupling._compute(self, self.input_buffer)

    def print_str(self):
        return ScaleRate.print_str(self) + \
               "\n" + JansenRitCoupling.print_str(self)


class JansenRitCouplingRatesToSpikes(RatesToSpikes, JansenRitCoupling, ABC):
    __metaclass__ = ABCMeta

    def configure(self):
        RatesToSpikes.configure(self)
        JansenRitCoupling.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        self.output_buffer = []
        for iP, (scale_factor, proxy_buffer) in enumerate(zip(self._scale_factor, self.input_buffer)):
            self.output_buffer.append(self._compute(scale_factor*JansenRitCoupling._compute(self, proxy_buffer),
                                                    iP, *args, **kwargs))


class JansenRitCouplingRatesToSpikesElephantPoisson(RatesToSpikesElephantPoisson,
                                                    JansenRitCouplingRatesToSpikes,
                                                    JansenRitCoupling):

    def configure(self):
        RatesToSpikesElephantPoisson.configure(self)
        JansenRitCoupling.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitCouplingRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoisson.print_str(self) + \
               JansenRitCoupling.print_str(self)


class JansenRitCouplingRatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonSingleInteraction,
                                                                     JansenRitCouplingRatesToSpikes,
                                                                     JansenRitCoupling):

    def configure(self):
        RatesToSpikesElephantPoissonSingleInteraction.configure(self)
        JansenRitCoupling.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitCouplingRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoissonSingleInteraction.print_str(self) + \
               JansenRitCoupling.print_str(self)


class JansenRitCouplingRatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonMultipleInteraction,
                                                                       JansenRitCouplingRatesToSpikes,
                                                                       JansenRitCoupling):

    def configure(self):
        RatesToSpikesElephantPoissonMultipleInteraction.configure(self)
        JansenRitCoupling.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        JansenRitCouplingRatesToSpikes.compute(self, *args, **kwargs)

    def print_str(self):
        return RatesToSpikesElephantPoissonMultipleInteraction.print_str(self) + \
               JansenRitCoupling.print_str(self)


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
                                                                    self.Rmin), self.Rmax) - 1)/self.r

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
        for scale_factor, proxy_buffer in zip(self._scale_factor, self.input_buffer):
            output_buffer.append(
                JansenRitInverseSigmoidal._compute(self, scale_factor * self._compute(proxy_buffer, *args, **kwargs))
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
    RATE = JansenRitCouplingScaleRate
    SPIKES = JansenRitCouplingRatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = JansenRitCouplingRatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = JansenRitCouplingRatesToSpikesElephantPoissonMultipleInteraction


class DefaultSpikeNetToTVBTransformersJansenRitInverseSigmoidal(Enum):
    SPIKES = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
    SPIKES_TO_RATE = ElephantSpikesRateJansenRitInverseSigmoidal
    SPIKES_TO_HIST = ElephantSpikesHistogramJansenRitInverseSigmoidal
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
