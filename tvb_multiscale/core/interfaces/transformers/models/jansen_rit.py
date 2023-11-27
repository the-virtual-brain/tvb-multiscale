# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import Int, NArray, Range

from tvb_multiscale.core.interfaces.transformers.models.base import \
    LinearRate, RatesToSpikes, SpikesToRates, LinearPotential
from tvb_multiscale.core.interfaces.transformers.models.integration import LinearIntegration
from tvb_multiscale.core.interfaces.transformers.models.elephant import \
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
        return self.cmax / (1.0 + np.exp(self.r * (self.midpoint - (input_buffer[..., self.ind_Ein] -
                                                                    input_buffer[..., self.ind_Iin]))))


class JansenRitSigmoidalLinearRate(LinearRate, JansenRitSigmoidal):

    def configure(self):
        LinearRate.configure(self)
        JansenRitSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method that just scales and translates the input buffer data to compute the output buffer data."""
        return LinearRate._compute(self,
                                   JansenRitSigmoidal._compute(self, input_buffer))


class JansenRitSigmoidalRatesToSpikesElephantPoisson(RatesToSpikesElephantPoisson,
                                                     JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoisson.configure(self)
        JansenRitSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        return RatesToSpikesElephantPoisson._compute(self,
                                                     JansenRitSigmoidal._compute(self, input_buffer))


class JansenRitSigmoidalRatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonSingleInteraction,
                                                                      JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoissonSingleInteraction.configure(self)
        JansenRitSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        return RatesToSpikesElephantPoissonSingleInteraction._compute(self,
                                                                      JansenRitSigmoidal._compute(self, input_buffer))


class JansenRitSigmoidalRatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonMultipleInteraction,
                                                                        JansenRitSigmoidal):

    def configure(self):
        RatesToSpikesElephantPoissonMultipleInteraction.configure(self)
        JansenRitSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        return RatesToSpikesElephantPoissonMultipleInteraction._compute(self,
                                                                        JansenRitSigmoidal._compute(self, input_buffer))


class JansenRitInverseSigmoidal(HasTraits):

    Rmax = NArray(
        label=r":math:`\R_max`",
        default=np.array([16., ]),
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


class ElephantSpikesHistogramJansenRitInverseSigmoidal(ElephantSpikesHistogram,  JansenRitInverseSigmoidal):

    def configure(self):
        SpikesToRatesJansenRitInverseSigmoidal.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return JansenRitInverseSigmoidal._compute(self,
                                                  ElephantSpikesHistogram._compute(self, input_buffer))


class ElephantSpikesHistogramRateJansenRitInverseSigmoidal(ElephantSpikesHistogramRate, JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return JansenRitInverseSigmoidal._compute(self,
                                                  ElephantSpikesHistogramRate._compute(self, input_buffer))


class ElephantSpikesRateJansenRitInverseSigmoidal(ElephantSpikesRate, JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesRate.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return JansenRitInverseSigmoidal._compute(self,
                                                  ElephantSpikesRate._compute(self, input_buffer))


class ElephantSpikesHistogramJansenRitInverseSigmoidalLinearIntegration(
    ElephantSpikesHistogram, LinearIntegration, JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesHistogram.configure(self)
        LinearIntegration.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        return LinearIntegration._compute(
            self, JansenRitInverseSigmoidal._compute(self,
                                                     ElephantSpikesHistogram._compute(self, input_buffer)))


class ElephantSpikesHistogramRateJansenRitInverseSigmoidalLinearIntegration(
    ElephantSpikesHistogramRate, LinearIntegration, JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        LinearIntegration.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        return LinearIntegration._compute(
            self, JansenRitInverseSigmoidal._compute(self,
                                                     ElephantSpikesHistogramRate._compute(self, input_buffer)))


class ElephantSpikesRateJansenRitInverseSigmoidalLinearIntegration(
    ElephantSpikesRate, LinearIntegration, JansenRitInverseSigmoidal):

    def configure(self):
        ElephantSpikesRate.configure(self)
        LinearIntegration.configure(self)
        JansenRitInverseSigmoidal.configure(self)

    def _compute(self, input_buffer):
        return LinearIntegration._compute(
            self, JansenRitInverseSigmoidal._compute(self,
                                                     ElephantSpikesRate._compute(self, input_buffer)))


class DefaultTVBtoSpikeNetTransformersJansenRitCoupling(object):
    RATE = JansenRitSigmoidalLinearRate
    SPIKES = JansenRitSigmoidalRatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = JansenRitSigmoidalRatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = JansenRitSigmoidalRatesToSpikesElephantPoissonMultipleInteraction


class DefaultSpikeNetToTVBTransformersJansenRitInverseSigmoidal(object):
    SPIKES = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
    SPIKES_TO_RATE = ElephantSpikesRateJansenRitInverseSigmoidal
    SPIKES_TO_HIST = ElephantSpikesHistogramJansenRitInverseSigmoidal
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRateJansenRitInverseSigmoidal
    SPIKES_LINEAR_INTEGRATION = ElephantSpikesHistogramRateJansenRitInverseSigmoidalLinearIntegration
    SPIKES_TO_RATE_LINEAR_INTEGRATION = ElephantSpikesRateJansenRitInverseSigmoidalLinearIntegration
    SPIKES_TO_HIST_LINEAR_INTEGRATION = ElephantSpikesHistogramJansenRitInverseSigmoidalLinearIntegration
    SPIKES_TO_HIST_RATE_LINEAR_INTEGRATION = ElephantSpikesHistogramRateJansenRitInverseSigmoidalLinearIntegration
    POTENTIAL = LinearPotential
