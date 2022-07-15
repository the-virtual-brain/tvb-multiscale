# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits
from tvb.basic.neotraits._attr import NArray

from tvb_multiscale.core.interfaces.base.transformers.models.base import SpikesToRates
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantSpikesHistogram, ElephantSpikesHistogramRate, ElephantSpikesRate


class ThalamoCorticalWCInverseSigmoidal(HasTraits):

    Rmin = NArray(
        label=r":math:`R_max`",
        default=np.array([0.1, ]),
        doc="""[Hz]. Minimum rate.""")

    sigma = NArray(
        label=r":math:sigma`",
        default=np.array([0.0, ]),
        doc="""[mV] Firing threshold (PSP) for which a 50% firing rate is achieved.
                In other words, it is the value of the average membrane potential
                corresponding to the inflection point of the sigmoid.
                The usual value for this parameter is around 6.0.""")

    beta = NArray(
        label=r":math:`\beta`",
        default=np.array([20.0, ]),
        doc="""Steepness of the sigmoidal transformation.""")

    w = NArray(
        label=r":math:`w`",
        default=np.array([1.0, ]),
        doc="""Rate weight scaling.""")

    def _compute(self, input_buffer):
        return self.sigma - np.log(1.0 / np.maximum(self.w * input_buffer, self.Rmin) - 1) / self.beta


class SpikesToRatesThalamoCorticalWCInverseSigmoidal(SpikesToRates, ThalamoCorticalWCInverseSigmoidal, ABC):
    __metaclass__ = ABCMeta

    def configure(self):
        super().configure()
        ThalamoCorticalWCInverseSigmoidal.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return ThalamoCorticalWCInverseSigmoidal._compute(self, super()._compute(input_buffer, *args, **kwargs))


class ElephantSpikesHistogramThalamoCorticalWCInverseSigmoidal(ElephantSpikesHistogram,
                                                       SpikesToRatesThalamoCorticalWCInverseSigmoidal,
                                                       ThalamoCorticalWCInverseSigmoidal):

    def configure(self):
        SpikesToRatesThalamoCorticalWCInverseSigmoidal.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return SpikesToRatesThalamoCorticalWCInverseSigmoidal._compute(self, input_buffer, *args, **kwargs)


class ElephantSpikesHistogramRateThalamoCorticalWCInverseSigmoidal(ElephantSpikesHistogramRate,
                                                           SpikesToRatesThalamoCorticalWCInverseSigmoidal,
                                                           ThalamoCorticalWCInverseSigmoidal):

    def configure(self):
        SpikesToRatesThalamoCorticalWCInverseSigmoidal.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return SpikesToRatesThalamoCorticalWCInverseSigmoidal._compute(self, input_buffer, *args, **kwargs)


class ElephantSpikesRateThalamoCorticalWCInverseSigmoidal(ElephantSpikesRate,
                                                          SpikesToRatesThalamoCorticalWCInverseSigmoidal,
                                                          ThalamoCorticalWCInverseSigmoidal):

    def configure(self):
        SpikesToRatesThalamoCorticalWCInverseSigmoidal.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of instantaneous mean spiking rates to result."""
        return SpikesToRatesThalamoCorticalWCInverseSigmoidal._compute(self, input_buffer, *args, **kwargs)


class DefaultSpikeNetToTVBTransformersThalamoCorticalWCInverseSigmoidal(Enum):
    SPIKES = ElephantSpikesHistogramRateThalamoCorticalWCInverseSigmoidal
    SPIKES_TO_RATE = ElephantSpikesRateThalamoCorticalWCInverseSigmoidal
    SPIKES_TO_HIST = ElephantSpikesHistogramThalamoCorticalWCInverseSigmoidal
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRateThalamoCorticalWCInverseSigmoidal
