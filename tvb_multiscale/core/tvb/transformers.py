# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.interfaces.transformers import Scale, RatesToSpikes, SpikesToRates

# TODO: Deal with abstract methods _compute for RatesToSpikes and SpikesToRates!!!


class TVBtoSpikeNetRateTransformer(Scale):

    """TVBtoSpikeNetRateTransformer class that just scales TVB mean field rates to Spiking Network spiking rates,
       including any unit conversions and conversions from mean field to total rates"""

    def compute(self, *args, **kwargs):
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBtoSpikeNetRateTransformer, self).compute(*args, **kwargs)


class TVBRateToSpikeNetSpikesTransformer(RatesToSpikes):

    """TVBRateToSpikeNetSpikesTransformer class that transforms TVB mean field rates to Spiking Network spikes (times),
       including any unit conversions and conversions from mean field to total rates"""

    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBRateToSpikeNetSpikesTransformer, self).compute(*args, **kwargs)


class TVBtoSpikeNetCurrentTransformer(Scale):

    """TVBRateToSpikeNetSpikesTransformer class that just scales TVB mean field currents to Spiking Network currents,
       including any unit conversions and conversions from mean field to total currents"""

    def compute(self, *args, **kwargs):
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBtoSpikeNetCurrentTransformer, self).compute(*args, **kwargs)


class SpikeNetSpikesToTVBRateTransformer(SpikesToRates):

    """SpikeNetSpikesToTVBRateTransformer abstract base class that transforms Spiking Network spikes (times) counts to
       TVB mean field rates, including any unit conversions and conversions from total to mean field rates"""

    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        super(SpikeNetSpikesToTVBRateTransformer, self).compute(*args, **kwargs)
        # Assume a Spiking Network (proxy, time) to TVB (time, voi, proxy) reshaping
        self.output_buffer = self.output_buffer.T[:, np.newaxis, :]
