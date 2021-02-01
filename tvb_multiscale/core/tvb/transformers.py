# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_multiscale.core.interfaces.transformers import Base, Scale


class TVBtoSpikeNetRateTransformer(Scale):

    """TVBtoSpikeNetRateTransformer class that just scales TVB mean field rates to Spiking Network spiking rates,
       including any unit conversions and conversions from mean field to total rates"""

    def compute_time(self):
        self.output_time = self.dt * np.arange(self.input_time[0], self.input_time[1] + 1)

    def compute(self):
        super(TVBtoSpikeNetRateTransformer, self).compute()
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.output_buffer = self.output_buffer[:, 0, :].T  # Remove voi


class TVBRateToSpikeNetSpikesTransformer(Base):
    __metaclass__ = ABCMeta

    """TVBRateToSpikeNetSpikesTransformer class that transforms TVB mean field rates to Spiking Network spikes (times),
       including any unit conversions and conversions from mean field to total rates"""

    def compute_time(self):
        self.output_time = self.dt * self.input_time

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        pass


class TVBtoSpikeNetCurrentTransformer(Scale):

    """TVBRateToSpikeNetSpikesTransformer class that just scales TVB mean field currents to Spiking Network currents,
       including any unit conversions and conversions from mean field to total currents"""

    def compute_time(self):
        self.output_time = self.dt * np.arange(self.input_time[0], self.input_time[1] + 1)

    def compute(self):
        super(TVBtoSpikeNetCurrentTransformer, self).compute()
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.output_buffer = self.output_buffer[:, 0, :].T  # Remove voi


class SpikeNetSpikesToTVBRateTransformer(Base):
    __metaclass__ = ABCMeta

    """SpikeNetSpikesToTVBRateTransformer abstract base class that transforms Spiking Network spikes (times) counts to
       TVB mean field rates, including any unit conversions and conversions from total to mean field rates"""

    def compute_time(self):
        self.output_time = np.array([np.ceil(self.output_time.min() / self.dt).item(),
                                     np.ceil(self.output_time.max() / self.dt).item()]).astype("i")

    def reshape(self):
        # Assume a Spiking Network (proxy, time) to TVB (time, voi, proxy) reshaping
        self.output_buffer = self.output_buffer.T[:, np.newaxis, :]

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        pass
