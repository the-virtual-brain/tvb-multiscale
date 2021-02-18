# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from tvb_multiscale.core.interfaces.base.transformers import Scale, RatesToSpikesElephantPoisson, \
    RatesToSpikesElephantPoissonSingleInteraction, RatesToSpikesElephantPoissonMultipleInteraction, \
    SpikesToRatesElephantHistogram, SpikesToRatesElephantRate, Transformers
from tvb_multiscale.core.utils.data_structures_utils import combine_enums


# TODO: Deal with abstract methods _compute for RatesToSpikes and SpikesToRates!!!


class TVBtoSpikeNetRateTransformer(Scale):

    """TVBtoSpikeNetRateTransformer class that just scales TVB mean field rates to Spiking Network rates,
       including any unit conversions and conversions from mean field to total rates"""

    def compute(self, *args, **kwargs):
        """Method for the scaling on the input buffer rates' data
           for the output buffer data also of rates to result."""
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBtoSpikeNetRateTransformer, self).compute(*args, **kwargs)


class TVBRatesToSpikesElephantPoisson(RatesToSpikesElephantPoisson):

    """
        TVBRatesToSpikesElephantPoisson Transformer class,
        using elephant functions inhomogeneous_poisson_process and homogeneous_poisson_process,
        depending on whether rate varies with time or not.
        This class can be used to produce independent spike trains per proxy node.
    """

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBRatesToSpikesElephantPoisson, self).compute(*args, **kwargs)


class TVBRatesToSpikesElephantPoissonSingleInteraction(RatesToSpikesElephantPoissonSingleInteraction):
    """
        TVBRatesToSpikesElephantPoissonSingleInteraction Transformer class,
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

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBRatesToSpikesElephantPoissonSingleInteraction, self).compute(*args, **kwargs)


class TVBRatesToSpikesElephantPoissonMultipleInteraction(RatesToSpikesElephantPoissonMultipleInteraction):
    """
        TVBRatesToSpikesElephantPoissonMultipleInteraction Transformer class,
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

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer rates' data
           for the output buffer data of spike trains to result."""
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBRatesToSpikesElephantPoissonMultipleInteraction, self).compute(*args, **kwargs)


class TVBtoSpikeNetCurrentTransformer(Scale):

    """TVBRateToSpikeNetSpikesTransformer class that just scales TVB mean field currents to Spiking Network currents,
       including any unit conversions and conversions from mean field to total currents"""

    def compute(self, *args, **kwargs):
        # Assume a TVB (time, voi, proxy) to Spiking Network (proxy, time) reshaping
        self.input_buffer = self.input_buffer[:, 0, :].T  # Remove voi
        super(TVBtoSpikeNetCurrentTransformer, self).compute(*args, **kwargs)


class TVBSpikesToRatesElephantHistogram(SpikesToRatesElephantHistogram):

    """
        TVBSpikesToRatesElephantHistogram Transformer class using the function time_histogram of elephant software.
        The algorithm is based just on computing a time histogram of the spike trains.
    """

    def compute(self, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.time_histogram function."""
        super(TVBSpikesToRatesElephantHistogram, self).compute(*args, **kwargs)
        # Assume a Spiking Network (proxy, time) to TVB (time, voi, proxy) reshaping
        self.output_buffer = self.output_buffer.T[:, np.newaxis, :]


class TVBSpikesToRatesElephantRate(SpikesToRatesElephantRate):

    """
        SpikesToRatesElephantRate Transformer class using the function instantaneous_rate of elephant software.
        The algorithm is based on convolution of spike trains with a kernel.
    """

    def compute(self, *args, **kwargs):
        """Method for the computation of spike trains data transformation
           to instantaneous mean spiking rates, using elephant.statistics.instantaneous_rate function."""
        super(TVBSpikesToRatesElephantRate, self).compute(*args, **kwargs)
        # Assume a Spiking Network (proxy, time) to TVB (time, voi, proxy) reshaping
        self.output_buffer = self.output_buffer.T[:, np.newaxis, :]


class TVBRatesToSpikesTransformers(Enum):
    ELEPHANT_POISSON = TVBRatesToSpikesElephantPoisson
    ELEPHANT_POISSON_SINGLE_INTERACTION = TVBRatesToSpikesElephantPoissonSingleInteraction
    ELEPHANT_POISSON_MULTIPLE_INTERACTION = TVBRatesToSpikesElephantPoissonMultipleInteraction


class TVBOutputScaleTransformers(Enum):
    RATE = TVBtoSpikeNetRateTransformer
    CURRENT = TVBtoSpikeNetCurrentTransformer


TVBOutputTransformers = combine_enums("TVBOutputTransformers", TVBRatesToSpikesTransformers, TVBOutputScaleTransformers)


class TVBSpikesToRatesTransformers(Enum):
    ELEPHANT_HISTOGRAM = TVBSpikesToRatesElephantHistogram
    ELEPHANT_RATE = TVBSpikesToRatesElephantRate


TVBInputTransformers = TVBSpikesToRatesTransformers


TVBTransformers = combine_enums("TVBTransformers", TVBOutputTransformers, TVBInputTransformers)
