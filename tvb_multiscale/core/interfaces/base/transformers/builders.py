# -*- coding: utf-8 -*-

from logging import Logger
import inspect
from enum import Enum
from abc import ABCMeta, abstractmethod
from six import string_types

import numpy as np

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import Attr, Float

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.base.transformers.models.base import Transformer, ScaleRate, ScaleCurrent
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantSpikesRate, ElephantSpikesHistogramRate, ElephantSpikesHistogram,  \
    RatesToSpikesElephantPoisson, RatesToSpikesElephantPoissonMultipleInteraction, \
    RatesToSpikesElephantPoissonSingleInteraction
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels


class DefaultTVBtoSpikeNetTransformers(Enum):
    RATE = ScaleRate
    SPIKES = RatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction
    CURRENT = ScaleCurrent


class DefaultSpikeNetToTVBTransformers(Enum):
    SPIKES = ElephantSpikesHistogramRate
    SPIKES_TO_RATE = ElephantSpikesRate
    SPIKES_TO_HIST = ElephantSpikesHistogram
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRate


class DefaultTVBtoSpikeNetModels(Enum):
    RATE = DefaultTVBtoSpikeNetTransformers.RATE.name
    SPIKES = DefaultTVBtoSpikeNetTransformers.SPIKES_SINGLE_INTERACTION.name
    CURRENT = DefaultTVBtoSpikeNetTransformers.CURRENT.name


class DefaultSpikeNetToTVBModels(Enum):
    SPIKES = DefaultSpikeNetToTVBTransformers.SPIKES_TO_HIST_RATE.name


class TransformerBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """TransformerBuilder abstract class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True, default=0.0)

    def _configure_transformer_model(self, interface, interface_models, default_transformer_models, transformer_models):
        # Return a model or an Enum
        model = interface.get("transformer", interface.pop("transformer_model", None))
        if model is None:
            model = interface.get("model", interface_models[0])
            model = model.upper()
            assert model in interface_models
            model = getattr(default_transformer_models, model).value
        if isinstance(model, Enum):
            # Enum input:
            assert model in transformer_models
        elif isinstance(model, string_types):
            # String input:
            model = model.upper()
            model = getattr(transformer_models, model)  # Get an Enum
        elif inspect.isclass(model):
            # if it is a Transformer type
            assert issubclass(model, Transformer)
        else:
            # or it has to be a Transformer model
            assert isinstance(model, Transformer)
        interface["transformer"] = model

    def build_transformer(self, model, **kwargs):
        return model(**kwargs)

    def set_transformer_parameters(self, transformer, params):
        for p, pval in params.items():
            setattr(transformer, p, pval)

    build_transformer
    @abstractmethod
    def configure_and_build_transformer(self, interfaces):
        pass


class TVBtoSpikeNetTransformerBuilder(TransformerBuilder):

    """TVBtoSpikeNetTransformerBuilder abstract class"""

    _tvb_to_spikeNet_models = list(TVBtoSpikeNetModels.__members__)
    _default_tvb_to_spikeNet_models = DefaultTVBtoSpikeNetModels
    _tvb_to_spikeNet_transformer_models = DefaultTVBtoSpikeNetTransformers

    def configure_and_build_transformer(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._tvb_to_spikeNet_models,
                                              self._default_tvb_to_spikeNet_models,
                                              self._tvb_to_spikeNet_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                if interface["transformer"] == DefaultTVBtoSpikeNetTransformers.SPIKES:
                    # If the transformer is "SPIKES", but there are parameters that concern correlations...
                    correlation_factor = params.pop("correlation_factor", None)
                    if correlation_factor:
                        interaction = params.pop("interaction", "multiple")
                        if interaction == "single":
                            interface["transformer"] = \
                                self.build_transformer(DefaultTVBtoSpikeNetTransformers.SPIKES_SINGLE_INTERACTION.value,
                                                       correlation_factor=correlation_factor, **params)
                        else:
                            interface["transformer"] = \
                                self.build_transformer(
                                    DefaultTVBtoSpikeNetTransformers.SPIKES_MULTIPLE_INTERACTION.value,
                                    correlation_factor=correlation_factor, **params)
                    else:
                        # SPIKES without correlations:
                        interface["transformer"] = self.build_transformer(interface["transformer"].value, **params)
                else:
                    # Other than SPIKES, e.g., RATE, CURRENT
                    interface["transformer"] = self.build_transformer(interface["transformer"].value, **params)
            elif inspect.isclass(interface["transformer"]):
                # ...or a Transformer type:
                interface["transformer"] = self.build_transformer(interface["transformer"], **params)
            else:
                # ...or an already built model
                self.set_transformer_parameters(interface["transformer"], params)


class SpikeNetToTVBTransformerBuilder(TransformerBuilder):

    """SpikeNetToTVBTransformerBuilder abstract class"""

    _spikeNet_to_tvb_models = list(SpikeNetToTVBModels.__members__)
    _default_spikeNet_to_tvb_transformer_models = DefaultSpikeNetToTVBModels
    _spikeNet_to_tvb_transformer_models = DefaultSpikeNetToTVBTransformers

    def configure_and_build_transformer(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._spikeNet_to_tvb_models,
                                              self._default_spikeNet_to_tvb_transformer_models,
                                              self._spikeNet_to_tvb_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                interface["transformer"] = interface["transformer"].value(**params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)
