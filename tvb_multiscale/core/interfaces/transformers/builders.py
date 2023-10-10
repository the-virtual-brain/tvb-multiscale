# -*- coding: utf-8 -*-

from logging import Logger
import inspect
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC
from six import string_types

from tvb.basic.neotraits._attr import Attr, Float

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.data_structures_utils import get_enum_values
from tvb_multiscale.core.interfaces.transformers.models.base import \
     LinearRate, LinearCurrent, LinearPotential
from tvb_multiscale.core.interfaces.transformers.models.elephant import \
    ElephantSpikesRate, ElephantSpikesHistogramRate, ElephantSpikesHistogram,  \
    RatesToSpikesElephantPoisson, RatesToSpikesElephantPoissonMultipleInteraction, \
    RatesToSpikesElephantPoissonSingleInteraction
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.interfaces.base.builders import InterfaceBuilder, RemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.transformers.interfaces import TransformerInterface, TransformerInterfaces, \
    RemoteTransformerInterface, RemoteTransformerInterfaces, \
    TVBtoSpikeNetTransformerInterface, TVBtoSpikeNetTransformerInterfaces, \
    SpikeNetToTVBTransformerInterface, SpikeNetToTVBTransformerInterfaces, \
    TVBtoSpikeNetRemoteTransformerInterface, TVBtoSpikeNetRemoteTransformerInterfaces, \
    SpikeNetToTVBRemoteTransformerInterface, SpikeNetToTVBRemoteTransformerInterfaces


class DefaultTVBtoSpikeNetTransformers(Enum):
    RATE = LinearRate
    SPIKES = RatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction
    CURRENT = LinearCurrent


class DefaultSpikeNetToTVBTransformers(Enum):
    SPIKES = ElephantSpikesHistogramRate
    SPIKES_TO_RATE = ElephantSpikesRate
    SPIKES_TO_HIST = ElephantSpikesHistogram
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRate
    POTENTIAL = LinearPotential


class DefaultTVBtoSpikeNetModels(Enum):
    RATE = DefaultTVBtoSpikeNetTransformers.RATE.name
    SPIKES = DefaultTVBtoSpikeNetTransformers.SPIKES_SINGLE_INTERACTION.name
    CURRENT = DefaultTVBtoSpikeNetTransformers.CURRENT.name


class DefaultSpikeNetToTVBModels(Enum):
    SPIKES = DefaultSpikeNetToTVBTransformers.SPIKES_TO_HIST_RATE.name
    POTENTIAL = DefaultSpikeNetToTVBTransformers.POTENTIAL.name


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

    _config_attrs = ["default_coupling_mode", "exclusive_nodes", "proxy_inds", "dt"]

    def _configure_transformer_model(self, interface, interface_models, default_transformer_models, transformer_models):
        # Return a model or an Enum
        model = interface.get("transformer", interface.pop("transformer_model", None))
        if model is None:
            model = interface.get("model", None)
            if model is None:
                model = list(interface_models)[0].name  # a string at this point
            else:
                model = model.upper()
            assert model in list(interface_models.__members__)  # Enum names (strings)
            model = getattr(default_transformer_models, model).value  # string name of transformer type
        if isinstance(model, Enum):
            # Enum input:
            assert model in transformer_models
        elif isinstance(model, string_types):
            # String input:
            model = model.upper()
            model = getattr(transformer_models, model)  # Get an Enum
        else:
            transformer_models_tuple = tuple(get_enum_values(transformer_models))
            if inspect.isclass(model):
                # assert it is a Transformer type...
                assert issubclass(model, transformer_models_tuple)
            else: # ...or instance
                assert isinstance(model, transformer_models_tuple)
        interface["transformer"] = model

    def build_transformer(self, model, **kwargs):
        kwargs["config"] = self.config
        return model(**kwargs)

    def set_transformer_parameters(self, transformer, params):
        for p, pval in params.items():
            setattr(transformer, p, pval)

    @abstractmethod
    def configure_and_build_transformers(self, interfaces):
        pass


class TVBtoSpikeNetTransformerBuilder(TransformerBuilder):

    """TVBtoSpikeNetTransformerBuilder abstract class"""

    _tvb_to_spikeNet_models = list(TVBtoSpikeNetModels.__members__)
    _default_tvb_to_spikeNet_models = DefaultTVBtoSpikeNetModels
    _tvb_to_spikeNet_transformer_models = DefaultTVBtoSpikeNetTransformers

    def configure_and_build_transformers(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._tvb_to_spikeNet_models,
                                              self._default_tvb_to_spikeNet_models,
                                              self._tvb_to_spikeNet_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                if interface["transformer"] == self._default_tvb_to_spikeNet_models.SPIKES:
                    # If the transformer is "SPIKES", but there are parameters that concern correlations...
                    correlation_factor = params.pop("correlation_factor", None)
                    if correlation_factor:
                        interaction = params.pop("interaction", "multiple")
                        if interaction == "single":
                            interface["transformer"] = \
                                self.build_transformer(
                                    self._default_tvb_to_spikeNet_models.SPIKES_SINGLE_INTERACTION.value,
                                    correlation_factor=correlation_factor, **params)
                        else:
                            interface["transformer"] = \
                                self.build_transformer(
                                    self._default_tvb_to_spikeNet_models.SPIKES_MULTIPLE_INTERACTION.value,
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

    def configure_and_build_transformers(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._spikeNet_to_tvb_models,
                                              self._default_spikeNet_to_tvb_transformer_models,
                                              self._spikeNet_to_tvb_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                interface["transformer"] = self.build_transformer(interface["transformer"].value, **params)
            elif inspect.isclass(interface["transformer"]):
                # ...or a Transformer type:
                interface["transformer"] = self.build_transformer(interface["transformer"], **params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)


class TransformerInterfaceBuilder(InterfaceBuilder, TransformerBuilder, ABC):

    """TransformerInterfaceBuilder class"""

    input_label = Attr(field_type=str, default="InToTrans", required=True, label="Input label",
                       doc="""Input label of interface builder,
                              to be used for files' names and Receiver class instance labels, 
                              for the communication of data towards this CoSimulator""")

    output_label = Attr(field_type=str, default="OutFromTrans", required=True, label="Output label",
                        doc="""Output label of interface builder,
                               to be used for files' names and Sender class instance labels, 
                               for the communication of data starting from this CoSimulator""")

    _output_interface_type = TransformerInterface
    _input_interface_type = TransformerInterface

    _output_interfaces_type = TransformerInterfaces
    _input_interfaces_type = TransformerInterfaces

    tvb_simulator_serialized = Attr(label="TVB simulator serialized",
                                    doc="""Dictionary of TVB simulator serialization""",
                                    field_type=dict,
                                    required=True,
                                    default={})

    @property
    def tvb_dt(self):
        return self.tvb_simulator_serialized.get("integrator.dt", self.config.DEFAULT_DT)

    @property
    def synchronization_time(self):
        return self.tvb_simulator_serialized.get("synchronization_time", 0.0)

    @property
    def synchronization_n_step(self):
        return int(self.tvb_simulator_serialized.get("synchronization_n_step", 0))

    def _configure_and_build_output_transformers(self):
        self.configure_and_build_transformers(self.output_interfaces)

    def _configure_and_build_input_transformers(self):
        self.configure_and_build_transformers(self.input_interfaces)

    def _configure_output_interfaces(self):
        self._configure_and_build_output_transformers()

    def _configure_input_interfaces(self):
        self._configure_and_build_input_transformers()

    def configure(self):
        if self.dt == 0.0:
            # From TVBInterfaceBuilder to TransformerBuilder:
            self.dt = self.tvb_dt
        super(TransformerInterfaceBuilder, self).configure()
        self._configure_output_interfaces()
        self._configure_input_interfaces()

    def build(self):
        self.build_interfaces()
        output_interfaces = \
            self._output_interfaces_type(interfaces=self._output_interfaces,
                                         synchronization_time=self.synchronization_time,
                                         synchronization_n_step=self.synchronization_n_step)
        input_interfaces = \
            self._input_interfaces_type(interfaces=self._input_interfaces,
                                        synchronization_time=self.synchronization_time,
                                        synchronization_n_step=self.synchronization_n_step)
        return output_interfaces, input_interfaces


class TVBtoSpikeNetTransformerInterfaceBuilder(TransformerInterfaceBuilder, TVBtoSpikeNetTransformerBuilder):

    """TVBtoSpikeNetTransformerInterfaceBuilder class"""

    output_label = Attr(field_type=str, default="TVBtToSpikeNetTrans", required=True, label="Output label",
                        doc="""Output label of interface builder,
                               to be used for files' names and Sender class instance labels, 
                               for the communication of data starting from this CoSimulator""")

    _output_interface_type = TVBtoSpikeNetTransformerInterface
    _output_interfaces_type = TVBtoSpikeNetTransformerInterfaces

    def _configure_and_build_output_transformers(self):
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformers(self, self.output_interfaces)

    def _configure_and_build_input_transformers(self):
        pass

    def _get_input_interface_arguments(self, interface, ii=0):
        pass

    def _configure_input_interfaces(self):
        pass

    def build(self):
        self.build_interfaces()
        output_interfaces = \
            self._output_interfaces_type(interfaces=self._output_interfaces,
                                         synchronization_time=self.synchronization_time,
                                         synchronization_n_step=self.synchronization_n_step)
        return output_interfaces


class SpikeNetToTVBTransformerInterfaceBuilder(TransformerInterfaceBuilder, SpikeNetToTVBTransformerBuilder):

    """SpikeNetToTVBTransformerInterfaceBuilder class"""

    input_label = Attr(field_type=str, default="SpikeNetToTVBtrans", required=True, label="Input label",
                       doc="""Input label of interface builder,
                              to be used for files' names and Receiver class instance labels, 
                              for the communication of data towards this CoSimulator""")

    _input_interface_type = SpikeNetToTVBTransformerInterface
    _input_interfaces_type = SpikeNetToTVBTransformerInterfaces

    def _configure_and_build_input_transformers(self):
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformers(self, self.input_interfaces)

    def _configure_and_build_output_transformers(self):
        pass

    def _get_output_interface_arguments(self, interface, ii=0):
        pass

    def _configure_output_interfaces(self):
        pass

    def build(self):
        self.build_interfaces()
        input_interfaces = \
            self._input_interfaces_type(interfaces=self._input_interfaces,
                                        synchronization_time=self.synchronization_time,
                                        synchronization_n_step=self.synchronization_n_step)
        return input_interfaces


class RemoteTransformerInterfaceBuilder(TransformerInterfaceBuilder, RemoteInterfaceBuilder, ABC):

    __metaclass__ = ABCMeta

    """RemoteTransformerInterfaceBuilder class"""

    _output_interface_type = RemoteTransformerInterface
    _input_interface_type = RemoteTransformerInterface

    _output_interfaces_type = RemoteTransformerInterfaces
    _input_interfaces_type = RemoteTransformerInterfaces

    def _configure_output_interfaces(self):
        self._assert_output_interfaces_component_config(
            self._remote_sender_types, ["sender", "sender_model"], self._default_remote_sender_type)
        self._assert_output_interfaces_component_config(
            self._remote_receiver_types, ["receiver", "receiver_model"], self._default_remote_receiver_type)
        super(RemoteTransformerInterfaceBuilder, self)._configure_output_interfaces()

    def _configure_input_interfaces(self):
        self._assert_input_interfaces_component_config(
            self._remote_sender_types, ["sender", "sender_model"], self._default_remote_sender_type)
        self._assert_input_interfaces_component_config(
            self._remote_receiver_types, ["receiver", "receiver_model"], self._default_remote_receiver_type)
        super(RemoteTransformerInterfaceBuilder, self)._configure_input_interfaces()

    def _get_output_interface_arguments(self, interface, ii=0):
        interface = super(RemoteTransformerInterfaceBuilder, self)._get_output_interface_arguments(interface, ii)
        interface = self._build_communicator(interface, "receiver", ii)
        interface = self._build_communicator(interface, "sender", ii)
        return interface

    def _get_input_interface_arguments(self, interface, ii=0):
        interface = super(RemoteTransformerInterfaceBuilder, self)._get_input_interface_arguments(interface, ii)
        interface = self._build_communicator(interface, "receiver", ii)
        interface = self._build_communicator(interface, "sender", ii)
        return interface


class TVBtoSpikeNetRemoteTransformerInterfaceBuilder(TVBtoSpikeNetTransformerInterfaceBuilder,
                                                     RemoteTransformerInterfaceBuilder):
    """TVBtoSpikeNetRemoteTransformerInterfaceBuilder class"""

    input_label = Attr(field_type=str, default="TVBToTrans", required=True, label="Input label",
                       doc="""Input label of interface builder,
                              to be used for files' names and Receiver class instance labels, 
                              for the communication of data towards this CoSimulator""")

    output_label = Attr(field_type=str, default="TransToSpikeNet", required=True, label="Output label",
                        doc="""Output label of interface builder,
                               to be used for files' names and Sender class instance labels, 
                               for the communication of data starting from this CoSimulator""")

    _output_interface_type = TVBtoSpikeNetRemoteTransformerInterface
    _output_interfaces_type = TVBtoSpikeNetRemoteTransformerInterfaces

    def _configure_output_interfaces(self):
        return RemoteTransformerInterfaceBuilder._configure_output_interfaces(self)

    def _get_output_interface_arguments(self, interface, ii=0):
        return RemoteTransformerInterfaceBuilder._get_output_interface_arguments(self, interface, ii)


class SpikeNetToTVBRemoteTransformerInterfaceBuilder(SpikeNetToTVBTransformerInterfaceBuilder,
                                                     RemoteTransformerInterfaceBuilder):
    """SpikeNetToTVBRemoteTransformerInterfaceBuilder class"""

    input_label = Attr(field_type=str, default="SpikeNetToTrans", required=True, label="Input label",
                       doc="""Input label of interface builder,
                              to be used for files' names and Receiver class instance labels, 
                              for the communication of data towards this CoSimulator""")

    output_label = Attr(field_type=str, default="TransToTVB", required=True, label="Output label",
                        doc="""Output label of interface builder,
                               to be used for files' names and Sender class instance labels, 
                               for the communication of data starting from this CoSimulator""")

    _input_interface_type = SpikeNetToTVBRemoteTransformerInterface
    _input_interfaces_type = SpikeNetToTVBRemoteTransformerInterfaces

    def _configure_input_interfaces(self):
        return RemoteTransformerInterfaceBuilder._configure_input_interfaces(self)

    def _get_input_interface_arguments(self, interface, ii=0):
        return RemoteTransformerInterfaceBuilder._get_input_interface_arguments(self, interface, ii)
