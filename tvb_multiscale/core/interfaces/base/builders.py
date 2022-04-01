# -*- coding: utf-8 -*-

import os
import glob
import inspect
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC
from logging import Logger
from six import string_types
from collections import OrderedDict

import numpy as np

from tvb.basic.neotraits._attr import Attr, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.base.transformers.builders import \
    TransformerBuilder, TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.data_structures_utils import summary_info, get_enum_values
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict, load_pickled_dict
from tvb_multiscale.core.interfaces.base.interfaces import SenderInterface, ReceiverInterface, \
    RemoteTransformerInterface, RemoteTransformerInterfaces, \
    TVBtoSpikeNetRemoteTransformerInterface, TVBtoSpikeNetRemoteTransformerInterfaces, \
    SpikeNetToTVBRemoteTransformerInterface, SpikeNetToTVBRemoteTransformerInterfaces
from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers, WriterToFile, ReaderFromFile


class InterfaceBuilder(HasTraits, ABC):
    __metaclass__ = ABCMeta

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

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of proxy nodes",
        doc="""Indices of proxy nodes""",
        required=True,
    )

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

    default_coupling_mode = Attr(
        label="Default coupling mode",
        field_type=str,
        doc="""Default coupling mode. Set to 'TVB' for large scale coupling to be computed in TVB 
               before being sent to a cosimulator. Default 'spikeNet', which entails that 
               large scale coupling for regions modeled outside TVB is handled by the cosimulator.""",
        required=True,
        default="spikeNet"
    )

    _output_interfaces = None
    _input_interfaces = None

    def _loop_to_get_from_interface_configs(self, interfaces, attr, default=None):
        output = []
        for interface in interfaces:
            output += ensure_list(interface.get(attr, default))
        return output

    def _loop_to_get_unique_from_interface_configs(self, interfaces, attr, default=None):
        return np.unique(self._loop_to_get_from_interface_configs(interfaces, attr, default))

    @property
    @abstractmethod
    def synchronization_time(self):
        pass

    @property
    @abstractmethod
    def synchronization_n_step(self):
        pass

    @property
    def out_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.output_interfaces, "proxy_inds",
                                                               default=self.proxy_inds)

    @property
    def in_proxy_inds(self):
        return self._loop_to_get_unique_from_interface_configs(self.input_interfaces, "proxy_inds",
                                                               default=self.proxy_inds)

    @property
    def number_of_out_proxy_nodes(self):
        return len(self.out_proxy_inds)

    @property
    def number_of_in_proxy_nodes(self):
        return len(self.in_proxy_inds)

    @property
    def number_of_output_interfaces(self):
        return len(self.output_interfaces)

    @property
    def number_of_input_interfaces(self):
        return len(self.output_interfaces)

    @staticmethod
    def _label_to_ind(these_labels, all_labels):
        all_labels = ensure_list(all_labels)
        if isinstance(these_labels, (list, tuple, np.ndarray)):
            return_array = True
        else:
            labels = ensure_list(these_labels)
            return_array = False
        inds = []
        for label in ensure_list(labels):
            inds.append(all_labels.index(label))
        if return_array:
            return np.array(inds)
        else:
            return inds[0]

    def _only_ind(self, value, labels):
        if isinstance(value, string_types):
            return self._label_to_ind(value, labels)
        else:
            return value

    def _only_inds(self, values, labels):
        inds = []
        for iV, value in enumerate(ensure_list(values)):
            inds.append(self._only_ind(value, labels))
        return inds

    def _only_inds_for_interfaces(self, interfaces, attr, labels, default=None):
        return self._only_inds(self._loop_to_get_unique_from_interface_configs(interfaces, attr, default), labels)

    @staticmethod
    def _assert_interfaces_component_config(interfaces_list, component_types, component_names, default_type):
        """This method will assert that all interfaces' components types are appropriate,
           i.e., included in the given communicator_types argument.
        """
        for ii, interface in enumerate(interfaces_list):
            component_names = ensure_list(component_names)
            component = component_names[0]
            for component_name in component_names:
                interface[component] = interface.pop(component_name, None)
                if interface[component] is not None:
                    break
            if interface[component] is None:
                interface[component] = default_type.value  # type <- Enum.value
            elif isinstance(interface[component], string_types):  # type <- Enum name
                interface[component] = getattr(component_types, interface[component])
            elif isinstance(interface[component], Enum):  # type <- Enum
                assert interface[component] in component_types
                interface[component] = interface[component].value
            else:
                # It is already a type or an instance
                component_types_tuple = tuple(get_enum_values(component_types))
                if inspect.isclass(interface[component]):
                    # assert it is a type...
                    assert issubclass(interface[component], component_types_tuple)
                else:  # ...or instance
                    assert isinstance(interface[component], component_types_tuple)
        return interfaces_list

    def _assert_input_interfaces_component_config(self, types_tuple, component_names, default_type):
        self.input_interfaces = \
            self._assert_interfaces_component_config(self.input_interfaces, types_tuple,
                                                     component_names, default_type)

    def _assert_output_interfaces_component_config(self, types_tuple, component_names, default_type):
        self.output_interfaces = \
            self._assert_interfaces_component_config(self.output_interfaces, types_tuple,
                                                     component_names, default_type)

    def set_coupling_mode(self, interface):
        interface["coupling_mode"] = interface.get("coupling_mode", self.default_coupling_mode)
        return interface

    def is_tvb_coupling_interface(self, interface):
        return self.set_coupling_mode(interface)["coupling_mode"].upper() == "TVB"

    def _get_output_interface_arguments(self, interface, ii=0):
        return interface

    def _get_input_interface_arguments(self, interface, ii=0):
        return interface

    def build_output_interface(self, interface, ii=0):
        interface = self._get_output_interface_arguments(interface, ii)
        return self._output_interface_type(**interface)

    def build_input_interface(self, interface, ii=0):
        return self._input_interface_type(**self._get_input_interface_arguments(interface, ii))

    def build_interfaces(self):
        self._output_interfaces = []
        for ii, interface in enumerate(self.output_interfaces):
            self._output_interfaces.append(self.build_output_interface(interface, ii))
        self._input_interfaces = []
        for ii, interface in enumerate(self.input_interfaces):
            self._input_interfaces.append(self.build_input_interface(interface, ii))

    def build(self):
        self.build_interfaces()
        return self._output_interfaces, self._input_interfaces

    @property
    def output_interfaces_filepath(self):
        return os.path.join(self.config.out.FOLDER_RES, self.__class__.__name__ + '_output_interfaces')

    @property
    def input_interfaces_filepath(self):
        return os.path.join(self.config.out.FOLDER_RES, self.__class__.__name__ + '_input_interfaces')

    def dump_interface(self, interface, filepath):
        dump_pickled_dict(interface, filepath)

    def load_interface(self, filepath):
        return load_pickled_dict(filepath)

    def dump_interfaces(self, interfaces, interfaces_filepath):
        number_of_leading_zeros = int(len(interfaces) / 10) + 1
        for ii, interface, in enumerate(interfaces):
            self.dump_interface(interface, interfaces_filepath + "_" + str(ii).zfill(number_of_leading_zeros))

    def dump_output_interfaces(self):
        self.dump_interfaces(self.output_interfaces, self.output_interfaces_filepath)

    def dump_input_interfaces(self):
        self.dump_interfaces(self.input_interfaces, self.input_interfaces_filepath)

    def dump_all_interfaces(self):
        self.dump_output_interfaces()
        self.dump_input_interfaces()

    def load_interfaces(self, input_or_output):
        input_or_output = input_or_output.lower()
        interfaces = []
        for ii, filepath in enumerate(glob.glob(getattr(self, "%s_interfaces_filepath" % input_or_output) + "*")):
            interfaces.append(self.load_interface(filepath))
        setattr(self, "%s_interfaces" % input_or_output, interfaces)

    def load_output_interfaces(self):
        self.load_interfaces("output")

    def load_input_interfaces(self):
        self.load_interfaces("input")

    def load_all_interfaces(self):
        self.load_output_interfaces()
        self.load_input_interfaces()

    def info(self, recursive=0):
        info = super(InterfaceBuilder, self).info(recursive=recursive)
        info['synchronization_time'] = self.synchronization_time
        info['synchronization_n_step'] = self.synchronization_n_step
        info['number_of_input_interfaces'] = self.number_of_input_interfaces
        info['number_of_output_interfaces'] = self.number_of_output_interfaces
        info['number_of_in_proxy_nodes'] = self.number_of_in_proxy_nodes
        info['number_of_out_proxy_nodes'] = self.number_of_out_proxy_nodes
        return info

    def _info_details_interfaces(self, interfaces, input_or_output):
        info = OrderedDict()
        for interface in interfaces:
            info.update({"%s_interfaces" % input_or_output: "properties:"})
            info.update(summary_info(interface))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = super(InterfaceBuilder, self).info_details(recursive=recursive, **kwargs)
        for input_or_output in ["input", "output"]:
            info.update(self._info_details_interfaces(getattr(self, "%s_interfaces" % input_or_output),
                                                      input_or_output))
        return info


class RemoteInterfaceBuilder(InterfaceBuilder):
    """RemoteInterfaceBuilder class"""

    __metaclass__ = ABCMeta

    _output_interface_type = SenderInterface
    _input_interface_type = ReceiverInterface

    _remote_sender_types = RemoteSenders
    _remote_receiver_types = RemoteReceivers

    _default_remote_sender_type = RemoteSenders.WRITER_TO_NUMPY
    _default_remote_receiver_type = RemoteReceivers.READER_FROM_NUMPY

    input_label = Attr(field_type=str, default="", required=True, label="Input label",
                       doc="""Input label of interface builder,
                              to be used for files' names and Receiver class instance labels, 
                              for the communication of data towards this CoSimulator""")

    output_label = Attr(field_type=str, default="", required=True, label="Output label",
                       doc="""Output label of interface builder,
                              to be used for files' names and Sender class instance labels, 
                              for the communication of data starting from this CoSimulator""")

    def configure(self):
        super(RemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(
            self._remote_sender_types, ["sender", "sender_model"], self._default_remote_sender_type)
        self._assert_input_interfaces_component_config(
            self._remote_receiver_types, ["receiver", "receiver_model"], self._default_remote_receiver_type)

    def _interface_communicator_label(self, label, ii=0):
        return "%s_%d" % (label, ii)

    def _file_path(self, label):
        return os.path.join(self.config.out.FOLDER_RES, "%s" % label)

    def _build_communicator(self, interface, communicator, ii):
        params = interface.pop(communicator + "_params", {})
        try:
            # Generate the communicator instance assuming a type
            interface[communicator] = interface[communicator](**params)
        except:
            # This is the case that the communicator instance is already generated
            for p, pval in params.items():
                setattr(interface[communicator], p, pval)
        # Set the interface communicator label if it is not already set by the user:
        if len(interface[communicator].label) == 0:
            interface[communicator].label = \
                    self._interface_communicator_label(np.where(communicator == "sender",
                                                                self.output_label, self.input_label).item(), ii)
        # If it is a file communicator, and the target/source filepath is not already set by the user
        # define a default filepath for the a file communicator
        if isinstance(interface[communicator], (WriterToFile, ReaderFromFile)):
            source_or_target = np.where(communicator == "sender", "target", "source").item()
            try:
                assert len(getattr(interface[communicator], source_or_target)) > 0
            except:
                setattr(interface[communicator], source_or_target, self._file_path(interface[communicator].label))
        return interface

    def _get_output_interface_arguments(self, interface, ii=0):
        return self._build_communicator(
            super(RemoteInterfaceBuilder, self)._get_output_interface_arguments(interface, ii), "sender", ii)

    def _get_input_interface_arguments(self, interface, ii=0):
        return self._build_communicator(
            super(RemoteInterfaceBuilder, self)._get_input_interface_arguments(interface, ii), "receiver", ii)


class RemoteTransformerBuilder(RemoteInterfaceBuilder, TransformerBuilder):
    """RemoteTransformerBuilder class"""

    _output_interface_type = RemoteTransformerInterface
    _input_interface_type = RemoteTransformerInterface

    _output_interfaces_type = RemoteTransformerInterfaces
    _input_interfaces_type = RemoteTransformerInterfaces

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
        self._assert_output_interfaces_component_config(
            self._remote_sender_types, ["sender", "sender_model"], self._default_remote_sender_type)
        self._assert_output_interfaces_component_config(
            self._remote_receiver_types, ["receiver", "receiver_model"], self._default_remote_receiver_type)
        self._configure_and_build_output_transformers()

    def _configure_input_interfaces(self):
        self._assert_input_interfaces_component_config(
            self._remote_sender_types, ["sender", "sender_model"], self._default_remote_sender_type)
        self._assert_input_interfaces_component_config(
            self._remote_receiver_types, ["receiver", "receiver_model"], self._default_remote_receiver_type)
        self._configure_and_build_input_transformers()

    def configure(self):
        if self.dt == 0.0:
            # From TVBInterfaceBuilder to TransformerBuilder:
            self.dt = self.tvb_dt
        super(RemoteInterfaceBuilder, self).configure()
        self._configure_output_interfaces()
        self._configure_input_interfaces()

    def _get_output_interface_arguments(self, interface, ii=0):
        interface = super(RemoteInterfaceBuilder, self)._get_output_interface_arguments(interface, ii)
        interface = self._build_communicator(interface, "receiver", ii)
        interface = self._build_communicator(interface, "sender", ii)
        return interface

    def _get_input_interface_arguments(self, interface, ii=0):
        interface = super(RemoteInterfaceBuilder, self)._get_input_interface_arguments(interface, ii)
        interface = self._build_communicator(interface, "receiver", ii)
        interface = self._build_communicator(interface, "sender", ii)
        return interface

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


class TVBtoSpikeNetRemoteTransformerBuilder(RemoteTransformerBuilder, TVBtoSpikeNetTransformerBuilder):
    """TVBtoSpikeNetRemoteTransformerBuilder class"""

    _output_interface_type = TVBtoSpikeNetRemoteTransformerInterface

    _output_interfaces_type = TVBtoSpikeNetRemoteTransformerInterfaces

    def _configure_and_build_output_transformers(self):
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformers(self, self.output_interfaces)

    def _configure_input_interfaces(self):
        pass

    def build(self):
        self.build_interfaces()
        output_interfaces = \
            self._output_interfaces_type(interfaces=self._output_interfaces,
                                         synchronization_time=self.synchronization_time,
                                         synchronization_n_step=self.synchronization_n_step)
        return output_interfaces


class SpikeNetToTVBRemoteTransformerBuilder(RemoteTransformerBuilder, SpikeNetToTVBTransformerBuilder):
    """SpikeNetToTVBRemoteTransformerBuilder class"""

    _input_interface_type = SpikeNetToTVBRemoteTransformerInterface

    _input_interfaces_type = SpikeNetToTVBRemoteTransformerInterfaces

    def _configure_and_build_input_transformers(self):
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformers(self, self.input_interfaces)

    def _configure_output_interfaces(self):
        pass

    def build(self):
        self.build_interfaces()
        input_interfaces = \
            self._input_interfaces_type(interfaces=self._input_interfaces,
                                        synchronization_time=self.synchronization_time,
                                        synchronization_n_step=self.synchronization_n_step)
        return input_interfaces


class TVBspikeNetRemoteTransformerBuilder(RemoteTransformerBuilder,
                                          TVBtoSpikeNetTransformerBuilder, SpikeNetToTVBTransformerBuilder):
    """TVBspikeNetRemoteTransformerBuilder class"""

    _output_interface_type = TVBtoSpikeNetRemoteTransformerInterface
    _input_interface_type = SpikeNetToTVBRemoteTransformerInterface

    _output_interfaces_type = TVBtoSpikeNetRemoteTransformerInterfaces
    _input_interfaces_type = SpikeNetToTVBRemoteTransformerInterfaces

    def _configure_and_build_output_transformers(self):
        TVBtoSpikeNetTransformerBuilder.configure_and_build_transformers(self, self.output_interfaces)

    def _configure_and_build_input_transformers(self):
        SpikeNetToTVBTransformerBuilder.configure_and_build_transformers(self, self.input_interfaces)
