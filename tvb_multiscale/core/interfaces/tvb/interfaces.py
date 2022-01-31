# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, Int, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.datatypes import HasTraits
from tvb_multiscale.core.interfaces.base.interfaces import BaseInterface, \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetInputInterface, SpikeNetOutputInterface, SpikeNetOutputInterfaces, SpikeNetInputInterfaces
from tvb_multiscale.core.interfaces.base.transformers.models.base import Transformer


class TVBInterface(HasTraits):
    __metaclass__ = ABCMeta

    """TVBInterface base class for interfaces sending/receivng data from/for TVB to/from a transformer of cosimulator"""

    proxy_inds = NArray(
        dtype=np.int,
        label="Indices of TVB proxy nodes",
        doc="""Indices of TVB proxy nodes""",
        required=True,
    )

    voi = NArray(
        dtype=int,
        label="Cosimulation model state variables' indices",
        doc="""Indices of model's variables of interest (VOI)""",
        required=True)

    voi_loc = np.array([])

    voi_labels = NArray(
        dtype='U128',
        label="Cosimulation model state variables' labels",
        doc=""""Labels of model's variables of interest (VOI)""",
        required=True,
    )

    @property
    def label(self):
        return "%s: %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                extract_integer_intervals(self.proxy_inds))

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds.shape[0]

    @property
    def n_voi(self):
        return self.voi.shape[0]

    def _set_local_indices(self, inds, simulator_inds):
        simulator_inds_list = list(simulator_inds)
        return np.array([simulator_inds_list.index(ind) for ind in inds])

    def set_local_voi_indices(self, monitor_voi):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor or CosimHistory"""
        self.voi_loc = self._set_local_indices(self.voi, monitor_voi)

    @abstractmethod
    def set_local_indices(self, *args):
        pass

    def print_str(self, sender_not_receiver=None):
        if sender_not_receiver is True:
            tvb_source_or_target = "Sender"
        elif sender_not_receiver is False:
            tvb_source_or_target = "Receiver"
        else:
            tvb_source_or_target = ""
        return "\nTVB state variable indices: %s" \
               "\nTVB %s proxy nodes' indices: %s" % \
               (str(self.voi.tolist()),
                tvb_source_or_target, extract_integer_intervals(self.proxy_inds, print=True))

    def info(self):
        info = super(TVBInterface, self).info()
        info["n_voi"] = self.n_voi
        info["number_of_proxy_nodes"] = self.number_of_proxy_nodes
        return info


class TVBOutputInterface(TVBInterface):

    """TVBOutputInterface base class for interfaces sending data from TVB to a transformer or cosimulator"""

    monitor_ind = Int(label="Monitor indice",
                      doc="Indice of monitor to get data from",
                      required=True,
                      default=0)

    coupling_mode = Attr(
        label="Coupling mode",
        field_type=str,
        doc="""Interface coupling mode. 'TVB' indicates that large scale coupling is computed in TVB 
                   before being sent to a cosimulator. 
                   Default 'spikeNet', which entails that 
                   large scale coupling for regions modeled outside TVB is handled by the cosimulator.""",
        required=True,
        default="spikeNet"
    )

    @property
    def label(self):
        return "%s: %s (%s) ->" % (self.__class__.__name__, str(self.voi_labels),
                                   extract_integer_intervals(self.proxy_inds))

    def set_local_indices(self, monitor_voi):
        self.set_local_voi_indices(monitor_voi)

    def print_str(self):
        return super(TVBOutputInterface, self).print_str(sender_not_receiver=True)

    def __call__(self, data):
        # Assume a single mode, and reshape from TVB (time, voi, proxy)...
        if data.shape[1] == 1:
            # ...to (proxy, time)
            return data[:, 0, :].T
        else:
            # ...or (proxy, time, voi)
            return np.transpose(data, (2, 0, 1))


class TVBInputInterface(TVBInterface):

    """TVBInputInterface base class for interfaces receiving data for TVB from a transformer or cosimulator"""

    proxy_inds_loc = np.array([])

    @property
    def label(self):
        return "%s: %s (%s) <-" % (self.__class__.__name__, str(self.voi_labels),
                                   extract_integer_intervals(self.proxy_inds))

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        self.set_local_voi_indices(simulator_voi)
        self.proxy_inds_loc = self._set_local_indices(self.proxy_inds, simulator_proxy_inds)

    def print_str(self):
        return super(TVBInputInterface, self).print_str(sender_not_receiver=False)

    def __call__(self, data):
        # Assume a single mode, and reshape from (proxy, (voi,) time) to TVB (time, voi, proxy)
        if data.ndim < 3:
            # if there was no voi dimension
            data = data.T
            data = data[:, None, :]
        else:
            # or if there was already a voi dimension
            data = np.transpose(data, (1, 2, 0))
        return data


class TVBSenderInterface(SenderInterface, TVBOutputInterface):

    """TVBSenderInterface class to send data to a remote transformer or cosimulator.
    """

    def __call__(self, data):
        return self.send(TVBOutputInterface.__call__(self, data))

    def info(self):
        info = SenderInterface.info(self)
        info.update(TVBOutputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = SenderInterface.info_details(self, **kwargs)
        info.update(TVBOutputInterface.info_details(self))
        return info


class TVBTransformerSenderInterface(TransformerSenderInterface, TVBOutputInterface):

    """TVBTransformerSenderInterface class to get data from TVB, transform them locally,
       and, then, send them to a -potentially remote- cosimulator.
    """

    def __call__(self, data):
        return self.transform_send(TVBOutputInterface.__call__(self, data))

    def info(self):
        info = TransformerSenderInterface.info(self)
        info.update(TVBOutputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = TransformerSenderInterface.info_details(self, **kwargs)
        info.update(TVBOutputInterface.info_details(self))
        return info


class TVBReceiverInterface(ReceiverInterface, TVBInputInterface):

    """TVBReceiverInterface class to receive data for TVB from a remote transformer or cosimulator.
    """

    def __call__(self):
        return TVBInputInterface.__call__(self, self.receive())

    def info(self):
        info = ReceiverInterface.info(self)
        info.update(TVBInputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = ReceiverInterface.info_details(self, **kwargs)
        info.update(TVBInputInterface.info_details(self))
        return info


class TVBReceiverTransformerInterface(ReceiverTransformerInterface, TVBInputInterface):

    """TVBReceiverTransformerInterface class receive data from a -potentially remote- cosimulator,
       and, then, transform them and set them to TVB locally.
    """

    def __call__(self):
        return TVBInputInterface.__call__(self, self.receive_transform())

    def info(self):
        info = ReceiverTransformerInterface.info(self)
        info.update(TVBInputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = ReceiverTransformerInterface.info_details(self, **kwargs)
        info.update(TVBInputInterface.info_details(self))
        return info


class TVBtoSpikeNetInterface(TVBOutputInterface, SpikeNetInputInterface, BaseInterface):

    """TVBtoSpikeNetInterface class to get data from TVB, transform them,
       and finally set them to the Spiking Network cosimulator, all processes taking place in shared memmory.
    """

    transformer = Attr(
        label="Transformer",
        field_type=Transformer,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    @property
    def label(self):
        return "%s: %s (%s) -> %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                           extract_integer_intervals(self.proxy_inds),
                                           str(self.populations), extract_integer_intervals(self.spiking_proxy_inds))

    def configure(self):
        super(TVBtoSpikeNetInterface, self).configure()
        self.transformer.configure()

    def reshape_data(self, data):
        return TVBOutputInterface.__call__(self, data)

    def __call__(self, data):
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = self.reshape_data(data[1])
        self.transformer()
        return self.set_proxy_data([self.transformer.output_time, self.transformer.output_buffer])

    def info(self):
        info = BaseInterface.info(self)
        info.update(SpikeNetInputInterface.info(self))
        info.update(TVBOutputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = BaseInterface.info_details(self)
        info.update(SpikeNetInputInterface.info_details(self, **kwargs))
        info.update(TVBOutputInterface.info_details(self))
        return info


class SpikeNetToTVBInterface(TVBInputInterface, SpikeNetOutputInterface, BaseInterface):

    """SpikeNetToTVBInterface class to get data the Spiking Network co-simulator, transform them,
       and finally set them to TVB, all processes taking place in shared memmory.
    """

    transformer = Attr(
        label="Transformer",
        field_type=Transformer,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    def configure(self):
        super(SpikeNetToTVBInterface, self).configure()
        self.transformer.configure()

    @property
    def label(self):
        return "%s: %s (%s) <- %s (%s)" % (self.__class__.__name__, str(self.voi_labels),
                                           extract_integer_intervals(self.proxy_inds),
                                           str(self.populations), extract_integer_intervals(self.spiking_proxy_inds))

    def reshape_data(self):
        return TVBInputInterface.__call__(self, self.transformer.output_buffer)

    def __call__(self):
        data = self.get_proxy_data()
        if data[0][1] <= data[0][0]:
            return None
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return [self.transformer.output_time, self.reshape_data()]

    def info(self):
        info = BaseInterface.info(self)
        info.update(SpikeNetOutputInterface.info(self))
        info.update(TVBInputInterface.info(self))
        return info

    def info_details(self, **kwargs):
        info = BaseInterface.info_details(self)
        info.update(SpikeNetOutputInterface.info_details(self, **kwargs))
        info.update(TVBInputInterface.info_details(self))
        return info


class TVBInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """SpikeNetInterfaces abstract base class"""

    @property
    def voi(self):
        return self._loop_get_from_interfaces("voi")

    @property
    def voi_unique(self):
        return np.unique(self._loop_get_from_interfaces("voi"))

    @property
    def voi_labels(self):
        return self._loop_get_from_interfaces("voi_labels")

    @property
    def voi_labels_unique(self):
        return np.unique(self._loop_get_from_interfaces("voi_labels"))

    @property
    def proxy_inds(self):
        return self._loop_get_from_interfaces("spiking_proxy_inds")

    @property
    def proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def n_vois(self):
        return self.voi_unique.shape[0]

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    @abstractmethod
    def set_local_indices(self, *args):
        pass

    def info(self):
        info = super(TVBInterfaces, self).info()
        info["n_vois"] = self.n_vois
        info["voi"] = self.voi_unique
        info["voi_labels"] = self.voi_labels_unique
        info["number_of_proxy_nodes"] = self.number_of_proxy_nodes
        return info

    def info_details(self, **kwargs):
        info = super(TVBInterfaces, self).info_details(**kwargs)
        info["proxy_inds_unique"] = self.proxy_inds_unique
        return info


class TVBOutputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBOutputInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and sends data to them"""

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    def set_local_indices(self, cosim_monitors):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(cosim_monitors[interface.monitor_ind].voi)

    def _compute_interface_times(self, interface, data):
        times = np.array([np.round(data[interface.monitor_ind][0][0] / self.dt),  # start_time_step
                          np.round(data[interface.monitor_ind][0][-1] / self.dt)]).astype("i")  # end_time_step
        if interface.coupling_mode.upper() != "TVB":
            times += self.synchronization_n_step  # adding the synchronization time when not a coupling interface
        return times

    def __call__(self, data):
        for interface in self.interfaces:
            #                 data values !!! assuming only 1 mode!!! -> shape (times, vois, proxys):
            interface([self._compute_interface_times(interface, data),
                       data[interface.monitor_ind][1][:, interface.voi_loc][:, :, interface.proxy_inds, 0]])

    def info(self):
        info = BaseInterfaces.info()
        info.update(TVBInterfaces.info())
        return info

    def info_details(self, **kwargs):
        info = BaseInterfaces.info_details(self, **kwargs)
        info.update(TVBInterfaces.info_details(self, **kwargs))
        return info


class TVBInputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBInputInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them"""

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and spiking_proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSimulators' vois and spiking_proxy_inds,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(simulator_voi, simulator_proxy_inds)

    def _set_data_from_interface(self, cosim_updates, interface, data, good_cosim_update_values_shape):
        # Convert start and end input_time step to a vector of integer input_time steps:
        time_steps = np.arange(data[0][0], data[0][1] + 1).astype("i")
        cosim_updates[
            (time_steps % good_cosim_update_values_shape[0])[:, None, None],
            interface.voi_loc[None, :, None],  # indices specific to cosim_updates needed here
            interface.proxy_inds_loc[None, None, :],  # indices specific to cosim_updates needed here
            0] = np.copy(data[1])  # !!! assuming only 1 mode!!!
        return cosim_updates, time_steps

    def _get_from_interface(self, interface, cosim_updates, all_time_steps, good_cosim_update_values_shape):
        data = interface()  # [start_and_time_steps, values]
        if data is not None:
            cosim_updates, time_steps = \
                self._set_data_from_interface(cosim_updates, interface, data, good_cosim_update_values_shape)
            all_time_steps += time_steps.tolist()
            return cosim_updates, all_time_steps
        else:
            return cosim_updates, all_time_steps

    def _prepare_cosim_upadate(self, good_cosim_update_values_shape):
        cosim_updates = np.empty(good_cosim_update_values_shape).astype(float)
        cosim_updates[:] = np.NAN
        all_time_steps = []
        return cosim_updates, all_time_steps

    def get_inputs(self, cosim_updates, all_time_steps, good_cosim_update_values_shape):
        if len(all_time_steps):
            all_time_steps = np.unique(all_time_steps)
            return [all_time_steps, cosim_updates[all_time_steps % good_cosim_update_values_shape[0]]]
        else:
            return [all_time_steps, cosim_updates]

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates, all_time_steps = self._prepare_cosim_upadate(good_cosim_update_values_shape)
        for interface in self.interfaces:
            cosim_updates, all_time_steps = \
                self._get_from_interface(interface, cosim_updates, all_time_steps, good_cosim_update_values_shape)
        return self.get_inputs(cosim_updates, all_time_steps, good_cosim_update_values_shape)

    def info(self):
        info = BaseInterfaces.info()
        info.update(TVBInterfaces.info())
        return info

    def info_details(self, **kwargs):
        info = BaseInterfaces.info_details(self, **kwargs)
        info.update(TVBInterfaces.info_details(self, **kwargs))
        return info


class TVBtoSpikeNetInterfaces(TVBOutputInterfaces, SpikeNetInputInterfaces):

    """TVBtoSpikeNetInterfaces"""

    def info(self):
        info = SpikeNetInputInterfaces.info()
        info.update(TVBOutputInterfaces.info())
        return info

    def info_details(self, **kwargs):
        info = SpikeNetInputInterfaces.info_details(self, **kwargs)
        info.update(TVBOutputInterfaces.info_details(self))
        return info


class SpikeNetToTVBInterfaces(TVBInputInterfaces, SpikeNetOutputInterfaces):

    """SpikeNetToTVBInterfaces"""

    def info(self):
        info = SpikeNetOutputInterfaces.info()
        info.update(TVBInputInterfaces.info())
        return info

    def info_details(self, **kwargs):
        info = SpikeNetOutputInterfaces.info_details(self, **kwargs)
        info.update(TVBInputInterfaces.info_details(self))
        return info


class TVBtoSpikeNetModels(Enum):
    RATE = 0
    SPIKES = 1
    CURRENT = 2


class SpikeNetToTVBModels(Enum):
    SPIKES = 0
    VOLTAGE = 1
