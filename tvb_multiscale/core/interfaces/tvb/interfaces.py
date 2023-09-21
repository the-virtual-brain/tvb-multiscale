# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, Int, NArray, List
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.interfaces.base.interfaces import BaseInterface, SenderInterface, ReceiverInterface, \
    BaseInterfaces
from tvb_multiscale.core.interfaces.base.transformers.interfaces import TransformerInterface, TransformerInterfaces, \
    TransformerSenderInterface, ReceiverTransformerInterface
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetInputInterface, SpikeNetOutputInterface, SpikeNetOutputInterfaces, SpikeNetInputInterfaces
from tvb_multiscale.core.interfaces.base.transformers.models.base import Transformer


class TVBInterface(BaseInterface):
    __metaclass__ = ABCMeta

    """TVBInterface base class for interfaces sending/receivng data from/for TVB to/from a transformer of cosimulator"""

    proxy_inds = NArray(
        dtype=int,
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

    def set_local_voi_indices(self, voi, monitor_voi):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor or CosimHistory"""
        self.voi_loc = self._set_local_indices(voi, monitor_voi)

    @abstractmethod
    def set_local_indices(self, *args):
        pass

    def info(self, recursive=0):
        info = super(TVBInterface, self).info(recursive=recursive)
        info["n_voi"] = self.n_voi
        info["number_of_proxy_nodes"] = self.number_of_proxy_nodes
        return info


class TVBOutputInterface(TVBInterface):

    """TVBOutputInterface base class for interfaces sending data from TVB to a transformer or cosimulator"""

    monitor_ind = Int(label="Monitor indice",
                      doc="Indice of monitor to get data from",
                      required=True,
                      default=0)

    cvoi = NArray(
        dtype=int,
        label="Cosimulation model coupling variables' indices",
        doc="""Indices of model's variables of interest (VOI) for TVB coupling interfaces""",
        required=False)

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

    def set_local_voi_indices(self, monitor_voi):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor or CosimHistory"""
        if self.coupling_mode.upper() == "TVB":
            self.voi_loc = self._set_local_indices(self.cvoi, monitor_voi)
        else:
            self.voi_loc = self._set_local_indices(self.voi, monitor_voi)

    def set_local_indices(self, monitor_voi):
        self.set_local_voi_indices(monitor_voi)

    def __call__(self, data):
        # Assume a single mode, and reshape from TVB (time, voi, proxy)...
        if data[1].shape[1] == 1:
            # ...to (proxy, time)
            data[1] = data[1][:, 0, :].T
        else:
            # ...or (proxy, time, voi)
            data[1] = np.transpose(data[1], (2, 0, 1))
        return data


class TVBInputInterface(TVBInterface):

    """TVBInputInterface base class for interfaces receiving data for TVB from a transformer or cosimulator"""

    proxy_inds_loc = np.array([])

    @property
    def label(self):
        return "%s: %s (%s) <-" % (self.__class__.__name__, str(self.voi_labels),
                                   extract_integer_intervals(self.proxy_inds))

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        self.set_local_voi_indices(self.voi, simulator_voi)
        self.proxy_inds_loc = self._set_local_indices(self.proxy_inds, simulator_proxy_inds)

    def __call__(self, data):
        if data is None:
            return None
        # Assume a single mode, and reshape from (proxy, (voi,) time) to TVB (time, voi, proxy)
        if data[1].ndim < 3:
            # if there was no voi dimension
            data[1] = data[1].T
            data[1] = data[1][:, None, :]
        else:
            # or if there was already a voi dimension
            data[1] = np.transpose(data[1], (1, 2, 0))
        return data


class TVBTransformerOutputInterface(TVBOutputInterface, TransformerInterface):

    """TVBTransformerOutputInterface class to get data from TVB, transform them locally,
       and, then, output them towards a cosimulator.
    """

    def __call__(self, data):
        return TransformerInterface.__call__(self, TVBOutputInterface.__call__(self, data))


class TVBSenderInterface(TVBOutputInterface, SenderInterface):

    """TVBSenderInterface class to send data to a remote transformer or cosimulator.
    """

    def __call__(self, data):
        return SenderInterface.send(self, TVBOutputInterface.__call__(self, data))


class TVBTransformerSenderInterface(TVBOutputInterface, TransformerSenderInterface):

    """TVBTransformerSenderInterface class to get data from TVB, transform them locally,
       and, then, send them to a -potentially remote- cosimulator.
    """

    def __call__(self, data):
        return TransformerSenderInterface.transform_and_send(self, TVBOutputInterface.__call__(self, data))


class TVBInputTransformerInterface(TVBInputInterface, TransformerInterface):

    """TVBInputTransformerInterface class to get data for TVB from a remote transformer or cosimulator
       and transform them.
    """

    def __call__(self, data):
        return TVBInputInterface.__call__(self, TransformerInterface.__call__(self, data))


class TVBReceiverInterface(TVBInputInterface, ReceiverInterface):

    """TVBReceiverInterface class to receive data for TVB from a remote transformer or cosimulator.
    """

    def __call__(self):
        return TVBInputInterface.__call__(self, ReceiverInterface.receive(self))


class TVBReceiverTransformerInterface(TVBInputInterface, ReceiverTransformerInterface):

    """TVBReceiverTransformerInterface class receive data from a -potentially remote- cosimulator,
       and, then, transform them and set them to TVB locally.
    """

    def __call__(self):
        return TVBInputInterface.__call__(self, ReceiverTransformerInterface.receive_transform(self))


class TVBtoSpikeNetInterface(TVBOutputInterface, SpikeNetInputInterface):

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

    def transform_data(self, data):
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return [self.transformer.output_time, self.transformer.output_buffer]

    def send_data(self, data=None):
        if data is None:
            data = [self.transformer.output_time, self.transformer.output_buffer]
        return self.set_proxy_data(data)

    def __call__(self, data):
        return self.send_data(
                    self.transform_data(
                        self.reshape_data(data)))

    def info(self, recursive=0):
        info = SpikeNetInputInterface.info(self, recursive=recursive)
        info.update(TVBOutputInterface.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = SpikeNetInputInterface.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBOutputInterface.info_details(self, recursive=recursive))
        return info


class SpikeNetToTVBInterface(TVBInputInterface, SpikeNetOutputInterface):

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

    def receive_data(self):
        return self.get_proxy_data()

    def reshape_data(self, data=None):
        if data is None:
            data = [self.transformer.output_time, self.transformer.output_buffer]
        return TVBInputInterface.__call__(self, data)

    def transform_data(self, data):
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return [self.transformer.output_time, self.transformer.output_buffer]

    def __call__(self):
        data = self.receive_data()
        if data is None or data[0][1] < data[0][0]:
            return None
        return self.reshape_data(
            self.transform_data(data))

    def info(self, recursive=0):
        info = SpikeNetOutputInterface.info(self, recursive=recursive)
        info.update(TVBInputInterface.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = SpikeNetOutputInterface.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBInputInterface.info_details(self, recursive=recursive))
        return info


class TVBInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """SpikeNetInterfaces abstract base class"""

    interfaces = List(of=TVBInterface)

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
        return self._loop_get_from_interfaces("proxy_inds")

    @property
    def proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("proxy_inds"))

    @property
    def n_vois(self):
        return self.voi_unique.shape[0]

    @property
    def number_of_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    @abstractmethod
    def set_local_indices(self, *args):
        pass

    def info(self, recursive=0):
        info = super(TVBInterfaces, self).info(recursive=recursive)
        info["n_vois"] = self.n_vois
        info["voi"] = self.voi_unique
        info["voi_labels"] = self.voi_labels_unique
        info["number_of_proxy_nodes"] = self.number_of_proxy_nodes
        return info

    def info_details(self, recursive=0, **kwargs):
        info = super(TVBInterfaces, self).info_details(recursive=recursive, **kwargs)
        info["proxy_inds_unique"] = self.proxy_inds_unique
        return info


class TVBOutputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBOutputInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and outputs data to them"""

    interfaces = List(of=TVBOutputInterface)

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True)

    min_idelay = Int(label="Minimum idelay",
                     doc="TVB minimum delay time steps",
                     required=True)

    def set_local_indices(self, cosim_monitors):
        """Method to set the correct voi indices with reference to the linked TVB CosimMonitor,
           for each cosimulation"""
        for interface in self.interfaces:
            interface.set_local_indices(cosim_monitors[interface.monitor_ind].voi)

    def _compute_interface_times(self, interface, data):
        times = np.array([np.round(data[interface.monitor_ind][0][0] / self.dt),  # start_time_step
                          np.round(data[interface.monitor_ind][0][-1] / self.dt)]).astype("i")  # end_time_step
        if interface.coupling_mode.upper() != "TVB":
            # As we cannot schedule spikeNet devices with times in the past,
            # we need to add here TVB min_delay time,
            # and subtract it from the connectome delays, in case coupling is "spikeNet".
            # Nothing needs to be done for coupling "TVB", which is scheduled "just in time",
            # i.e., for the next synchronization_time period, to "spikeNet" devices
            times += self.min_idelay
        return times

    def __call__(self, data):
        outputs = []
        for ii, interface in enumerate(self.interfaces):
            #                 data values !!! assuming only 1 mode!!! -> shape (times, vois, proxys):
            outputs.append(interface([self._compute_interface_times(interface, data),
                           data[interface.monitor_ind][1][:, interface.voi_loc][:, :, interface.proxy_inds, 0],
                           ii]))
        return outputs

    def info(self, recursive=0):
        info = BaseInterfaces.info(self, recursive=recursive)
        info.update(TVBInterfaces.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = BaseInterfaces.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBInterfaces.info_details(self, recursive=recursive, **kwargs))
        return info


class TVBTransformerOutputInterfaces(TVBOutputInterfaces):
    """TVBTransformerSenderInterfaces class holds a list of TVB interfaces to cosimulator
        and transforms and sends data to them"""

    interfaces = List(of=TVBTransformerOutputInterface)

    pass


class TVBSenderInterfaces(TVBOutputInterfaces):

    """TVBSenderInterfaces class holds a list of TVB interfaces to transformer/cosimulator
        and sends data to them"""

    interfaces = List(of=TVBSenderInterface)

    pass


class TVBTransformerSenderInterfaces(TVBSenderInterfaces):
    """TVBTransformerSenderInterfaces class holds a list of TVB interfaces to cosimulator
        and transforms and sends data to them"""

    interfaces = List(of=TVBTransformerSenderInterface)

    pass


class TVBInputInterfaces(BaseInterfaces, TVBInterfaces):

    """TVBInputInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them"""

    interfaces = List(of=TVBInputInterface)

    def set_local_indices(self, simulator_voi, simulator_proxy_inds):
        """Method to get the correct indices of voi and proxy_inds,
           adjusted to the contents, shape etc of the cosim_updates,
           based on TVB CoSimulators' vois and proxy_inds,
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

    def _get_from_interface(self, input_data, interface, cosim_updates, all_time_steps, good_cosim_update_values_shape):
        if input_data is not None:
            cosim_updates, time_steps = \
                self._set_data_from_interface(cosim_updates, interface, input_data, good_cosim_update_values_shape)
            all_time_steps += time_steps.tolist()
            return cosim_updates, all_time_steps
        else:
            return cosim_updates, all_time_steps

    def _prepare_cosim_update(self, good_cosim_update_values_shape):
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

    def __call__(self, input_datas, good_cosim_update_values_shape):
        cosim_updates, all_time_steps = self._prepare_cosim_update(good_cosim_update_values_shape)
        for ii, (interface, input_data) in enumerate(zip(self.interfaces, input_datas)):
            if len(input_data) > 2:
                assert input_data[2] == ii
                input_data = input_data[:2]
            cosim_updates, all_time_steps = \
                self._get_from_interface(interface(input_data), interface, cosim_updates,
                                         all_time_steps, good_cosim_update_values_shape)
        return self.get_inputs(cosim_updates, all_time_steps, good_cosim_update_values_shape)

    def info(self, recursive=0):
        info = BaseInterfaces.info(self, recursive=recursive)
        info.update(TVBInterfaces.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = BaseInterfaces.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBInterfaces.info_details(self, recursive=recursive, **kwargs))
        return info


class TVBInputTransformerInterfaces(TVBInputInterfaces):

    """TVBInputTransformerInterfaces class holds a list of TVB interfaces from cosimulator
       and gets data from them and transforms them"""

    interfaces = List(of=TVBInputTransformerInterface)

    pass


class TVBReceiverInterfaces(TVBInputInterfaces):

    """TVBReceiverInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them"""

    interfaces = List(of=TVBReceiverInterface)

    def _get_from_interface(self, interface, cosim_updates, all_time_steps, good_cosim_update_values_shape):
        data = interface()  # [start_and_time_steps, values]
        return super(TVBReceiverInterfaces, self)._get_from_interface(
            data, interface, cosim_updates, all_time_steps, good_cosim_update_values_shape)

    def __call__(self, good_cosim_update_values_shape):
        cosim_updates, all_time_steps = self._prepare_cosim_update(good_cosim_update_values_shape)
        for interface in self.interfaces:
            cosim_updates, all_time_steps = \
                self._get_from_interface(interface, cosim_updates, all_time_steps, good_cosim_update_values_shape)
        return self.get_inputs(cosim_updates, all_time_steps, good_cosim_update_values_shape)


class TVBReceiverTransformerInterfaces(TVBReceiverInterfaces):

    """TVBReceiverTransformerInterfaces class holds a list of TVB interfaces from cosimulator
       and receives data from them and transforms them"""

    interfaces = List(of=TVBReceiverTransformerInterface)

    pass


class TVBtoSpikeNetInterfaces(TVBOutputInterfaces, SpikeNetInputInterfaces):

    """TVBtoSpikeNetInterfaces"""

    interfaces = List(of=TVBtoSpikeNetInterface)

    def info(self, recursive=0):
        info = SpikeNetInputInterfaces.info(self, recursive=recursive)
        info.update(TVBOutputInterfaces.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = SpikeNetInputInterfaces.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBOutputInterfaces.info_details(self, recursive=recursive))
        return info


class SpikeNetToTVBInterfaces(TVBReceiverInterfaces, SpikeNetOutputInterfaces):

    """SpikeNetToTVBInterfaces"""

    interfaces = List(of=SpikeNetToTVBInterface)

    def info(self, recursive=0):
        info = SpikeNetOutputInterfaces.info(self, recursive=recursive)
        info.update(TVBInputInterfaces.info(self, recursive=recursive))
        return info

    def info_details(self, recursive=0, **kwargs):
        info = SpikeNetOutputInterfaces.info_details(self, recursive=recursive, **kwargs)
        info.update(TVBInputInterfaces.info_details(self, recursive=recursive))
        return info


class TVBtoSpikeNetTransformerInterface(TransformerInterface):
    """TVBtoSpikeNetTransformerInterface  class for TVB -> spikeNet transformations
       - setting data received for a Transformer from TVB,
       - performing the Transformer computation,
       - and outputing data to be sent to the (spiking) cosimulator.
    """

    pass


class SpikeNetToTVBTransformerInterface(TransformerInterface):
    """SpikeNetToTVBTransformerInterface  class for spikeNet -> TVB transformations
       - setting data from a (spiking) cosimulator for a Transformer,
       - performing the Transformer computation,
       - and outputing data to be sent to TVB.
    """

    pass


class TVBtoSpikeNetTransformerInterfaces(TransformerInterfaces):

    """TVBtoSpikeNetTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetTransformerInterface)


class SpikeNetToTVBTransformerInterfaces(TransformerInterfaces):

    """SpikeNetToTVBTransformerInterfaces"""

    interfaces = List(of=SpikeNetToTVBTransformerInterface)


class TVBtoSpikeNetModels(Enum):
    RATE = 0
    SPIKES = 1
    CURRENT = 2


class SpikeNetToTVBModels(Enum):
    SPIKES = 0
    VOLTAGE = 1
