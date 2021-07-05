# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr, Float, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals, list_of_dicts_to_dict_of_lists

from tvb_multiscale.core.interfaces.base.interfaces import \
    SenderInterface, ReceiverInterface, TransformerSenderInterface, ReceiverTransformerInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class SpikeNetInterface(HasTraits):

    __metaclass__ = ABCMeta

    """SpikeNetInterface abstract base class."""

    spiking_network = None
    # spiking_network = Attr(label="Spiking Network",
    #                        doc="""The instance of SpikingNetwork class""",
    #                        field_type=SpikingNetwork,
    #                        required=True)

    spiking_proxy_inds = NArray(
        dtype=np.int,
        label="Indices of Spiking Network proxy nodes",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
    )

    populations = NArray(
        dtype='U128',
        label="Spiking Network populations",
        doc="""Spiking Network populations associated to the interface""",
        required=True)

    @property
    def label(self):
        return "%s: %s (%s)" % (self.__class__.__name__, str(self.populations),
                                extract_integer_intervals(self.spiking_proxy_inds))

    @property
    def number_of_proxy_nodes(self):
        return self.spiking_proxy_inds.shape[0]

    @property
    def number_of_populations(self):
        return self.populations.shape[0]

    def _get_proxy_gids(self, nest_devices):
        return np.array(nest_devices.do_for_all("gids", return_type="values")).flatten()

    @property
    @abstractmethod
    def proxy_gids(self):
        pass

    @property
    def number_of_proxy_gids(self):
        return self.proxy_gids.shape[0]

    def print_str(self, sender_not_receiver=None):
        if sender_not_receiver is True:
            spikeNet_source_or_target = "Sender"
        elif sender_not_receiver is False:
            spikeNet_source_or_target = "Receiver"
        else:
            spikeNet_source_or_target = ""
        return "\nSpiking Network populations: %s" \
               "\nSpiking Network %s proxy nodes' \n" \
               "indices: %s\nand gids: %s" % \
               (str(self.populations.tolist()),
                spikeNet_source_or_target,
                extract_integer_intervals(self.spiking_proxy_inds, print=True),
                extract_integer_intervals(self.proxy_gids, print=True))


class SpikeNetOutputInterface(SpikeNetInterface):

    """SpikeNetOutputInterface base class."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of SpikeNetOutputDeviceSet implementing a proxy node 
                        sending outputs from the spiking network to the co-simulator""",
                 field_type=SpikeNetOutputDeviceSet,
                 required=True)

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    times = NArray(
        dtype=np.int,
        label="Indices of Spiking Network proxy nodes",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
        default=np.array([1, 0])
    )

    def configure(self):
        super(SpikeNetOutputInterface, self).configure()
        self.proxy.configure()
        if len(self.model) == 0:
            self.model = self.proxy.model

    @property
    def label(self):
        return "%s: %s (%s) ->" % (self.__class__.__name__, str(self.populations),
                                   extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        return super(SpikeNetOutputInterface, self).print_str(sender_not_receiver=True)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)

    def get_proxy_data(self):
        data = self.proxy()
        if len(data[0]) == 2:
            # This will work for multimeters:
            self.times = np.array([np.round(data[0][0] / self.dt),  # start_time_step
                                   np.round(data[0][1] / self.dt)]).astype("i")  # end_time_step
        else:
            # This will work for spike recorders:
            time = np.int(np.round(self.time / self.dt))
            times = self.times.copy()
            if time > times[1]:
                times[0] = times[1] + 1
                times[1] = time
            self.times = times
        return [self.times, data[-1]]


class SpikeNetInputInterface(SpikeNetInterface):

    """SpikeNetInputInterface base class."""

    proxy = Attr(label="Proxy",
                 doc="""An instance of SpikeNetInputDeviceSet implementing a proxy node 
                        receiving inputs from the co-simulator as an input to the spiking network""",
                 field_type=SpikeNetInputDeviceSet,
                 required=True)

    def configure(self):
        super(SpikeNetInputInterface, self).configure()
        self.proxy.configure()
        if len(self.model) == 0:
            self.model = self.proxy.model

    @property
    def label(self):
        return "%s: %s (%s) <-" % (self.__class__.__name__, str(self.populations),
                                   extract_integer_intervals(self.spiking_proxy_inds))

    def print_str(self):
        return super(SpikeNetInputInterface, self).print_str(sender_not_receiver=False)

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)

    def set_proxy_data(self, data):
        return self.proxy(data)


class SpikeNetSenderInterface(SpikeNetOutputInterface, SenderInterface):

    """SpikeNetSenderInterface class."""

    def configure(self):
        SpikeNetOutputInterface.configure(self)
        SenderInterface.configure(self)

    def __call__(self):
        return self.send(self.get_proxy_data())

    def print_str(self):
        return SenderInterface.print_str(self) + SpikeNetOutputInterface.print_str(self)


class SpikeNetReceiverInterface(SpikeNetInputInterface, ReceiverInterface):

    """SpikeNetReceiverInterface class."""

    def configure(self):
        SpikeNetInputInterface.configure(self)
        ReceiverInterface.configure(self)

    def __call__(self):
        return self.set_proxy_data(self.receive())

    def print_str(self):
        return ReceiverInterface.print_str(self) + SpikeNetInputInterface.print_str(self)


class SpikeNetTransformerSenderInterface(SpikeNetOutputInterface, TransformerSenderInterface):

    """SpikeNetTransformerSenderInterface class."""

    def configure(self):
        SpikeNetOutputInterface.configure(self)
        TransformerSenderInterface.configure(self)

    def __call__(self):
        return self.transform_send(self.get_proxy_data())

    def print_str(self):
        return TransformerSenderInterface.print_str(self) + SpikeNetOutputInterface.print_str(self)


class SpikeNetReceiverTransformerInterface(SpikeNetInputInterface, ReceiverTransformerInterface):

    """SpikeNetReceiverTransformerInterface class."""

    def configure(self):
        SpikeNetInputInterface.configure(self)
        ReceiverTransformerInterface.configure(self)

    def __call__(self):
        return self.set_proxy_data(self.receive_transform())

    def print_str(self):
        return ReceiverTransformerInterface.print_str(self) + SpikeNetInputInterface.print_str(self)


class SpikeNetInterfaces(HasTraits):

    """SpikeNetInterfaces abstract base class"""

    interfaces = List(of=SpikeNetInterface)

    @property
    def spiking_network(self):
        if len(self.interfaces):
            return self.interfaces[0].spiking_network
        else:
            return None

    @property
    def populations(self):
        return self._loop_get_from_interfaces("populations")

    @property
    def populations_unique(self):
        return np.unique(self._loop_get_from_interfaces("populations"))

    @property
    def spiking_proxy_inds(self):
        return self._loop_get_from_interfaces("spiking_proxy_inds")

    @property
    def spiking_proxy_inds_unique(self):
        return np.unique(self._loop_get_from_interfaces("spiking_proxy_inds"))

    @property
    def number_of_populations(self):
        return self.populations_unique.shape[0]

    @property
    def number_of_spiking_proxy_nodes(self):
        return self.proxy_inds_unique.shape[0]

    @property
    def spiking_proxy_receivers(self):
        return self._loop_get_from_interfaces("spiking_proxy_receiver")

    @property
    def spiking_proxy_senders(self):
        return self._loop_get_from_interfaces("spiking_proxy_sender")


class SpikeNetOutputInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetOutputInterfaces"""

    pass


class SpikeNetOutputRemoteInterfaces(SpikeNetOutputInterfaces):

    """SpikeNetOutputRemoteInterfaces"""

    def __call__(self):
        for interface in self.interfaces:
            interface()


class SpikeNetInputInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetInputInterfaces"""

    pass


class SpikeNetInputRemoteInterfaces(SpikeNetInputInterfaces):

    """SpikeNetInputRemoteInterfaces"""

    def __call__(self, *args):
        for interface in self.interfaces:
            interface(*args)
