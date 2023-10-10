# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from decimal import Decimal

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, List, NArray
from tvb.contrib.scripts.utils.data_structures_utils import extract_integer_intervals, list_of_dicts_to_dict_of_lists

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.interfaces.base.interfaces import \
    BaseInterface, SenderInterface, ReceiverInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.transformers.interfaces import TransformerInterface, \
    TransformerSenderInterface, ReceiverTransformerInterface
from tvb_multiscale.core.interfaces.spikeNet.io import SpikeNetInputDeviceSet, SpikeNetOutputDeviceSet
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


class SpikeNetInterface(BaseInterface):

    __metaclass__ = ABCMeta

    """SpikeNetInterface abstract base class."""

    spiking_network = Attr(label="Spiking Network",
                           doc="""The instance of SpikingNetwork class""",
                           field_type=SpikingNetwork,
                           required=True)

    spiking_proxy_inds = NArray(
        dtype=int,
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
    def spiking_simulator(self):
        return self.spiking_network.spiking_simulator_module

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

    def info(self, recursive=0):
        info = super(SpikeNetInterface, self).info(recursive=recursive)
        info["populations"] = self.populations
        info["number_of_proxy_nodes"] = self.number_of_proxy_nodes
        info["spiking_proxy_inds"] = self.spiking_proxy_inds
        info["number_of_proxy_gids"] = self.number_of_proxy_gids
        info["proxy_gids"] = self.proxy_gids
        return info


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
        dtype=int,
        label="Time step indices.",
        doc="""Time step indices of last Spiking Network output values.""",
        required=True,
        default=np.array([1, 0])
    )

    _number_of_dt_decimals = None

    def configure(self):
        super(SpikeNetOutputInterface, self).configure()
        self.proxy.configure()
        if len(self.model) == 0:
            self.model = self.proxy.model
        self._number_of_dt_decimals = np.abs(Decimal('%g' % self.dt).as_tuple().exponent)

    @property
    @abstractmethod
    def _time(self):
        pass

    @property
    def time(self):
        return np.around(self._time, decimals=self._number_of_dt_decimals)

    @property
    def label(self):
        return "%s: %s (%s) ->" % (self.__class__.__name__, str(self.populations),
                                   extract_integer_intervals(self.spiking_proxy_inds))

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.source)

    def get_proxy_data(self):
        data = self.proxy()
        if data is not None:
            if len(data[0]) == 2:
                # This will work for multimeters:
                self.times = np.array([np.round(data[0][0] / self.dt),  # start_time_step
                                       np.round(data[0][1] / self.dt)]).astype("i")  # end_time_step
            else:
                # This will work for spike recorders:
                time = int(np.round(self.time / self.dt))
                times = self.times.copy()
                if time > times[1]:
                    times[0] = times[1] + 1
                    times[1] = time
                self.times = times
            return [self.times, data[-1]]
        return None

    def __call__(self):
        return self.get_proxy_data()


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

    @property
    def proxy_gids(self):
        return self._get_proxy_gids(self.proxy.target)

    def set_proxy_data(self, data):
        if data is not None:
            return self.proxy(data)
        else:
            return None

    def __call__(self, data):
        return self.set_proxy_data(data)


class SpikeNetOutputTransformerInterface(SpikeNetOutputInterface, TransformerInterface):

    """SpikeNetOutputTransformerInterface class."""

    def configure(self):
        SpikeNetOutputInterface.configure(self)
        TransformerInterface.configure(self)

    def __call__(self):
        return self.transform(self.get_proxy_data())


class SpikeNetInputTransformerInterface(SpikeNetInputInterface, TransformerInterface):

    """SpikeNetInputTransformerInterface class."""

    def configure(self):
        SpikeNetInputInterface.configure(self)
        TransformerInterface.configure(self)

    def __call__(self, data):
        return self.set_proxy_data(self.transform(data))


class SpikeNetSenderInterface(SpikeNetOutputInterface, SenderInterface):

    """SpikeNetSenderInterface class."""

    def configure(self):
        SpikeNetOutputInterface.configure(self)
        SenderInterface.configure(self)

    def __call__(self):
        return self.send(self.get_proxy_data())


class SpikeNetReceiverInterface(SpikeNetInputInterface, ReceiverInterface):

    """SpikeNetReceiverInterface class."""

    def configure(self):
        SpikeNetInputInterface.configure(self)
        ReceiverInterface.configure(self)

    def __call__(self):
        return self.set_proxy_data(self.receive())


class SpikeNetTransformerSenderInterface(SpikeNetOutputInterface, TransformerSenderInterface):

    """SpikeNetTransformerSenderInterface class."""

    def configure(self):
        SpikeNetOutputInterface.configure(self)
        TransformerSenderInterface.configure(self)

    def __call__(self):
        return self.transform_and_send(self.get_proxy_data())


class SpikeNetReceiverTransformerInterface(SpikeNetInputInterface, ReceiverTransformerInterface):

    """SpikeNetReceiverTransformerInterface class."""

    def configure(self):
        SpikeNetInputInterface.configure(self)
        ReceiverTransformerInterface.configure(self)

    def __call__(self):
        return self.set_proxy_data(self.receive_transform())


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

    def info(self, recursive=0):
        info = super(SpikeNetInterfaces, self).info(recursive=recursive)
        info["number_of_interfaces"] = self.number_of_interfaces
        info["number_of_populations"] = self.number_of_populations
        info["populations_unique"] = self.populations_unique
        info["spiking_proxy_inds"] = self.spiking_proxy_inds
        return info


class SpikeNetOutputInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetOutputInterfaces"""

    interfaces = List(of=SpikeNetOutputInterface)

    def __call__(self):
        outputs = []
        for ii, interface in enumerate(self.interfaces):
            output = interface()
            if output is not None:
                output += [ii]
            outputs.append(output)
        return outputs


class SpikeNetOutputTransformerInterfaces(SpikeNetOutputInterfaces):

    """SpikeNetOutputTransformerInterfaces"""

    interfaces = List(of=SpikeNetOutputTransformerInterface)


class SpikeNetSenderInterfaces(SpikeNetOutputInterfaces):

    """SpikeNetSenderInterfaces"""

    interfaces = List(of=SpikeNetSenderInterface)


class SpikeNetTransformerSenderInterfaces(SpikeNetOutputInterfaces):

    """SpikeNetTransformerSenderInterfaces"""

    interfaces = List(of=SpikeNetTransformerSenderInterface)


class SpikeNetReceiverInterfaces(BaseInterfaces, SpikeNetInterfaces):

    """SpikeNetReceiverInterfaces"""

    interfaces = List(of=SpikeNetReceiverInterface)

    def __call__(self):
        results = []
        for interface in self.interfaces:
            results.append(interface())
        return results


class SpikeNetReceiverTransformerInterfaces(SpikeNetReceiverInterfaces):

    """SpikeNetReceiverTransformerInterfaces"""

    interfaces = List(of=SpikeNetReceiverTransformerInterface)


class SpikeNetInputInterfaces(SpikeNetReceiverInterfaces):

    """SpikeNetInputInterfaces"""

    interfaces = List(of=SpikeNetInputInterface)

    def __call__(self, input_datas):
        results = []
        if input_datas is not None:
            for ii, (interface, input_data) in enumerate(zip(self.interfaces, input_datas)):
                if len(input_data) > 2:
                    assert input_data[2] == ii
                    input_data = input_data[:2]
                results.append(interface(input_data))
        return results


class SpikeNetInputTransformerInterfaces(SpikeNetInputInterfaces):

    """SpikeNetInputTransformerInterfaces"""

    interfaces = List(of=SpikeNetInputTransformerInterface)
