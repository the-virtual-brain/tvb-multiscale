from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr, Int, Float, List
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.interfaces.base.io import Sender, Receiver


class BaseInterface(HasTraits):
    __metaclass__ = ABCMeta

    """Cosimulation BaseInterface abstract base class."""

    model = Attr(
        label="Model",
        field_type=str,
        doc="""Name of interface model (string).""",
        required=False,
        default=""
    )

    @abstractmethod
    def __call__(self, *args):
        pass


class SenderInterface(BaseInterface):

    """SenderInterface base class sending data to/from a transformer/cosimulator
    """

    sender = Attr(
        label="Sender",
        field_type=Sender,
        doc="""A Sender class instance to send data to the transformer/cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        super(SenderInterface, self).configure()
        self.sender.configure()

    def send(self, data):
        return self.sender(data)

    def __call__(self, data):
        return self.sender(data)


class ReceiverInterface(BaseInterface):

    """ReceiverInterface base class receiving data from a transformer/cosimulator
    """

    receiver = Attr(
        label="Receiver",
        field_type=Receiver,
        doc="""A Receiver class instance to receive data from the transformer/cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        super(ReceiverInterface, self).configure()
        self.receiver.configure()

    def receive(self):
        return self.receiver()

    def __call__(self):
        return self.receiver()


class BaseInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """BaseInterfaces
       This class holds a list of interfaces"""

    interfaces = List(of=BaseInterface)

    synchronization_time = Float(
        label="Synchronization Time",
        default=0.0,
        required=True,
        doc="""Cosimulation synchronization time (ms) for exchanging data 
               in milliseconds, must be an integral multiple of integration-step size.""")

    synchronization_n_step = Int(
        label="Synchronization time steps",
        default=0,
        required=True,
        doc="""Cosimulation synchronization time steps (int) for exchanging data.""")

    @property
    def number_of_interfaces(self):
        return len(self.interfaces)

    def _loop_get_from_interfaces(self, attr):
        out = []
        for interfaces in self.interfaces:
            out += ensure_list(getattr(interfaces, attr))
        return out

    def configure(self):
        """Method to configure the interfaces"""
        super(BaseInterfaces, self).configure()
        for interface in self.interfaces:
            interface.configure()

    @property
    def labels(self):
        labels = ""
        for interface in self.interfaces:
            labels += "\n" + interface.label
        return labels

    @abstractmethod
    def __call__(self):
        pass

    def info(self, recursive=0):
        info = super(BaseInterfaces, self).info(recursive=recursive)
        info["number_of_interfaces"] = self.number_of_interfaces
        return info
