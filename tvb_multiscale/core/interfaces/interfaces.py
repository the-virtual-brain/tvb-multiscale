from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr, List
from tvb.basic.neotraits._core import HasTraits

from tvb_multiscale.core.config import LINE
from tvb_multiscale.core.interfaces.io import Communicator


class BaseInterface(HasTraits):
    __metaclass__ = ABCMeta

    """Cosimulation BaseInterface abstract base class."""

    label = Attr(label="BaseInterface label",
                 doc="""BaseInterface label""",
                 field_type=str,
                 required=True)

    def configure(self):
        """Method to configure the cosimulation"""
        super(BaseInterface, self).configure()

    @abstractmethod
    def __call__(self, *args):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def print_str(self, *args):
        return "\nLabel: %s, Type: %s" % (self.label, self.__repr__())

    def __str__(self):
        return self.print_str()


class CommunicatorInterface(BaseInterface):
    """CommunicatorInterface abstract base class sending/receiving data
        directly to/from a transformer/cosimulator
    """

    communicator = Attr(
        label="Communicator",
        field_type=Communicator,
        doc="""A Communicator class instance to send/receive data to/from the transformer/cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        self.communicator.configure()
        super(CommunicatorInterface, self).configure()

    def __call__(self, *args):
        return self.communicator(*args)

    def print_str(self, sender_not_receiver=None):
        out = super(CommunicatorInterface, self).print_str()
        if sender_not_receiver is True:
            return out + "\nSender: %s" % str(self.communicator)
        elif sender_not_receiver is False:
            return out + "\nReceiver: %s" % str(self.communicator)
        else:
            return out + "\nCommunicator: %s" % str(self.communicator)


class TransformerInterface(BaseInterface):
    """TransformerInterface abstract base class
       - sending/receiving data to/from a Transformer,
       - performing the Transformer computation,
       - and receiving/sending data from/to the cosimulator.
    """

    communicator1 = Attr(
        label="Communicator before transformation",
        field_type=Communicator,
        doc="""A Communicator class instance to send/receive data to/from the transformer.""",
        required=True
    )

    transformer = Attr(
        label="Transformer",
        field_type=Communicator,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    communicator2 = Attr(
        label="Communicator after transformation",
        field_type=Communicator,
        doc="""A Communicator class instance to send/receive data to/from the cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        self.communicator1.configure()
        self.transformer.configure()
        self.communicator2.configure()
        super(TransformerInterface, self).configure()

    def __call__(self, *args):
        self.communicator1(*args)
        self.transformer()
        out = self.communicator2()
        return out

    def print_str(self, sender_not_receiver=None):
        if sender_not_receiver is True:
            comm_str = "Sender"
        elif sender_not_receiver is False:
            comm_str = "Receiver"
        else:
            comm_str = "Communicator"
        out = super(TransformerInterface, self).print_str()
        out += "\n%s to Transformer: %s" % (comm_str, str(self.communicator1))
        out += "\nTransformer: %s" % str(self.transformer)
        out += "\n%s from Transformer: %s" % (comm_str, str(self.communicator2))


class BaseInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """This class holds a list of interfaces"""

    interfaces = List(of=BaseInterface)

    @property
    def number_of_interfaces(self):
        return len(self.interfaces)

    def _loop_get_from_interfaces(self, attr):
        out = []
        for interfaces in self.interfaces:
            out += list(getattr(interfaces, attr))
        return out

    def configure(self):
        """Method to configure the interfaces"""
        for interface in self.interfaces:
            interface.configure()
        super().configure()

    @abstractmethod
    def __call__(self, *args):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str()

    def print_str(self):
        output = 2 * LINE + "%s\n\n" % self.__repr__()
        for ii, interface in enumerate(self.interfaces):
            output += "%d. %s" % (ii, interface.print_str())
            output += LINE + "\n"
        return output