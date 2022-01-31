from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr, Int, Float, List

from tvb_multiscale.core.datatypes import HasTraits
from tvb_multiscale.core.interfaces.base.io import Communicator, Sender, Receiver
from tvb_multiscale.core.interfaces.base.transformers.models.base import Transformer


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


class CommunicatorInterface(BaseInterface):
    __metaclass__ = ABCMeta

    """CommunicatorInterface abstract base class sending/receiving data to/from a transformer/cosimulator
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

    @abstractmethod
    def __call__(self, *args):
        pass

    def info(self):
        info = super(CommunicatorInterface, self).info()
        info.update(self.communicator.info())
        return info

    def info_details(self, **kwargs):
        info = super(CommunicatorInterface, self).info_details()
        info.update(self.communicator.info_details(**kwargs))
        return info


class SenderInterface(CommunicatorInterface):

    """SenderInterface base class sending data to/from a transformer/cosimulator
    """

    communicator = Attr(
        label="Sender Communicator",
        field_type=Sender,
        doc="""A Sender Communicator class instance to send data to the transformer/cosimulator.""",
        required=True
    )

    @property
    def sender(self):
        """A property method to return the Sender class used to send data to the transformer/cosimulator."""
        return self.communicator

    def send(self, data):
        return self.communicator(data)

    def __call__(self, data):
        return self.communicator(data)


class ReceiverInterface(CommunicatorInterface):

    """ReceiverInterface base class receiving data from a transformer/cosimulator
    """

    communicator = Attr(
        label="Receiver Communicator",
        field_type=Sender,
        doc="""A Receiver Communicator class instance to receive data from the transformer/cosimulator.""",
        required=True
    )

    @property
    def receiver(self):
        """A property method to return the Sender class used to send data to the transformer."""
        return self.communicator

    def receive(self):
        return self.communicator()

    def __call__(self):
        return self.communicator()


class CommunicatorTransformerInterface(BaseInterface):
    """TransformerInterface abstract base class
       - setting/getting data to/from a Transformer,
       - performing the Transformer computation,
       - and receiving/sending data from/to the cosimulator.
    """

    communicator = Attr(
        label="Communicator to/from transformation",
        field_type=Communicator,
        doc="""A Communicator class instance to send/receive data for/from the transformer.""",
        required=True
    )

    transformer = Attr(
        label="Transformer",
        field_type=Transformer,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        self.communicator.configure()
        self.transformer.configure()
        super(CommunicatorTransformerInterface, self).configure()

    @abstractmethod
    def __call__(self, *args):
        pass

    def info(self):
        info = super(CommunicatorTransformerInterface, self).info()
        info.update(self.communicator.info())
        info.update(self.transformer.info())
        return info

    def info_details(self, **kwargs):
        info = super(CommunicatorTransformerInterface, self).info_details()
        info.update(self.communicator.info_details(**kwargs))
        info.update(self.transformer.info_details())
        return info


class TransformerSenderInterface(CommunicatorTransformerInterface):
    """TransformerSenderInterface base class
       - setting data to a Transformer,
       - performing the Transformer computation,
       - and sending data to the cosimulator.
    """

    communicator = Attr(
        label="Sender Communicator",
        field_type=Sender,
        doc="""A Sender Communicator class instance to send data to the cosimulator.""",
        required=True
    )

    @property
    def sender(self):
        """A property method to return the Sender class used to send data from the transformer."""
        return self.communicator

    def transform_send(self, data):
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return self.communicator2([self.transformer.output_time, self.transformer.output_buffer])

    def __call__(self, data):
        return self.transform_send(data)

    def info(self):
        info = super(TransformerSenderInterface, self).info()
        info.update(self.transformer.info())
        info.update(self.communicator.info())
        return info

    def info_details(self, **kwargs):
        info = super(TransformerSenderInterface, self).info_details()
        info.update(self.transformer.info_details())
        info.update(self.communicator.info_details(**kwargs))
        return info


class ReceiverTransformerInterface(CommunicatorTransformerInterface):
    """ReceiverTransformerInterface base class
       - receiving data from a cosimulator,
       - performing the Transformer computation,
       - and outputing data to the other cosimulator.
    """

    communicator = Attr(
        label="Receiver Communicator",
        field_type=Receiver,
        doc="""A Receiver Communicator class instance to receive data from the cosimulator.""",
        required=True
    )

    @property
    def receiver(self):
        """A property method to return the Receiver class used to receive data for the transformer."""
        return self.communicator

    def receive_transform(self):
        data = self.communicator()
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return [self.transformer.output_time, self.transformer.output_buffer]

    def __call__(self):
        return self.receive_transform()


class RemoteTransformerInterface(BaseInterface):
    """RemoteTransformerInterface base class
       - receiving data for a Transformer,
       - performing the Transformer computation,
       - and sending data to the cosimulator.
    """

    receiver = Attr(
        label="Receiver communicator",
        field_type=Receiver,
        doc="""A Communicator class instance to receive data for the transformer.""",
        required=True
    )

    transformer = Attr(
        label="Transformer",
        field_type=Transformer,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    sender = Attr(
        label="Communicator after transformation",
        field_type=Sender,
        doc="""A Communicator class instance to send data to the cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the RemoteTransformerInterface"""
        self.receiver.configure()
        self.transformer.configure()
        self.sender.configure()
        super(RemoteTransformerInterface, self).configure()

    def receive_transform_send(self):
        data = self.receiver()
        self.transformer.time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return self.sender([self.transformer.time, self.transformer.output_buffer])

    def __call__(self):
        self.receive_transform_send()

    def info(self):
        info = super(RemoteTransformerInterface, self).info()
        info.update(self.receiver.info())
        info.update(self.transformer.info())
        info.update(self.sender.info())
        return info

    def info_details(self, **kwargs):
        info = super(RemoteTransformerInterface, self).info_details()
        info.update(self.receiver.info_details(**kwargs))
        info.update(self.transformer.info_details())
        info.update(self.sender.info_details(**kwargs))
        return info


class BaseInterfaces(HasTraits):
    __metaclass__ = ABCMeta

    """This class holds a list of interfaces"""

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
            out += list(getattr(interfaces, attr))
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
    def __call__(self, *args):
        pass

    def info(self):
        info = super(BaseInterfaces, self).info()
        info["number_of_interfaces"] = self.number_of_interfaces
        for interface in self.interfaces:
            info.update(interface.info())
        return info

    def info_details(self, **kwargs):
        info = super(BaseInterfaces, self).info_details()
        for interface in self.interfaces:
            info.update(interface.info_details(**kwargs))
        return info
