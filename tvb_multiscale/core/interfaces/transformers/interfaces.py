from tvb.basic.neotraits._attr import Attr, List

from tvb_multiscale.core.interfaces.base.interfaces import BaseInterface, BaseInterfaces
from tvb_multiscale.core.interfaces.base.io import Sender, Receiver
from tvb_multiscale.core.interfaces.transformers.models.base import Transformer


class TransformerInterface(BaseInterface):

    """TransformerInterface base class
       - setting data to a Transformer,
       - performing the Transformer computation,
       - and outputing the data.
    """

    transformer = Attr(
        label="Transformer",
        field_type=Transformer,
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        super(TransformerInterface, self).configure()
        self.transformer.configure()

    def transform(self, data):
        if data is not None:
            self.transformer.input_time = data[0]
            self.transformer.input_buffer = data[1]
            self.transformer()
            return [self.transformer.output_time, self.transformer.output_buffer]
        else:
            return None

    def __call__(self, data):
        return self.transform(data)


class TVBtoSpikeNetTransformerInterface(TransformerInterface):
    """TVBtoSpikeNetTransformerInterface class
           - setting TVB data to a Transformer,
           - performing the Transformer computation,
           - and outputing the data for the spiking network cosimulator.
    """
    pass


class SpikeNetToTVBTransformerInterface(TransformerInterface):
    """SpikeNetToTVBTransformerInterface class
           - setting data from a spiking network cosimulator to a Transformer,
           - performing the Transformer computation,
           - and outputing the data for TVB.
    """
    pass


class TransformerSenderInterface(TransformerInterface):
    """TransformerSenderInterface base class
       - setting data to a Transformer,
       - performing the Transformer computation,
       - and sending data to the cosimulator.
    """

    sender = Attr(
        label="Sender",
        field_type=Sender,
        doc="""A Sender class instance to send data to the cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        super(TransformerSenderInterface, self).configure()
        self.sender.configure()

    def transform_and_send(self, data):
        if data is not None:
            return self.sender(self.transform(data))
        else:
            return None

    def __call__(self, data):
        return self.transform_and_send(data)


class ReceiverTransformerInterface(TransformerInterface):
    """ReceiverTransformerInterface base class
       - receiving data from a cosimulator,
       - performing the Transformer computation,
       - and outputing data to the other cosimulator.
    """

    receiver = Attr(
        label="Receiver Communicator",
        field_type=Receiver,
        doc="""A Receiver Communicator class instance to receive data from the cosimulator.""",
        required=True
    )

    def configure(self):
        """Method to configure the CommunicatorInterface"""
        super(ReceiverTransformerInterface, self).configure()
        self.receiver.configure()

    def receive_transform(self):
        data = self.receiver()
        if data is None:
            return None
        return self.transform(data)

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
        if data is None:
            return None
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        self.transformer()
        return self.sender([self.transformer.input_time, self.transformer.output_buffer])

    def __call__(self):
        self.receive_transform_send()


class TVBtoSpikeNetRemoteTransformerInterface(RemoteTransformerInterface):
    """TVBtoSpikeNetRemoteTransformerInterface  class for TVB -> spikeNet transformations
       - receiving data for a Transformer from TVB,
       - performing the Transformer computation,
       - and sending data to the (spiking) cosimulator.
    """

    pass


class SpikeNetToTVBRemoteTransformerInterface(RemoteTransformerInterface):
    """SpikeNetToTVBRemoteTransformerInterface  class for spikeNet -> TVB transformations
       - receiving data from a (spiking) cosimulator for a Transformer,
       - performing the Transformer computation,
       - and sending data to TVB.
    """

    pass


class TransformerInterfaces(BaseInterfaces):

    """TransformerInterfaces"""

    interfaces = List(of=TransformerInterface)

    def __call__(self, cosim_updates):
        outputs = []
        if cosim_updates is not None:
            for ii, (interface, cosim_update) in enumerate(zip(self.interfaces, cosim_updates)):
                if cosim_update is not None and len(cosim_update) > 2:
                    assert cosim_update[2] == ii
                    cosim_update = cosim_update[:2]
                outputs.append(interface(cosim_update) + [ii])
        return outputs


class TVBtoSpikeNetTransformerInterfaces(TransformerInterfaces):

    """TVBtoSpikeNetTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetTransformerInterface)


class SpikeNetToTVBTransformerInterfaces(TransformerInterfaces):

    """SpikeNetToTVBTransformerInterfaces"""

    interfaces = List(of=SpikeNetToTVBTransformerInterface)


class RemoteTransformerInterfaces(BaseInterfaces):

    """RemoteTransformerInterfaces"""

    interfaces = List(of=RemoteTransformerInterface)

    def __call__(self):
        outputs = []
        for interface in self.interfaces:
            outputs.append(interface())
        return outputs


class TVBtoSpikeNetRemoteTransformerInterfaces(RemoteTransformerInterfaces):

    """TVBtoSpikeNetRemoteTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetRemoteTransformerInterface)


class SpikeNetToTVBRemoteTransformerInterfaces(RemoteTransformerInterfaces):

    """SpikeNetToTVBRemoteTransformerInterfaces"""

    interfaces = List(of=SpikeNetToTVBRemoteTransformerInterface)
