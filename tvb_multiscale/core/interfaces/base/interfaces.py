from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr, Int, Float, List

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.interfaces.base.io import Sender, Receiver
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

    def transform_send(self, data):
        if data is not None:
            return self.sender(self.transform(data))
        else:
            return None

    def __call__(self, data):
        return self.transform_send(data)


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
        self.transformer.input_time = data[0]

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

    def info(self, recursive=0):
        info = super(BaseInterfaces, self).info(recursive=recursive)
        info["number_of_interfaces"] = self.number_of_interfaces
        return info


class TransformerInterfaces(BaseInterfaces):

    """TransformerInterfaces"""

    interfaces = List(of=TransformerInterface)

    def __call__(self, *args):
        for interface in self.interfaces:
            interface()


class TVBtoSpikeNetTransformerInterfaces(TransformerInterfaces):

    """TVBtoSpikeNetTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetTransformerInterface)

    pass


class SpikeNetToTVBTransformerInterfaces(TransformerInterfaces):

    """SpikeNetToTVBTransformerInterfaces"""

    interfaces = List(of=SpikeNetToTVBTransformerInterface)

    pass


class RemoteTVBtoSpikeNetTransformerInterfaces(TVBtoSpikeNetTransformerInterfaces):

    """RemoteTVBtoSpikeNetTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetTransformerInterface)

    receiver = Attr(
        label="Receiver communicator",
        field_type=Receiver,
        doc="""A Communicator class instance to receive data from TVB for the transformers.""",
        required=True
    )

    sender = Attr(
        label="Communicator after transformation",
        field_type=Sender,
        doc="""A Communicator class instance to send data to the spiking network cosimulators.""",
        required=True
    )

    def __call__(self, *args):
        data = self.receiver()
        if data is not None:
            time, data, _ = tuple(data)
            datas = []
            for ii, interface in enumerate(self.interfaces):
                #                 data values !!! assuming only 1 mode!!! -> shape (times, vois, proxys):
                datas.append(interface([time,
                             data[:, interface.voi_loc][:, :, interface.proxy_inds],
                             ii]))
        self.sender(datas)


class RemoteSpikeNetToTVBTransformerInterfaces(SpikeNetToTVBTransformerInterfaces):

    """SpikeNetToTVBTransformerInterfaces"""

    interfaces = List(of=RemoteSpikeNetToTVBTransformerInterfaces)

    receiver = Attr(
        label="Receiver communicator",
        field_type=Receiver,
        doc="""A Communicator class instance to receive data from the spiking network cosimulators for the transformers.""",
        required=True
    )

    sender = Attr(
        label="Communicator after transformation",
        field_type=Sender,
        doc="""A Communicator class instance to send data to TVB.""",
        required=True
    )

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
        datas = self.receiver()
        cosim_updates, all_time_steps = self._prepare_cosim_upadate(good_cosim_update_values_shape)
        for interface, data in zip(self.interfaces, datas):
            cosim_updates, all_time_steps = \
                self._get_from_interface(interface, cosim_updates, all_time_steps, good_cosim_update_values_shape)
        return self.get_inputs(cosim_updates, all_time_steps, good_cosim_update_values_shape)


class RemoteTransformerInterfaces(BaseInterfaces):

    """RemoteTransformerInterfaces"""

    interfaces = List(of=RemoteTransformerInterface)

    def __call__(self, *args):
        for interface in self.interfaces:
            interface()


class TVBtoSpikeNetRemoteTransformerInterfaces(RemoteTransformerInterfaces):

    """TVBtoSpikeNetRemoteTransformerInterfaces"""

    interfaces = List(of=TVBtoSpikeNetRemoteTransformerInterfaceRemoteTransformerInterface)

    pass


class SpikeNetToTVBRemoteTransformerInterfaces(RemoteTransformerInterfaces):

    """SpikeNetToTVBRemoteTransformerInterfaces"""

    interfaces = List(of=SpikeNetToTVBRemoteTransformerInterface)

    pass
