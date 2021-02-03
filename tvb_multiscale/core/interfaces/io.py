# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import os
from enum import Enum

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr

from tvb_multiscale.core.interfaces.transformers import Base as Transformer
from tvb_multiscale.core.spiking_models.devices import InputDevice, OutputDevice, DeviceSet


class Communicator(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract Communicator class to transfer data (time and values).
    """

    def __repr__(self):
        return self.__class__.__name__

    def print_str(self, sender_not_receiver_flag=None):
        if sender_not_receiver_flag is True:
            source_or_target = "\nSource: %s" % str(self.source)
        elif sender_not_receiver_flag is False:
            source_or_target = "\nTarget: %s" % str(self.target)
        else:
            source_or_target = ""
        return "\n%s:%s" % (self.__repr__(), source_or_target)

    @abstractmethod
    def __call__(self, *args):
        pass


class Sender(Communicator):
    __metaclass__ = ABCMeta

    """
        Abstract Communicator class to send data (time and values).
        It comprises of:
            - a target attribute, i.e., the location to send data to,
            - an abstract method to send data to the target.
    """

    target = None

    @abstractmethod
    def send(self, data):
        pass

    def __call__(self, data):
        self.send(data)

    def print_str(self):
        return super(Sender, self).print_str(True)


class Receiver(Communicator):
    __metaclass__ = ABCMeta

    """
        Abstract Communicator class to receive data (time and values).
        It comprises of:
            - a source attribute, i.e., the location to read data from,
            - an abstract method to read data from the source.
    """

    source = None

    @abstractmethod
    def receive(self):
        pass

    def __call__(self):
        return self.receive()

    def print_str(self):
        return super(Receiver, self).print_str(False)


class SetToMemory(Sender):
    __metaclass__ = ABCMeta

    """
       SetToMemory class to set data directly to an object in memory.
       It comprises of:
           - a target attribute, i.e., the object to send data to,
           - an abstract method to send data to the target.
    """

    @abstractmethod
    def send(self, data):
        pass


class GetFromMemory(Receiver):
    __metaclass__ = ABCMeta
    """
       GetFromMemory class to receive data from an object in memory.
       It comprises of:
           - a source attribute, i.e., the object to receive (i.e., copy) data from,
           - an abstract method to receive data from the source.
    """

    @abstractmethod
    def receive(self):
        pass


class SetToTransformer(SetToMemory):

    """
       SetToTransformer class to set data directly to a Transformer instance in memory.
       It comprises of:
           - a target attribute, i.e., the Transformer instance to send data to,
           - an abstract method to set data to the target.
    """

    target = Attr(
        label="Target Transformer",
        field_type=Transformer,
        doc="""Target Transformer to send (i.e., copy) data to.""",
        required=True
    )

    def send(self, data):
        self.target.input_time = np.copy(data[0])
        self.target.input_buffer = np.copy(data[1])


class GetFromTransformer(GetFromMemory):

    """
       GetFromTransformer class to get data from a Transformer instance in memory.
       It comprises of:
           - a source attribute, i.e., the Transformer instance to get (i.e., copy) data from,
           - an abstract method to get data from the source.
    """

    source = Attr(
        label="Source Transformer",
        field_type=Transformer,
        doc="""Source Transformer to receive (i.e., copy) data from.""",
        required=True
    )

    def receive(self):
        return [np.copy(self.source.output_time), np.copy(self.source.output_buffer)]


class SpikeNetInputDevice(SetToMemory):
    __metaclass__ = ABCMeta
    """
        SpikeNetInputDevice class to set data directly to an InputDevice instance,
        or a DeviceSet of such instances in memory.
        It comprises of:
            - a target attribute, i.e., the InputDevice or DeviceSet instance to send data to,
            - an abstract method to set data to the target, depending on the specific InputDevice.
    """

    target = Attr(field_type=(InputDevice, DeviceSet),
                  required=True,
                  label="Target of Spiking Network",
                  doc="""Spiking Network InputDevice or DeviceSet to set data to.""")

    @abstractmethod
    def send(self, data):
        pass


class SpikeNetEventsFromOutpuDevice(GetFromMemory):
    """
        SpikeNetEventsFromOutpuDevice class to read events' data
         (times, senders and values from NEST Multimeters-like devices) from an OutputDevice,
         or a DeviceSet of such instances
        It comprises of:
            - a source attribute, i.e., the OutputDevice or DeviceSet instance to get (i.e., copy) data from,
            - a method to get data from the source.
    """

    source = Attr(field_type=(OutputDevice, DeviceSet),
                  required=True,
                  label="Source of Spiking Network events",
                  doc="""Spiking Network OutputDevice or DeviceSet to get events from.""")

    def receive(self):
        events = self.source.get_events()
        self.source.reset
        return events


class RemoteSender(Sender):
    """
           Abstract RemoteSender base class to send data (time and values) to a remote location (e.g., file, MPI).
           It comprises of:
           - a target attribute, to send data to,
           - an abstract method to write data to the target.
    """
    pass


class RemoteReceiver(Receiver):
    """
           Abstract RemoteReceiver base class to receive data (time and values) from a remote location (e.g., file, MPI).
           It comprises of:
           - a source attribute, to receive data from,
           - an abstract method to receive from the target.
    """
    pass


class WriterToFile(Sender):
    __metaclass__ = ABCMeta

    """
       Abstract base WriterToFile class to write data (time and values) to a file.
       It comprises of:
       - a target attribute, i.e., the absolute path to the file to write data to,
       - an abstract method to write data to the target.
    """

    target = Attr(field_type=os.PathLike, default="", required=True,
                  label="Path to target file", doc="""Full path to .npy file to write data to.""")

    @abstractmethod
    def send(self, data):
        pass


class ReaderFromFile(Receiver):
    __metaclass__ = ABCMeta

    """
       Abstract base ReaderFromFile class to read data (time and values) from a file.
       It comprises of:
           - a source attribute, i.e., the absolute path to the file to read data from,
           - an abstract method to read data from the source.
    """

    source = Attr(field_type=os.PathLike, default="", required=True,
                  label="Path to source file", doc="""Full path to .npy file to read data from.""")

    @abstractmethod
    def receive(self):
        pass


class NPZWriter(WriterToFile):

    """
       NPZWriter class to write data (time and values) to a .npz file.
       It comprises of:
           - a target attribute, i.e., the absolute path to the .npz file to write data to,
           - an abstract method to write data to the target.
    """

    def send(self, data):
        np.savez(self.target, time=data[0], values=data[1])


class NPZReader(ReaderFromFile):

    """
        NPZReader class to read data (time and values) from a .npz file.
        It comprises of:
            - a source attribute, i.e., the absolute path to the .npz file to read data from,
            - an abstract method to read data from the source.
    """

    def receive(self):
        data = []
        with np.load(self.source) as datafile:
            data.append(datafile["time"])
            data.append(datafile["values"])
        return data


class WritersToFile(Enum):
    WRITER_TO_NUMPY = NPZWriter


class ReadersFromFile(Enum):
    READER_FROM_NUMPY = NPZReader


class SettersToMemory(Enum):
    SET_TO_TRANSFORMER = SetToTransformer


class GettersFromMemory(Enum):
    GET_FROM_TRANSFORMER = GetFromTransformer
    SPIKE_NET_OUTPUT_DEVICE = SpikeNetEventsFromOutpuDevice


class RemoteSenders(WritersToFile):
    pass


class RemoteReceivers(ReadersFromFile):
    pass


class Senders(SettersToMemory, RemoteSenders):
    pass


class Receivers(GettersFromMemory, RemoteReceivers):
    pass


class Communicators(Senders, Receivers):
    pass
