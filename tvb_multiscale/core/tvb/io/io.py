# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import os

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Attr

from tvb_multiscale.core.interfaces.transformers import Base as Transformer


class TVBIO(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract base TVB IO class to transfer TVB data (time and values).
    """

    @abstractmethod
    def __call__(self, *args):
        pass


class TVBSender(TVBIO):
    __metaclass__ = ABCMeta

    """
        Abstract base TVBWriter class to send TVB data (time and values) .
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


class TVBReceiver(TVBIO):
    __metaclass__ = ABCMeta

    """
        Abstract base TVBReader class to read TVB data (time and values) .
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


class TVBSenderFILE(TVBSender):
    __metaclass__ = ABCMeta

    """
       Abstract base TVBWriterFILE class to write data (time and values) to a file.
       It comprises of:
       - a target attribute, i.e., the absolute path to the file to write data to,
       - an abstract method to write data to the target.
    """

    target = Attr(field_type=os.PathLike, default="", required=True,
                  label="Path to target file", doc="""Full path to .npy file to write data to.""")

    @abstractmethod
    def send(self, data):
        pass


class TVBReceiverFILE(TVBReceiver):
    __metaclass__ = ABCMeta

    """
       Abstract base TVBReaderFILE class to read data (time and values) from a file.
       It comprises of:
           - a source attribute, i.e., the absolute path to the file to read data from,
           - an abstract method to read data from the source.
    """

    source = Attr(field_type=os.PathLike, default="", required=True,
                  label="Path to source file", doc="""Full path to .npy file to read data from.""")

    @abstractmethod
    def receive(self):
        pass


class TVBWriterNUMPY(TVBSenderFILE):

    """
       TVBWriterNUMPY class to write data (time and values) to a .npz file.
       It comprises of:
           - a target attribute, i.e., the absolute path to the .npz file to write data to,
           - an abstract method to write data to the target.
    """

    def send(self, data):
        np.savez(self.target, time=data[0], values=data[1])


class TVBReaderNUMPY(TVBReceiverFILE):

    """
        TVBReaderNUMPY class to read data (time and values) from a .npz file.
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


class TVBSenderMEMORY(TVBSender):

    """
       TVBWriterMEMORY class to send data directly to a Transformer object in memory.
       It comprises of:
           - a target attribute, i.e., the Transformer object to send data to,
           - an abstract method to send data to the target.
    """

    target = Attr(
        label="Target Transfomer",
        field_type=Transformer,
        doc="""Target Transformer to send (i.e., copy) data to.""",
        required=True
    )

    def send(self, data):
        self.target.time = np.copy(data[0])
        self.target.input_buffer = np.copy(data[1])


class TVBReceiverMEMORY(TVBReceiver):

    """
       TVBReaderMEMORY class to receive data from a Transformer object in memory.
       It comprises of:
           - a source attribute, i.e., the Transformer object to receive (i.e., copy) data from,
           - an abstract method to receive data from the source.
    """

    source = Attr(
        label="Source Transfomer",
        field_type=Transformer,
        doc="""Source Transformer to receive (i.e., copy) data from.""",
        required=True
    )

    def receive(self):
        return [np.copy(self.source.time), np.copy(self.source.output_buffer)]
