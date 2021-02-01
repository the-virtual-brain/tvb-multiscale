# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb.basic.neotraits.api import HasTraits, Float, NArray


class Base(HasTraits):
    __metaclass__ = ABCMeta

    """
        Abstract Base Transformer class comprising:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - an abstract method for the computations applied 
              upon the input buffer data for the output buffer data to result.
    """

    input_buffer = NArray(
        label="Input buffer",
        doc="""Buffer of incoming data.""",
        required=True,
        default=np.array([])
    )

    output_buffer = NArray(
        label="Output buffer",
        doc="""Buffer of outgoing data.""",
        required=True,
        default=np.array([])
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    input_time = NArray(
        label="Input time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the input buffer.""",
        required=True,
        default=np.array([])
    )

    output_time = NArray(
        label="Output time vector",
        doc="""Buffer of time (float) or time steps (integer) corresponding to the output bufer.""",
        required=True,
        default=np.array([])
    )

    def compute_time(self):
        self.output_time = np.copy(self.input_time)

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Abstract method for the computation on the input buffer data for the output buffer data to result."""
        pass

    def __call__(self):
        self.compute_time()
        self.compute()

    def configure(self):
        if self.receiver:
            self.receiver.configure()
        if self.sender:
            self.sender.configure()
        super(Base, self).configure()

    def print_str(self):
        output = "\n%s, dt = %g" % (self.__repr__(), self.dt)
        if self.receiver:
            output += "\nReceiver: %s" % str(self.receiver)
        if self.sender:
            output += "\nSender: %s" % str(self.sender)


# A few basic examples:


class Elementary(Base):
    """
        Elementary Transformer just copies the input to the output without any computation.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a method to copy the input buffer data to the output buffer.
    """

    def compute(self):
        """Method that just copies input buffer data to the output buffer"""
        self.output_buffer = np.copy(self.input_buffer)


class Scale(Base):
    """
        Scale Transformer scales the input with a scale factor in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a scale factor numpy.array,
            - a method to multiply the input buffer data by the scale factor for the output buffer data to result.
    """

    scale_factor = NArray(
        label="Scale factor",
        doc="""Array to scale input buffer.""",
        required=True,
        default=np.array([1.0])
    )

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = self.scale_factor * self.input_buffer


class DotProduct(Base):
    """
        DotProduct Transformer computes the dot product of the input with a scale factor
        in order to compute the output.
        It comprises of:
            - an input buffer data numpy.array,
            - an output buffer data numpy.array,
            - a dot factor numpy.array,
            - a method to perform the left dot product upon the input buffer for the output buffer data to result.
    """

    dot_factor = NArray(
        label="Dot factor",
        doc="""Array to perform the left dot product upon the input buffer.""",
        required=True,
        default=np.array([1.0])
    )

    def compute(self):
        """Method that just scales input buffer data to compute the output buffer data."""
        self.output_buffer = np.dot(self.dot_factor * self.input_buffer)
