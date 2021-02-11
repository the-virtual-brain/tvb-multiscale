from abc import ABCMeta, abstractmethod

from tvb_multiscale.core.interfaces.io import SetToMemory, GetFromMemory
from tvb_multiscale.core.spiking_models.devices import InputDevice, DeviceSet, OutputDevice


class SpikeNetInputDevice(SetToMemory):
    __metaclass__ = ABCMeta
    """
        SpikeNetInputDevice class to set data directly to an InputDevice instance,
        or a DeviceSet of such instances in memory.
        It comprises of:
            - a target attribute, i.e., the InputDevice or DeviceSet instance to send data to,
            - an abstract method to set data to the target, depending on the specific InputDevice.
    """

    # target = Attr(field_type=(InputDevice, DeviceSet),
    #               required=True,
    #               label="Target of Spiking Network",
    #               doc="""Spiking Network InputDevice or DeviceSet to set data to.""")

    def configure(self):
        assert isinstance(self.target, InputDevice, DeviceSet)
        super(SpikeNetInputDevice, self).configure()

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

    # TODO: find a way to use here multiple options for field_type!

    # source = Attr(field_type=(OutputDevice, DeviceSet),
    #               required=True,
    #               label="Source of Spiking Network events",
    #               doc="""Spiking Network OutputDevice or DeviceSet to get events from.""")

    def configure(self):
        assert isinstance(self.source, OutputDevice, DeviceSet)
        super(OutputDevice, self).configure()

    def receive(self):
        events = self.source.get_events()
        self.source.reset
        return events
