# -*- coding: utf-8 -*-

import ray

import numpy as np

from tvb_multiscale.core.neotraits import HasTraits, Attr
from tvb_multiscale.core.ray.client import RayClient
from tvb_multiscale.core.interfaces.base.io import Sender, Receiver
from tvb_multiscale.core.interfaces.transformers.models import Transformer
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetInterface, SpikeNetToTVBInterface, \
    TVBSenderInterface, TVBReceiverInterface, TVBTransformerSenderInterface, TVBReceiverTransformerInterface, \
    TVBOutputInterfaces, TVBInputInterfaces, SpikeNetInputInterfaces, SpikeNetOutputInterfaces


class RaySenderInterface(HasTraits):

    sending_ref_obj = None

    def __init__(self, **kwargs):
        self.sending_ref_obj = None
        super(RaySenderInterface, self).__init__(**kwargs)

    def _send_data_block(self):
        ray.get(self.sending_ref_obj)
        self.sending_ref_obj = None
        return self.sending_ref_obj

    def _send_data(self, block=False):
        if isinstance(self.sending_ref_obj, ray._raylet.ObjectRef):
            if block:
                return self._send_data_block()
            else:
                running, done = ray.wait([self.sending_ref_obj])
                if len(running) == 0:
                    self.sending_ref_obj = None
        return self.sending_ref_obj


class RayReceiverInterface(HasTraits):

    receiving_ref_obj = None

    def __init__(self, **kwargs):
        self.receiving_ref_obj = None
        super(RayReceiverInterface, self).__init__(**kwargs)

    def _receive_data_block(self):
        data = ray.get(self.receiving_ref_obj)
        self.receiving_ref_obj = None
        return data

    def _receive_data(self, block=False):
        if isinstance(self.receiving_ref_obj, ray._raylet.ObjectRef):
            if block:
                return self._receive_data_block()
            else:
                running, done = ray.wait([self.receiving_ref_obj])
                if len(running) == 0:
                    return self._receive_data_block()
            return self.receiving_ref_obj
        elif self.receiving_ref_obj is not None:
            data = self.receiving_ref_obj.copy()
            self.receiving_ref_obj = None
            if data[0][1] < data[0][0]:
                return None
            return data
        else:
            return self.receiving_ref_obj


class RayTVBSenderInterface(TVBSenderInterface, RaySenderInterface):

    """RayTVBSenderInterface sends TVB data to a remote TVBtoSpikeNetTransformerInterface"""

    sender = Attr(
        label="Sender",
        field_type=Sender,
        doc="""A Sender class instance to send data to the transformer/cosimulator.""",
        required=False
    )

    def configure(self):
        self.sending_ref_obj = None
        super(RayTVBSenderInterface, self).configure()

    def send_data(self, data):
        return super(RayTVBSenderInterface, self).__call__(data)

    def __call__(self, data=None, block=False):
        # Return:
        # None, if nothing to be done, i.e., sending has been finalized
        # sending_ref_obj if sending is running
        if data is not None:
            self.sending_ref_obj = self.send_data(data)
        return self._send_data(block=block)


class RayTVBReceiverInterface(TVBReceiverInterface, RayReceiverInterface):

    """RayTVBReceiverInterface receives TVB data from a remote SpikeNetToTVBTransformerInterface"""

    receiver = Attr(
        label="Receiver",
        field_type=Receiver,
        doc="""A Receiver class instance to receive data from the transformer/cosimulator.""",
        required=False
    )

    def configure(self):
        self.receiving_ref_obj = None
        super(RayTVBReceiverInterface, self).configure()

    def receive_data(self):
        return super(RayTVBReceiverInterface, self).__call__()

    def __call__(self, block=False):
        # Return:
        # data, if nothing to be done, i.e., receiving has been finalized
        # receiving_ref_obj if receiving is running
        if self.receiving_ref_obj is None:
            self.receiving_ref_obj = self.receive_data()
        return self._receive_data(block)


class RayTVBTransformerSenderInterface(TVBTransformerSenderInterface, RaySenderInterface):

    """RayTVBTransformerSenderInterface transforms TVB data and sends them to a remote spikeNet simulator"""

    def configure(self):
        self.sending_ref_obj = None
        super(RayTVBTransformerSenderInterface, self).configure()

    def transform_and_send(self, data):
        return super(RayTVBTransformerSenderInterface, self).__call__(data)

    def __call__(self, data=None, block=False):
        # Return:
        # None, if nothing to be done, i.e., transforming and sending has been finalized
        # send_ref_obj if sending is running
        if data is not None:
            self.sending_ref_obj = self.transform_and_send(data)
        return self._send_data(block=block)


class RayTVBReceiverTransformerInterface(TVBReceiverTransformerInterface, RayReceiverInterface):

    """RayTVBReceiverTransformerInterface receives data from a remote spikeNet simulator
        and transforms them to deliver them to TVB"""

    def configure(self):
        self.receiving_ref_obj = None
        super(RayTVBReceiverTransformerInterface, self).configure()

    def receive_and_transform_data(self):
        return super(RayTVBReceiverTransformerInterface, self).__call__()

    def __call__(self, block=False):
        # Return:
        # data, if nothing to be done, i.e., receiving and transforming has been finalized
        # receiving_ref_obj if receiving is running
        # spiking_simulator.run_task_ref_obj if the spikeNet simulator is still running
        if self.receiving_ref_obj is None:
            if self.spiking_simulator.is_running:
                if block:
                    self.spiking_simulator.block_run  # BLOCK until spikeNet integration is over
                else:
                    return self.spiking_simulator.run_task_ref_obj
            self.receiving_ref_obj = self.receive_and_transform_data()
        return self._receive_data(block)


# def check_data(data, msg=""):
#     print("\n" + msg)
#     print("type=%s" % str(type(data)))
#     print("len=%d" % len(data))
#     types = [type(d) for d in data]
#     print("types=%s" % str(types))
#     shapes = [d.shape for d in data]
#     print("shapes=%s" % str(shapes))
#     if data[-1].size > 0:
#         try:
#             isnans = np.isnan(data[-1])
#             if np.all(isnans):
#                 print("All NaNs!")
#             elif np.any(isnans):
#                 print("Some NaNs found!")
#         except Exception as e:
#             pass
#             #print("\nWTF?: %s" % str(data))


class RayTVBtoSpikeNetInterface(TVBtoSpikeNetInterface, RaySenderInterface):

    """RayTVBtoSpikeNetInterface transforms TVB data via an optionally remote Transformer
        and sends them to a remote spikeNet simulator"""

    transformer_ref_obj = None
    ray_transformer_flag = False
    _data = None  # temporary data buffer

    transformer = Attr(
        label="Transformer",
        field_type=(Transformer, RayClient),
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    def __init__(self, **kwargs):
        self.transformer_ref_obj = None
        self.ray_transformer_flag = False
        self._data = None
        super(RayTVBtoSpikeNetInterface, self).__init__(**kwargs)

    def configure(self):
        self.transformer_ref_obj = None
        self.sending_ref_obj = None
        self._data = None
        assert isinstance(self.transformer, (Transformer, RayClient))
        self.ray_transformer_flag = False
        if self.transformer.__class__.__name__.find("Ray") > -1:
            self.ray_transformer_flag = True
        super(RayTVBtoSpikeNetInterface, self).configure()

    def _send_data(self, data=None, block=False):
        if data is None and self._data is not None:
            data = self._data.copy()
        if self.sending_ref_obj is None:
            if self.spiking_simulator.is_running:
                if block:
                    self.spiking_simulator.block_run
                else:
                    self._data = data
                    return self.spiking_simulator.run_task_ref_obj
            if data is not None:
                # check_data(data, msg="Sending transformed data from TVB!")
                self.sending_ref_obj = self.send_data(data)
            self._data = None
        return super(RayTVBtoSpikeNetInterface, self)._send_data(block=block)

    def _from_transforming_to_sending(self, block=False):
        self._data = ray.get(self.transformer_ref_obj)
        self.transformer_ref_obj = None
        return self._send_data(self._data, block=block)

    def __call__(self, data=None, block=False):
        # Return:
        # None, if nothing to be done or if transforming and sending has been finalized
        # transformer_ref_obj if transforming is running
        # a ref object, i.e., send_ref_obj if sending is running, or a spikeNet one if spikeNet simulator is running,
        if isinstance(self.transformer_ref_obj, ray._raylet.ObjectRef):
            # This is the case when transformer is remote and running....
            if block:
                return self._from_transforming_to_sending(block)
            else:
                running, done = ray.wait([self.transformer_ref_obj])
                if len(running) == 0:
                    return self._from_transforming_to_sending(block)
                else:
                    return self.transformer_ref_obj
        elif data is not None:
            # This is the case when the interface has to start the process by transforming TVB data...
            data = self.reshape_data(data)
            if self.ray_transformer_flag:
                self.transformer_ref_obj = self.transformer(data=data[1], time=data[0])
                if block:
                    return self._from_transforming_to_sending(block)
                return self.transformer_ref_obj
            else:
                return self._send_data(data=self.transform_data(data),
                                       block=block)
        else:
            return self._send_data(block=block)


class RaySpikeNetToTVBInterface(SpikeNetToTVBInterface, RayReceiverInterface):

    """RaySpikeNetToTVBInterface transforms receives data from a remote spikeNet simulator
        and transforms them via an optionally remote Transformer"""

    transformer_ref_obj = None
    ray_transformer_flag = False

    transformer = Attr(
        label="Transformer",
        field_type=(Transformer, RayClient),
        doc="""A Transformer class instance to process data.""",
        required=True
    )

    def __init__(self, **kwargs):
        self.transformer_ref_obj = None
        self.ray_transformer_flag = False
        super(RaySpikeNetToTVBInterface, self).__init__(**kwargs)

    def configure(self):
        self.transformer_ref_obj = None
        self.receive_ref_obj = None
        assert isinstance(self.transformer, (Transformer, RayClient))
        self.ray_transformer_flag = False
        if self.transformer.__class__.__name__.find("Ray") > -1:
            self.ray_transformer_flag = True
        super(RaySpikeNetToTVBInterface, self).configure()

    def _get_transformed_data_block(self):
        data = self.reshape_data(
            ray.get(self.transformer_ref_obj))
        self.transformer_ref_obj = None
        return data

    def _ray_transform(self, data=None, block=False):
        if isinstance(self.transformer_ref_obj, ray._raylet.ObjectRef):
            running, done = ray.wait([self.transformer_ref_obj])
            if block or len(running) == 0:
                return self._get_transformed_data_block()
            else:
                return self.transformer_ref_obj
        elif data is not None:
            self.transformer_ref_obj = self.transformer(data=data[1], time=data[0])
            if block:
                # ...run it with blocking:
                return self._get_transformed_data_block()
            else:
                # ...run it without blocking
                return self.transformer_ref_obj
        else:
            return None

    def receive_data(self):
        return super(RaySpikeNetToTVBInterface, self).receive_data()

    def _receive_and_transform(self, block=False):
        # Spiking simulation is done, receive data:
        if self.receiving_ref_obj is None:
            self.receiving_ref_obj = self.receive_data()
        data = self._receive_data(block)
        if isinstance(data, ray._raylet.ObjectRef) or data is None:
            return data
        elif data[0][1] < data[0][0]:
            return None
        # check_data(data, msg="Received data from NEST!")
        # Send data to transformer and start transforming:
        if self.ray_transformer_flag:
            return self._ray_transform(data=data, block=block)
        else:
            return self.reshape_data(
                self.transform_data(data))

    def __call__(self, block=False):
        # Return:
        # data, if receiving and transforming data has been finalized
        # transformer_ref_obj if transforming is running
        # receive_ref_obj if receiving is running, i.e., if spiking simulator is running,
        # because we assume that receiving of data is sequential and blocking anyway
        if self.transformer_ref_obj:
            # This is the case when a remote transformer is running...
            return self._ray_transform(block=block)
        else:
            return self._receive_and_transform(block=block)


class RayTVBOutputInterfaces(TVBOutputInterfaces):

    """TVBOutputInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and sends data to them."""

    running_tasks_refs = list()

    def __init__(self, **kwargs):
        self.running_tasks_refs = list()
        super(RayTVBOutputInterfaces, self).__init__(**kwargs)

    @property
    def is_running(self):
        return len(self.running_tasks_refs) > 0

    @property
    def data_ready(self):
        if len(self.running_tasks_refs):
            running, done = ray.wait([task_ref for task_ref in self.running_tasks_refs if task_ref is not None])
            return len(running) == 0
        return None

    def __call__(self, data=None, block=False):
        if data is None:
            for ii, (interface, running_task_ref) in enumerate(zip(self.interfaces, self.running_tasks_refs)):
                if running_task_ref is not None:
                    # Only if there is a transforming or sending task pending, rerun the interface:
                    self.running_tasks_refs[ii] = interface(block=block)
        else:
            # This is the case we need to transform data before sending them:
            for interface in self.interfaces:
                #                 data values !!! assuming only 1 mode!!! -> shape (times, vois, proxys):
                self.running_tasks_refs.append(
                    interface([self._compute_interface_times(interface, data),
                               data[interface.monitor_ind][1][:, interface.voi_loc][:, :, interface.proxy_inds, 0]],
                              block=block))
        self.running_tasks_refs = [ref_obj for ref_obj in self.running_tasks_refs if ref_obj is not None]
        # Return pending task references or empty list:
        return self.running_tasks_refs


class RayTVBInputInterfaces(TVBInputInterfaces):

    """TVBInputInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them."""

    running_tasks_refs = list()
    cosim_updates = np.array(list())
    all_time_steps = list()

    def __init__(self, **kwargs):
        self.running_tasks_refs = list()
        self.cosim_updates = np.array(list())
        self.all_time_steps = list()
        super(RayTVBInputInterfaces, self).__init__(**kwargs)

    @property
    def is_running(self):
        return len(self.running_tasks_refs) > 0

    @property
    def data_ready(self):
        if len(self.running_tasks_refs):
            running, done = ray.wait([task_ref for task_ref in self.running_tasks_refs
                                      if isinstance(task_ref, ray._raylet.ObjectRef)])
            return len(running) == 0
        return None

    def __call__(self, good_cosim_update_values_shape=None, block=False):
        if not self.is_running:
            # print("\nInitializing cosim_updates for this syncrun...")
            # Initialize at first call for this instance of synchronization:
            self.cosim_updates, self.all_time_steps = self._prepare_cosim_update(good_cosim_update_values_shape)
            self.running_tasks_refs = [1] * self.number_of_interfaces
        for ii, (interface, running_task_ref) in enumerate(zip(self.interfaces, self.running_tasks_refs)):
            if self.running_tasks_refs[ii] is not None:
                # print("\nNot running...Get data...")
                # Get data or reference to a remote task of receiving annd/or transforming data:
                self.running_tasks_refs[ii] = interface(block=block)
                if not isinstance(self.running_tasks_refs[ii], ray._raylet.ObjectRef)  \
                    and self.running_tasks_refs[ii] is not None:
                    # It is data, place them to cosim_updates
                    data = self.running_tasks_refs[ii].copy()
                    # print("\nIt must be data: %s..." % str(type(data)))
                    self.cosim_updates, time_steps = \
                        self._set_data_from_interface(self.cosim_updates, interface,
                                                      data, good_cosim_update_values_shape)
                    self.all_time_steps += time_steps.tolist()
                    self.running_tasks_refs[ii] = None
        self.running_tasks_refs = [ref_obj for ref_obj in self.running_tasks_refs if ref_obj is not None]
        if len(self.running_tasks_refs) == 0:
            # print("\nAll interfaces finished. Returning data!")
            # Return cosim_updates data:
            inputs = self.get_inputs(self.cosim_updates, self.all_time_steps, good_cosim_update_values_shape)
            self.cosim_updates = np.array([])
            self.all_time_steps = []
            return inputs
        else:
            # print("\nStill some interfaces running....!")
            return self.running_tasks_refs


class RayTVBtoSpikeNetInterfaces(RayTVBOutputInterfaces, SpikeNetInputInterfaces):

    """RayTVBtoSpikeNetInterfaces"""

    pass


class RaySpikeNetToTVBInterfaces(TVBInputInterfaces, SpikeNetOutputInterfaces):

    """SpikeNetToTVBInterfaces"""

    pass
