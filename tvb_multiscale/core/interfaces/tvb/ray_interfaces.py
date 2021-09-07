# -*- coding: utf-8 -*-

import ray

import numpy as np

from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetInterface, SpikeNetToTVBInterface, \
    TVBOutputInterfaces, TVBInputInterfaces, SpikeNetInputInterfaces, SpikeNetOutputInterfaces


class RayTVBtoSpikeNetInterface(TVBtoSpikeNetInterface):

    ray_transformer_flag = False

    transformer_ref_obj = None

    send_ref_obj = None

    def configure(self):
        super(RayTVBtoSpikeNetInterface, self).configure()
        self.ray_transformer_flag = False
        if self.transformer.__class__.__name__.find("Ray") > -1:
            self.ray_transformer_flag = True

    def _send_data_block(self):
        self.send_ref_obj = None
        return self.set_proxy_data([self.transformer.output_time, self.transformer.output_buffer])

    def __call__(self, data=None, block=False):
        # Return:
        # None, if nothing to be done or if transforming and sending has been finalized
        # transformer_ref_obj if transforming is running
        # send_ref_obj if sending is running
        if isinstance(self.transformer_ref_obj, ray._raylet.ObjectRef):
            # This is the case when transformer is remote and running....
            if block:
                ray.get(self.transformer_ref_obj)
                self.transformer_ref_obj = None
                self.send_ref_obj = 1
                return self.__call__(block=block)
            else:
                return self.transformer_ref_obj
        elif data is not None:
            # This is the case when the interface has to start the process by transforming TVB data....
            self.transformer.input_time = data[0]
            self.transformer.input_buffer = self.reshape_data(data[1])
            if self.ray_transformer_flag:
                self.transformer_ref_obj = self.transformer()
            else:
                self.transformer()
            return self.__call__(block=block)
        else:
            # This is the case when we need to send the data.
            if block:
                # This is the blocking case
                ray.get(self.spiking_simulator.run_task_ref_obj)  # BLOCK until spikeNet integration is over
                return self._send_data_block()
            else:
                running, done = ray.wait([self.spiking_simulator.run_task_ref_obj])
                if len(done):
                    # Spiking simulation is done, send data and block
                    return self._send_data_block()
                else:
                    # Spiking simulation is not done yet, so, do not block.
                    self.send_ref_obj = running[0]
                    return self.send_ref_obj


class RaySpikeNetToTVBInterface(SpikeNetToTVBInterface):

    ray_transformer_flag = False

    transformer_ref_obj = None

    receive_ref_obj = None

    def configure(self):
        super(RaySpikeNetToTVBInterface, self).configure()
        self.ray_transformer_flag = False
        if self.transformer.__class__.__name__.find("Ray") > -1:
            self.ray_transformer_flag = True

    def _receive_and_transform(self, block=False):
        # Spiking simulation is done, receive data, which is blocking by definition:
        data = self.get_proxy_data()
        self.receive_ref_obj = None
        if data[0][1] < data[0][0]:
            return None
        # Having received the data, get them to the transformer...
        self.transformer.input_time = data[0]
        self.transformer.input_buffer = data[1]
        if self.ray_transformer_flag:
            # If the transformer is remote...
            if block:
                # ...run it with blocking:
                return ray.get(self.transformer())
            else:
                # ...run it without blocking
                self.transformer_ref_obj = self.transformer()
                return self.transformer_ref_obj
        else:
            # ...else if the transformer is not remote, run it locally and block:
            return self.transformer()

    def _get_transformed_data_block(self):
        transformer_outputs = ray.get(self.transformer_ref_obj)
        self.transformer_ref_obj = None
        return transformer_outputs

    def __call__(self, block=False):
        # Return:
        # data, if receiving and transforming data has been finalized
        # transformer_ref_obj if transforming is running
        # receive_ref_obj if receiving is running
        if self.transformer_ref_obj:
            # This is the case when a remote transformer is running...
            if block:
                # ...and we want to block:
                return self._get_transformed_data_block()
            else:
                # ...or not...
                running, done = ray.wait([self.transformer_ref_obj])
                if len(done):
                    return self._get_transformed_data_block()
                else:
                    return self.transformer_ref_obj
        else:
            # This is the case when we need to receive data...
            if self.receive_ref_obj is None:
                self.receive_ref_obj = self.spiking_simulator.run_task_ref_obj
            if block:
                ray.get(self.receive_ref_obj)  # BLOCK until spikeNet integration is over
            else:
                running, done = ray.wait([self.receive_ref_obj])
                if len(running):
                    # Spiking simulation is not done yet. Do not block.
                    self.receive_ref_obj = running[0]
                    return self.receive_ref_obj
            return self._receive_and_transform(block)


class RayTVBOutputInterfaces(TVBOutputInterfaces):

    """TVBOutputInterfaces class holds a list of TVB interfaces to transformer/cosimulator
       and sends data to them."""

    running_tasks_refs = []

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
            for interface, running_task_ref in zip(self.interfaces, self.running_tasks_refs):
                if running_task_ref is not None:
                    # Only if there is a transforming or sending task pending, rerun the interface:
                    self.running_tasks_refs.append(interface(block=block))
        else:
            # This is the case we need to transform data before sending them:
            self.running_tasks_refs = []
            for interface in self.interfaces:
                #                 data values !!! assuming only 1 mode!!! -> shape (times, vois, proxys):
                self.running_tasks_refs.append(
                    interface([self._compute_interface_times(interface, data),
                               data[interface.monitor_ind][1][:, interface.voi_loc][:, :, interface.proxy_inds, 0]],
                              block=block))
        if np.all([running_task_ref is None for running_task_ref in self.running_tasks_refs]):
            self.running_tasks_refs = []
        # Return pending task references or empty list:
        return self.running_tasks_refs


class RayTVBInputInterfaces(TVBInputInterfaces):

    """TVBInputInterfaces class holds a list of TVB interfaces from transformer/cosimulator
       and receives data from them."""

    running_tasks_refs = []
    cosim_updates = np.array([])
    all_time_steps = []

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
            # Initialize at first call for this instance of synchronization:
            self.cosim_updates, self.all_time_steps = self._prepare_cosim_upadate(good_cosim_update_values_shape)
            self.running_tasks_refs = [1] * self.number_of_interfaces
        for ii, (interface, running_task_ref) in enumerate(zip(self.interfaces, self.running_tasks_refs)):
            if self.running_tasks_refs[ii] is not None:
                # Get data or reference to a remote task of receiving or transforming data:
                data = interface(block=block)
                if isinstance(data, ray._raylet.ObjectRef):
                    # If it is a reference to a remote task, set it:
                    self.running_tasks_refs[ii] = data
                elif data is not None:
                    # If it is data, place them to cosim_updates
                    self.cosim_updates, time_steps = \
                        self._set_data_from_interface(self.cosim_updates, interface, data, good_cosim_update_values_shape)
                    self.all_time_steps += time_steps.tolist()
                    self.running_tasks_refs[ii] = None  # mark this interface as done
        if np.all([running_task_ref is None for running_task_ref in self.running_tasks_refs]):
            self.running_tasks_refs = []
            # Return cosim_updates data:
            inputs = self.get_inputs(self.cosim_updates, self.all_time_steps, good_cosim_update_values_shape)
            self.cosim_updates = np.array([])
            self.all_time_steps = []
            return inputs
        else:
            # Return pending task references:
            return [task_ref for task_ref in self.running_tasks_refs if task_ref is not None]


class RayTVBtoSpikeNetInterfaces(RayTVBOutputInterfaces, SpikeNetInputInterfaces):

    """RayTVBtoSpikeNetInterfaces"""

    pass


class RaySpikeNetToTVBInterfaces(TVBInputInterfaces, SpikeNetOutputInterfaces):

    """SpikeNetToTVBInterfaces"""

    pass
