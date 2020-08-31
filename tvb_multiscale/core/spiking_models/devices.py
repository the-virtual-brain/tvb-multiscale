# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from six import string_types
from collections import OrderedDict

import pandas as pd
import xarray as xr
import numpy as np

from tvb_multiscale.core.config import initialize_logger, LINE

from tvb_multiscale.core.data_structures_utils import filter_neurons, filter_events, summarize

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils \
    import ensure_list, flatten_list, list_of_dicts_to_dict_of_lists, \
    sort_events_by_x_and_y, data_xarray_from_continuous_events, extract_integer_intervals, is_integer
from tvb.contrib.scripts.utils.computations_utils import spikes_rate_convolution, compute_spikes_counts


LOG = initialize_logger(__name__)


"""Classes for creating:
    - output devices that can measure and summarize the activity of a whole neuronal population
    - input devices that induce some activity by stimulation to a whole neuronal population.
Output devices assume that data is structured in discrete events tagged by time and sender neuron.
"""


class Device(object):
    __metaclass__ = ABCMeta

    """Class to wrap around a measuring or stimulating device"""

    device = None  # the index of the device in the spiking network
    model = "device"  # the device model name
    _number_of_connections = 0  # total number of devices' connections to neurons

    # Modify accordingly for other simulators than NEST, by overwriting to the inheriting class:
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, device, *args, **kwargs):
        self.device = device   # the index of the device in the spiking network
        self.model = "device"  # the device model name
        self._number_of_connections = self.number_of_connections

    def __repr__(self):
        return "%s - Model: %s, gid: %d" % (self.__class__.__name__, self.model, self.device[0])

    def __str__(self):
        return self.print_str()

    def print_str(self, connectivity=False):
        output = "\n" + self.__repr__() + "\nparameters: %s" % str(self.get_attributes())
        if connectivity:
            neurons = self.neurons
            output += ",\nconnections to %d neurons: %s," \
                      "\nweights: %s," \
                      "\ndelays: %s," \
                      "\nreceptors: %s" % \
                      (len(neurons), extract_integer_intervals(neurons, print=True),
                       str(self.get_weights(summary="stats")),
                       str(self.get_delays(summary="stats")),
                       str(self.get_receptors(summary=1)))
        return output

    @abstractmethod
    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        pass

    # Methods to get or set attributes for devices and/or their connections:

    @abstractmethod
    def Set(self, values_dict):
        """Method to set attributes of the device
           Arguments:
            values_dict: dictionary of attributes names' and values.
        """
        pass

    @abstractmethod
    def Get(self, attrs=None):
        """Method to get attributes of the device.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all neurons' attributes.
           Returns:
            Dictionary of attributes.
        """
        pass

    @abstractmethod
    def _GetConnections(self, **kwargs):
        """Method to get attributes of the connections from/to the device
           Return:
            the connections' objects
        """
        pass

    @abstractmethod
    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device to/from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        pass

    @abstractmethod
    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the device
            Arguments:
             connections: connections' objects.
             values_dict: dictionary of attributes names' and values.
            Returns:
             Dictionary of (numpy) arrays of connections' attributes.
        """
        pass

    @abstractmethod
    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            connections: connections' objects.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of (numpy) arrays of connections' attributes.
        """
        pass

    def filter_neurons(self, neurons=None, exclude_neurons=[]):
        """This method will select/exclude the connected neurons, depending on user inputs
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            tuple of neurons.
        """
        # Method to select or exclude some of the connected neurons to the device:
        if neurons is None:
            neurons = self.neurons
        return filter_neurons(neurons, exclude_neurons)

    def get_attributes(self):
        """Method to get all attributes of the device.
           Returns:
            Dictionary of (numpy) arrays of neurons' attributes.
        """
        return self.Get()

    def get_number_of_connections(self, neurons=None, exclude_neurons=[]):
        """Method to get the number of  connections of the device to/from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            int: number of connections
        """
        return len(self.GetConnections(neurons=neurons, exclude_neurons=exclude_neurons))

    def SetToConnections(self, values_dict, neurons=None, exclude_neurons=[]):
        """Method to set attributes of the connections from/to the device.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: collection (list, tuple, array) of neurons the attribute of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
        """
        self.SetToConnections(values_dict, self.GetConnections(neurons, exclude_neurons))

    def GetFromConnections(self, attrs=None, neurons=None, exclude_neurons=[], summary=None):
        """Method to get attributes of the connections from/to the device.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            neurons: collection (list, tuple, array) of neurons the attribute of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of (numpy) arrays of connections' attributes.
        """
        attributes = self._GetFromConnections(self.GetConnections(neurons, exclude_neurons), attrs)
        if summary:
            return summarize(attributes, summary)
        else:
            return attributes

    def get_weights(self, neurons=None, exclude_neurons=[], summary=None):
        """Method to get weights of the connections from/to the device.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the weights of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of connections' weights
        """
        return self.GetFromConnections(self._weight_attr, neurons, exclude_neurons, summary)[self._weight_attr]

    def get_delays(self, neurons=None, exclude_neurons=[], summary=None):
        """Method to get delays of the connections from/to the device.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the delays of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of connections' delays
        """
        return self.GetFromConnections(self._delay_attr, neurons, exclude_neurons, summary)[self._delay_attr]

    def get_receptors(self, neurons=None, exclude_neurons=[], summary=None):
        """Method to get the receptors of the connections from/to the device.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the attribute of which should be set.
                     Default = None, corresponds to all neurons of the population.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Array of connections' receptors
        """
        return self.GetFromConnections(self._receptor_attr, neurons, exclude_neurons, summary)[self._receptor_attr]

    # attributes of device connections across all neurons connected to it

    @property
    @abstractmethod
    def connections(self):
        """Method to get all connections of the device to/from the device.
           Returns:
            connections' objects.
        """
        pass

    @property
    @abstractmethod
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        pass

    @property
    def weights(self):
        """Method to get all connections' weights of the device to/from the device."""
        return self.get_weights()

    @property
    def delays(self):
        """Method to get all connections' delays of the device to/from the device."""
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get all connections' receptors of the device to/from the device."""
        return self.get_receptors()

    # Summary attributes of device across all neurons connected to it

    @property
    def number_of_connections(self):
        """Method to get the number of all connections from/to the device."""
        if self._number_of_connections == 0 or self._number_of_connections is None:
            self._number_of_connections = self.get_number_of_connections()
        return self._number_of_connections

    @property
    def node_weight(self):
        """Method to get the mean of all connections' weights of the device to/from neurons."""
        return np.mean(self.weights)

    @property
    def node_delay(self):
        """Method to get the mean of all connections' delays of the device to/from neurons."""
        return np.mean(self.delays)

    @property
    def node_receptors(self):
        """Method to get all unique connections' receptors of the device to/from neurons."""
        return np.unique(self.receptors)


class InputDevice(Device):
    model = "input_device"

    def __init__(self, device, *args, **kwargs):
        super(InputDevice, self).__init__(device, *args, **kwargs)
        self.model = "input_device"

    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device to neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.device,
                                    target=self.filter_neurons(neurons, exclude_neurons))

    @property
    def connections(self):
        """Method to get all connections of the device to/from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(source=self.device)


InputDeviceDict = {}


class OutputDevice(Device):
    model = "output_device"
    
    def __init__(self, device, *args, **kwargs):
        super(OutputDevice, self).__init__(device, *args, **kwargs)
        self.model = "output_device"

    def GetConnections(self, neurons=None, exclude_neurons=[]):
        """Method to get connections of the device from neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons which should be included in the output.
                     Default = None, corresponds to all neurons the device is connected to.
            exclude_neurons: collection (list, tuple, array) of neurons which should be excluded. Default = [].
           Returns:
            connections' objects.
        """
        return self._GetConnections(source=self.filter_neurons(neurons, exclude_neurons),
                                    target=self.device)

    @property
    def connections(self):
        """Method to get all connections of the device from neurons.
           Returns:
            connections' objects.
        """
        return self.GetConnections(target=self.device)

    def filter_events(self, events=None, variables=None, neurons=None, times=None,
                      exclude_neurons=[], exclude_times=[]):
        """This method will select/exclude part of the measured events, depending on user inputs
            Arguments:
                events: dictionary of events
                variables: collection (list, tuple, array) of variables to be included in the output,
                           assumed to correspond to keys of the events dict.
                           Default=None, corresponds to all keys of events.
                neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                         Default = None, corresponds to all neurons found as senders of events.
                times: collection (list, tuple, array) of times the events of which should be included in the output.
                         Default = None, corresponds to all events' times.
                exclude_neurons: collection (list, tuple, array) of neurons
                                 the events of which should be excluded from the output. Default = [].
                exclude_times: collection (list, tuple, array) of times
                               the events of which should be excluded from the output. Default = [].
            Returns:
              the filtered dictionary of events
        """
        # The events of the device
        if events is None:
            events = self.events
        return filter_events(events, variables, neurons, times, exclude_neurons, exclude_times)

    @property
    @abstractmethod
    def events(self):
        """This method returns the dictionary of events"""
        pass

    @property
    @abstractmethod
    def number_of_events(self):
        """This method returns the number (integer) of events"""
        pass

    @property
    @abstractmethod
    def reset(self):
        """This method resets the device by deleting all events"""
        pass

    @property
    def senders(self):
        """This method returns the senders neurons of events"""
        return self.events['senders']

    @property
    def times(self):
        """This method returns the times of events"""
        return self.events['times']


class SpikeDetector(OutputDevice):
    model = "spike_detector"

    def __init__(self, device, *args, **kwargs):
        super(SpikeDetector, self).__init__(device, *args, **kwargs)
        self.model = "spike_detector"

    def get_spikes_events(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will select/exclude part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             the filtered dictionary of spikes' events
        """
        return self.filter_events(None, None, neurons, times, exclude_neurons, exclude_times)

    def get_spikes_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return spikes' times
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             the spikes' times in an array
        """
        return self.filter_events(None, "times", neurons, times, exclude_neurons, exclude_times)["times"]

    def get_spikes_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return spikes' times
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                           the events of which should be excluded from the output. Default = [].
           Returns:
            the spikes' times in an array
        """
        return self.filter_events(None, "senders", neurons, times, exclude_neurons, exclude_times)["senders"]

    # The following attributes are time summaries without taking into consideration spike timing:

    def get_number_of_spikes(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return the total number of spikes
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             int: the spikes' total number
        """
        return len(self.get_spikes_times(neurons, times, exclude_neurons, exclude_times))

    def get_mean_number_of_spikes(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return the total number of spikes divided by the total number of connected neurons
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             float: total number of spikes divided by the total number of connected neurons
        """
        n_neurons = self.get_number_of_connections(neurons, exclude_neurons)
        if n_neurons > 0:
            return len(self.get_spikes_times(neurons, times, exclude_neurons, exclude_times)) / n_neurons
        else:
            return 0.0

    def get_spikes_rate(self, dt=1.0, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return the mean spike rate,
           as the total number of spikes divided by the total number of connected neurons and time duration
           after selecting/excluding part of the detected spikes' events, depending on user inputs
           Arguments:
            dt: total time duration, Default = 1.0 (in msec)
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                           the events of which should be excluded from the output. Default = [].
           Returns:
            float: total number of spikes divided by the total number of connected neurons
        """
        return self.get_mean_number_of_spikes(neurons, times, exclude_neurons, exclude_times) / dt

    def get_spikes_times_by_neurons(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[],
                                    full_senders=False):
        """This method will return the spikes' times per neuron,
             after selecting/excluding part of the detected spikes' events, depending on user inputs
             Arguments:
              neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                       Default = None, corresponds to all neurons found as senders of events.
              times: collection (list, tuple, array) of times the events of which should be included in the output.
                     Default = None, corresponds to all events' times.
              exclude_neurons: collection (list, tuple, array) of neurons
                               the events of which should be excluded from the output. Default = [].
              exclude_times: collection (list, tuple, array) of times
                             the events of which should be excluded from the output. Default = [].
              full_senders: if True, we include neurons' indices that have sent no spike at all. Default=False
             Returns:
              dictionary of spike events sorted by sender neuron
         """
        sorted_events = sort_events_by_x_and_y(self.events, x="senders", y="times",
                                               filter_x=neurons, filter_y=times,
                                               exclude_x=exclude_neurons, exclude_y=exclude_times)
        if full_senders:
            # In this case we also include neurons with 0 spikes in the output
            sender_neurons = self.filter_neurons(neurons=neurons, exclude_neurons=exclude_neurons)
            output = OrderedDict()
            for neuron in sender_neurons:
                output[neuron] = np.array([])
            output.update(sorted_events)
            return output
        else:
            return sorted_events

    def get_spikes_neurons_by_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return the spikes' senders per spike time,
             after selecting/excluding part of the detected spikes' events, depending on user inputs
             Arguments:
              neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                       Default = None, corresponds to all neurons found as senders of events.
              times: collection (list, tuple, array) of times the events of which should be included in the output.
                     Default = None, corresponds to all events' times.
              exclude_neurons: collection (list, tuple, array) of neurons
                               the events of which should be excluded from the output. Default = [].
              exclude_times: collection (list, tuple, array) of times
                             the events of which should be excluded from the output. Default = [].
             Returns:
              dictionary of spike events sorted by time
         """
        return sort_events_by_x_and_y(self.events, x="times", y="senders",
                                      filter_x=times, filter_y=neurons,
                                      exclude_x=exclude_times, exclude_y=exclude_neurons)

    @property
    def spikes_times(self):
        """This method will return spikes' times in an array"""
        return self.times

    @property
    def spikes_senders(self):
        """This method will return spikes' sender neurons in an array"""
        return self.senders

    @property
    def number_of_spikes(self):
        """This method will return the total number (int) of spike events"""
        return self.number_of_events

    @property
    def mean_number_of_spikes(self):
        """This method will return the total number of spikes divided by the total number of connected neurons"""
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        """This method will return the mean spike rate (float) as the
           total number of spikes divided by the total number of connected neurons, assuimng a duration of 1 ms
        """
        return self.get_spikes_rate()

    # The following attributes are computed across time:

    def compute_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points=None,
                                        spikes_kernel=None, mode="per_neuron",
                                        name=None, **kwargs):
        """This method computes spike rate across time.
           Arguments:
            time: the time vector
            spikes_kernel_width: float: the width of the sliding window kernel in ms
            spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                           Default=None, and computed using the time vector
            spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
            mode: if "per_neuron" the output is returned for each neuron separetely.
                  Otherwise, it is computed across all neurons. Default = "per_neuron"
            name: name of the data to be computed. Default = None,
                  which defaults to device_model_name + " - Total spike rate across time"
           Returns:
            xarray.DataArray with spike rates' time series
        """
        if spikes_kernel is None:
            if spikes_kernel_width_in_points is None:
                spikes_kernel_width_in_points = np.maximum(1, int(np.ceil(spikes_kernel_width / np.diff(time))))
            # Default spikes' kernel is just a rectangular one, normalized with its width.
            spikes_kernel = np.ones((spikes_kernel_width_in_points, )) / spikes_kernel_width

        if name is None:
            name = self.model + " - Total spike rate across time"

        if mode == "per_neuron":
            senders_neurons = []
            rates = []
            # Computing separately per neuron
            for neuron, spikes_times in self.get_spikes_times_by_neurons(full_senders=True, **kwargs).items():
                senders_neurons.append(neuron)
                if len(spikes_times) > 0:
                    # Getting spikes counts per time interval
                    spikes_counts = compute_spikes_counts(spikes_times, time)
                    # Computing rate as a convolution with spikes_kernel
                    rates.append(spikes_rate_convolution(spikes_counts, spikes_kernel))
                else:
                    rates.append(np.zeros(time.shape))
            return xr.DataArray(rates, dims=["Neuron", "Time"], coords={"Neuron": senders_neurons,
                                                                        "Time": time})
        else:
            # Computing for the whole population(s) alltogether
            spikes_times = self.get_spikes_times(**kwargs)
            if len(spikes_times) > 0:
                # Getting spikes counts per time interval
                spikes_counts = compute_spikes_counts(spikes_times, time)
                # Computing rate as some kind of convolution with spikes_kernel
                rates = spikes_rate_convolution(spikes_counts, spikes_kernel)
            else:
                rates = np.zeros(time.shape)
            return xr.DataArray(rates, dims=["Time"], coords={"Time": time}, name=name)

    def compute_mean_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                             spikes_kernel=None, name=None, **kwargs):
        """This method computes mean spike rate across time
           by dividing the total spike rate with the total number of neurons.
           Arguments:
            time: the time vector
            spikes_kernel_width: float: the width of the sliding window kernel in ms
            spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                           Default=None, and computed using the time vector
            spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
            mode: if "per_neuron" the output is returned for each neuron separetely.
                  Otherwise, it is computed across all neurons. Default = "per_neuron"
            name: name of the data to be computed. Default = None,
                  which defaults to device_model_name + " - Mean spike rate accross time"
           Returns:
            xarray.DataArray with spike rates' time series
        """
        if name is None:
            name = self.model + " - Mean spike rate accross time"
        n_neurons = self.get_number_of_connections(**kwargs)
        if n_neurons > 0:
            return self.compute_spikes_rate_across_time(time, spikes_kernel_width, spikes_kernel_width_in_points,
                                                        spikes_kernel, "total", name, **kwargs) / n_neurons
        else:
            return xr.DataArray(0.0 * time, dims=["Time"], coords={"Time": time}, name=name)


class Multimeter(OutputDevice):
    model = "multimeter"

    def __init__(self, device, *args, **kwargs):
        super(Multimeter, self).__init__(device)
        self.model = "multimeter"

    @property
    @abstractmethod
    def record_from(self):
        """This method returns a list of the recorded variabls of the device"""
        pass

    @property
    def time(self):
        """This method returns the time vector (array) of the device."""
        return np.unique(self.times)

    def _determine_variables(self, variables=None):
        """This method selects (a subset of) variables after asserting that the multimeter records from them.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
           Returns:
            a list of (selected) variables' names the multimeter records from
        """
        if variables is not None:
            variables = ensure_list(variables)
            for variable in variables:
                assert variable in self.record_from
        else:
            variables = self.record_from
        return variables

    def get_data(self, variables=None, neurons=None, exclude_neurons=[],
                 name=None, dims_names=["Variable", "Neuron", "Time"]):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        if name is None:
            name = self.model
        events = dict(self.events)
        times = events.pop("times")
        senders = events.pop("senders")
        # We assume that the multimeter captures events even for continuous variables as it is the case in NEST.
        # Therefore, we have to re-arrange the output to get all variables separated following time order.
        return data_xarray_from_continuous_events(events, times, senders,
                                                  variables=self._determine_variables(variables),
                                                  filter_senders=neurons, exclude_senders=exclude_neurons,
                                                  name=name, dims_names=dims_names)

    def get_mean_data(self, variables=None, neurons=None, exclude_neurons=[],
                      name=None, dims_names=["Variable", "Time"]):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], "Neuron", dims_names[1]]
        data = self.get_data(variables, neurons, exclude_neurons, name, dims_names)
        return data.mean(dim="Neuron")

    def get_total_data(self, variables=None, neurons=None, exclude_neurons=[],
                       name=None, dims_names=["Variable", "Time"]):
        """This method returns time series' data recorded by the multimeter, summed across the neurons' dimension.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dims_names[0], "Neuron", dims_names[1]]
        data = self.get_data(variables, neurons, exclude_neurons, name, dims_names)
        return data.sum(dim="Neuron")

    @property
    def data(self):
        """This method returns time series' data recorded by the multimeter.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data()

    @property
    def data_mean(self):
        """This method returns time series' data recorded by the multimeter, averaged across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data()

    @property
    def data_total(self):
        """This method returns time series' data recorded by the multimeter, summed across the neurons' dimension.
           Returns:
            a xarray DataArray with the output data with dimensions ["Variable", "Time"]
        """
        return self.get_mean_data()

    def current_data(self, variables=None, neurons=None, exclude_neurons=[],
                     name=None, dims_names=["Variable", "Neuron"]):
        """This method returns the last time point of the data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        # This method will return current time data
        if name is None:
            name = self.model
        coords = OrderedDict()
        variables = self._determine_variables(variables)
        events = self.events
        times = events["times"]
        coords[dims_names[0]] = variables
        if len(times) > 0:
            # Get only the last time stamp events:
            output_inds = events["times"] == events["times"][-1]
            # Optionally filter sender neurons
            senders = events["senders"]
            if neurons is not None:
                output_inds = np.logical_and(output_inds,
                                             [sender in flatten_list(neurons) for sender in senders])
            else:
                neurons = ensure_list(self.neurons)
            if len(exclude_neurons) > 0:
                output_inds = np.logical_and(output_inds,
                                             [sender not in flatten_list(exclude_neurons) for sender in senders])
            coords[dims_names[1]] = neurons
            data = np.empty((len(variables), len(neurons)))
            for i_var, var in enumerate(variables):
                data[i_var] = events[var][output_inds]
        else:
            # The multimeter is still empty, so return zeros
            if neurons is None:
                neurons = ensure_list(self.neurons)
            else:
                neurons = flatten_list(neurons)
            for neuron in flatten_list(exclude_neurons):
                try:
                    neurons.remove(neuron)
                except:
                    pass
            n_neurons = len(neurons)
            coords[dims_names[1]] = neurons
            data = np.empty((len(variables), len(neurons)))
            for i_var in range(len(variables)):
                data[i_var] = np.zeros((n_neurons, ))
        return xr.DataArray(data, coords=coords, dims=list(coords.keys()), name=name)

    def current_data_mean(self, variables=None, neurons=None, exclude_neurons=[], name=None, dim_name="Variable"):
        """This method returns the last time point of the data recorded by the multimeter, averaged across neurons.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            a xarray DataArray with the output data
        """
        dims_names = [dim_name, "Neuron"]
        output = self.current_data(variables, neurons, exclude_neurons, name, dims_names)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, variables=None, neurons=None, exclude_neurons=[]):
        """This method returns the last time point of the data recorded by the multimeter, averaged across neurons,
           in a list of values
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
           Returns:
            a list of values with output data
        """
        return self.current_data_mean(variables, neurons, exclude_neurons).values.tolist()


class Voltmeter(Multimeter):
    # The Voltmer is just a Mutlimeter measuring only a voltage quantity
    model = "voltmeter"

    def __init__(self, device, *args, **kwargs):
        super(Voltmeter, self).__init__(device)
        self.model = "voltmeter"

    @property
    @abstractmethod
    def var(self):
        """A method to return the string of the voltage variable's name, e.g, "V_m" for membrane potential in NEST"""
        pass

    def get_data(self, neurons=None, exclude_neurons=[], name=None, dims_names=["Variable", "Neuron", "Time"]):
        """This method returns time series' data of the membrane potential recorded by the voltmeter.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the voltmeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        return super(Voltmeter, self).get_data(self.var, neurons, exclude_neurons,  name, dims_names)

    def get_mean_data(self, neurons=None, exclude_neurons=[], name=None, dims_names=["Variable", "Time"]):
        """This method returns time series' data of the membrane potential recorded by the voltmeter,
           averaged across neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the voltmeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        data = self.get_data(neurons, exclude_neurons)
        return data.mean(dim="Neuron")

    @property
    def data(self):
        """This method returns time series' data of the membrane potential recorded by the voltmeter.
           Returns:
            a xarray DataArray with the output data and dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_data()

    @property
    def data_mean(self):
        """This method returns time series' data of the membrane potential recorded by the voltmeter,
           averaged across neurons.
           Returns:
            a xarray DataArray with the output data and dimensions ["Variable", "Neuron", "Time"]
        """
        return self.get_mean_data()

    def current_data(self, neurons=None, exclude_neurons=[], name=None, dims_names=["Variable", "Neuron"]):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the voltmeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        return super(Voltmeter, self).current_data([self.var], neurons, exclude_neurons, name, dims_names)

    def current_data_mean(self, neurons=None, exclude_neurons=[], name=None, dim_name="Variable"):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter,
           averaged across neurons.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            a xarray DataArray with the output data
        """
        output = self.current_data(neurons, exclude_neurons, name, dim_name)
        return output.mean(dim="Neuron")

    def current_data_mean_values(self, neurons=None, exclude_neurons=[], name=None, dim_name="Variable"):
        """This method returns the last time point of the membrane's voltage data recorded by the voltmeter,
           averaged across neurons, in a list of values.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dim_name: the dimension label (string) for the output array. Default = "Variable"
           Returns:
            list of values with the output data
        """
        return self.current_data_mean(neurons, exclude_neurons, name, dim_name).values.tolist()


class SpikeMultimeter(Multimeter, SpikeDetector):
    """This is a multimeter device that measures a continuous spike variable,
       i.e., a variable that equals zero for every time moment when there is no spike emitted/received, and
        a positive or negative spike weight, where there is a spike.
    """
    model = "spike_multimeter"
    spike_var = "spikes"

    def __init__(self, device, *args, **kwargs):
        super(SpikeMultimeter, self).__init__(device)
        self.model = "spike_multimeter"

    def get_spikes(self, neurons=None, exclude_neurons=[], name=None, dims_names=["Variable", "Neuron", "Time"]):
        """This method returns time series' data of spike weights recorded by the spike multimeter.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            name: label of output. Default = None
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Variable", "Neuron", "Time"]
           Returns:
            a xarray DataArray with the output data
        """
        return self.get_data([self.spike_var], neurons, exclude_neurons, name, dims_names)

    def get_spikes_inds(self, neurons=None, exclude_neurons=[]):
        """This method returns all events' indices for which there is a spike, i.e., the spike weight is not zero.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons the multimeter records from.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
           Returns:
            a numpy array with the output indices
        """
        spikes = self.events[self.spike_var]
        spikes_inds = spikes != 0.0
        senders = self.senders
        if neurons is not None:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender in flatten_list(neurons) for sender in senders])
        if len(exclude_neurons) > 0:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender not in flatten_list(exclude_neurons) for sender in senders])
        return np.where(spikes_inds)[0]

    def get_spikes_events(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method returns an event structure similar to a spike_detectors,
           i.e., where there are events only for spike times, not for continuous time.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                           the events of which should be excluded from the output. Default = [].
            Returns:
            the filtered dictionary of events
        """
        events = dict(self.events)
        inds = events[self.spike_var] != 0
        for var, val in events.items():
            events[var] = val[inds]
        events["weights"] = np.array(events[self.spike_var])
        del events[self.spike_var]
        return self.filter_events(events, None, neurons, times, exclude_neurons, exclude_times)

    def get_spikes_weights(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return spikes' weights
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             the spikes' times in an array
        """
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["weights"]

    def get_spikes_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return spikes' times
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             the spikes' times in an array
        """
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["times"]

    def get_spikes_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method will return spikes' senders neurons' indices,
            after selecting/excluding part of the detected spikes' events, depending on user inputs
            Arguments:
             neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                      Default = None, corresponds to all neurons found as senders of events.
             times: collection (list, tuple, array) of times the events of which should be included in the output.
                    Default = None, corresponds to all events' times.
             exclude_neurons: collection (list, tuple, array) of neurons
                              the events of which should be excluded from the output. Default = [].
             exclude_times: collection (list, tuple, array) of times
                            the events of which should be excluded from the output. Default = [].
            Returns:
             the spikes' senders neurons' indices in an array
        """
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["senders"]

    # The following attributes are time summaries without taking into consideration spike timing:

    def get_total_spikes_activity(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method returns the total spikes' activity by adding spike weights
           and dividing by the total number of neurons,
           after filtering spikes' events based on user inputs.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                           the events of which should be excluded from the output. Default = [].
           Returns:
            float: total spikes' activity
        """
        return np.sum(self.get_spikes_weights(neurons, times, exclude_neurons, exclude_times))

    def get_mean_spikes_activity(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        """This method returns the mean spike activity by adding spike weights
           and dividing by the total number of neurons,
           after filtering spikes' events based on user inputs.
           Arguments:
            neurons: collection (list, tuple, array) of neurons the events of which should be included in the output.
                     Default = None, corresponds to all neurons found as senders of events.
            times: collection (list, tuple, array) of times the events of which should be included in the output.
                   Default = None, corresponds to all events' times.
            exclude_neurons: collection (list, tuple, array) of neurons
                             the events of which should be excluded from the output. Default = [].
            exclude_times: collection (list, tuple, array) of times
                           the events of which should be excluded from the output. Default = [].
           Returns:
            float: mean spikes' activity
        """
        n_neurons = self.get_number_of_connections(neurons, exclude_neurons)
        if n_neurons > 0:
            return self.get_total_spikes_activity(neurons, times, exclude_neurons, exclude_times) / n_neurons
        else:
            return 0.0

    @property
    def spikes(self):
        """This method returns time series' data of spike weights recorded by the spike multimeter.
           Returns:
            a xarray DataArray with the output data
        """
        return self.get_spikes()

    @property
    def spikes_times(self):
        """This method will return spikes' times.
            Returns:
             the spikes' times in an array
        """
        return self.get_spikes_times()

    @property
    def spikes_senders(self):
        """This method will return spikes' senders neurons' indices,
            Returns:
             the pikes' senders neurons' indices in an array
        """
        return self.get_spikes_senders()

    @property
    def spikes_weights(self):
        """This method will return spikes' weights.
            Returns:
             the spikes' weights in an array
        """
        return self.get_spikes_weights()

    @property
    def number_of_spikes(self):
        """This method will return the total number of spikes with weight not equal to 0."""
        return np.sum(self.spikes != 0)

    @property
    def mean_number_of_spikes(self):
        """This method will return the total number of spikes,
         divided by the total number of neurons recorded by the device."""
        return self.get_mean_number_of_spikes()

    @property
    def spikes_rate(self):
        """This method will return the mean spike rate (float) as the
           total number of spikes divided by the total number of connected neurons, assuimng a duration of 1 ms
        """
        return self.get_spikes_rate()

    @property
    def total_spikes_activity(self):
        """This method returns the total spike activity by adding spike weights.
           Returns:
            float: total spikes' activity
        """
        return self.get_total_spikes_activity()

    @property
    def mean_spikes_activity(self):
        """This method returns the mean spike activity by adding spike weights
           and dividing by the total number of neurons.
           Returns:
            float: mean spikes' activity
        """
        return self.get_mean_spikes_activity()

    # The following attributes are computed across time:

    def compute_spikes_activity_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points=None,
                                            spikes_kernel=None, mode="per_neuron",
                                            name=None, rate_mode="activity",  **kwargs):
        """This method computes spike activity (i.e., weights) across time.
           Arguments:
            time: the time vector
            spikes_kernel_width: float: the width of the sliding window kernel in ms
            spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                           Default=None, and computed using the time vector
            spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
            mode: if "per_neuron" the output is returned for each neuron separetely.
                  Otherwise, it is computed across all neurons. Default = "per_neuron"
            name: name of the data to be computed. Default = None,
                  which defaults to device_model_name + " - Total spike rate across time"
           Returns:
            xarray.DataArray with spike rates' time series
        """
        if name is None:
            name = self.model + " - Total spike activity accross time"
        spikes = self.get_spikes(**kwargs).values
        if rate_mode == "rate":
            for i_spike, spike in spikes:
                spikes[i_spike] = np.heaviside(spike, 0.0)

        if spikes_kernel is None:
            spikes_kernel = np.ones((spikes_kernel_width_in_points, ))
            if rate_mode.find("rate") > -1:
                # For spike rate computation we have to normalize with the kernel width in time units
                spikes_kernel /= spikes_kernel_width
            else:
                # For "activity" computation we have to normalize with the kernel width in time steps
                spikes_kernel /= spikes_kernel_width_in_points

        if mode == "per_neuron":
            # Returning output per neuron
            activity = []
            for spike in spikes:
                activity.append(spikes_rate_convolution(spike, spikes_kernel))
            return xr.DataArray(np.array(activity), dims=["Neuron", "Time"],
                                coords={"Neuron": spikes.coords[spikes.dims[0]].item(),
                                        "Time": time})
        else:
            # Returning output as for all neurons together
            spikes = np.sum(spikes, axis=0).squeeze()
            activity = spikes_rate_convolution(spikes, spikes_kernel)

        return xr.DataArray(activity, dims=["Time"], coords={"Time": time}, name=name)

    def compute_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points=None,
                                        spikes_kernel=None, mode="per_neuron", name=None, **kwargs):
        """This method computes spike rate across time.
           Arguments:
            time: the time vector
            spikes_kernel_width: float: the width of the sliding window kernel in ms
            spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                           Default=None, and computed using the time vector
            spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
            name: name of the data to be computed. Default = None,
                  which defaults to device_model_name + " - Total spike rate across time"
           Returns:
            xarray.DataArray with spike rates' time series
        """
        return self.compute_spikes_activity_across_time(time, spikes_kernel_width, spikes_kernel_width_in_points=None,
                                                        spikes_kernel=spikes_kernel, mode=mode,
                                                        name=name, rate_mode="rate",  **kwargs)

    def compute_mean_spikes_activity_across_time(self, time, spike_kernel_width,
                                                 spikes_kernel=None, name=None, **kwargs):
        """This method computes spike activity (i.e., weights) across time, averaged across neurons.
           Arguments:
            time: the time vector
            spikes_kernel_width: float: the width of the sliding window kernel in ms
            spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                           Default=None, and computed using the time vector
            spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
            name: name of the data to be computed. Default = None,
                  which defaults to device_model_name + " - Total spike rate across time"
           Returns:
            xarray.DataArray with spike rates' time series
        """
        if name is None:
            name = self.model + " - Mean spike activity accross time"
        n_neurons = self.get_number_of_connections(**kwargs)
        if n_neurons > 0:
            return self.compute_spikes_activity_across_time(time, spike_kernel_width, spikes_kernel,
                                                            "total", name, **kwargs) / n_neurons
        else:
            return 0.0 * time


OutputDeviceDict = {"spike_detector": SpikeDetector,
                    "multimeter": Multimeter,
                    "spike_multimeter": SpikeMultimeter,
                    "voltmeter": Voltmeter}


OutputSpikeDeviceDict = {"spike_detector": SpikeDetector,
                         "spike_multimeter": SpikeMultimeter}


class DeviceSet(pd.Series):

    _number_of_connections = 0

    def __init__(self, name="", model="", device_set=None, **kwargs):
        super(DeviceSet, self).__init__(device_set, **kwargs)
        if np.any([not isinstance(device, Device) for device in self]):
            raise ValueError("Input device_set is not a Series of Device objects!:\n%s" %
                             str(device_set))
        self.name = str(name)
        self.model = str(model)
        self.update_model()
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def __getitem__(self, items):
        """This method will return a subset of the DeviceSet if the argument is a sequence,
        or a single Device if the argument is an integer indice or a string label."""
        if isinstance(items, string_types) or is_integer(items):
            return super(DeviceSet, self).__getitem__(items)
        return DeviceSet(name=self.name, model=self.model, device_set=super(DeviceSet, self).__getitem__(items))

    def _repr(self):
        return "%s - Name: %s, Model: %s" % \
               (self.__class__.__name__, self.name, self.model)

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self.print_str()

    def print_str(self, connectivity=False):
        output = "\n" + self._repr()
        output += ",\nDevices:"
        for node_index, node in self.iteritems():
            output += LINE + node.print_str(connectivity)
        return output

    def _input_nodes(self, nodes=None):
        """This method returns (a subset of) the DeviceSet nodes' labels in a list."""
        if nodes is None:
            # no input nodes
            return list(self.index)
        else:
            if nodes in list(self.index) or nodes in list(range(len(self))):
                # input nodes is a single index or label
                return [nodes]
            else:
                # input nodes is a sequence of indices or labels
                return list(nodes)

    def _return_by_type(self, values_dict, return_type="dict", concatenation_index_name="Region", name=None):
        """This method returns data collected from the Devices of the DeviceSet in a desired output format, among
           dict (Default), pandas.Series, xarray.DataArray or a list of values, depending on user input.
           Arguments:
            values_dict: dictionary of attributes and values
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "dict". "values" stands for a list of values, without labelling the output.
            concatenation_index_name: The dimension name along which the concatenation across devices happens.
                                      Default = "Region".
            name: Label of the output data. Default = None.
           Returns:
            output data in the selected type.
        """
        if return_type == "values":
            return list(values_dict.values())
        elif return_type == "dict":
            return values_dict
        elif return_type == "Series":
            return pd.Series(values_dict, name=name)
        elif return_type == "DataArray":
            if name is None:
                name = self.name
            for key, val in values_dict.items():
                if isinstance(val, xr.DataArray):
                    val.name = key
                else:
                    raise_value_error("DataArray concatenation not possible! "
                                      "Not all outputs are DataArrays!:\n %s" % str(values_dict))
            dims = list(values_dict.keys())
            values = list(values_dict.values())
            if len(values) == 0:
                return xr.DataArray([])
            output = xr.concat(values, dim=pd.Index(dims, name=concatenation_index_name))
            output.name = name
            return output
        else:
            return values_dict

    def do_for_all_devices(self, attr, *args, nodes=None, return_type="values",
                           concatenation_index_name="Region", name=None, **kwargs):
        """This method will perform the required action (Device method or property),
           and will return the output data collected from (a subset of) the Devices of the DeviceSet,
           in a desired output format, among dict (Default), pandas.Series, xarray.DataArray or a list of values,
           depending on user input.
           Arguments:
            attr: the name of the method/property of Device requested
            *args: possible position arguments of attr
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
            concatenation_index_name: The dimension name along which the concatenation across devices happens.
                                      Default = "Region".
            name: Label of the output data. Default = None.
            **kwargs: possible keyword arguments of attr
           Returns:
            output data in the selected type.
        """
        values_dict = OrderedDict()
        for node in self._input_nodes(nodes):
            val = getattr(self[node], attr)
            if hasattr(val, "__call__"):
                values_dict.update({node: val(*args, **kwargs)})
            else:
                values_dict.update({node: val})
        return self._return_by_type(values_dict, return_type, concatenation_index_name, name)

    @property
    def number_of_connections(self):
        """This method will return the total number of connections of each Device of the DeviceSet.
           Returns:
            a list of Devices' numbers of connections
        """
        self._number_of_connections = self.do_for_all_devices("number_of_connections")
        if len(self._number_of_connections) == 0:
            self._number_of_connections = 0
        return self._number_of_connections

    @property
    def times(self):
        """This method will return the event times recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' times
        """
        return self.do_for_all_devices("times", return_type="values")

    @property
    def time(self):
        """This method will return the time arrays of the events recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' time (numpy) arrays
        """
        return self.do_for_all_devices("time", return_type="values")

    @property
    def senders(self):
        """This method will return the event senders neurons' indices recorded from each Device of the DeviceSet.
           Returns:
            a list of Devices' events' senders neurons' indices
        """
        return self.do_for_all_devices("senders", return_type="values")

    @property
    def weights(self):
        """This method will return the weights of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the weights of the connections
        """
        return self.do_for_all_devices("node_weight", return_type="values")

    @property
    def delays(self):
        """This method will return the delays of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the delays of the connections
        """
        return self.do_for_all_devices("node_delay", return_type="values")

    @property
    def receptors(self):
        """This method will return the receptors of the connections of each Device of the DeviceSet.
           Returns:
            a list of (numpy) arrays of the receptors of the connections
        """
        return self.do_for_all_devices("node_receptors", return_type="values")

    def record_from_per_node(self, nodes=None, return_type="values"):
        """A method to get the attribute "record_from" from (a subset of) all Devices of the DeviceSet
           This method will lead to an error if the device set does not consist of OutputDevices.
           Arguments:
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
           Returns:
            the output in the selected type
        """
        values_dict = self.do_for_all_devices("record_from", nodes=nodes, return_type="dict")
        if return_type == "dict":
            return values_dict
        values = np.array(list(values_dict.values()))
        unique_values, indices = np.unique(values, return_inverse=True)
        if return_type == "values":
            return unique_values.squeeze().tolist()
        nodes = np.array(list(values_dict.keys()))
        output = OrderedDict()
        for i_output, val in enumerate(unique_values):
            key = np.where(indices == i_output)[0]
            output[tuple(nodes[key])] = val
        return output

    @property
    def record_from(self):
        """A method to get the attribute "record_from" from all Devices of the DeviceSet
           This method will lead to an error if the device set does not consist of OutputDevices.
           Returns:
            the outputs of each device in a list
        """
        return self.record_from_per_node()

    def update_model(self):
        """Assert that all Devices of the set are of the same model."""
        if len(self) > 0:
            models = self.do_for_all_devices("model")
            if np.any([model != self.model for model in models]):
                raise ValueError("Not all devices of the DeviceSet are of the same model!:\n %s" % str(models))

    def update(self, device_set=None):
        if device_set:
            super(DeviceSet, self).update(device_set)
        self.update_model()
        self._number_of_connections = self.number_of_connections

    def Get(self, attrs=None, nodes=None, return_type="dict", name=None):
        """A method to get attributes from (a subset of) all Devices of the DevoceSet.
            attr: the name of the method/property of Device requested
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
            return_type: string selecting one of the return types ["values", "dict", "Series", "DataArray"].
                         Default = "values". "values" stands for a list of values, without labelling the output.
            name: Label of the output data. Default = None.
        """
        if attrs is None:
            # Get dictionary of all attributes
            values_dict = []
            for node in self._input_nodes(nodes):
                values_dict.append(self[node].Get())
            values_dict = list_of_dicts_to_dict_of_lists(values_dict)
        else:
            values_dict = OrderedDict({})
            for attr in ensure_list(attrs):
                this_attr = []
                for node in self._input_nodes(nodes):
                    this_attr.append(self[node].Get(attr))
                values_dict.update({attr: this_attr})
        return self._return_by_type(values_dict, return_type, name)

    def Set(self, value_dict, nodes=None):
        """A method to set attributes to (a subset of) all Devices of the DevoceSet.
            value_dict: dict of attributes and values to be set
            nodes: a subselection of Device nodes of the DeviceSet the action should be performed upon
        """
        # A method to set attributes to all devices of the set
        # TODO: find a nice way to treat arrays or lists attributes.
        #  Some NEST device attributes have to be arrays and some scalars,
        #  but we don't know what is what beforehand...
        def get_scalar_dict1(din, index):
            dout = dict()
            for key, val in din.items():
                try:
                    dout[key] = val[index] * np.ones(1)
                except:
                    dout[key] = val
            return dout

        def get_scalar_dict2(din, index):
            dout = dict()
            for key, val in din.items():
                try:
                    dout[key] = val[index]
                except:
                    dout[key] = val
            return dout

        for i_n, node in enumerate(self._input_nodes(nodes)):
            try:
                # Good for spike times and weights of spike generator
                value_dict_i_n = get_scalar_dict1(value_dict, i_n)
                self[node].Set(value_dict_i_n)
            except:
                # Good for amplitude of dc generator and rate of poisson generator
                value_dict_i_n = get_scalar_dict2(value_dict, i_n)
                self[node].Set(value_dict_i_n)
