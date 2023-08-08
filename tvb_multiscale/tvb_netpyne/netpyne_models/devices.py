import numpy as np
from tvb_multiscale.core.spiking_models.devices import InputDevice, SpikeRecorder, Multimeter
from tvb.basic.neotraits.api import HasTraits, Attr, Int, List
from tvb_multiscale.tvb_netpyne.netpyne.utils import generateSpikesForPopulation
from xarray import DataArray
from copy import copy, deepcopy

class NetpyneDevice(HasTraits):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        self.netpyne_instance = netpyne_instance
        HasTraits.__init__(self)
        self.params = kwargs.get('params')

    @property
    def spiking_simulator_module(self):
        return self.netpyne_instance

    def _assert_device(self):
        """Method to assert that the node of the network is a device"""
        pass

    def _assert_spiking_simulator(self):
        if self.netpyne_instance is None:
            raise ValueError("No NetPyNE instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    @property
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of nodes's elements"""
        pass

    def _assert_nodes(self, nodes=None):
        """Method to assert that the node of the network is valid"""
        pass

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the device
            Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects. Default = None, corresponding to all device's connections
        """
        pass

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the device
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all device's attributes
            connections: connections' objects. Default = None, corresponding to all device's connections
           Returns:
            Dictionary of sequences (tuples, lists, arrays) of connections' attributes.
        """
        pass

    def _Set(self, values_dict, nodes=None):
        """Method to set attributes of the SpikingNodeCollection's nodes.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
        """
        pass

    def _Get(self, attr=None, nodes=None):
        """Method to get attributes of the SpikingNodeCollection's nodes.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes' attributes.
        """
        pass

    def _GetConnections(self, nodes=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingNodeCollection node.
           Arguments:
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to nodes
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        # So far, tvb implementation is interested only in number of connections.
        # Thus it's equivalent to number of neurons this device is connected to/from.
        return self.neurons

    def get_neurons(self):
        """Method to get the indices of all the neurons the device is connected to/from."""
        return self.neurons

class NetpyneInputDevice(NetpyneDevice, InputDevice):

    """NetpyneInputDevice class to wrap around a NetPyNE input (stimulating) device"""

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "netpyne_input_device") #TODO: wrong model
        NetpyneDevice.__init__(self, device, netpyne_instance, *args, **kwargs)
        super(NetpyneInputDevice, self).__init__(device, netpyne_instance, *args, **kwargs)

    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected to."""
        return self.netpyne_instance.neuronsConnectedWith(self.label)
    
    @property
    def spiking_simulator_module(self):
        return self.device

    _own_neurons = None
    @property
    def own_neurons(self):
        """Method to get gids of artificial spike generating neurons that constitute this device as a proxy node"""
        if self._own_neurons is None:
            self._own_neurons = self.netpyne_instance.cellGidsForPop(self.label)
        return self._own_neurons

class NetpynePoissonGenerator(NetpyneInputDevice):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        kwargs["model"] = kwargs.pop("model", "poisson_generator")
        super(NetpynePoissonGenerator, self).__init__(device, netpyne_instance, *args, **kwargs)
        self.spikesPerNeuron = {}

    def _Set(self, values_dict, nodes=None):

        rates = values_dict["rate_values"]
        dts = values_dict["rate_times"]

        spikesPerNeuronIndex = generateSpikesForPopulation(len(self.own_neurons), rates, dts)
        
        # append to already collected ones
        for index, spikes in spikesPerNeuronIndex.items():
            gid = self.own_neurons[index]
            if gid not in self.spikesPerNeuron:
                self.spikesPerNeuron[gid] = []
            self.spikesPerNeuron[gid].extend(spikes)

NetpyneSpikeInputDeviceDict = {"poisson_generator": NetpynePoissonGenerator}

NetpyneInputDeviceDict = {}
NetpyneInputDeviceDict.update(NetpyneSpikeInputDeviceDict)

# Output devices

class NetpyneOutputDevice(NetpyneDevice):

    def __init__(self, device, netpyne_instance, *args, **kwargs):
        NetpyneDevice.__init__(self, device, netpyne_instance, *args, **kwargs)
    
    @property
    def neurons(self):
        """Method to get the indices of all the neurons the device is connected from."""
        return self.netpyne_instance.cellGidsForPop(self.population_label)

class NetpyneSpikeRecorder(NetpyneOutputDevice, SpikeRecorder):

    latestRecordTime = 0.0

    def __init__(self, device=None, netpyne_instance=None, **kwargs):
        # kwargs["model"] = kwargs.get("model", "spike_recorder")
        NetpyneOutputDevice.__init__(self, device, netpyne_instance, **kwargs)
        SpikeRecorder.__init__(self, device, **kwargs)
    
    @property
    def events(self):
        spktimes, spkgids = self.netpyne_instance.getSpikes(generatedBy=self.neurons)
        return {'senders': spkgids, 'times': spktimes}

    @property
    def number_of_events(self):
        return self.device.numberOfSpikes(self.population_label)

    def reset(self):
        pass


    def get_new_events(self, variables=None, **filter_kwargs):
        spktimes, spkgids = self.netpyne_instance.getSpikes(generatedBy=self.neurons, startingFrom=self.latestRecordTime)

        numSpikes = len(spktimes)
        if numSpikes > 0:
            period = self.netpyne_instance.time - self.latestRecordTime
            rate = 1000 * numSpikes / len(self.neurons) / period
            print(f"Netpyne:: recorded {len(spktimes)} spikes from {self.population_label}. Approx. rate: {rate}. Timeframe {self.latestRecordTime} + {period}")

        self.latestRecordTime = self.netpyne_instance.time

        return {'senders': spkgids, 'times': spktimes}

    @property
    def new_events(self):
        return self.get_new_events()

    @property
    def number_of_new_events(self):
        """This method returns the number (integer) of events"""
        return self.device.numberOfSpikes(self.population_label)

    @property
    def spiking_simulator_module(self):
        return self.device.netpyne_instance


class NetpyneMultimeter(NetpyneOutputDevice, Multimeter):

    _output_events_index = 0

    def _variables(self):
        return self.params.get('variables', {})

    def _events(self, onlyNew):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        result = {}
        if onlyNew:
            timeSlice = slice(self._output_events_index, None)
        else:
            timeSlice = slice(None)

        time = self.netpyne_instance.getRecordedTime()[timeSlice]

        for var in self._variables():
            varData = self.netpyne_instance.getTraces(var, self.neurons, timeSlice)
            # turning data of shape (n_neurons, time) to flat array [time_0_neur_0, ..., time_0_neur_n, ..., time_n_neur_0, ..., time_n_neur_n]
            result[var] = varData.T.flatten()
        # repeat time and neurons to match the format as in comment above
        result['times'] = time.repeat(self.number_of_neurons)
        result['senders'] = np.tile(self.neurons, len(time))

        if not onlyNew:
            self._output_events_index = len(time)

        return result


    def get_data(self, variables=None, events_inds=None,
                 name=None, dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True, new=False):
        """This method returns time series' data recorded by the multimeter.
           Arguments:
            variables: a sequence of variables' names (strings) to be selected.
                       Default = None, corresponds to all variables the multimeter records from.
            name: label of output. Default = None, which defaults to the label of the Device
            dims_names: sequence of dimensions' labels (strings) for the output array.
                        Default = ["Time", "Variable", "Neuron"]
           Returns:
            a xarray DataArray with the output data
        """
        if not variables:
            variables = self.record_from # variable names

        time = self.netpyne_instance.getRecordedTime()

        # shape (vars, neurs, time)
        data = np.zeros((len(variables), len(self.neurons), len(time)))

        for varInd, var in enumerate(variables):
            data[varInd] = self.netpyne_instance.getTraces(var, self.neurons)

        # reshape to (time, vars, neurs)
        data = data.transpose(2, 0, 1)

        m_data = DataArray(
            data,
            dims=["Time", "Variable", "Neuron"],
            coords={"Time": time,
                    "Variable": variables,
                    "Neuron": copy(self.neurons)},
            name=self.label)

        return m_data

    @property
    def events(self):
        return self._events(newOnly=False)
    
    def get_new_events(self, variables=None, name=None,
                      dims_names=["Time", "Variable", "Neuron"], flatten_neurons_inds=True):
        """Method to convert and place continuous time data measured from Monitors, to an events dictionary."""
        return self._events(newOnly=True)

    @property
    def number_of_events(self):
        # time * neurons
        return len(self.netpyne_instance.getRecordedTime()) * len(self.neurons)

    @property
    def number_of_new_events(self):
        # (time - oldEventsNum) * neurons
        return (len(self.netpyne_instance.getRecordedTime()) - self._output_events_index) * len(self.neurons)

    @property
    def record_from(self):
        return list(self._variables().keys())

    def reset(self):
        self._output_events_index = 0


NetpyneOutputSpikeDeviceDict = {"spike_recorder": NetpyneSpikeRecorder}
NetpyneOutputContinuousTimeDeviceDict = {"multimeter": NetpyneMultimeter}


NetpyneOutputDeviceDict = {}
NetpyneOutputDeviceDict.update(NetpyneOutputSpikeDeviceDict)
NetpyneOutputDeviceDict.update(NetpyneOutputContinuousTimeDeviceDict)
