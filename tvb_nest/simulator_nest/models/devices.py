# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils \
    import ensure_list, flatten_list, sort_events_by_x_and_y, data_frame_from_continuous_events
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict, OrderedDict


LOG = initialize_logger(__name__)


# Classes for creating:
# - output devices that can measure and summarize the activity of a whole neuronal population
# - input devices that induce some activity by stimulation to a whole neuronal population.
# For the moment, these classes wrap around NEST commands.
# Therefore output devices assume that data is structured in discrete events tagged by time and sender neuron.


class NESTDevice(object):
    __metaclass__ = ABCMeta

    nest_instance = None
    device = None
    number_of_connections = 0
    model = "device"

    def __init__(self, nest_instance, device):
        self.nest_instance = nest_instance
        self.model = "device"
        try:
            self.nest_instance.GetStatus(device)[0]["element_type"]
            self.device = device
        except:
            raise ValueError("Failed to GetStatus of device %s!" % str(device))
        self.update_number_of_connections()

    @property
    def nest_model(self):
        return self.nest_instance.GetStatus(self.device)[0]["model"].split("<SLILiteral: ")[1].split(">")[0]

    @abstractmethod
    def connections(self):
        pass

    @abstractmethod
    def neurons(self):
        pass

    @property
    def number_of_neurons(self):
        return self.get_number_of_neurons()

    def update_number_of_connections(self):
        neurons = self.neurons
        for neuron in neurons:
            element_type = self.nest_instance.GetStatus((neuron,))[0]["element_type"]
            if element_type != "neuron":
                raise_value_error("Node %d is not a neuron but a %s!" % (neuron, element_type))
        self.number_of_connections = len(neurons)

    def SetStatus(self, values_dict):
        self.nest_instance.SetStatus(self.device, values_dict)


class NESTInputDevice(NESTDevice):
    model = "input_device"

    def __init__(self, nest_instance, device):
        super(NESTInputDevice, self).__init__(nest_instance, device)
        self.model = "input_device"
        
    def __getattr__(self, attr):
        try:
            self.nest_instance.GetStatus(self.device, attr)[0]
        except:
            raise ValueError("Failed to get %s from device %s" % (str(attr), str(self.device)))

    @property
    def connections(self):
        return self.nest_instance.GetConnections(target=self.device)

    @property
    def neurons(self):
        return tuple([conn[0] for conn in self.connections])


class NESTPoissonGenerator(NESTInputDevice):
    model = "poisson_generator"

    def __init__(self, nest_instance, device):
        super(NESTPoissonGenerator, self).__init__(nest_instance, device)
        self.model = "poisson_generator"


class NESTSinusoidalPoissonGenerator(NESTInputDevice):
    model = "sinusoidal_poisson_generator"

    def __init__(self, nest_instance, device):
        super(NESTSinusoidalPoissonGenerator, self).__init__(nest_instance, device)
        self.model = "sinusoidal_poisson_generator"


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):
    model = "inhomogeneous_poisson_generator"

    def __init__(self, nest_instance, device):
        super(NESTInhomogeneousPoissonGenerator, self).__init__(nest_instance, device)
        self.model = "inhomogeneous_poisson_generator"


class NESTMIPGenerator(NESTInputDevice):
    model = "mip_generator"

    def __init__(self, nest_instance, device):
        super(NESTMIPGenerator, self).__init__(nest_instance, device)
        self.model = "mip_generator"


class NESTDGammaSupGenerator(NESTInputDevice):
    model = "gamma_sup_generator"

    def __init__(self, nest_instance, device):
        super(NESTDGammaSupGenerator, self).__init__(nest_instance, device)
        self.model = "gamma_sup_generator"


class NESTDPPDSupGenerator(NESTInputDevice):
    model = "ppd_sup_generator"

    def __init__(self, nest_instance, device):
        super(NESTDPPDSupGenerator, self).__init__(nest_instance, device)
        self.model = "ppd_sup_generator"


class NESTSpikeGenerator(NESTInputDevice):
    model = "spike_generator"

    def __init__(self, nest_instance, device):
        super(NESTSpikeGenerator, self).__init__(nest_instance, device)
        self.model = "spike_generator"


class NESTPulsePacketGenerator(NESTInputDevice):
    model = "pulse_packet_generator"

    def __init__(self, nest_instance, device):
        super(NESTPulsePacketGenerator, self).__init__(nest_instance, device)
        self.model = "pulse_packet_generator"


class NESTDCGenerator(NESTInputDevice):
    model = "dc_generator"

    def __init__(self, nest_instance, device):
        super(NESTDCGenerator, self).__init__(nest_instance, device)
        self.model = "dc_generator"


class NESTStepCurrentGenerator(NESTInputDevice):
    model = "step_current_generator"

    def __init__(self, nest_instance, device):
        super(NESTStepCurrentGenerator, self).__init__(nest_instance, device)
        self.model = "step_current_generator"


class NESTACGenerator(NESTInputDevice):
    model = "ac_generator"

    def __init__(self, nest_instance, device):
        super(NESTACGenerator, self).__init__(nest_instance, device)
        self.model = "ac_generator"


class NESTStepRateGenerator(NESTInputDevice):
    model = "step_rate_generator"

    def __init__(self, nest_instance, device):
        super(NESTStepRateGenerator, self).__init__(nest_instance, device)
        self.model = "step_rate_generator"


class NESTNoiseGenerator(NESTInputDevice):
    model = "noise_generator"

    def __init__(self, nest_instance, device):
        super(NESTNoiseGenerator, self).__init__(nest_instance, device)
        self.model = "noise_generator"


NESTInputDeviceDict = {"poisson_generator": NESTPoissonGenerator,
                       "sinusoidal_poisson_generator": NESTSinusoidalPoissonGenerator,
                       "inhomogeneous_poisson_generator": NESTInhomogeneousPoissonGenerator,
                       "mip_generator": NESTMIPGenerator,
                       "gamma_sup_generator": NESTDGammaSupGenerator,
                       "ppd_sup_generator": NESTDPPDSupGenerator,
                       "spike_generator": NESTSpikeGenerator,
                       "pulse_packet_generator": NESTPulsePacketGenerator,
                       "dc_generator": NESTDCGenerator,
                       "step_current_generator": NESTStepCurrentGenerator,
                       "ac_generator": NESTACGenerator,
                       "step_rate_generator": NESTStepRateGenerator,
                       "noise_generator": NESTNoiseGenerator
                       }


class NESTOutputDevice(NESTDevice):
    model = "output_device"
    
    def __init__(self, nest_instance, device):
        super(NESTOutputDevice, self).__init__(nest_instance, device)
        self.model = "output_device"

    def filter_events(self, vars=None, events=None, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        if events is None:
            events = self.events
        spike_times = np.array(events["times"])
        senders = np.array(events["senders"])
        inds = np.ones((self.n_events,))
        if len(inds) > 0:
            if times is not None:
                if senders is not None:
                    inds = np.logical_and(inds, [time in flatten_list(times) and
                                                 time not in flatten_list(exclude_times) and
                                                 sender in flatten_list(neurons) and
                                                 sender not in flatten_list(exclude_neurons)
                                                 for time, sender in zip(spike_times, senders)])
                else:
                    inds = np.logical_and(inds, [time in flatten_list(times) and
                                                 time not in flatten_list(exclude_times) and
                                                 sender not in flatten_list(exclude_neurons)
                                                 for time, sender in zip(spike_times, senders)])
            else:
                if neurons is not None:
                    inds = np.logical_and(inds, [time not in flatten_list(exclude_times) and
                                                 sender in flatten_list(neurons) and
                                                 sender not in flatten_list(exclude_neurons)
                                                 for time, sender in zip(spike_times, senders)])
                else:
                    inds = np.logical_and(inds, [time not in flatten_list(exclude_times) and
                                                 sender not in flatten_list(exclude_neurons)
                                                 for time, sender in zip(spike_times, senders)])
        if vars is None:
            vars = events.keys()
        output_events = OrderedDict()
        for var in ensure_list(vars):
            output_events[var] = events[var][inds]
        return output_events

    @property
    def connections(self):
        return self.nest_instance.GetConnections(source=self.device)

    @property
    def neurons(self):
        return tuple([conn[1] for conn in self.connections])

    @property
    def events(self):
        return self.nest_instance.GetStatus(self.device)[0]['events']

    @property
    def senders(self):
        return self.events['senders']

    @property
    def times(self):
        return self.events['times']

    @property
    def number_of_events(self):
        return self.nest_instance.GetStatus(self.device, 'n_events')[0]

    @property
    def n_events(self):
        return self.number_of_events

    def __getattr__(self, attr):
        try:
            self.events[attr]
        except:
            raise ValueError("Failed to get %s from device %s" % (str(attr), str(self.device)))

    @property
    def reset(self):
        self.nest_instance.SetStatus(self.device, {'n_events': 0})


class NESTSpikeDetector(NESTOutputDevice):
    model = "spike_detector"

    def __init__(self, nest_instance, device):
        super(NESTSpikeDetector, self).__init__(nest_instance, device)
        self.model = "spike_detector"

    @property
    def connections(self):
        return self.nest_instance.GetConnections(target=self.device)

    @property
    def neurons(self):
        return tuple([conn[0] for conn in self.connections])

    def get_spike_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.filter_events("times", None, neurons, times, exclude_neurons, exclude_times)

    def get_spike_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.filter_events("senders", None, neurons, times, exclude_neurons, exclude_times)

    def get_spikes_times_by_neurons(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return sort_events_by_x_and_y(self.events, x="senders", y="times",
                                      filter_x=neurons, filter_y=times,
                                      exclude_x=exclude_neurons, exclude_y=exclude_times)

    def get_spikes_neurons_by_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return sort_events_by_x_and_y(self.events, x="times", y="senders",
                                      filter_x=times, filter_y=neurons,
                                      exclude_x=exclude_times, exclude_y=exclude_neurons)

    def get_mean_spikes_rate(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return len(self.get_spike_times(neurons, times, exclude_neurons, exclude_times)) / \
               self.get_number_of_neurons(neurons, exclude_neurons)

    @property
    def spikes_times(self):
        return self.times

    @property
    def spikes_senders(self):
        return self.senders

    @property
    def number_of_spikes(self):
        return self.number_of_events

    @property
    def mean_spikes_rate(self):
        return self.get_mean_spikes_rate()

    def compute_spike_rate(self, time, spike_counts_kernel_width, spike_rate_fun=None):

        def spike_rate(spikes_times, time, spike_counts_kernel_width):
            spikes_times = np.array(spikes_times)
            spike_counts_kernel_width2 = spike_counts_kernel_width / 2
            spike_counts = []
            for t in time:
                spike_counts.append(np.sum(np.logical_and(spikes_times >= (t - spike_counts_kernel_width2),
                                                          spikes_times < (t + spike_counts_kernel_width2))))
            return np.array(spike_counts) / spike_counts_kernel_width * 1000  # in spikes/sec

        if spike_rate_fun is None:
            spike_rate_fun = spike_rate

        # Compute rate
        return spike_rate_fun(self.spikes_times, time, spike_counts_kernel_width)

    def compute_mean_spike_rate(self, time, spike_counts_kernel_width, spike_rate_fun=None):
        return self.compute_spike_rate(time, spike_counts_kernel_width, spike_rate_fun) / self.number_of_neurons


class NESTMultimeter(NESTOutputDevice):
    model = "multimeter"

    def __init__(self, nest_instance, device):
        super(NESTMultimeter, self).__init__(nest_instance, device)
        self.model = "multimeter"

    @property
    def record_from(self):
        return self.nest_instance.GetStatus(self.device)[0]['record_from']

    def get_data(self, variables=[], neurons=None, exclude_neurons=[]):
        variables = ensure_list(variables)
        if len(variables) == 0:
            variables = self.record_from
        else:
            for variable in variables:
                assert variable in self.record_from
        events = dict(self.events)
        times = events.pop("times")
        senders = events.pop("senders")
        data, times, neurons = \
            data_frame_from_continuous_events(events, times, senders, variables=variables,
                                              filter_senders=neurons, exclude_senders=exclude_neurons)
        return data, times, neurons

    def get_mean_data(self, variables=[], neurons=None, exclude_neurons=[]):
        data, times, neurons = self.get_data(variables, neurons, exclude_neurons)
        n_neurons = len(neurons)
        for var, values in data.items():
            data[var] = values / n_neurons
        return data, times, neurons

    @property
    def data(self):
        return self.get_data()[0]

    @property
    def data_mean(self):
        return self.get_mean_data()[0]

    def current_data(self, variables=[], neurons=None, exclude_neurons=[]):
        output = OrderedDict()
        variables = ensure_list(variables)
        if len(variables) == 0:
            variables = self.record_from
        else:
            for variable in variables:
                assert variable in self.record_from
        events = self.events
        times = events["times"]
        if len(times) > 0:
            output_inds = events["times"] == events["times"][-1]
            senders = events["senders"]
            if neurons is not None:
                output_inds = np.logical_and(output_inds,
                                             [sender in flatten_list(neurons) for sender in senders])
            if len(exclude_neurons) > 0:
                output_inds = np.logical_and(output_inds,
                                             [sender not in flatten_list(exclude_neurons) for sender in senders])

            for var in variables:
                output[var] = events[var][output_inds]
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
            for var in variables:
                output[var] = np.zeros((len(neurons), ))
        return output

    def current_data_mean(self, variables=[], neurons=None, exclude_neurons=[]):
        output = self.current_data(variables, neurons, exclude_neurons)
        for var in output.keys():
            output[var] = np.mean(output[var])
        return output

    def current_data_mean_values(self, variables=[], neurons=None, exclude_neurons=[]):
        return list(self.current_data_mean(variables, neurons, exclude_neurons).values())


class NESTVoltmeter(NESTMultimeter):
    model = "voltmeter"

    def __init__(self, nest_instance, device):
        super(NESTVoltmeter, self).__init__(nest_instance, device)
        self.model = "voltmeter"
        assert "V_m" in self.record_from

    def get_V_m(self, neurons=None, exclude_neurons=[]):
        return self.get_data("V_m", neurons, exclude_neurons)[0]["V_m"]

    @property
    def V_m(self):
        return self.get_V_m()


class NESTSpikeMultimeter(NESTMultimeter):
    model = "spike_multimeter"
    spike_var = "spikes"

    def __init__(self, nest_instance, device):
        super(NESTSpikeMultimeter, self).__init__(nest_instance, device)
        self.model = "spike_multimeter"

    def get_spikes(self, neurons=None, exclude_neurons=[]):
        return self.get_data(self.spike_var, neurons, exclude_neurons)[0][self.spike_var]

    def get_spikes_inds(self, neurons=None, exclude_neurons=[]):
        spikes = self.events[self.spike_var]
        spikes_inds = spikes != 0.0
        senders = self.senders
        if neurons is not None:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender in flatten_list(neurons) for sender in senders])
        if len(exclude_neurons) > 0:
            spikes_inds = np.logical_and(spikes_inds,
                                         [sender not in flatten_list(exclude_neurons) for sender in senders])
        return np.where(spikes_inds)

    def get_spikes_events(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        events = dict(self.events)
        inds = events[self.spike_var] != 0
        for var, val in events.items():
            events[var] = val[inds]
        events["weights"] = np.array(events[self.spike_var])
        del events[self.spike_var]
        return self.filter_events(None, events, neurons, times, exclude_neurons, exclude_times)

    def get_spikes_weights(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["weights"]

    def get_spikes_times(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["times"]

    def get_spikes_senders(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return self.get_spikes_events(self, neurons, times, exclude_neurons, exclude_times)["senders"]

    def get_mean_spikes_rate(self, neurons=None, times=None, exclude_neurons=[], exclude_times=[]):
        return len(self.get_spikes_times(neurons, times, exclude_neurons, exclude_times)) / \
               self.get_number_of_neurons(neurons, exclude_neurons)

    def get_mean_spikes_activity(self, neurons=None, exclude_neurons=[]):
        return np.sum(self.get_spikes_weights(neurons, exclude_neurons)) / \
               self.get_number_of_neurons(neurons, exclude_neurons)

    def get_total_spikes_activity(self, neurons=None, exclude_neurons=[]):
        return np.sum(self.get_spikes_weights(neurons, exclude_neurons))

    @property
    def spikes(self):
        return self.get_spikes()

    @property
    def spikes_times(self):
        return self.get_spikes_times()

    @property
    def spikes_senders(self):
        return self.get_spikes_senders()

    @property
    def spikes_weights(self):
        return self.get_spikes_weights()

    @property
    def number_of_spikes(self):
        return np.sum(self.spikes > 0)

    @property
    def mean_spikes_rate(self):
        return self.get_mean_spikes_rate()

    @property
    def mean_spikes_activity(self):
        return self.get_mean_spikes_activity()

    @property
    def total_spikes_activity(self):
        return self.get_total_spikes_activity()


NESTOutputDeviceDict = {"spike_detector": NESTSpikeDetector,
                        "multimeter": NESTMultimeter,
                        "spike_multimeter": NESTSpikeMultimeter,
                        "voltmeter": NESTVoltmeter}


NESTOutputSpikeDeviceDict = {"spike_detector": NESTSpikeDetector,
                             "spike_multimeter": NESTSpikeMultimeter}


class NESTDeviceSet(IndexedOrderedDict):

    def __init__(self, name="", model="", device_set=OrderedDict({})):
        self.name = str(name)
        self.model = str(model)
        if not (isinstance(device_set, dict) and
                np.all([isinstance(device, NESTDevice)
                        for device in device_set.values()])):
            raise ValueError("Input device_set is not a IndexedOrderedDict of NESTDevice objects!:\n%s" %
                             str(device_set))
        super(NESTDeviceSet, self).__init__(device_set)
        self.update_model()
        LOG.info("%s of model %s for %s created!" % (self.__class__, self.model, self.name))

    def update_model(self):
        if len(self._dict) > 0:
            self.model = list(self._dict.values())[0].model
            if np.any([self.model != device.model
                       for device in self._dict.values()]):
                raise ValueError("Not all devices of the device_set are of the same model!:\n %s"
                                 % str([device.model for device in self._dict.values()]))

    def update(self, *args, **kwargs):
        super(NESTDeviceSet, self).update(*args, **kwargs)
        self.update_model()

    def SetStatus(self, value_dict, nodes=None):
        # TODO: find a nice way to treat arrays or lists attributes.
        #  Some NEST device parameters have to be arrays and some scalars,
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

        if nodes is None or len(nodes) == 0:
            nodes = self._dict.keys()
        nodes = ensure_list(nodes)
        for i_n, node in enumerate(nodes):
            try:
                # Good for rate, spike times and weights of spike generator
                value_dict_i_n = get_scalar_dict1(value_dict, i_n)
                self._dict[node].SetStatus(value_dict_i_n)
            except:
                # Good for amplitude of dc generator and poisson generator
                try:
                    value_dict_i_n = get_scalar_dict2(value_dict, i_n)
                    self._dict[node].SetStatus(value_dict_i_n)
                except:
                    print("This is the problem I was talking about!")

    def GetStatus(self, attrs, nodes=None, return_values=False):
        if nodes is None or len(nodes) == 0:
            nodes = self._dict.keys()
        vals = OrderedDict({})
        for attr in ensure_list(attrs):
            this_attr = []
            for node in ensure_list(nodes):
                this_attr.append(getattr(self._dict[node], attr))
            vals.update({attr: this_attr})
        if return_values:
            vals = vals.values()
            if len(vals) == 1:
                return vals[0]
            else:
                return vals
        else:
            return vals

    def __getattr__(self, attr):
        values = []
        for node in self._dict.values():
            val = getattr(node, attr)
            if hasattr(val, "__call__"):
                values.append(val())
            else:
                values.append(val)
        return values
