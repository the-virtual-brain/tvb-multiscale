# -*- coding: utf-8 -*-

import numpy as np
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import flatten_tuple, ensure_list
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict, OrderedDict

LOG = initialize_logger(__name__)


class NESTDevice(object):
    nest_instance = None
    device = None
    number_of_connections = 0

    def __init__(self, nest_instance, device, device_model):
        self.nest_instance = nest_instance
        self.__model = device_model
        try:
            self.nest_instance.GetStatus(device, "element_type")
            self.device = device
        except:
            raise ValueError("Failed to GetStatus of device %s!" % str(device))
        self.update_number_of_connections()

    @property
    def model(self):
        return self.__model

    @property
    def connections(self):
        return self.nest_instance.GetConnections(source=self.device)

    @property
    def neurons(self):
        return tuple([conn[0] for conn in self.connections])

    @property
    def number_of_neurons(self):
        return self.number_of_connections

    def update_number_of_connections(self):
        try:
            # Confirm that connections are valid
            neurons = self.neurons
            assert np.all([self.nest_instance.GetStatus([neuron],
                                                        "element_type")[0] == "neuron"
                           for neuron in neurons])
        except:
            raise ValueError("Failed to GetStatus of devices' %s\n connections %s!"
                             % (str(self.device), str(self.connections)))
        self.number_of_connections = len(neurons)

    def SetStatus(self, values_dict):
        self.nest_instance.SetStatus(self.device, values_dict)


class NESTInputDevice(NESTDevice):

    def __getattr__(self, attr):
        try:
            self.nest_instance.GetStatus(self.device, attr)[0]
        except:
            raise ValueError("Failed to get %s from device %s" % (str(attr), str(self.device)))

    @property
    def connections(self):
        return self.nest_instance.GetConnections(target=self.device)


class NESTPoissonGenerator(NESTInputDevice):
    __model = "poisson_generator"


class NESTSinusoidalPoissonGenerator(NESTInputDevice):
    __model = "sinusoidal_poisson_generator"


class NESTInhomogeneousPoissonGenerator(NESTInputDevice):
    __model = "inhomogeneous_poisson_generator"


class NESTMIPGenerator(NESTInputDevice):
    __model = "mip_generator"


class NESTDGammaSupGenerator(NESTInputDevice):
    __model = "gamma_sup_generator"


class NESTDPPDSupGenerator(NESTInputDevice):
    __model = "ppd_sup_generator"


class NESTSpikeGenerator(NESTInputDevice):
    __model = "spike_generator"


class NESTPulsePacketGenerator(NESTInputDevice):
    __model = "pulse_packet_generator"


class NESTDCGenerator(NESTInputDevice):
    __model = "dc_generator"


class NESTStepCurrentGenerator(NESTInputDevice):
    __model = "step_current_generator"


class NESTACGenerator(NESTInputDevice):
    __model = "ac_generator"


class NESTStepRateGenerator(NESTInputDevice):
    __model = "step_rate_generator"


class NESTNoiseGenerator(NESTInputDevice):
    __model = "noise_generator"


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

    def __init__(self, nest_instance, device, device_model):
        super(NESTOutputDevice, self).__init__(nest_instance, device, device_model)

    def get_times(self, neurons=[]):
        times = self.nest_instance.GetStatus(self.device)[0]['events']['times'].tolist()
        if neurons is not None:
            neurons = flatten_tuple(neurons)
            senders = self.nest_instance.GetStatus(self.device)[0]['events']['senders'].tolist()
            new_times = []
            for time, sender in zip(times, senders):
                if sender in neurons:
                    new_times.append(time)
            times = new_times
        return times

    def get_senders(self, neurons=None, sort=False):
        senders = self.nest_instance.GetStatus(self.device)[0]['events']['senders'].tolist()
        if neurons is not None:
            neurons = flatten_tuple(neurons)
            new_senders = []
            for sender in senders:
                if sender in neurons:
                    new_senders.append(sender)
            senders = new_senders
        if sort:
            senders.sort()
        return senders

    def get_number_of_events(self, neurons=None):
        if neurons is None:
            return self.number_of_events
        else:
            return len(self.get_senders(neurons))

    @property
    def senders(self):
        return self.get_senders(None)

    @property
    def times(self):
        return self.get_times(None)

    @property
    def number_of_events(self):
        return self.nest_instance.GetStatus(self.device, 'n_events')[0]

    def __getattr__(self, attr):
        try:
            self.nest_instance.GetStatus(self.device)[0]['events'][attr].tolist()
        except:
            raise ValueError("Failed to get %s from device %s" % (str(attr), str(self.device)))

    def reset(self):
        self.nest_instance.SetStatus(self.device, {'n_events': 0})


class NESTSpikeDetector(NESTOutputDevice):
    __model = "spike_detector"

    @property
    def connections(self):
        return self.nest_instance.GetConnections(target=self.device)

    def get_spike_times(self, neurons=[]):
        return self.get_times(neurons)

    def get_number_of_spikes(self, neurons=[]):
        return self.get_number_of_events(neurons)

    @property
    def spike_times(self):
        return self.times

    @property
    def spike_senders(self):
        return self.senders

    @property
    def number_of_spikes(self):
        return self.number_of_events

    def compute_spike_rate(self, time, spike_counts_kernel_width, spike_rate_fun=None):

        def spike_rate(spike_times, time, spike_counts_kernel_width):
            spike_times = np.array(spike_times)
            spike_counts_kernel_width2 = spike_counts_kernel_width / 2
            spike_counts = []
            for t in time:
                spike_counts.append(np.sum(np.logical_and(spike_times >= (t - spike_counts_kernel_width2),
                                                          spike_times < (t + spike_counts_kernel_width2))))
            return np.array(spike_counts) / spike_counts_kernel_width * 1000  # in spikes/sec

        if spike_rate_fun is None:
            spike_rate_fun = spike_rate

        # Compute rate
        return spike_rate_fun(self.spike_times, time, spike_counts_kernel_width)

    def compute_mean_spike_rate(self, time, spike_counts_kernel_width, spike_rate_fun=None):
        return self.compute_spike_rate(time, spike_counts_kernel_width, spike_rate_fun) / self.number_of_neurons


class NESTMultimeter(NESTOutputDevice):
    __model = "multimeter"

    @property
    def record_from(self):
        return self.nest_instance.GetStatus(self.device, "record_from")


class NESTVoltmeter(NESTMultimeter):
    __model = "voltmeter"

    def __init__(self, nest_instance, device, device_model, neurons=()):
        super(NESTOutputDevice, self).__init__(nest_instance, device, device_model, neurons)
        assert ("V_m",) in self.record_from


NESTOutputDeviceDict = {"spike_detector": NESTSpikeDetector,
                        "multimeter": NESTMultimeter,
                        "voltmeter": NESTVoltmeter}


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
            self.model = self._dict.values()[0].model
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
            values.append(val)
        return values

    # def reset(self, nodes=None):
    #     if nodes is None or len(nodes) == 0:
    #         nodes = self._dict.keys()
    #     for node in ensure_list(nodes):
    #         self._dict[node].reset()
