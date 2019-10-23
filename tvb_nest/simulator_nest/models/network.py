# -*- coding: utf-8 -*-

import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.nest_factory import load_nest
from tvb_nest.simulator_nest.models.region_node import NESTRegionNode
from tvb_nest.simulator_nest.models.devices import NESTDeviceSet, NESTOutputDeviceDict, NESTInputDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.indexed_ordered_dict import IndexedOrderedDict

LOG = initialize_logger(__name__)


def compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                        spike_rate_fun=None, mean_or_sum_fun="compute_spike_rate"):
    rates = IndexedOrderedDict({})
    max_rate = 0
    for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors.items()):
        rates.update({pop_label: IndexedOrderedDict({})})
        for i_region, (reg_label, region_spike_detector) in enumerate(pop_spike_detector.items()):
            rates[pop_label].update(
                {reg_label: getattr(region_spike_detector, mean_or_sum_fun)(
                    time, spike_counts_kernel_width, spike_rate_fun)})
            temp = np.max(rates[pop_label][reg_label])
            if temp > max_rate:
                max_rate = temp
    return rates, max_rate


class NESTNetwork(object):
    # IndexedOrderedDict of NESTRegionNode objects: nodes['rh-insula']['E']
    region_nodes = IndexedOrderedDict({})
    # These devices are distinct from the ones for the TVB-NEST interface
    output_devices = IndexedOrderedDict({})  # output_devices['Excitatory']['rh-insula']
    #
    stimulation_devices = IndexedOrderedDict({})  # input_devices['Inhibitory']['rh-insula']

    def __init__(self, nest_instance=None,
                 region_nodes=IndexedOrderedDict({}),
                 output_devices=IndexedOrderedDict({}),
                 stimulation_devices=IndexedOrderedDict({}), config=CONFIGURED):
        self.config = config
        if nest_instance is None:
            nest_instance = load_nest()
        self.nest_instance = nest_instance
        if isinstance(region_nodes, IndexedOrderedDict) and \
                np.all([isinstance(node, NESTRegionNode)
                        for node in region_nodes.values()]):
            self.region_nodes = region_nodes
        else:
            raise ValueError("Input region_nodes is not a IndexedOrderedDict of NESTRegionNode objects!: \n %s" %
                             str(region_nodes))

        self.output_devices = output_devices
        if isinstance(output_devices, IndexedOrderedDict) and \
                np.all([isinstance(dev, NESTDeviceSet) and (dev.model in NESTOutputDeviceDict.keys() or
                                                            len(dev.model) == 0)
                        for dev in output_devices.values()]):
            self.output_devices = output_devices
        else:
            raise ValueError("Input output_devices is not a IndexedOrderedDict of output NESTDeviceSet objects!:\n %s" %
                             str(output_devices))

        if isinstance(stimulation_devices, IndexedOrderedDict) and \
                np.all([isinstance(dev, NESTDeviceSet) and (dev.model in NESTInputDeviceDict.keys() or
                                                            len(dev.model) == 0)
                        for dev in stimulation_devices.values()]):
            self.stimulation_devices = stimulation_devices
        else:
            raise ValueError(
                "Input stimulation_devices is not a IndexedOrderedDict of input NESTDeviceSet objects!:\n %s" %
                str(stimulation_devices))

        LOG.info("%s created!" % self.__class__)

    @property
    def nodes_labels(self):
        return self.region_nodes.keys()

    @property
    def number_of_nodes(self):
        return len(self.region_nodes)

    def get_devices_by_model(self, model):
        devices = IndexedOrderedDict({})
        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.items()):
            if pop_device.model == model:
                devices[pop_label] = pop_device
        return devices

    def _prepare_to_compute_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                        spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None):

        try:
            spike_detectors = self.get_devices_by_model("spike_detector")
        except:
            spike_detectors = self.get_devices_by_model("spike_multimeter")

        first_spike_time = self.config.calcul.MAX_SINGLE_VALUE
        last_spike_time = 0.0
        mean_spike_interval = self.config.calcul.MAX_SINGLE_VALUE
        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.items()):
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_device.items()):
                n_spikes = len(region_spike_detector.spike_times)
                if n_spikes > 0:
                    temp = np.min(region_spike_detector.spike_times)
                    if temp < first_spike_time:
                        first_spike_time = temp
                    temp = np.max(region_spike_detector.spike_times)
                    if temp > last_spike_time:
                        last_spike_time = temp
                    if n_spikes > 1:
                        temp = np.mean(np.diff(np.unique(region_spike_detector.spike_times)))
                        if temp < mean_spike_interval:
                            mean_spike_interval = temp

        if min_spike_interval is None:
            min_spike_interval = self.nest_instance.GetKernelStatus("min_delay")

        if time is None:

            # The kernel width should ideally be spike_counts_kernel_n_intervals times the mean_spike_interval
            # The mean_spike_interval should be longer than min_spike_interval (equal to min NEST delay = 0.001 ms)
            # and shorter than about time_duration / spike_counts_kernel_n_intervals **2

            if spike_counts_kernel_width is None:
                time_duration = last_spike_time - first_spike_time
                mean_spike_interval = np.minimum(np.maximum(min_spike_interval, mean_spike_interval),
                                                 time_duration / spike_counts_kernel_n_intervals ** 2)
                spike_counts_kernel_width = spike_counts_kernel_n_intervals * mean_spike_interval
            time_step = (1 - spike_counts_kernel_overlap) * spike_counts_kernel_width
            time = np.arange(first_spike_time, last_spike_time + time_step, time_step)

        else:

            # In this case it is the input time vector that determines
            if spike_counts_kernel_width is None:
                time_duration = time[-1] - time[0]
                mean_spike_interval = np.minimum(np.maximum(min_spike_interval, mean_spike_interval),
                                                 time_duration / spike_counts_kernel_n_intervals ** 2)
                spike_counts_kernel_width = spike_counts_kernel_n_intervals * mean_spike_interval

        return spike_detectors, time, spike_counts_kernel_width

    def compute_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                            spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None, spike_rate_fun=None):
        spike_detectors, time, spike_counts_kernel_width = \
            self._prepare_to_compute_spike_rates(spike_counts_kernel_width, spike_counts_kernel_n_intervals,
                                                 spike_counts_kernel_overlap, min_spike_interval, time)
        rates, max_rate = compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                                              spike_rate_fun, "compute_spike_rate")
        return rates, max_rate, spike_detectors, time

    def compute_mean_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                 spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                 spike_rate_fun=None):
        spike_detectors, time, spike_counts_kernel_width = \
            self._prepare_to_compute_spike_rates(spike_counts_kernel_width, spike_counts_kernel_n_intervals,
                                                 spike_counts_kernel_overlap, min_spike_interval, time)
        rates, max_rate = compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                                              spike_rate_fun, "compute_mean_spike_rate")
        return rates, max_rate, spike_detectors, time

