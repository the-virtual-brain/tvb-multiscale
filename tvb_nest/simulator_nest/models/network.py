# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.nest_factory import load_nest
from tvb_nest.simulator_nest.models.region_node import NESTRegionNode
from tvb_nest.simulator_nest.models.devices \
    import NESTDeviceSet, NESTOutputDeviceDict, NESTInputDeviceDict, NESTOutputSpikeDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import list_of_dicts_to_dicts_of_ndarrays


LOG = initialize_logger(__name__)


def compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                        spike_rate_fun=None, mean_or_sum_fun="compute_spike_rate"):
    rates = Series()
    max_rate = 0
    for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors.iteritems()):
        rates[pop_label] = Series()
        for i_region, (reg_label, region_spike_detector) in enumerate(pop_spike_detector.iteritems()):
            rates[pop_label][reg_label] = \
                    getattr(region_spike_detector, mean_or_sum_fun)(time, spike_counts_kernel_width, spike_rate_fun)
            temp = np.max(rates[pop_label][reg_label])
            if temp > max_rate:
                max_rate = temp
    return rates, max_rate


class NESTNetwork(object):
    # Series of NESTRegionNode objects: nodes['rh-insula']['E']
    region_nodes = Series()
    # These devices are distinct from the ones for the TVB-NEST interface
    output_devices = Series()  # output_devices['Excitatory']['rh-insula']
    #
    stimulation_devices = Series()  # input_devices['Inhibitory']['rh-insula']

    def __init__(self, nest_instance=None,
                 region_nodes=Series(),
                 output_devices=Series(),
                 stimulation_devices=Series(),
                 config=CONFIGURED):
        self.config = config
        if nest_instance is None:
            nest_instance = load_nest()
        self.nest_instance = nest_instance
        self.region_nodes = Series()
        self.output_devices = Series()
        self.stimulation_devices = Series()
        if isinstance(region_nodes, Series):
            if len(region_nodes) > 0 and \
                    np.any([not isinstance(node, NESTRegionNode) for node in region_nodes]):
                raise ValueError("Input region_nodes is neither a NESTRegionNode "
                                 "nor a Series of NESTRegionNode objects!: \n %s" %
                                 str(region_nodes))
            self.region_nodes = region_nodes

        if isinstance(output_devices, Series):
            if len(output_devices) > 0 \
                    and np.any([not isinstance(dev, NESTDeviceSet) for dev in output_devices]):
                raise ValueError("Input output_devices is not a Series of output NESTDeviceSet objects!:\n %s" %
                                 str(output_devices))
            self.output_devices = output_devices
        if isinstance(stimulation_devices, Series):
            if len(stimulation_devices) > 0 and \
                    np.any([not isinstance(dev, NESTDeviceSet) for dev in stimulation_devices]):
                raise ValueError("Input stimulation_devices is not a Series of input NESTDeviceSet objects!:\n %s" %
                                 str(stimulation_devices))
            self.stimulation_devices = stimulation_devices

        LOG.info("%s created!" % self.__class__)

    @property
    def nodes_labels(self):
        return list(self.region_nodes.index)

    @property
    def number_of_nodes(self):
        return len(self.region_nodes)

    def get_devices_by_model(self, model):
        devices = Series()
        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.iteritems()):
            if pop_device.model == model:
                devices[pop_label] = pop_device
        return devices

    def _prepare_to_compute_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                        spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None):

        for device_name in NESTOutputSpikeDeviceDict.keys():
            spike_detectors = self.get_devices_by_model(device_name)
            if len(spike_detectors.keys()) > 0:
                break   # If this is not an empty dict of devices
        if len(spike_detectors.keys()) == 0:
            LOG.warning("No spike measuring device in this NEST network!")
            return None, None, None

        first_spike_time = self.config.calcul.MAX_SINGLE_VALUE
        last_spike_time = 0.0
        mean_spike_interval = self.config.calcul.MAX_SINGLE_VALUE
        for i_pop, (pop_label, pop_device) in enumerate(spike_detectors.iteritems()):
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_device.iteritems()):
                n_spikes = len(region_spike_detector.spikes_times)
                if n_spikes > 0:
                    temp = np.min(region_spike_detector.spikes_times)
                    if temp < first_spike_time:
                        first_spike_time = temp
                    temp = np.max(region_spike_detector.spikes_times)
                    if temp > last_spike_time:
                        last_spike_time = temp
                    if n_spikes > 1:
                        temp = np.mean(np.diff(np.unique(region_spike_detector.spikes_times)))
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
        if spike_detectors is not None:
            rates, max_rate = compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                                                  spike_rate_fun, "compute_spike_rate")
            return rates, max_rate, spike_detectors, time
        else:
            return None, None, None, None

    def compute_mean_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                 spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                 spike_rate_fun=None):
        spike_detectors, time, spike_counts_kernel_width = \
            self._prepare_to_compute_spike_rates(spike_counts_kernel_width, spike_counts_kernel_n_intervals,
                                                 spike_counts_kernel_overlap, min_spike_interval, time)
        if spike_detectors is not None:
            rates, max_rate = compute_spike_rates(spike_detectors, time, spike_counts_kernel_width,
                                                  spike_rate_fun, "compute_mean_spike_rate")
            return rates, max_rate, spike_detectors, time
        else:
            return None, None, None, None

    def get_mean_data_from_multimeter(self, *args, **kwargs):
        mean_data = Series()
        multimeters = self.get_devices_by_model("multimeter")
        for device_name, device_set in multimeters.iteritems():
            outputs = device_set.do_for_all_devices("get_mean_data", *args, **kwargs)
            time = []
            data = []
            for output in outputs:
                data.append(output[0])
                time = output[1]
            mean_data[device_name] = Series(list_of_dicts_to_dicts_of_ndarrays(data))
        return DataFrame(mean_data), time
