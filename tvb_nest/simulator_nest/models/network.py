# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.nest_factory import load_nest
from tvb_nest.simulator_nest.models.region_node import NESTRegionNode
from tvb_nest.simulator_nest.models.devices \
    import NESTDeviceSet, NESTOutputSpikeDeviceDict
from tvb_scripts.utils.computations_utils import compute_spikes_rates
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list, nested_to_multiindex_pandas_dataframe


LOG = initialize_logger(__name__)


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

    def get_devices_by_model(self, model, nodes=None):
        devices = Series()
        if nodes is None:
            get_device = lambda device, nodes: device
        else:
            nodes = ensure_list(nodes)
            get_device = lambda device, nodes: device[nodes]

        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.iteritems()):
            if pop_device.model == model:
                devices[pop_label] = get_device(pop_device, nodes)
        return devices

    def _prepare_to_compute_spike_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                        spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                        regions=None):

        for device_name in NESTOutputSpikeDeviceDict.keys():
            spike_detectors = self.get_devices_by_model(device_name, nodes=regions)
            if len(spike_detectors) > 0:
                break   # If this is not an empty dict of devices
        if len(spike_detectors) == 0:
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

    def compute_spikes_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                             spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None, spike_rate_fun=None):
        spike_detectors, time, spike_counts_kernel_width = \
            self._prepare_to_compute_spike_rates(spike_counts_kernel_width, spike_counts_kernel_n_intervals,
                                                 spike_counts_kernel_overlap, min_spike_interval, time)
        if spike_detectors is not None:
            rates, max_rate = compute_spikes_rates(spike_detectors, time, spike_counts_kernel_width,
                                                   spike_rate_fun, "compute_spike_rate")
            return rates, max_rate, spike_detectors, time
        else:
            return None, None, None, None

    def compute_mean_spikes_rates(self, spike_counts_kernel_width=None, spike_counts_kernel_n_intervals=10,
                                  spike_counts_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                  spike_rate_fun=None, regions=None):
        spike_detectors, time, spike_counts_kernel_width = \
            self._prepare_to_compute_spike_rates(spike_counts_kernel_width, spike_counts_kernel_n_intervals,
                                                 spike_counts_kernel_overlap, min_spike_interval, time, regions)
        if spike_detectors is not None:
            rates, max_rate = compute_spikes_rates(spike_detectors, time, spike_counts_kernel_width,
                                                   spike_rate_fun, "compute_mean_spike_rate")
            return rates, max_rate, spike_detectors, time
        else:
            return None, None, None, None

    def get_mean_data_from_multimeter(self, population_devices=None, variables=None, regions=None, **kwargs):
        mean_data = Series()
        multimeters = self.get_devices_by_model("multimeter", nodes=regions)
        if len(multimeters) == 0:
            LOG.warning("No multimeter device in this NEST network!")
            return None, None
        index = list(multimeters.index)
        time = None
        if population_devices is None:
            population_devices = index
        else:
            population_devices = np.intersect1d(index, population_devices).tolist()
        for i_dev, device_name in enumerate(population_devices):
            mean_data[device_name] = \
                Series(multimeters[device_name].do_for_all_devices("get_mean_data", return_type="Series",
                                                                   variables=variables, **kwargs))
            new_time = multimeters[0].time[0]
            if time is None:
                time = new_time
            else:
                if np.any(new_time != time):
                    raise_value_error("Time vector of device %s is not identical to previous devices %s!"
                                      % (device_name, str()))
        mean_data = nested_to_multiindex_pandas_dataframe(mean_data, names=["Population", "Region", "Variable"])
        mean_data = mean_data.reorder_levels([2, 1, 0]).transpose()
        return mean_data, time
