# -*- coding: utf-8 -*-
from collections import OrderedDict
import pandas as pd
import xarray as xr
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.nest_factory import load_spiking_simulator
from tvb_nest.simulator_nest.models.region_node import NESTRegionNode
from tvb_nest.simulator_nest.models.devices \
    import NESTDeviceSet, NESTOutputSpikeDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class NESTNetwork(object):
    # pd.Series of NESTRegionNode objects: nodes['rh-insula']['E']
    region_nodes = pd.Series()
    # These devices are distinct from the ones for the TVB-NEST interface
    output_devices = pd.Series()  # output_devices['Excitatory']['rh-insula']
    #
    stimulation_devices = pd.Series()  # input_devices['Inhibitory']['rh-insula']
    nodes_min_delay = 0.0

    def __init__(self, nest_instance=None,
                 region_nodes=pd.Series(),
                 output_devices=pd.Series(),
                 stimulation_devices=pd.Series(),
                 nodes_min_delay=0.0,
                 config=CONFIGURED):
        self.config = config
        if nest_instance is None:
            nest_instance = load_spiking_simulator()
        self.nest_instance = nest_instance
        if nodes_min_delay <= 0.0:
            self.nodes_min_delay = self.nest_instance.GetKernelStatus("resolution")
        else:
            self.nodes_min_delay = nodes_min_delay
        self.region_nodes = pd.Series()
        self.output_devices = pd.Series()
        self.stimulation_devices = pd.Series()
        if isinstance(region_nodes, pd.Series):
            if len(region_nodes) > 0 and \
                    np.any([not isinstance(node, NESTRegionNode) for node in region_nodes]):
                raise ValueError("Input region_nodes is neither a NESTRegionNode "
                                 "nor a pandas.Series of NESTRegionNode objects!: \n %s" %
                                 str(region_nodes))
            self.region_nodes = region_nodes

        if isinstance(output_devices, pd.Series):
            if len(output_devices) > 0 \
                    and np.any([not isinstance(dev, NESTDeviceSet) for dev in output_devices]):
                raise ValueError("Input output_devices is not a pandas.Series of output NESTDeviceSet objects!:\n %s" %
                                 str(output_devices))
            self.output_devices = output_devices
        if isinstance(stimulation_devices, pd.Series):
            if len(stimulation_devices) > 0 and \
                    np.any([not isinstance(dev, NESTDeviceSet) for dev in stimulation_devices]):
                raise ValueError("Input stimulation_devices is not a pandas.Series of input NESTDeviceSet objects!:\n %s" %
                                 str(stimulation_devices))
            self.stimulation_devices = stimulation_devices

        LOG.info("%s created!" % self.__class__)

    @property
    def nestconfig(self):
        return self.config.nest

    @property
    def nodes_labels(self):
        return list(self.region_nodes.index)

    @property
    def number_of_nodes(self):
        return len(self.region_nodes)

    def get_devices_by_model(self, model, nodes=None):
        devices = pd.Series()
        if nodes is None:
            get_device = lambda device, nodes: device
        else:
            nodes = ensure_list(nodes)
            get_device = lambda device, nodes: device[nodes]

        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.iteritems()):
            if pop_device.model == model:
                devices[pop_label] = get_device(pop_device, nodes)
        return devices

    def _prepare_to_compute_spike_rates(self, population_devices=None, regions=None, mode="rate",
                                        spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                        spikes_kernel_overlap=0.5, min_spike_interval=None, time=None):

        if mode.find("activity") > -1:
            spike_detectors = self.get_devices_by_model("spike_multimeter", nodes=regions)
        else:
            for device_name in NESTOutputSpikeDeviceDict.keys():
                spike_detectors = self.get_devices_by_model(device_name, nodes=regions)
                if len(spike_detectors) > 0:
                    break   # If this is not an empty dict of devices
        if len(spike_detectors) == 0:
            LOG.warning("No spike measuring device in this NEST network!")
            return None, None, None

        if population_devices is not None:
            population_devices = np.intersect1d(list(spike_detectors.index),
                                                ensure_list(population_devices)).tolist()
            if len(population_devices) == 0:
                LOG.warning("No spike measuring device left after user selection!")
                return None, None, None
            spike_detectors = spike_detectors[population_devices]

        if regions is not None:
            regions = ensure_list(regions)

        first_spike_time = self.config.calcul.MAX_SINGLE_VALUE
        last_spike_time = 0.0
        mean_spike_interval = self.config.calcul.MAX_SINGLE_VALUE
        for i_pop, (pop_label, pop_device) in enumerate(spike_detectors.iteritems()):
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_device.iteritems()):
                if regions is not None and reg_label not in regions:
                    break
                spike_times = np.array(region_spike_detector.spikes_times)
                n_spikes = len(spike_times)
                if n_spikes > 0:
                    temp = np.min(spike_times)
                    if temp < first_spike_time:
                        first_spike_time = temp
                    temp = np.max(spike_times)
                    if temp > last_spike_time:
                        last_spike_time = temp
                    if n_spikes > 1:
                        temp = np.mean(np.diff(np.unique(spike_times)))
                        if temp < mean_spike_interval:
                            mean_spike_interval = temp

        if min_spike_interval is None:
            min_spike_interval = self.nest_instance.GetKernelStatus("min_delay")

        if time is None:

            # The kernel width should ideally be spikes_kernel_n_intervals times the mean_spike_interval
            # The mean_spike_interval should be longer than min_spike_interval (equal to min NEST delay = 0.001 ms)
            # and shorter than about time_duration / spikes_kernel_n_intervals **2

            if spikes_kernel_width is None:
                time_duration = last_spike_time - first_spike_time
                mean_spike_interval = np.minimum(np.maximum(min_spike_interval, mean_spike_interval),
                                                 time_duration / spikes_kernel_n_intervals ** 2)
                spikes_kernel_width = spikes_kernel_n_intervals * mean_spike_interval
            time_step = (1 - spikes_kernel_overlap) * spikes_kernel_width
            time = np.arange(first_spike_time, last_spike_time + time_step, time_step)

        else:

            # In this case it is the input time vector that determines
            if spikes_kernel_width is None:
                time_duration = time[-1] - time[0]
                mean_spike_interval = np.minimum(np.maximum(min_spike_interval, mean_spike_interval),
                                                 time_duration / spikes_kernel_n_intervals ** 2)
                spikes_kernel_width = spikes_kernel_n_intervals * mean_spike_interval
            time_step = np.mean(np.diff(time))

        return spike_detectors, time, int(np.maximum(1, np.ceil(spikes_kernel_width / time_step)))

    def compute_spikes_rates(self, mode="total_rate", population_devices=None, regions=None,
                             devices_dim_name="Population", name="Spikes rates from NEST network",
                             spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                             spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                             spikes_kernel=None):
        spike_detectors, time, spikes_kernel_width = \
            self._prepare_to_compute_spike_rates(population_devices, regions, mode,
                                                 spikes_kernel_width, spikes_kernel_n_intervals,
                                                 spikes_kernel_overlap, min_spike_interval, time)
        if spike_detectors is not None:
            kwargs = {}
            mode = mode.lower()
            if mode.find("activity") > -1:
                fun = "spikes_activity_across_time"
            else:
                fun = "spikes_rate_across_time"
            if mode.find("mean") > -1:
                fun = "compute_mean_" + fun
            else:
                fun = "compute_" + fun
                if mode.find("total") > -1:
                    kwargs.update({"mode": "total"})
            shape = spike_detectors[0].shape
            equal_shape_per_population = True
            rates = []
            population_devices = []
            for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors.iteritems()):
               rates.append(pop_spike_detector.do_for_all_devices(fun, time, spikes_kernel_width,
                                                                  spikes_kernel=spikes_kernel,
                                                                  return_type="xarray", **kwargs))
               population_devices.append(pop_label)
               equal_shape_per_population = pop_spike_detector.shape == shape
            if equal_shape_per_population:
                rates = xr.concat(rates, dim=pd.Index(list(spike_detectors.index), name=devices_dim_name))
                if mode == 'per_neuron':
                    # Reorder dimensions
                    #           0           1       2       3
                    # from: "Population, Region,   Neuron  Time"
                    # to:      "Time,  Population  Region, Neuron"
                    rates = rates.transpose(rates.dims[-1], rates.dims[0], rates.dims[1], rates.dims[2])
                else:
                    # Reorder dimensions
                    #           0           1       2
                    # from: "Population, Region,   Time
                    # to:      "Time,  Population  Region, "
                    rates = rates.transpose(rates.dims[-1], rates.dims[0], rates.dims[1])
            else:
                if mode == 'per_neuron':
                    for i_r, r in enumerate(rates):
                        # We cannot assume that all populations have the same number of neurons (and/or regions).
                        # Therefore, we need a Series data structure along populations
                        # Reorder dimensions
                        #           0       1        2
                        # from:   Region, Neuron   Time
                        # to:     "Time,  Neuron   Region"
                        rates[i_r] = r.transpose(r.dims[-1], r.dims[1], r.dims[0])
                else:
                    for i_r, r in enumerate(rates):
                        # We cannot assume that all populations have the same number of neurons (and/or regions).
                        # Therefore, we need a Series data structure along populations
                        # Reorder dimensions
                        #           0       2
                        # from:   Region,  Time
                        # to:     "Time,   Region"
                        rates[i_r] = r.transpose(r.dims[-1], r.dims[0])
                rates = pd.Series(rates, index=pd.Index(population_devices, name=devices_dim_name))
            rates.name = name
            return rates, spike_detectors
        else:
            return None, None

    def compute_spikes_activities(self, mode="total", population_devices=None, regions=None,
                                 devices_dim_name="Population", name="Spikes activities from NEST network",
                                 spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                 spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                 spikes_kernel=None):
        return self.compute_spikes_rates(mode+"_activity", population_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def compute_mean_spikes_rates(self, population_devices=None, regions=None, mode="rate",
                                  devices_dim_name="Population", name="Mean spikes rates from NEST network",
                                  spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                  spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                  spikes_kernel=None):
        return self.compute_spikes_rates(mode + "_mean", population_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def compute_mean_spikes_activities(self, population_devices=None, regions=None,
                                       devices_dim_name="Population", name="Mean spikes activity from NEST network",
                                       spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                       spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                       spikes_kernel=None):
        return self.compute_spikes_rates("mean_activity", population_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def get_data_from_multimeter(self, mode="total", population_devices=None, variables=None, regions=None,
                                 devices_dim_name="Population", name="Data from NEST multimeter",
                                 **kwargs):
        if mode == "mean":
            fun = "get_mean_data"
        else:
            fun = "get_data"
            if mode == "total":
                kwargs.update({"mode": mode})
        multimeters = self.get_devices_by_model("multimeter", nodes=regions)
        if len(multimeters) == 0:
            LOG.warning("No multimeter device in this NEST network!")
            return None, None
        index = list(multimeters.index)
        if population_devices is None:
            population_devices = index
        else:
            population_devices = np.intersect1d(index, ensure_list(population_devices)).tolist()
        shape = multimeters[population_devices[0]].shape
        equal_shape_per_population = True
        data = []
        for i_dev, device_name in enumerate(population_devices):
            data.append(multimeters[device_name].do_for_all_devices(fun, return_type="xarray",
                                                                    variables=variables, **kwargs))
            data[-1].name = device_name
            equal_shape_per_population = multimeters[device_name].shape == shape
        if equal_shape_per_population:
            data = xr.concat(data, dim=pd.Index(population_devices, name=devices_dim_name))
            if mode == 'per_neuron':
                # Reorder dimensions
                #           0           1         2       3       4
                # from: "Population, Region,   Variable  Neuron  Time"
                # to:      "Time,   Variable,  Region, Population, Neuron"
                data = data.transpose(data.dims[4], data.dims[2], data.dims[1], data.dims[0], data.dims[3])
            else:
                # Reorder dimensions
                #            0          1         2       3
                # from: "Population, Region,   Variable, Time "
                # to:      "Time,   Variable,   Region, Population   "
                data = data.transpose(data.dims[3], data.dims[2], data.dims[1], data.dims[0])
        else:
            if mode == 'per_neuron':
                for i_d, d in enumerate(data):
                    # We cannot assume that all populations have the same number of neurons (and/or regions).
                    # Therefore, we need a Series data structure along populations
                    # Reorder dimensions
                    #           0       1        2       3
                    # from:   Region, Variable Neuron   Time "
                    # to:     "Time,  Variable  Region  Neuron"
                    data[i_d] = d.transpose(d.dims[3], d.dims[1], d.dims[0], d.dims[2])
            else:
                for i_d, d in enumerate(data):
                    # We cannot assume that all populations have the same number of neurons (and/or regions).
                    # Therefore, we need a Series data structure along populations
                    # Reorder dimensions
                    #           0         1      2
                    # from:   Region,  Variable, Time  "
                    # to:     "Time,   Variable, Region"
                    data[i_d] = d.transpose(d.dims[2], d.dims[1], d.dims[0])
            data = pd.Series(data, index=pd.Index(population_devices, name=devices_dim_name))
        data.name = name
        return data

    def get_mean_data_from_multimeter(self, population_devices=None, variables=None, regions=None,
                                      devices_dim_name="Population",
                                      name="Mean data from NEST multimeter", **kwargs):
        data = self.get_data_from_multimeter("mean", population_devices, variables, regions,
                                             devices_dim_name, **kwargs)
        data.name = name
        return data
