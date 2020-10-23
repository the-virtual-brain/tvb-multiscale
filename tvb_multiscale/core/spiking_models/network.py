# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd
import xarray as xr
import numpy as np

from tvb_multiscale.core.config import CONFIGURED, initialize_logger, LINE
from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode
from tvb_multiscale.core.spiking_models.devices import DeviceSet, OutputSpikeDeviceDict

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class SpikingNetwork(object):
    __metaclass__ = ABCMeta

    """
        SpikingNetwork is a class representing a spiking network comprising of:
        - a SpikingBrain class, i.e., neural populations organized per brain region they reside and neural model,
        - a pandas.Series of DeviceSet classes of output (measuring/recording/monitor) devices,
        - a pandas.Series of DeviceSet classes of input (stimulating) devices,
        all of which are implemented as indexed mappings by inheriting from pandas.Series class.
        The class also includes methods to return measurements (mean, sum/total data, spikes, spikes rates etc)
        from output devices, as xarray.DataArrays.
        e.g. SpikingPopulations can be indexed as:
        spiking_network.brain_regions['rh-insula']['E'] for population "E" residing in region node "rh-insula",
        and similarly for an output device:
        spiking_network.output_devices['Excitatory']['rh-insula'], 
        which measures a quantity labelled following the target population ("Excitatory"),
        residing in region node "rh-insula".
    """

    brain_regions = None  # spiking_brain['rh-insula']['E']
    # These devices are distinct from the ones for the TVB-Spiking Network interface
    output_devices = None  # output_devices['Excitatory']['rh-insula']
    #
    input_devices = None  # input_devices['Inhibitory']['rh-insula']

    def __init__(self,
                 brain_regions=None,
                 output_devices=None,
                 input_devices=None,
                 config=CONFIGURED):
        self.config = config

        self.brain_regions = brain_regions
        self.output_devices = output_devices
        self.input_devices = input_devices

        if isinstance(brain_regions, pd.Series):
            if len(brain_regions) > 0 and \
                    np.any([not isinstance(node, SpikingRegionNode) for node in brain_regions]):
                raise ValueError("Input spiking_brain is neither a SpikingRegionNode "
                                 "nor a pandas.Series of SpikingRegionNode objects!: \n %s" %
                                 str(brain_regions))
            self.brain_regions = brain_regions

        if isinstance(output_devices, pd.Series):
            if len(output_devices) > 0 \
                    and np.any([not isinstance(dev, DeviceSet) for dev in output_devices]):
                raise ValueError("Input output_devices is not a pandas.Series of output DeviceSet objects!:\n %s" %
                                 str(output_devices))
            self.output_devices = output_devices
        if isinstance(input_devices, pd.Series):
            if len(input_devices) > 0 and \
                    np.any([not isinstance(dev, DeviceSet) for dev in input_devices]):
                raise ValueError("Input input_devices is not a pandas.Series of input DeviceSet objects!:\n %s" %
                                 str(input_devices))
            self.input_devices = input_devices

        LOG.info("%s created!" % self.__class__)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.print_str()

    def print_str(self, connectivity=False):
        spiking_brain = LINE + self.brain_regions.print_str(connectivity)
        input_devices = 2*LINE + "\n\nInput Devices:\n"
        for node_name, node in self.input_devices.iteritems():
            input_devices += LINE + node.print_str(connectivity)
        output_devices = 2*LINE + "\n\nOutput Devices:\n"
        for node_name, node in self.output_devices.iteritems():
            output_devices += LINE + node.print_str(connectivity)
        outputs = 3*LINE + "%s:\n" % self.__class__.__name__
        for output_name, output in zip(["Spiking Brain Regions", "Input Devices", "Output Devices"],
                                       [spiking_brain, input_devices, output_devices]):
            outputs += output
        return outputs

    @abstractmethod
    def configure(self, *args, **kwargs):
        """Method to configure a simulation just before execution.
        """
        pass

    @abstractmethod
    def Run(self, simulation_length, *args, **kwargs):
        """Method to simulate the spiking network for a specific simulation_length (in ms)."""
        pass

    @property
    @abstractmethod
    def min_delay(self):
        pass

    @property
    def nodes_labels(self):
        return list(self.brain_regions.index)

    @property
    def number_of_nodes(self):
        return len(self.brain_regions)

    def get_devices_by_model(self, model, regions=None):
        """This method will loop though all network's devices to return all devices of a given model.
           Arguments:
            - model: the model name (string),
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
           Returns:
            - a Series of selected DeviceSet instances
        """
        # Get all devices set of a given model
        devices = pd.Series()
        if regions is None:
            get_device = lambda device, regions: device
        else:
            nodes = ensure_list(regions)
            get_device = lambda device, regions: device[nodes]
        for i_pop, (pop_label, pop_device) in enumerate(self.output_devices.iteritems()):
            if pop_device.model == model:
                devices[pop_label] = get_device(pop_device, regions)
        return devices

    def get_spike_devices(self, mode="events", regions=None, populations_devices=None):
        """This method will loop though all network's devices to return all devices of a given model.
           Arguments:
            - mode: if "activity",
                     the method looks for "multimeter" devices that record spikes' weights continuously in time.
                    Default = "events", looking for devices that record spikes' events.
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of selected DeviceSet instances
        """
        spike_devices = pd.Series()
        if mode.find("activity") > -1:
            spike_devices = self.get_devices_by_model("spike_multimeter", regions=regions)
        else:
            for device_name in OutputSpikeDeviceDict.keys():
                spike_devices = self.get_devices_by_model(device_name, regions=regions)
                if len(spike_devices) > 0:
                    break  # If this is not an empty dict of devices
        if len(spike_devices) == 0:
            LOG.warning("No spike measuring device in this Spiking Network network!")
            return spike_devices
        if populations_devices is not None:
            populations_devices = np.intersect1d(list(spike_devices.index),
                                                 ensure_list(populations_devices)).tolist()
            if len(populations_devices) == 0:
                LOG.warning("No spike measuring device left after user selection!")
                return spike_devices
            spike_devices = spike_devices[populations_devices]
        return spike_devices

    def get_spikes(self, mode="events", regions=None, populations_devices=None, **kwargs):
        """This method returns all spikes' events from any kind of spikes' recording devices.
           Arguments:
            - mode: if "activity",
                     the method looks for "multimeter" devices that record spikes' weights continuously in time.
                    Default = "events", looking for devices that record spikes' events.
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of spikes' events per region and population.
        """
        spike_devices = self.get_spike_devices(mode, regions, populations_devices)
        spikes = pd.Series()
        for i_pop, (pop_label, pop_spike_device) in enumerate(spike_devices.iteritems()):
            spikes[pop_label] = \
                pop_spike_device.do_for_all_devices("get_spikes_events", **kwargs)
        return spikes

    def _prepare_to_compute_spike_rates(self, populations_devices=None, regions=None, mode="rate",
                                        spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                        spikes_kernel_overlap=0.5, min_spike_interval=None, time=None):
        """This method gets spikes recording devices and
           precomputes parameters necessary for the computation of spikes' rates,
           based on user inputs.
        """
        spike_devices = self.get_spike_devices(mode, regions, populations_devices)

        if regions is not None:
            regions = ensure_list(regions)

        # If there is no time vector user input, we need to get a feeling of the spike intervals' statistics
        if time is None or spikes_kernel_width is None:
            first_spike_time = self.config.calcul.MAX_SINGLE_VALUE
            last_spike_time = 0.0
            mean_spike_interval = self.config.calcul.MAX_SINGLE_VALUE
            for i_pop, (pop_label, pop_device) in enumerate(spike_devices.iteritems()):
                for i_region, (reg_label, region_spike_device) in enumerate(pop_device.iteritems()):
                    if regions is not None and reg_label not in regions:
                        break
                    spike_times = np.array(region_spike_device.spikes_times)
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
                min_spike_interval = self.min_delay

        if time is None:

            # The kernel width should ideally be spikes_kernel_n_intervals times the mean_spike_interval
            # The mean_spike_interval should be longer than min_spike_interval (equal to min delay = 0.001 ms)
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

        spikes_kernel_width_in_points = int(np.maximum(1, np.ceil(spikes_kernel_width / time_step)))
        spikes_kernel_width = spikes_kernel_width_in_points * time_step
        return spike_devices, time, spikes_kernel_width, spikes_kernel_width_in_points

    def compute_spikes_rates(self, mode="total_rate", populations_devices=None, regions=None,
                             devices_dim_name="Population", name="Spikes rates from Spiking Network",
                             spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                             spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                             spikes_kernel=None, flatten_neurons_inds=True):
        """This method computes spikes rates, or activities, from spikes recording devices,
           and optionally the mean or total (sum) rate across neurons.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        spike_devices, time, spikes_kernel_width, spikes_kernel_width_in_points = \
            self._prepare_to_compute_spike_rates(populations_devices, regions, mode,
                                                 spikes_kernel_width, spikes_kernel_n_intervals,
                                                 spikes_kernel_overlap, min_spike_interval, time)
        if spike_devices is not None:
            kwargs = {}
            mode = mode.lower()
            if mode.find("activity") > -1:
                fun = "spikes_activity_across_time"  # computing with spike weights
            else:
                fun = "spikes_rate_across_time"
            # Computing mean or total (sum) quantities across neurons
            if mode.find("mean") > -1:
                fun = "compute_mean_" + fun
            else:
                fun = "compute_" + fun
                if mode.find("total") > -1:
                    kwargs.update({"mode": "total"})
                else:
                    mode = "per_neuron"
                    kwargs["flatten_neurons_inds"] = flatten_neurons_inds
            shape = spike_devices[0].shape
            equal_shape_per_population = True
            rates = []
            populations_devices = []
            for i_pop, (pop_label, pop_spike_device) in enumerate(spike_devices.iteritems()):
               rates.append(pop_spike_device.do_for_all_devices(fun, time,
                                                                  spikes_kernel_width, spikes_kernel_width_in_points,
                                                                  spikes_kernel=spikes_kernel,
                                                                  return_type="DataArray", **kwargs))
               populations_devices.append(pop_label)
               equal_shape_per_population = pop_spike_device.shape == shape
            if equal_shape_per_population:
                rates = xr.concat(rates, dim=pd.Index(list(spike_devices.index), name=devices_dim_name))
                if rates.size == 0:  # In case there is nothing to measure in Spiking Network
                    rates.name = name
                    return rates, spike_devices
                if mode == 'per_neuron':
                    # Reorder dimensions
                    #           0           1        2       3
                    # from: Population, Region,    Neuron  Time
                    # to:     Time,    Population  Region, Neuron"
                    rates = rates.transpose(rates.dims[-1], rates.dims[0], rates.dims[1], rates.dims[2])
                else:
                    # Reorder dimensions
                    #           0           1       2
                    # from: Population, Region,   Time
                    # to:   Time,     Population  Region
                    rates = rates.transpose(rates.dims[-1], rates.dims[0], rates.dims[1])
            else:
                if mode == 'per_neuron':
                    for i_r, r in enumerate(rates):
                        if len(r.dims) < 3:  # In case there is nothing to measure in Spiking Network
                            break
                        # We cannot assume that all populations have the same number of neurons (and/or regions).
                        # Therefore, we need a Series data structure along populations
                        # Reorder dimensions
                        #           0       1        2
                        # from:   Region, Neuron   Time
                        # to:     Time,   Region   Neuron
                        rates[i_r] = r.transpose(r.dims[-1], r.dims[0], r.dims[1])
                else:
                    for i_r, r in enumerate(rates):
                        if len(r.dims) < 2:  # In case there is nothing to measure in Spiking Network
                            break
                        # We cannot assume that all populations have the same number of neurons (and/or regions).
                        # Therefore, we need a Series data structure along populations
                        # Reorder dimensions
                        #           0       1
                        # from:   Region,  Time
                        # to:     Time,   Region
                        rates[i_r] = r.transpose(r.dims[-1], r.dims[0])
                rates = pd.Series(rates, index=pd.Index(populations_devices, name=devices_dim_name))
            rates.name = name
            return rates, spike_devices
        else:
            return None, None

    def compute_spikes_activities(self, mode="total", populations_devices=None, regions=None,
                                  devices_dim_name="Population Device", name="Spikes activities from Spiking Network",
                                  spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                  spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                  spikes_kernel=None):
        """This method computes spikes activities from spikes weights' recording devices.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        return self.compute_spikes_rates(mode + "_activity", populations_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def compute_mean_spikes_rates(self, populations_devices=None, regions=None, mode="rate",
                                  devices_dim_name="Population Device",
                                  name="Mean spikes rates from Spiking Network",
                                  spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                  spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                  spikes_kernel=None):
        """This method computes mean spikes rates across neurons from spikes recording devices.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        return self.compute_spikes_rates(mode + "_mean", populations_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def compute_mean_spikes_activities(self, populations_devices=None, regions=None,
                                       devices_dim_name="Population Device",
                                       name="Mean spikes activity from Spiking Network",
                                       spikes_kernel_width=None, spikes_kernel_n_intervals=10,
                                       spikes_kernel_overlap=0.5, min_spike_interval=None, time=None,
                                       spikes_kernel=None):
        """This method computes mean spikes activities across neurons from spikes recording devices.
            The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
            depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        return self.compute_spikes_rates("mean_activity", populations_devices, regions,
                                         devices_dim_name, name,
                                         spikes_kernel_width, spikes_kernel_n_intervals,
                                         spikes_kernel_overlap, min_spike_interval, time,
                                         spikes_kernel)

    def get_data_from_multimeter(self, mode="total", populations_devices=None, variables=None, regions=None,
                                 devices_dim_name="Population Device", name="Data from Spiking Network multimeter",
                                 flatten_neurons_inds=True, **kwargs):
        """This method returns data, from devices recording continuous variables in time,
           and optionally computes the mean or total sum across neurons.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        if mode == "mean":
            # Mean quantities across neurons
            fun = "get_mean_data"
        elif mode == "total":
            # Total (summing) quantities across neurons
            fun = "get_total_data"
        else:
            fun = "get_data"
            mode = "per_neuron"
            kwargs["flatten_neurons_inds"] = flatten_neurons_inds
        multimeters = self.get_devices_by_model("multimeter", regions=regions)
        if len(multimeters) == 0:
            LOG.warning("No multimeter device in this Spiking Network!")
            return None, None
        index = list(multimeters.index)
        if populations_devices is None:
            populations_devices = index
        else:
            populations_devices = np.intersect1d(index, ensure_list(populations_devices)).tolist()
        shape = multimeters[populations_devices[0]].shape
        equal_shape_per_population = True
        data = []
        for i_dev, device_name in enumerate(populations_devices):
            data.append(multimeters[device_name].do_for_all_devices(fun, return_type="DataArray",
                                                                    variables=variables, **kwargs))
            data[-1].name = device_name
            equal_shape_per_population = multimeters[device_name].shape == shape
        if equal_shape_per_population:
            data = xr.concat(data, dim=pd.Index(populations_devices, name=devices_dim_name))
            if data.size == 0:  # In case there is nothing to measure in Spiking Network
                data.name = name
                return data
            if mode == 'per_neuron':
                # Reorder dimensions
                #           0           1         2       3         4
                # from: Population, Region,   Variable  Neuron     Time
                # to:     Time,    Variable,   Region, Population, Neuron
                data = data.transpose(data.dims[-1], data.dims[2], data.dims[1], data.dims[0], data.dims[3])
            else:
                # Reorder dimensions
                #            0          1         2       3
                # from: "Population, Region,   Variable, Time "
                # to:      "Time,   Variable,   Region, Population   "
                data = data.transpose(data.dims[3], data.dims[2], data.dims[1], data.dims[0])
        else:
            if mode == 'per_neuron':
                for i_d, d in enumerate(data):
                    if len(d.dims) < 4:   # In case there is nothing to measure in Spiking Network
                        break
                    # We cannot assume that all populations have the same number of neurons (and/or regions).
                    # Therefore, we need a Series data structure along populations
                    # Reorder dimensions
                    #           0       1         2       3
                    # from:   Region, Variable  Neuron   Time
                    # to:      Time,  Variable  Region  Neuron
                    data[i_d] = d.transpose(d.dims[3], d.dims[1], d.dims[0], d.dims[2])
            else:
                for i_d, d in enumerate(data):
                    if len(d.dims) < 3:   # In case there is nothing to measure in Spiking Network
                        break
                    # We cannot assume that all populations have the same number of neurons (and/or regions).
                    # Therefore, we need a Series data structure along populations
                    # Reorder dimensions
                    #           0         1      2
                    # from:   Region,  Variable, Time
                    # to:      Time,   Variable, Region
                    data[i_d] = d.transpose(d.dims[2], d.dims[1], d.dims[0])
            data = pd.Series(data, index=pd.Index(populations_devices, name=devices_dim_name))
        data.name = name
        return data

    def get_mean_data_from_multimeter(self, populations_devices=None, variables=None, regions=None,
                                      devices_dim_name="Population device",
                                      name="Mean data from Spiking Network multimeter",
                                      **kwargs):
        """This method computes the mean data across neurons, from devices recording continuous variables in time.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        data = self.get_data_from_multimeter("mean", populations_devices, variables, regions,
                                             devices_dim_name, **kwargs)
        data.name = name
        return data

    def get_total_data_from_multimeter(self, populations_devices=None, variables=None, regions=None,
                                       devices_dim_name="Population device",
                                       name="Mean data from Spiking Network multimeter",
                                       **kwargs):
        """This method computes the mean data across neurons, from devices recording continuous variables in time.
           The output is returned either as a xarray.DataArray, or as a pandas.Series of xarray.DataArrays,
           depending on the homo/hetero-geneity of the distribution of populations per region, respectively.
        """
        data = self.get_data_from_multimeter("total", populations_devices, variables, regions,
                                             devices_dim_name, **kwargs)
        data.name = name
        return data
