# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from tvb_multiscale.core.config import CONFIGURED, initialize_logger, LINE
from tvb_multiscale.core.spiking_models.region_node import SpikingRegionNode
from tvb_multiscale.core.spiking_models.brain import SpikingBrain
from tvb_multiscale.core.spiking_models.devices import DeviceSet, OutputSpikeDeviceDict, OutputContinuousTimeDeviceDict

from tvb.basic.neotraits.api import HasTraits, Attr

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class SpikingNetwork(HasTraits):
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

    brain_regions = Attr(
        field_type=SpikingBrain,
        label="Spiking brain regions",
        default=None,
        required=True,
        doc="""A SpikingBrain instance holding all neural populations 
               organized per brain region they reside and neural model""")  # spiking_brain['rh-insula']['E']

    # These devices are distinct from the ones for the TVB-Spiking Network interface
    output_devices = Attr(
        field_type=pd.Series,
        label="Output devices.",
        default=pd.Series(),
        required=True,
        doc="""A pandas.Series of output (recording) devices of the SpikingNetwork, 
               organized by recorded population and brain region.""")
    # output_devices['Excitatory']['rh-insula']

    input_devices = Attr(
        field_type=pd.Series,
        label="Input devices.",
        default=pd.Series(),
        required=True,
        doc="""A pandas.Series of input (stimulating) devices of the SpikingNetwork, 
               organized by target population and brain region.""")
    # input_devices['Inhibitory']['rh-insula']

    # These devices are distinct from the ones for the TVB-Spiking Network interface
    output_proxies = Attr(
        field_type=pd.Series,
        label="Output proxies.",
        default=pd.Series(),
        required=True,
        doc="""A pandas.Series of output (recording) devices of the SpikingNetwork, 
               which record data to send to a co-simulator, organized by 
               co-simulator recorded variable (e.g., TVB state variable) and brain region.""")
    # output_devices['Excitatory']['rh-insula']

    input_proxies = Attr(
        field_type=pd.Series,
        label="Intput proxys of the SpikingNetwork.",
        default=pd.Series(),
        required=False,
        doc="""A pandas.Series of input (stimulating) devices of the SpikingNetwork, 
               that mimick a co-simulator's activity (e.g., TVB mean field state variables) 
                   organized by brain region and population.""")
    # input_devices['Inhibitory']['rh-insula']

    _OutputSpikeDeviceDict = OutputSpikeDeviceDict
    _OutputContinuousTimeDeviceDict = OutputContinuousTimeDeviceDict

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

        super(SpikingNetwork, self).__init__()

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

    @property
    def populations_sizes(self):
        return self.brain_regions.populations_sizes

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

    def _get_devices(self, output_device_dict, devices_type, regions=None, populations_devices=None,
                     **kwargs):
        """This method will return all Spiking Network's recording devices of a given type.
           Arguments:
            - output_device_dict: a dictionary mapping device names (keys) to device classes (values).
            - devices_type: the name of the device type (string).
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
            - **kwargs: other keyword arguments. See get_spikes_devices method.
           Returns:
            - a Series of selected DeviceSet instances
        """
        devices = pd.Series()
        mode = kwargs.get("mode", None)
        if mode and mode.find("activity") > -1:
            devices = self.spiking_network.get_devices_by_model("spike_multimeter", regions=regions)
        else:
            for device_name in output_device_dict.keys():
                devices = devices.append(self.get_devices_by_model(device_name, regions=regions))
        if len(devices) == 0:
            LOG.warning("No %s recording device in this Spiking Network network!" % devices_type)
            return devices
        if populations_devices is not None:
            populations_devices = np.intersect1d(list(devices.index),
                                                 ensure_list(populations_devices)).tolist()
            if len(populations_devices) == 0:
                LOG.warning("No s recording device left after user selection!" % devices_type)
                return devices
            devices = devices[populations_devices]
        return devices

    def get_spikes_devices(self, mode="events", regions=None, populations_devices=None):
        """This method will return all Spiking Network's spike recording devices.
           Arguments:
            - mode: if "activity",
                     the method looks for "multimeter" devices that record spikes_train' weights continuously in time.
                    Default = "events", looking for devices that record spikes_train' events.
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of selected DeviceSet instances
        """
        return self._get_devices(self._OutputSpikeDeviceDict, "spikes'",
                                 regions, populations_devices, mode=mode)

    def get_continuous_time_devices(self, regions=None, populations_devices=None):
        """This method will return all Spiking Network's continuous time data recording devices.
           Arguments:
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of selected DeviceSet instances
        """
        return self._get_devices(self._OutputContinuousTimeDeviceDict, "continuous_time_data",
                                 regions, populations_devices)

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
        spike_devices = self.get_spikes_devices(mode, regions, populations_devices)
        spikes = pd.Series()
        for i_pop, (pop_label, pop_spike_device) in enumerate(spike_devices.iteritems()):
            spikes[pop_label] = \
                pop_spike_device.do_for_all_devices("get_spikes_events", **kwargs)
        return spikes

    def get_data(self, regions=None, populations_devices=None, **kwargs):
        """This method returns all continuous time data events from recording devices.
           Arguments:
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of data xarray.DataArrays per region and population.
        """
        devices = self.get_continuous_time_devices(regions, populations_devices)
        data = pd.Series()
        for i_pop, (pop_label, pop_device) in enumerate(devices.iteritems()):
            data[pop_label] = \
                pop_device.do_for_all_devices("get_data", **kwargs)
        return data
