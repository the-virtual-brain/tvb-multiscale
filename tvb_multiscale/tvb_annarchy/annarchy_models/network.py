# -*- coding: utf-8 -*-

import os

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.tvb_annarchy.config import Config, CONFIGURED
from tvb_multiscale.tvb_annarchy.annarchy_models.brain import ANNarchyBrain
from tvb_multiscale.tvb_annarchy.annarchy_models.devices import \
    ANNarchyOutputSpikeDeviceDict, ANNarchyOutputContinuousTimeDeviceDict


class ANNarchyNetwork(SpikingNetwork):

    """
        ANNarchyNetwork is a class representing a ANNarchy spiking network comprising of:
        - a ANNarchyBrain class, i.e., neural populations organized per brain region they reside and neural model,
        - a pandas.Series of DeviceSet classes of output (measuring/recording/monitor) devices,
        - a pandas.Series of DeviceSet classes of input (stimulating) devices,
        all of which are implemented as indexed mappings by inheriting from pandas.Series class.
        The class also includes methods to return measurements (mean, sum/total data, spikes, spikes rates etc)
        from output devices, as xarray.DataArrays.
        e.g. ANNarchyPopulations can be indexed as:
        annarchy_network.brain_regions['rh-insula']['E'] for population "E" residing in region node "rh-insula",
        and similarly for an output device:
        annarchy_network.output_devices['Excitatory']['rh-insula'],
        which measures a quantity labelled following the target population ("Excitatory"),
        residing in region node "rh-insula".
    """

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    brain_regions = Attr(
        field_type=ANNarchyBrain,
        label="NEST brain regions",
        default=None,
        required=True,
        doc="""A NESTBrain instance holding all NEST neural populations 
                   organized per brain region they reside and neural model""")  # spiking_brain['rh-insula']['E']

    network_path = Attr(
        label="ANNarchy network path",
        field_type=str,
        doc="""Path to the compiled code of the ANNarchy network.""",
        required=False,
        default=""
    )

    annarchy_instance = None

    _network = None

    _dt = None

    _OutputSpikeDeviceDict = ANNarchyOutputSpikeDeviceDict
    _OutputContinuousTimeDeviceDict = ANNarchyOutputContinuousTimeDeviceDict

    def __init__(self, annarchy_instance=None, **kwargs):
        self.annarchy_instance = annarchy_instance
        self._dt = None
        self._network = None
        self.network_path = ""
        self.brain_regions = ANNarchyBrain()
        super(ANNarchyNetwork, self).__init__(**kwargs)

    @property
    def spiking_simulator_module(self):
        return self.annarchy_instance

    @property
    def network(self):
        if self._network is None:
            if self.annarchy_instance is not None:
                self._network = self.annarchy_instance.Network(everything=True)
        return self._network

    @property
    def dt(self):
        if self._dt is None:
            self._dt = self.annarchy_instance.Global.dt()
        return self._dt

    @property
    def min_delay(self):
        return self.dt

    def compile_network(self, *args, **kwargs):
        self.network_path = kwargs.pop("network_path", self.network_path)
        if not os.path.isdir(self.network_path):
            directory = str(kwargs.pop("directory", self.config.out.FOLDER_RES))
            cwd = os.getcwd()
            if directory.find(cwd) > -1:
                self.network_path = os.path.join(directory.split(cwd)[-1][1:].split("res")[0],
                                                 self.__class__.__name__)
        self.annarchy_instance.compile(directory=self.network_path, *args, **kwargs)

    def configure(self, *args, **kwargs):
        super(ANNarchyNetwork, self).configure()
        self.compile_network(*args, **kwargs)

    def Run(self, simulation_length=None, **kwargs):
        """Method to simulate the ANNarchy network for a specific simulation_length (in ms).
           It will run annarchy_instance.simulate(simulation_length, *args, **kwargs)
        """
        if simulation_length is None:
            simulation_length = self.dt
        measure_time = kwargs.pop("measure_time", False)
        for dev_name, out_dev_set in self.output_devices.items():
            out_dev_set.do_for_all("resume")
        self.annarchy_instance.simulate(simulation_length, measure_time=measure_time, **kwargs)
        for dev_name, out_dev_set in self.output_devices.items():
            out_dev_set.do_for_all("pause")

    def info(self, recursive=0):
        info = super(ANNarchyNetwork, self).info(recursive=recursive)
        info["ANNarchy_Network"] = self.network
        return info
