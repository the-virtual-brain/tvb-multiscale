# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.tvb_nest.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
from tvb_multiscale.tvb_nest.nest_models.devices import NESTOutputSpikeDeviceDict, NESTOutputContinuousTimeDeviceDict
from tvb_multiscale.core.spiking_models.network import SpikingNetwork


LOG = initialize_logger(__name__)


class NESTNetwork(SpikingNetwork):
    """
        NESTNetwork is a class representing a NEST spiking network comprising of:
        - a NESTBrain class, i.e., neural populations organized per brain region they reside and neural model,
        - a pandas.Series of DeviceSet classes of output (measuring/recording/monitor) NEST devices,
        - a pandas.Series of DeviceSet classes of input (stimulating) NEST devices,
        all of which are implemented as indexed mappings by inheriting from pandas.Series class.
        The class also includes methods to return measurements (mean, sum/total data, spikes, spikes rates etc)
        from output devices, as xarray.DataArrays.
        e.g. NESTPopulations can be indexed as:
        nest_network.brain_regions['rh-insula']['E'] for population "E" residing in region node "rh-insula",
        and similarly for an output device:
        nest_network.output_devices['Excitatory']['rh-insula'],
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
        field_type=NESTBrain,
        label="NEST brain regions",
        default=None,
        required=True,
        doc="""A NESTBrain instance holding all NEST neural populations 
               organized per brain region they reside and neural model""")  # spiking_brain['rh-insula']['E']

    nest_instance = None

    _OutputSpikeDeviceDict = NESTOutputSpikeDeviceDict
    _OutputContinuousTimeDeviceDict = NESTOutputContinuousTimeDeviceDict

    def __init__(self, nest_instance=None, **kwargs):
        self.nest_instance = nest_instance
        self.config = kwargs.get("config", CONFIGURED)
        kwargs["config"] = self.config
        self.brain_regions = NESTBrain()
        super(NESTNetwork, self).__init__(**kwargs)

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    def Run(self, time, **kwargs):
        if self.nest_instance is not None:
            self.nest_instance.Run(time, **kwargs)

    def Simulate(self, time, **kwargs):
        if self.nest_instance is not None:
            self.nest_instance.Prepare()
            self.nest_instance.Run(time, **kwargs)

    def Cleanup(self, **kwargs):
        if self.nest_instance is not None:
            self.nest_instance.Cleanup(**kwargs)

    @property
    def min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    @property
    def dt(self):
        return self.nest_instance.GetKernelStatus("resolution")
