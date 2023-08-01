from tvb_multiscale.core.spiking_models.network import SpikingNetwork

from tvb_multiscale.tvb_netpyne.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_netpyne.netpyne_models.devices import NetpyneOutputSpikeDeviceDict

class NetpyneNetwork(SpikingNetwork):

    """
        NetpyneNetwork is a class representing a spiking network comprising of:
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

    netpyne_instance = None

    _OutputSpikeDeviceDict = NetpyneOutputSpikeDeviceDict

    def __init__(self, netpyne_instance, **kwargs):
        self.netpyne_instance = netpyne_instance
        super(NetpyneNetwork, self).__init__(**kwargs)

    @property
    def spiking_simulator_module(self):
        return self.netpyne_instance

    @property
    def dt(self):
        return self.netpyne_instance.dt

    @property
    def min_delay(self):
        return self.netpyne_instance.minDelay()

    def configure(self, *args, **kwargs):
        super(NetpyneNetwork, self).configure(args, kwargs)
        simulationDuration = self.tvb_cosimulator.simulation_length
        self.netpyne_instance.prepareSimulation(simulationDuration)

    def Run(self, simulation_length, *args, **kwargs):
        """Method to simulate the NetPyNE network for a specific simulation_length (in ms).
        """
        self.netpyne_instance.run(simulation_length)