from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_templates import random_normal_weight, random_normal_tvb_weight, random_uniform_tvb_delay
from collections import OrderedDict
import numpy as np

class ThalamicVIMBuilder(NetpyneNetworkBuilder):

    def __init__(self, tvb_simulator={}, spiking_nodes_inds=[], netpyne_instance=None, config=None, logger=None):
        super(ThalamicVIMBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, netpyne_instance=netpyne_instance, config=config, logger=logger)

        self.scale_e = 1.2
        self.scale_i = 0.4

    def configure(self):
        try:
            # presuming that network is cloned to ./tvb_multiscale/tvb_netpyne/netpyne_models/models/thalamic_VIM_ET
            from .thalamic_VIM_ET.src.netParams import netParams
            from .thalamic_VIM_ET.src.cfg import cfg
        except ModuleNotFoundError:
            raise Exception('Spiking network should be cloned locally and imported here as `netParams` and `cfg`')

        # for external stimuli
        self.synMechE = 'exc'
        self.synMechI = 'inh'
        netParams.synMechParams[self.synMechE] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA
        netParams.synMechParams[self.synMechI] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA

        def intervalFunc(simTime):
            pass
        cfg.interval = 1
        cfg.intervalFunc = intervalFunc

        super(ThalamicVIMBuilder, self).configure(netParams, cfg, autoCreateSpikingNodes=False)
        self.global_coupling_scaling *= self.tvb_serial_sim.get("model.G", np.array([2.0]))[0].item()
        self.lamda = self.tvb_serial_sim.get("model.lamda", np.array([0.0]))[0].item()

    def proxy_node_synaptic_model_funcs(self):
        return {"E": lambda src_node, dst_node: self.synMechE,
                "I": lambda src_node, dst_node: self.synMechI}

    def set_defaults(self):
        self.set_populations()
        self.set_output_devices()

    def set_populations(self):
        self.populations = [
            {
                "label": "E", "model": None, # 'model' not used, as it is needed only in case of autoCreateSpikingNodes is True
                "nodes": None,  # None means "all"
                "params": {"global_label": "PY_pop"}, # population of spiking network to be interfaced with excitatory population of TVB node
                "scale": self.scale_e,
            },
            {
                "label": "I", "model": None,
                "nodes": None,  # None means "all"
                "params": {'global_label': 'FSI_pop'}, # population of spiking network to be interfaced with inhibitory population of TVB node
                "scale": self.scale_i
            }
        ]


    def set_output_devices(self):
        # Creating  devices to be able to observe NetPyNE activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_recorder()]

    def set_spike_recorder(self):
        connections = OrderedDict()
        # Keys are arbitrary. Value is either 'E' or 'I' for this model, and will be matched with items in self.populations dict
        connections["E"] = "E"
        connections["I"] = "I"
        params = self.config.NETPYNE_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
        # params["record_to"] = self.output_devices_record_to
        device = {"model": "spike_recorder", "params": params,
                #   "neurons_fun": lambda node_id, population: population[:np.minimum(100, len(population))],
                  "connections": connections, "nodes": None}  # None means all here
        # device.update(self.spike_recorder)
        return device

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(ThalamicVIMBuilder, self).build()