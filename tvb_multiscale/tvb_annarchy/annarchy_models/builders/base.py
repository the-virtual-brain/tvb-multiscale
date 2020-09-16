# -*- coding: utf-8 -*-
from pandas import Series
import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import \
    load_nest, compile_modules, create_conn_spec, create_device, connect_device
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder

from tvb.contrib.scripts.utils.log_error_utils import raise_value_error
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class NESTModelBuilder(SpikingModelBuilder):

    """This is a not so opinionated builder of a NEST Network"""

    config = CONFIGURED
    nest_instance = None
    default_min_spiking_dt = CONFIGURED.NEST_MIN_DT
    default_min_delay = CONFIGURED.NEST_MIN_DT
    modules_to_install = []
    _spiking_brain = NESTBrain()

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, logger=LOG):
        super(NESTModelBuilder, self).__init__(tvb_simulator, nest_nodes_ids, config, logger)
        # Setting or loading a nest instance:
        if nest_instance is not None:
            self.nest_instance = nest_instance
        else:
            self.nest_instance = load_nest(self.config, self.logger)

        self._spiking_brain = NESTBrain()

        # Setting NEST defaults from config
        self.default_population = {"model": self.config.DEFAULT_MODEL, "scale": 1, "params": {}, "nodes": None}

        self.default_synaptic_weight_scaling = \
            lambda weight, n_cons: self.config.DEFAULT_SPIKING_SYNAPTIC_WEIGHT_SCALING(weight, n_cons)

        self.default_populations_connection = dict(self.config.DEFAULT_CONNECTION)
        self.default_populations_connection["delay"] = self.default_min_delay
        self.default_populations_connection["nodes"] = None

        self.default_nodes_connection = dict(self.config.DEFAULT_CONNECTION)
        self.default_nodes_connection["delay"] = self.default_populations_connection["delay"]
        self.default_nodes_connection.update({"source_nodes": None, "target_nodes": None})

    def _configure_nest_kernel(self):
        self.nest_instance.ResetKernel()  # This will restart NEST!
        self._update_spiking_dt()
        self._update_default_min_delay()
        self.nest_instance.set_verbosity(self.config.NEST_VERBOCITY)  # don't print all messages from NEST
        self.nest_instance.SetKernelStatus({"resolution": self.spiking_dt, "print_time": self.config.NEST_PRINT_TIME})

    def _compile_install_nest_module(self, module):
        if module[-6:] == "module":
            module_name = module.split("module")[0]
        else:
            module_name = module
            module = module + "module"
        try:
            # Try to install it...
            self.logger.info("Trying to install module %s..." % module)
            self.nest_instance.Install(module)
            self.logger.info("DONE installing module %s!" % module)
        except:
            self.logger.info("FAILED! We need to first compile it!")
            # ...unless we need to first compile it:
            compile_modules(module_name, recompile=False, config=self.config)
            # and now install it...
            self.logger.info("Installing now module %s..." % module)
            self.nest_instance.Install(module)
            self.logger.info("DONE installing module %s!" % module)

    def compile_install_nest_modules(self, modules_to_install):
        if len(modules_to_install) > 0:
            self.logger.info("Starting to compile modules %s!" % str(modules_to_install))
            while len(modules_to_install) > 0:
                self._compile_install_nest_module(modules_to_install.pop())

    def confirm_compile_install_nest_models(self, models):
        nest_models = self.nest_instance.Models()
        models = ensure_list(models)
        for model in models:  # , module # zip(models, cycle(modules_to_install)):
            if model not in nest_models:
                self._compile_install_nest_module(model)

    def configure(self):
        self._configure_nest_kernel()
        super(NESTModelBuilder, self).configure()
        self.compile_install_nest_modules(self.modules_to_install)
        self.confirm_compile_install_nest_models(self._models)

    @property
    def min_delay(self):
        try:
            return self.nest_instance.GetKernelStatus("min_delay")
        except:
            return self.default_min_delay

    def _prepare_populations_connection_params(self, pop_src, pop_trg, conn_spec, syn_spec):
        return create_conn_spec(n_src=len(pop_src), n_trg=len(pop_trg),
                                src_is_trg=(pop_src == pop_trg), config=self.config, **conn_spec)

    def _get_minmax_delay(self, delay, minmax):
        if isinstance(delay, dict):
            if "distribution" in delay.keys():
                if delay["distribution"] == "uniform":
                    return delay[minmax]
                else:
                    raise_value_error("Only uniform distribution is allowed for delays to make sure that > min_delay!\n"
                                      "Distribution given is %s!" % delay["distribution"])
            else:
                raise_value_error("If delay is a dictionary it has to be a distribution dictionary!\n"
                                  "Instead, the delay given is %s\n" % str(delay))
        else:
            return delay

    def _get_min_delay(self, delay):
        return self._get_minmax_delay(delay, "low")

    def _get_max_delay(self, delay):
        return self._get_minmax_delay(delay, "high")

    def _assert_synapse_model(self, synapse_model, delay):
        if synapse_model is None:
            synapse_model = "static_synapse"
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %s != 0.0 is not possible!" % str(delay))
            elif self._get_min_delay(delay) == 0.0 and synapse_model == "rate_connection_delayed":
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %s <= 0.0 is not possible!" % str(delay))
            elif self._get_max_delay(delay) == 0.0:
                return "rate_connection_instantaneous"
            else:
                return "rate_connection_delayed"
        else:
            return synapse_model

    def _assert_delay(self, delay, synapse_model="static_synapse"):
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %s != 0.0 is not possible!" % str(delay))
            elif synapse_model == "rate_connection_delayed" and self._get_min_delay(delay) <= 0.0:
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %s <= 0.0 is not possible!" % str(delay))
            elif self._get_min_delay(delay) < 0.0:
                raise_value_error("Coupling rate neurons with negative delay = %s < 0.0 is not possible!" % str(delay))
        elif self._get_min_delay(delay) < self.spiking_dt:
            raise_value_error("Coupling spiking neurons with delay = %s < NEST integration step = %f is not possible!:"
                              "\n" % (str(delay), self.spiking_dt))
        return delay

    def connect_two_populations(self, source, target, conn_spec, syn_spec):
        syn_spec["synapse_model"] = self._assert_synapse_model(syn_spec.get("synapse_model",
                                                                            syn_spec.get("model", "static_synapse")),
                                                               syn_spec["delay"])
        if syn_spec["synapse_model"] == "rate_connection_instantaneous":
            del syn_spec["delay"]  # For instantaneous rate connections
        else:
            syn_spec["delay"] = self._assert_delay(syn_spec["delay"])
        self.nest_instance.Connect(source, target, conn_spec, syn_spec)

    def build_spiking_population(self, label, model, size, params, *args, **kwargs):
        return NESTPopulation(self.nest_instance.Create(model, int(np.round(size)), params=params),
                              label, model, self.nest_instance)

    def build_spiking_region_node(self, label="", input_node=None, *args, **kwargs):
        return NESTRegionNode(label, input_node, self.nest_instance)

    def build_and_connect_devices(self, devices):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         self._spiking_brain, self.config, nest_instance=self.nest_instance)

    def build(self):
        return NESTNetwork(self.nest_instance, self._spiking_brain,
                           self._output_devices, self._input_devices, config=self.config)
