# -*- coding: utf-8 -*-
from itertools import cycle
from pandas import Series
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.nest_models.region_node import NESTRegionNode
from tvb_nest.nest_models.network import NESTNetwork
from tvb_nest.nest_models.builders.nest_factory import \
    load_nest, compile_modules, create_conn_spec, create_device, connect_device
from tvb_multiscale.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.spiking_models.builders.templates import tvb_weight, tvb_delay
from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class NESTModelBuilder(SpikingModelBuilder):
    config = CONFIGURED
    nest_instance = None
    default_min_spiking_dt = CONFIGURED.nest.NEST_MIN_DT
    default_min_delay = CONFIGURED.nest.NEST_MIN_DT

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED, logger=LOG):
        super(NESTModelBuilder, self).__init__(tvb_simulator, nest_nodes_ids, config, logger)
        # Setting or loading a nest instance:
        if nest_instance is not None:
            self.nest_instance = nest_instance
        else:
            self.nest_instance = load_nest(self.config, self.logger)

        # Setting NEST defaults from config
        self.default_population = {"model": self.config.nest.DEFAULT_MODEL, "scale": 1, "params": {}, "nodes": None}

        self.default_synaptic_weight_scaling = \
            lambda weight, n_cons: self.config.DEFAULT_SPIKING_SYNAPTIC_WEIGHT_SCALING(weight, n_cons)

        self.default_populations_connection = dict(self.config.nest.DEFAULT_CONNECTION)
        self.default_populations_connection["delay"] = self.default_min_delay
        self.default_populations_connection["nodes"] = None

        self.default_nodes_connection = dict(self.config.nest.DEFAULT_CONNECTION)
        self.default_nodes_connection["delay"] = self.default_populations_connection["delay"]
        self.default_nodes_connection.update({"source_nodes": None, "target_nodes": None})

        # When any of the properties params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations = [{"label": "E", "model": self.default_population["model"], "params": {},
                             "scale": 1, "nodes": None}]  # None means "all"

        # When any of the properties weight, delay, receptor_type below
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.populations_connections = \
            [{"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
              "model": self.default_populations_connection["model"],
              "conn_spec": self.default_populations_connection["conn_spec"],
              "weight": 1.0, "delay": self.default_populations_connection["delay"],
              "receptor_type": 0,
              "nodes": None,  # None means "all"
              }]

        # When any of the properties weight, delay, receptor_type below
        # depends on regions, set a handle to a function with
        # arguments (source_region_index=None, target_region_index=None)

        # Between NEST node delays should be at least equal to NEST time resolution
        # Therefore, zero TVB delays will become spiking_dt delays in NEST

        self.nodes_connections = \
            [{"source": "E", "target": "E",
              "model": self.default_nodes_connection["model"],
              "conn_spec": self.default_nodes_connection["conn_spec"],
              "weight": tvb_weight,
              "delay": tvb_delay,
              "receptor_type": 0,
              "source_nodes": None, "target_nodes": None}  # None means "all"
             ]

        # Use these to observe NEST network behavior
        # Labels have to be different
        self.output_devices = [{"model": "spike_detector",
                                "params": self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"],
                                    #           label <- target population
                                    "connections": {"E": "E"}, "nodes": None},  # None means "all"
                                   {"model": "multimeter",
                                    "params": self.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"],
                                    #                     label <- target population
                                    "connections": {"Excitatory": "E"}, "nodes": None},  # None means "all"
                                   ]
        self.output_devices[1]["params"]["interval"] = self.monitor_period
        self.input_devices = []  # use these for possible external stimulation devices

    def _configure_nest_kernel(self):
        self.nest_instance.ResetKernel()  # This will restart NEST!
        self._update_spiking_dt()
        self._update_default_min_delay()
        self.nest_instance.set_verbosity(100)  # don't print all messages from NEST
        self.nest_instance.SetKernelStatus({"resolution": self.spiking_dt, "print_time": True})

    def _confirm_compile_install_nest_models(self, models, modules=[]):
        nest_models = self.nest_instance.Models()
        models = ensure_list(models)
        modules = ensure_list(modules)
        if len(modules) == 0:
            for model in models:
                modules.append("%smodule" % model)  # Assuming default naming for modules as modelmodule
        for model, module in zip(models, cycle(modules)):
            if model not in nest_models:
                try:
                    # Try to install it...
                    self.logger.info("Trying to install module %s..." % module)
                    self.nest_instance.Install(module)
                except:
                    self.logger.info("FAILED! We need to first compile it!")
                    # ...unless we need to first compile it:
                    compile_modules(model, recompile=False, config=self.config)
                    # and now install it...
                    self.logger.info("Installing now module %s..." % module)
                    self.nest_instance.Install(module)
                    self.logger.info("DONE installing module %s!" % module)
                nest_models = self.nest_instance.Models()

    def _configure_populations(self):
        super(NESTModelBuilder, self)._configure_populations()
        self._confirm_compile_install_nest_models(self.models)

    def configure(self):
        self._configure_nest_kernel()
        super(NESTModelBuilder, self).configure()

    @property
    def min_delay(self):
        try:
            return self.nest_instance.GetKernelStatus("min_delay")
        except:
            return self.default_min_delay

    def _prepare_populations_connection_params(self, pop_src, pop_trg, conn_spec, syn_spec):
        return create_conn_spec(n_src=len(pop_src), n_trg=len(pop_trg),
                                src_is_trg=(pop_src == pop_trg), config=self.config, **conn_spec)

    def _assert_synapse_model(self, synapse_model, delay):
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %f != 0.0 is not possible!" % delay)
            elif delay == 0.0 and synapse_model == "rate_connection_delayed":
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %f <= 0.0 is not possible!" % delay)
            elif delay == 0.0:
                return "rate_connection_instantaneous"
            else:
                return "rate_connection_delayed"
        else:
            return synapse_model

    def _assert_delay(self, delay, synapse_model="static_synapse"):
        if synapse_model.find("rate") > -1:
            if synapse_model == "rate_connection_instantaneous" and delay != 0.0:
                raise_value_error("Coupling neurons with rate_connection_instantaneous synapse "
                                  "and delay = %f != 0.0 is not possible!" % delay)
            elif synapse_model == "rate_connection_delayed" and delay <= 0.0:
                raise_value_error("Coupling neurons with rate_connection_delayed synapse "
                                  "and delay = %f <= 0.0 is not possible!" % delay)
            elif delay < 0.0:
                raise_value_error("Coupling rate neurons with negative delay = %f < 0.0 is not possible!" % delay)
        elif delay < self.spiking_dt:
            LOG.warning("Coupling spiking neurons with delay = %f < NEST integration step = %s is not possible!\n"
                        "Setting delay equal to NEST integration step!" % (delay, self.spiking_dt))
            return self.default_min_delay
        else:
            return delay

    def connect_two_populations(self, source, target, conn_spec, syn_spec):
        syn_spec["model"] = self._assert_synapse_model(syn_spec["model"], syn_spec["delay"])
        if syn_spec["model"] == "rate_connection_instantaneous":
            del syn_spec["delay"]  # For instantaneous rate connections
        else:
            syn_spec["delay"] = self._assert_delay(syn_spec["delay"])
        self.nest_instance.Connect(source, target, conn_spec, syn_spec)

    def build_spiking_populations(self, model, size, params, *args, **kwargs):
        return self.nest_instance.Create(model, int(np.round(size)), params=params)

    def build_spiking_region_node(self, label="", input_node=Series(), *args, **kwargs):
        return NESTRegionNode(self.nest_instance, label, input_node)

    def build_and_connect_devices(self, devices):
        return build_and_connect_devices(devices, create_device, connect_device,
                                         self.nodes, self.config, nest_instance=self.nest_instance)

    def build(self):
        return NESTNetwork(self.nest_instance, self.nodes,
                           self._output_devices, self.input_devices, config=self.config)
