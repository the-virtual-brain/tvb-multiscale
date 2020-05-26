# -*- coding: utf-8 -*-

import time
from collections import OrderedDict
import numpy as np
from xarray import DataArray, concat

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import Config, CONFIGURED
from tvb_nest.nest_models.builders.models.ww_deco2014 import WWDeco2014Builder
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_multiscale.examples.paperwork.workflow import Workflow as WorkflowBase
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesRegionX
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, ensure_list


CONFIGURED.NEST_MIN_DT = 0.01

CONFIGURED.DEFAULT_CONNECTION = {"model": "static_synapse", "weight": 1.0, "delay": 0.0, 'receptor_type': 0,
                                 "conn_spec": {"autapses": True, 'multapses': True, 'rule': "all_to_all",
                                               "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

CONFIGURED.NEST_OUTPUT_DEVICES_PARAMS_DEF = \
    {"multimeter": {"withtime": True, "withgid": True, 'record_from': ["V_m"]},
     "voltimeter": {"withtime": True, "withgid": True},
     "spike_detector": {"withgid": True, "withtime": True, 'precise_times': True},
     "spike_multimeter": {"withtime": True, "withgid": True, 'record_from': ["spike"]}}

CONFIGURED.NEST_INPUT_DEVICES_PARAMS_DEF = \
    {"poisson_generator": {},
     "mip_generator": {"p_copy": 0.5, "mother_seed": 0},
     "inhomogeneous_poisson_generator": {"allow_offgrid_times": False}}


class Workflow(WorkflowBase):
    config = Config(separate_by_run=True)

    tvb_nodes_ids = []

    nest_model_builder = WWDeco2014Builder
    nest_nodes_ids = []
    nest_populations_order = 100
    nest_stimulus_rate = 2400.0
    nest_network = None

    interface_builder = RedWWexcIOinhIBuilder
    tvb_to_nest_interface = "rate"
    nest_to_tvb_interface = None
    exclusive_nodes = True
    tvb_nest_model = None

    nest_ts = None
    nest_spikes = None

    def __init__(self, **pse_params):
        self.pse_params = pse_params

    @property
    def general_parameters(self):
        general_parameters = super(Workflow, self).general_parameters
        general_parameters.update({"nest_nodes_ids": self.nest_nodes_ids,
                                   "tvb_to_nest_interface": str(self.tvb_to_nest_interface),
                                   "nest_to_tvb_interface": str(self.nest_to_tvb_interface)})
        return general_parameters

    def write_nest_network(self):
        self.write_group(self.nest_model_builder.params_E, "nest_network/pop_E", "dictionary", False)
        self.write_group(self.nest_model_builder.params_I, "nest_network/pop_I", "dictionary", False)
        self.write_group(self.nest_model_builder.populations,
                         "nest_network/populations", "list_of_dictionaries", False)
        self.write_group(self.nest_model_builder.populations_connections,
                         "nest_network/populations_connections", "list_of_dictionaries", False)
        self.write_group(self.nest_model_builder.nodes_connections,
                         "nest_network/nodes_connections", "list_of_dictionaries", False)
        self.write_group(self.nest_model_builder.output_devices,
                         "nest_network/output_devices", "list_of_dictionaries", False)
        self.write_group(self.nest_model_builder.input_devices,
                         "nest_network/input_devices", "list_of_dictionaries", True)

    def prepare_nest_network(self):

        # Build a NEST network model with the corresponding builder
        # Using all default parameters for this example
        self.nest_model_builder = self.nest_model_builder(self.simulator, self.nest_nodes_ids, config=self.config)

        # ----------------------------------------------------------------------------------------------------------------
        # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
        # ----------------------------------------------------------------------------------------------------------------

        self.nest_model_builder.population_order = self.nest_populations_order

        exc_pop_scale = 1.6
        inh_pop_scale = 0.4
        self.N_E = int(self.nest_model_builder.population_order * exc_pop_scale)
        self.N_I = int(self.nest_model_builder.population_order * inh_pop_scale)

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        G = self.simulator.model.G[0].item() * self.simulator.coupling.a[0].item()
        lamda = self.simulator.model.lamda[0].item()
        w_p = self.simulator.model.w_p[0].item()
        J_i = self.simulator.model.J_i[0].item()

        common_params = {
            "V_th": -50.0,  # mV
            "V_reset": -55.0,  # mV
            "E_L": -70.0,  # mV
            "E_ex": 0.0,  # mV
            "E_in": -70.0,  # mV
            "tau_decay_AMPA": 2.0,  # ms
            "tau_decay_GABA_A": 10.0,  # ms
            "tau_decay_NMDA": 100.0,  # ms
            "tau_rise_NMDA": 2.0,  # ms
            "s_AMPA_ext_max": self.N_E * np.ones((self.nest_model_builder.number_of_nodes,)).astype("f"),
            "epsilon": 1.0,
            "alpha": 0.5,  # kHz
            "beta": 0.062,
            "lambda_NMDA": 0.28,
            "I_e": 0.0  # pA
        }
        self.nest_model_builder.params_E = dict(common_params)
        self.nest_model_builder.params_E.update({
            "C_m": 500.0,  # pF
            "g_L": 25.0,  # nS
            "t_ref": 2.0,  # ms
            "g_AMPA_ext": 3.37,  # nS
            "g_AMPA": 0.065,  # nS
            "g_NMDA": 0.20,  # nS
            "g_GABA_A": 10.94,  # nS
            "w_E": w_p,  # w+ in the paper
            "w_I": J_i,
            "N_E": self.N_E,
            "N_I": self.N_I
        })
        self.nest_model_builder.params_I = dict(common_params)
        self.nest_model_builder.params_I.update({
            "C_m": 200.0,  # pF
            "g_L": 20.0,  # nS
            "t_ref": 1.0,  # ms
            "g_AMPA_ext": 2.59,  # nS
            "g_AMPA": 0.051,  # nS
            "g_NMDA": 0.16,  # nS
            "g_GABA_A": 8.51,  # nS
            "w_E": 1.0,
            "w_I": 1.0,
            "N_E": self.N_E,
            "N_I": self.N_I
        })

        def param_fun(node_index, params, lamda=1.0):
            w_E_ext = lamda * G * \
                      self.nest_model_builder.tvb_weights[:,
                        list(self.nest_model_builder.spiking_nodes_ids).index(node_index)]
            w_E_ext[node_index] = 1.0  # this is external input weight to this node
            out_params = dict(params)
            out_params.update({"w_E_ext": w_E_ext})
            return out_params

        # Populations' configurations
        # When any of the properties model, params and scale below depends on regions,
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.nest_model_builder.populations = [
            {"label": "E", "model": "iaf_cond_deco2014",
             "nodes": None,  # None means "all"
             "params": lambda node_index: param_fun(node_index, self.nest_model_builder.params_E),
             "scale": exc_pop_scale},
            {"label": "I", "model": "iaf_cond_deco2014",
             "nodes": None,  # None means "all"
             "params": lambda node_index: param_fun(node_index, self.nest_model_builder.params_I,
                                                    lamda=lamda),
             "scale": inh_pop_scale}
        ]
        # Just in case we need them:
        self.populations = ["E", "I"]
        self.populations_size = [self.N_E, self.N_I]

        # Within region-node connections
        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # set a handle to a function with
        # arguments (region_index=None) returning the corresponding property
        self.nest_model_builder.populations_connections = [
            #              ->
            {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
             "model": "static_synapse",
             "conn_spec": self.nest_model_builder.config.DEFAULT_CONNECTION["conn_spec"],
             "weight": 1.0,
             "delay": 0.1,
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "E", "target": "I",  # E -> I
             "model": "static_synapse",
             "conn_spec": self.nest_model_builder.config.DEFAULT_CONNECTION["conn_spec"],
             "weight": 1.0,
             "delay": 0.1,
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "E",  # I -> E
             "model": "static_synapse",
             "conn_spec": self.nest_model_builder.config.DEFAULT_CONNECTION["conn_spec"],
             "weight": -1.0,
             "delay": 0.1,
             "receptor_type": 0, "nodes": None},  # None means "all"
            {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
             "model": "static_synapse",
             "conn_spec": self.nest_model_builder.config.DEFAULT_CONNECTION["conn_spec"],
             "weight": -1.0,
             "delay": 0.1,
             "receptor_type": 0, "nodes": None}  # None means "all"
        ]

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type

        # When any of the properties model, conn_spec, weight, delay, receptor_type below
        # depends on regions, set a handle to a function with
        # arguments (source_region_index=None, target_region_index=None)
        self.nest_model_builder.nodes_connections = []
        if len(self.nest_nodes_ids) > 1:
            self.nest_model_builder.nodes_connections = [
                #              ->
                {"source": "E", "target": ["E"],
                 "model": "static_synapse",
                 "conn_spec": self.nest_model_builder.config.DEFAULT_CONNECTION["conn_spec"],
                 #  weight scaling the TVB connectivity weight
                 "weight": 1.0,
                 # additional delay to the one of TVB connectivity
                 "delay": self.nest_model_builder.tvb_delay_fun,
                 # Each region emits spikes in its own port:
                 "receptor_type": self.nest_model_builder.receptor_by_source_region_fun,
                 "source_nodes": None, "target_nodes": None}  # None means "all"
            ]
            if self.nest_model_builder.tvb_model.lamda[0] > 0:
                self.nest_model_builder.nodes_connections[0]["target"] += ["I"]

        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.nest_model_builder.output_devices = []
        connections = OrderedDict({})
        #          label <- target population
        connections["E"] = "E"
        connections["I"] = "I"
        params = dict(self.nest_model_builder.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"])
        self.nest_model_builder.output_devices.append(
            {"model": "spike_detector", "params": params,
             "connections": connections, "nodes": None})  # None means "all"

        connections = OrderedDict({})
        #               label    <- target population
        connections["Excitatory"] = "E"
        connections["Inhibitory"] = "I"
        params = dict(self.nest_model_builder.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["interval"] = self.nest_model_builder.nest_instance.GetKernelStatus("resolution")
        params['record_from'] = ["V_m",
                                 "s_AMPA", "x_NMDA", "s_NMDA", "s_GABA",
                                 "I_AMPA", "I_NMDA", "I_GABA", "I_L", "I_e",
                                 "spikes_exc", "spikes_inh"
                                 ]
        for i_node in range(self.nest_model_builder.number_of_nodes):
            params['record_from'].append("s_AMPA_ext_%d" % i_node)
            params['record_from'].append("I_AMPA_ext_%d" % i_node)
            params['record_from'].append("spikes_exc_ext_%d" % i_node)

        self.nest_model_builder.output_devices.append(
            {"model": "multimeter", "params": params,
             "connections": connections, "nodes": None})  # None means "all"

        if self.nest_stimulus_rate is not None and self.nest_stimulus_rate > 0:
            connections = OrderedDict({})
            #          label <- target population
            connections["Stimulus"] = ["E", "I"]
            self.nest_model_builder.input_devices = [
                {"model": "poisson_generator",
                 "params": {"rate": self.nest_stimulus_rate, "origin": 0.0,
                            "start": 0.1,
                            # "stop": 100.0
                            },
                 "connections": connections, "nodes": None,
                 "weights": 1.0, "delays": 0.0,
                 "receptor_type": lambda target_node_id: int(target_node_id + 1)}
            ]

        # ----------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------

        if self.writer:
            self.write_nest_network()

        self.nest_network = self.nest_model_builder.build_spiking_network()

        return self.nest_network

    def prepare_rate_interface(self, lamda):
        from tvb_multiscale.spiking_models.builders.templates \
            import tvb_delay, receptor_by_source_region
        # For spike transmission from TVB to NEST devices acting as TVB proxy nodes with TVB delays:
        self.interface_builder.tvb_to_spikeNet_interfaces = [
            {"model": "inhomogeneous_poisson_generator",
             "params": {"allow_offgrid_times": False},
             # # ---------Properties potentially set as function handles with args (nest_node_id=None)-------------------------
             "interface_weights": 1.0 * self.N_E,
             # Applied outside NEST for each interface device
             # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
             #   To multiply TVB connectivity weight:
             "weights": 1.0,
             #                                     To add to TVB connectivity delay:
             "delays": lambda tvb_node_id, nest_node_id:
             tvb_delay(tvb_node_id, nest_node_id, self.interface_builder.tvb_delays),
             "receptor_type": lambda tvb_node_id, nest_node_id:
             receptor_by_source_region(tvb_node_id, nest_node_id, start=1),
             # --------------------------------------------------------------------------------------------------------------
             #             TVB sv -> NEST population
             "connections": {"R_e": ["E"]},
             "source_nodes": None, "target_nodes": None}]  # None means all here

        if lamda > 0.0:
            #       Coupling towards the inhibitory population as well:
            self.interface_builder.tvb_to_spikeNet_interfaces[0]["connections"]["R_e"] += ["I"]

        return self.interface_builder

    def prepare_dc_interface(self, G, lamda):
        from tvb_multiscale.spiking_models.builders.templates \
            import random_normal_tvb_weight, tvb_delay

        # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
        self.interface_builder.tvb_to_spikeNet_interfaces = [
            {"model": "dc_generator", "params": {},
             # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
             #   Applied outside NEST for each interface device
             "interface_weights": 1.0,
             # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
             #    To multiply TVB connectivity weight:
             "weights": lambda tvb_node_id, nest_node_id:
             random_normal_tvb_weight(tvb_node_id, nest_node_id,
                                      G * self.interface_builder.tvb_weights,
                                      sigma=0.1),
             #    To add to TVB connectivity delay:
             "delays": lambda tvb_node_id, nest_node_id:
             tvb_delay(tvb_node_id, nest_node_id, self.interface_builder.tvb_delays),
             # ----------------------------------------------------------------------------------------------------------------
             #    TVB sv -> NEST population
             "connections": {"S_e": ["E"]},
             "source_nodes": None, "target_nodes": None}]  # None means all here

        if lamda > 0.0:
            # Coupling to inhibitory populations as well (feedforward inhibition):
            self.interface_builder.tvb_to_spikeNet_interfaces.append(
                {"model": "dc_generator", "params": {},
                 # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
                 #   Applied outside NEST for each interface device
                 "interface_weights": 1.0 * self.N_E / self.N_I,
                 # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                 #    To multiply TVB connectivity weight:
                 "weights": lambda tvb_node_id, nest_node_id:
                 random_normal_tvb_weight(tvb_node_id, nest_node_id,
                                          G * lamda *
                                          self.interface_builder.tvb_weights, sigma=0.1),
                 #    To add to TVB connectivity delay:
                 "delays": lambda tvb_node_id, nest_node_id:
                 tvb_delay(tvb_node_id, nest_node_id, self.interface_builder.tvb_delays),
                 # ----------------------------------------------------------------------------------------------------------------
                 #    TVB sv -> NEST population
                 "connections": {"S_e": ["I"]},
                 "source_nodes": None, "target_nodes": None}
            )

        return self.interface_builder

    def prepare_Ie_interface(self, lamda):
        from tvb_multiscale.spiking_models.builders.templates \
            import random_normal_tvb_weight, tvb_delay

        # For directly setting an external current parameter in NEST neurons instantaneously:
        self.interface_builder.tvb_to_spikeNet_interfaces = [
            {"model": "current", "parameter": "I_e",
             # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
             "interface_weights": 1.0,
             # ----------------------------------------------------------------------------------------------------------------
             #                  TVB sv -> NEST population
             "connections": {"S_e": ["E"]},
             "nodes": None}]  # None means all here
        if self.interface_builder.tvb_model.lamda[0] > 0.0:
            # Coupling to inhibitory populations as well (feedforward inhibition):
            self.interface_builder.tvb_to_spikeNet_interfaces.append(
                {
                    "model": "current", "parameter": "I_e",
                    # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
                    "interface_weights": lamda * self.N_E / self.N_I,
                    # ----------------------------------------------------------------------------------------------------------------
                    #                     TVB sv -> NEST population
                    "connections": {"S_e": ["I"]},
                    "nodes": None}
            )

        return self.interface_builder

    def prepare_nest_to_tvb_interface(self):
        # NEST -> TVB:
        # Use S_e and S_i instead of r_e and r_i
        # for transmitting to the TVB state variables directly
        connections = OrderedDict()
        #            TVB <- NEST
        connections["R_e"] = ["E"]
        connections["R_i"] = ["I"]
        self.interface_builder.spikeNet_to_tvb_interfaces = \
            [{"model": "spike_detector", "params": {},
              # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
              "weights": 1.0, "delays": 0.0,
              # -----------------------------------------------------------------------------------------------------------------
              "connections": connections, "nodes": None}]  # None means all here

    def write_interface(self):
        interface_props = {}
        for prop in ["tvb_nodes_ids", "spiking_nodes_ids",
                     "w_tvb_to_spike_rate", "w_tvb_to_current", "w_tvb_to_potential",
                     "w_spikes_to_tvb", "w_spikes_var_to_tvb", "w_potential_to_tvb"]:
            interface_props[prop] = getattr(self.interface_builder, prop)
        self.write_group(interface_props, "tvb_nest_interface/props", "dictionary", close_file=False)
        self.write_group(self.interface_builder.tvb_to_spikeNet_interfaces,
                         "tvb_nest_interface/tvb_to_nest_interfaces", "list_of_dictionaries", False)
        self.write_group(self.interface_builder.spikeNet_to_tvb_interfaces,
                         "tvb_nest_interface/nest_to_tvb_interfaces", "list_of_dictionaries", True)

    def prepare_interface(self):

        G = self.simulator.model.G[0] * self.simulator.coupling.a[0]
        lamda = self.simulator.model.lamda[0]

        # Build a TVB-NEST interface with all the appropriate connections between the
        # TVB and NEST modelled regions
        # Using all default parameters for this example
        self.interface_builder = self.interface_builder(self.simulator, self.nest_network,
                                                        self.nest_nodes_ids, self.exclusive_nodes, self.N_E)

        # ----------------------------------------------------------------------------------------------------------------
        # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
        # ----------------------------------------------------------------------------------------------------------------

        # TVB -> NEST
        if self.tvb_to_nest_interface == "rate":
            self.interface_builder = self.prepare_rate_interface(lamda)
        elif self.tvb_to_nest_interface == "dc":
            self.interface_builder = self.prepare_dc_interface(G, lamda)
        elif self.tvb_to_nest_interface == "Ie":
            self.interface_builder = self.prepare_Ie_interface(lamda)

        if self.nest_to_tvb_interface is not None:
            self.prepare_nest_to_tvb_interface()

        if self.writer:
            self.write_interface()

        self.tvb_nest_model = self.interface_builder.build_interface(self.tvb_to_nest_interface,
                                                                     self.nest_to_tvb_interface)

        return self.tvb_nest_model

    def get_nest_data(self):

        self.nest_ts = \
            TimeSeriesRegionX(
                self.nest_network.get_data_from_multimeter(mode="per_neuron",
                                                           populations_devices=["Excitatory", "Inhibitory"]),
                connectivity=self.connectivity)

        if self.transient:
            self.nest_ts = self.nest_ts[self.transient:]
            exclude_times = [0.0, self.transient]
        else:
            exclude_times = None

        self.nest_spikes = self.nest_network.get_spikes(mode="events", return_type="Series",
                                                        exclude_times=exclude_times)

        return self.nest_ts, self.nest_spikes

    def cosimulate(self, cleanup=True):

        # Configure the simulator with the TVB-NEST interface...
        self.simulator.configure(self.tvb_nest_model)

        # ...and simulate!
        results = self.simulator.run(simulation_length=self.simulation_length)

        if self.simulator.tvb_spikeNet_interface is not None:
            # Integrate NEST one more NEST time step so that multimeters get the last time point
            # unless you plan to continue simulation later
            self.simulator.run_spiking_simulator(
                self.simulator.tvb_spikeNet_interface.nest_instance.GetKernelStatus("resolution"))
            # Clean-up NEST simulation
            self.simulator.tvb_spikeNet_interface.nest_instance.Cleanup()

        self.tvb_ts = TimeSeriesRegionX(results[0][1], time=results[0][0],
                                       connectivity=self.simulator.connectivity,
                                       labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                                       labels_dimensions={
                                          "State Variable": ensure_list(self.simulator.model.state_variables),
                                          "Region": self.simulator.connectivity.region_labels.tolist()},
                                       sample_period=self.simulator.integrator.dt)
        if self.transient:
            self.tvb_ts = self.tvb_ts[self.transient:]
        if self.writer:
            self.write_ts(self.tvb_ts, "TVB_TimeSeries", recursive=True)

        if self.tvb_nest_model is not None:
            self.nest_ts, self.nest_spikes = self.get_nest_data()
            if self.writer:
                self.write_ts(self.nest_ts, "NEST_TimeSeries", recursive=True)
                self.write_object(self.nest_spikes.to_dict(), "NEST_Spikes")

        return self.tvb_ts, self.nest_ts, self.nest_spikes

    def simulate_nest(self, cleanup=True):
        self.nest_network.configure()
        self.nest_network.Run(self.simulation_length)
        self.nest_ts, self.nest_spikes = self.get_nest_data()
        if cleanup:
            self.nest_network.nest_instance.Cleanup()
        if self.writer:
            self.write_ts(self.nest_ts, "NEST_TimeSeries", recursive=True)
            self.write_object(self.nest_spikes.to_dict(), "NEST_Spikes")
        return self.nest_ts, self.nest_spikes

    def get_nest_rates(self):
        rates = []
        pop_labels = []
        for i_pop, (pop_label, pop_spikes) in enumerate(self.nest_spikes.iteritems()):
            pop_labels.append(pop_label)
            rates.append([])
            reg_labels = []
            for reg_label, reg_spikes in pop_spikes.iteritems():
                reg_labels.append(reg_label)
                # rates (spikes/sec) =
                #   total_number_of_spikes (int) / total_time_duration (sec) / total_number_of_neurons_in_pop (int)
                rates[-1].append(len(reg_spikes["times"]) / self.duration / self.populations_size[i_pop])

        self.rates["NEST"] = DataArray(np.array(rates),
                                       dims=["Population", "Region"],
                                       coords={"Population": pop_labels, "Region": reg_labels})

        return self.rates["NEST"]

    def plot_tvb_ts(self):
        # For timeseries plot:
        self.tvb_ts.plot_timeseries(plotter=self.plotter, per_variable=True, figsize=self.figsize, add_legend=False)

        # For raster plot:
        if self.number_of_regions > 9:
            self.tvb_ts.plot_raster(plotter=self.plotter, per_variable=True, figsize=self.figsize, add_legend=False)

        spiking_nodes_ids = []
        if self.exclusive_nodes:
            spiking_nodes_ids = self.nest_nodes_ids

        n_spiking_nodes_ids = len(spiking_nodes_ids)
        if n_spiking_nodes_ids > 0:
            self.tvb_ts[:, :, spiking_nodes_ids].plot_timeseries(plotter=self.plotter, per_variable=True,
                                                                figsize=self.figsize,
                                                                figname="Spiking nodes TVB Time Series")
            if n_spiking_nodes_ids > 3:
                self.tvb_ts[:, :, spiking_nodes_ids].plot_raster(plotter=self.plotter, per_variable=True,
                                                                 figsize=self.figsize,
                                                                 figname="Spiking nodes TVB Time Series Raster")

    def compute_nest_mean_field(self):
        mean_field = TimeSeriesRegionX(self.nest_ts._data.mean(axis=-1), connectivity=self.nest_ts.connectivity,
                                       title="Mean field spiking nodes time series")
        # We place here all variables that relate to local excitatory synapses
        mean_field_exc = mean_field[:, ["spikes_exc", "s_AMPA", "I_AMPA", "x_NMDA", "s_NMDA", "I_NMDA"]]
        mean_field_exc._data.name = "Mean excitatory synapse data from NEST multimeter"

        # We place here all variables that relate to local inhibitory synapses
        mean_field_inh = mean_field[:, ["spikes_inh", "s_GABA", "I_GABA"]]
        mean_field_inh._data.name = "Mean inhibitory synapse data from NEST multimeter"

        # TODO: deal specifically with external input of node I to synapse I

        # Substitute the per-region synaptic variables with their sums to reduce outpumean_field:
        s_AMPA_ext_nest_nodes = \
            mean_field[:, ['s_AMPA_ext_%d' % node_id for node_id in [0] + self.nest_nodes_ids]]._data
        I_AMPA_ext_nest_nodes = \
            mean_field[:, ['I_AMPA_ext_%d' % node_id for node_id in [0] + self.nest_nodes_ids]]._data
        spikes_exc_ext_nest_nodes = mean_field[:,
                                    ['spikes_exc_ext_%d' % node_id for node_id in [0] + self.nest_nodes_ids]]._data

        s_AMPA_ext_tot = \
            mean_field[:, ['s_AMPA_ext_%d' % node_id for node_id in range(self.number_of_regions)]]. \
                _data.sum(axis=1).expand_dims(axis=1, dim={"Variable": ["s_AMPA_ext_tot"]})
        I_AMPA_ext_tot = \
            mean_field[:, ['I_AMPA_ext_%d' % node_id for node_id in range(self.number_of_regions)]]. \
                _data.sum(axis=1).expand_dims(axis=1, dim={"Variable": ["I_AMPA_ext_tot"]})
        spikes_exc_ext_tot = \
            mean_field[:, ['spikes_exc_ext_%d' % node_id for node_id in range(self.number_of_regions)]]. \
                _data.sum(axis=1).expand_dims(axis=1, dim={"Variable": ["spikes_exc_ext_tot"]})

        # We place here all variables that relate to large-scale excitatory synapses
        mean_field_ext = TimeSeriesRegionX(
            concat([spikes_exc_ext_tot, s_AMPA_ext_tot, I_AMPA_ext_tot,
                    spikes_exc_ext_nest_nodes, s_AMPA_ext_nest_nodes, I_AMPA_ext_nest_nodes], "Variable"),
            connectivity=mean_field.connectivity)
        mean_field_ext._data.name = "Mean external synapse data from NEST multimeter"

        # We place here all variables that refer to all neurons of all populations the same
        mean_field_neuron = mean_field[:, ["I_e", "V_m", "I_L"]]

        return mean_field, mean_field_ext, mean_field_exc, mean_field_inh, mean_field_neuron

    def plot_nest_ts(self):
        mean_field, mean_field_ext, mean_field_exc, mean_field_inh, mean_field_neuron = self.compute_nest_mean_field()

        mean_field_ext.plot_timeseries(plotter=self.plotter, per_variable=True, figsize=self.figsize)
        # Then plot the local excitatory...:
        mean_field_exc.plot_timeseries(plotter=self.plotter, per_variable=True, figsize=self.figsize)
        # ...and local inhibtiory synaptic activity that result:
        mean_field_inh.plot_timeseries(plotter=self.plotter, per_variable=True, figsize=self.figsize)
        # ...and finally the common neuronal variables:
        mean_field_neuron.plot_timeseries(plotter=self.plotter, per_variable=True, figsize=self.figsize)

        spiking_nodes_ids = []
        if self.exclusive_nodes:
            spiking_nodes_ids = self.nest_nodes_ids

        n_spiking_nodes_ids = len(spiking_nodes_ids)
        if n_spiking_nodes_ids > 3:
            mean_field_ext.plot_raster(plotter=self.plotter, per_variable=True, figsize=self.figsize)
            # Then plot the local excitatory...:
            mean_field_exc.plot_raster(plotter=self.plotter, per_variable=True, figsize=self.figsize)
            # ...and local inhibtiory synaptic activity that result:
            mean_field_inh.plot_raster(plotter=self.plotter, per_variable=True, figsize=self.figsize)
            # ...and finally the common neuronal variables:
            mean_field_neuron.plot_raster(plotter=self.plotter, per_variable=True, figsize=self.figsize)
        else:
            dims = self.nest_ts.labels_ordering
            self.nest_ts.plot_map(y=dims[4], row=dims[2], col=dims[3], per_variable=True,
                                  cmap="jet", figsize=self.figsize, plotter=self.plotter)

        self.plotter.plot_spike_events(self.nest_spikes)

    def run(self, **model_params):
        self.model_params = model_params
        if self.writer:
           self.write_general_params(close_file=False)
           if len(self.model_params) > 0:
               self.write_model_params(close_file=True)

        # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------
        print("Preparing TVB simulator...")
        self.prepare_simulator()

        self.tvb_nodes_ids = list(range(self.simulator.connectivity.number_of_regions))

        # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

        if len(self.nest_nodes_ids) > 0:
            for ind in self.nest_nodes_ids:
                self.tvb_nodes_ids.remove(ind)
            tic = time.time()
            print("Building NEST network...")
            self.prepare_nest_network()
            print("Done! in %f min" % ((time.time() - tic) / 60))

        # Not considering here the case of TVB spiking/multiscale models:
        self.mf_nodes_ids = self.tvb_nodes_ids
        self.spiking_regions_ids = self.nest_nodes_ids

        # -----------------------------------3. Build the TVB-NEST interface model -----------------------------------------
        if self.tvb_to_nest_interface is not None and self.nest_network is not None:
            print("Building TVB-NEST interface...")
            tic = time.time()
            self.prepare_interface()
            print("Done! in %f min" % ((time.time() - tic) / 60))
        else:
            self.tvb_nest_model = None

        # -----------------------------------4. Simulate and gather results-------------------------------------------------
        t_start = time.time()
        if self.tvb_nest_model is not None or len(self.tvb_nodes_ids) != 0:
            self.cosimulate()

        else:
            self.simulate_nest()
        print("\nSimulated in %f secs!" % (time.time() - t_start))

        self.duration = (self.simulation_length - self.transient) / 1000  # make it seconds

        # -----------------------------------5. Compute rate per region and population--------------------------------------
        self.rates = {}
        if self.nest_spikes is not None:
            self.rates["NEST"] = self.get_nest_rates()
            if self.writer:
                self.write_object(self.rates["NEST"].to_dict(), "NEST_rates")
        if self.tvb_ts is not None:
            self.rates["TVB"] = self.get_tvb_rates()
            if self.writer:
                self.write_object(self.rates["TVB"].to_dict(), "TVB_rates")

        # -------------------------------------------5. Plot results--------------------------------------------------------
        if self.plotter:
            if self.tvb_ts is not None:
                self.plot_tvb_ts()
            if self.nest_ts is not None:
                self.plot_nest_ts()

        return self.rates
