# -*- coding: utf-8 -*-
import time
from collections import OrderedDict
import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.examples.plot_results import plot_results
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_tvb.simulator import Simulator
from tvb_nest.interfaces.builders.models.red_ww_exc_io_inh_i import RedWWexcIOinhIBuilder
from tvb_nest.simulator_nest.builders.models.red_ww_exc_io_inh_i import RedWWExcIOInhIBuilder
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_nest.plot.plotter import Plotter
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


if __name__ == "__main__":

    config = CONFIGURED
    plotter = Plotter(config)

    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()
    plotter.plot_tvb_connectivity(connectivity)

    # ----------------------2. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.model = ReducedWongWangExcIOInhI()

    # Synaptic gating state variables S_e, S_i need to be in the interval [0, 1]
    simulator.connectivity = connectivity
    # TODO: Try to make this part of the __init__ of the Simulator!
    simulator.integrator.dt = \
        float(int(np.round(simulator.integrator.dt / config.nest.NEST_MIN_DT))) * config.nest.NEST_MIN_DT
    simulator.integrator.noise.nsig = np.array([0.001])
    # Some extra monitors for neuroimaging measures:
    mon_raw = Raw(period=simulator.integrator.dt)
    # mon_bold = Bold(period=2000.)
    # mon_eeg = EEG(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)  # mon_bold, mon_eeg

    # ------3. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)-------

    # Select the regions for the fine scale modeling with NEST spiking networks
    number_of_regions = simulator.connectivity.region_labels.shape[0]
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    for id in range(number_of_regions):
        if simulator.connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)


    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = \
        RedWWExcIOInhIBuilder(simulator, nest_nodes_ids, config=config)
    # Common order of neurons' number per population:
    nest_model_builder.populations_order = 100

    # or...
    #
    # # ----------------------------------------------------------------------------------------------------
    # # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------
    # V_th = -50.0,  # mV
    # V_reset = -55.0,  # mV
    # E_L = -70.0,  # mV
    # # exc neurons (AMPA,rec/ext, NMDA)
    # C_m_ex = 500.0,  # pF
    # g_L_ex = 25.0,  # nS
    # t_ref_ex = 2.0,  # ms
    # # inh neurons (GABA):
    # C_m_in = 200.0,  # pF
    # g_L_in = 20.0,  # nS
    # t_ref_in = 1.0,  # ms
    # # exc spikes (AMPA,rec/ext, NMDA):
    # E_ex = 0.0,  # mV
    # tau_decay_ex = 100.0,  # maximum(AMPA,rec, NMDA) = maximum(2.0, 100.0) ms
    # tau_rise_ex = 2.0,  # tau_rise_NMDA = 2.0 ms
    # # ext, exc spikes(AMPA, ext):
    # # inh spikes (GABA):
    # E_in = -70.0,  # mV
    # tau_decay_in = 10.0,  # tau_GABA = 10.0 ms
    # tau_rise_in = 1.0  # assuming tau_rise_GABA = 1.0 ms
    #
    # # Populations' configurations
    # # When any of the properties model, params and scale below depends on regions,
    # # set a handle to a function with
    # # arguments (region_index=None) returning the corresponding property
    #
    # common_params = {
    #     "V_th": V_th, "V_reset": V_reset, "E_L": E_L,
    #     "E_ex": E_ex, "E_in": E_in,
    #     "tau_rise_ex": tau_rise_ex, "tau_rise_in": tau_rise_in,
    #     "tau_decay_ex": tau_decay_ex, "tau_decay_in": tau_decay_in,
    # }
    #
    # nest_model_builder.params_ex = dict(common_params)
    # nest_model_builder.params_ex.update({
    #     "C_m": C_m_ex, "g_L": g_L_ex, "t_ref": t_ref_ex,
    # })
    # nest_model_builder.params_in = dict(common_params)
    # nest_model_builder.params_in.update({
    #     "C_m": C_m_in, "g_L": g_L_in, "t_ref": t_ref_in,
    # })
    # nest_model_builder.populations = \
    #     [{"label": "E", "model": nest_model_builder.default_population["model"],
    #       "nodes": None,  # None means "all"
    #        "params": nest_model_builder.params_ex,
    #        "scale": 1.0},
    #       {"label": "I", "model": nest_model_builder.default_population["model"],
    #        "nodes": None,  # None means "all"
    #        "params": nest_model_builder.params_in,
    #        "scale": 0.7}
    #     ]
    #
    # # Within region-node connections
    # # When any of the properties model, conn_spec, weight, delay, receptor_type below
    # # set a handle to a function with
    # # arguments (region_index=None) returning the corresponding property
    # nest_model_builder.populations_connections = [
    #     {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "conn_spec": nest_model_builder.default_populations_connection["conn_spec"],
    #      "weight": nest_model_builder.tvb_model.w_p[0],
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "E", "target": "I",  # E -> I
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "conn_spec": nest_model_builder.default_populations_connection["conn_spec"],
    #      "weight": 1.0,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "I", "target": "E",  # I -> E
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "conn_spec": nest_model_builder.default_populations_connection["conn_spec"],
    #      "weight": -nest_model_builder.tvb_model.J_i[0],
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "I", "target": "I",  # I -> I This is a self-connection for population "I"
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "conn_spec": nest_model_builder.default_populations_connection["conn_spec"],
    #      "weight": -1.0,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None}  # None means "all"
    # ]
    # # Among/Between region-node connections
    # # Given that only the AMPA population of one region-node couples to
    # # all populations of another region-node,
    # # we need only one connection type
    # nest_model_builder.nodes_connections = [
    #     {"source": "E", "target": ["E", "I"],
    #      "model": nest_model_builder.default_nodes_connection["model"],
    #      "conn_spec": nest_model_builder.default_nodes_connection["conn_spec"],
    #      "weight": 100 * nest_model_builder.tvb_simulator.model.G[0],  # weight scaling the TVB connectivity weight
    #      "delay": nest_model_builder.default_nodes_connection["delay"],  # additional delay to the one of TVB connectivity
    #      # Each region emits spikes in its own port:
    #      "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
    # ]
    #
    # # Creating  devices to be able to observe NEST activity:
    # # Labels have to be different
    # nest_model_builder.output_devices = []
    # connections = OrderedDict({})
    # #          label <- target population
    # connections["E"] = "E"
    # connections["I"] = "I"
    # nest_model_builder.output_devices.append({"model": "spike_detector", "params": {},
    #                                           "connections": connections, "nodes": None})  # None means all here
    # connections = OrderedDict({})
    # connections["Excitatory"] = "E"
    # connections["Inhibitory"] = "I"
    # params = dict(nest_model_builder.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
    # params["interval"] = nest_model_builder.monitor_period
    # nest_model_builder.output_devices.append({"model": "multimeter", "params": params,
    #                                           "connections": connections, "nodes": None})  # None means all here
    #
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------

    nest_network = nest_model_builder.build_nest_network()
    N_e = int(nest_model_builder.populations[0]["scale"] * nest_model_builder.populations_order)

    # -----------------------------------4. Build the TVB-NEST interface model -----------------------------------------

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = \
        RedWWexcIOinhIBuilder(simulator, nest_network, nest_nodes_ids, N_e=N_e, exclusive_nodes=True)

    # or...

    # # ----------------------------------------------------------------------------------------------------------------
    # # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------

    # # For directly setting an external current parameter in NEST neurons instantaneously:
    # tvb_nest_builder.tvb_to_nest_interfaces = [{"model": "current",  "parameter": "I_e",
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    #                                    "interface_weights": 5.0,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                               TVB sv -> NEST population
    #                                    "connections": {"S_e": ["E", "I"]},
    #                                    "nodes": None}]  # None means all here
    #
    # # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    # tvb_nest_builder.tvb_to_nest_interfaces = [{"model": "dc_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                                    "interface_weights": 1.0,  # Applied outside NEST for each interface device
    #                                    "weights": 50 * tvb_nest_builder.tvb_model.G[0],  # To multiply TVB connectivity weight
    # #                                 To add to TVB connectivity delay:
    # #                                   "delays": nest_network.nodes_min_delay,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                                 TVB sv -> NEST population
    #                                    "connections": {"S_e": ["E", "I"]},
    #                                    "source_nodes": None, "target_nodes": None}]  # None means all here

    # # For spike transmission from TVB to NEST devices acting as TVB proxy nodes with TVB delays:
    # # Options:
    # # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
    # # For spike trains with correlation probability p_copy set:
    # # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
    # # An alternative option to poisson_generator is:
    # # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
    # tvb_nest_builder.tvb_to_nest_interfaces =
    #                           [{"model": "inhomogeneous_poisson_generator",
    #                             "params": {"allow_offgrid_times": False},
    #                            # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                            "interface_weights": 1.0 * N_e,  # Applied outside NEST for each interface device
    #                            "weights": tvb_nest_builder.tvb_model.G[0],  # To multiply TVB connectivity weight
    #                            #                                 To add to TVB connectivity delay:
    #                            "delays": nest_network.nodes_min_delay,
    #                            "receptor_types": 0,
    #                            # ----------------------------------------------------------------------------------------------------------------
    #                            #                                        TVB sv or param -> NEST population
    #                            "connections": {"R_e": ["E", "I"]},
    #                            "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # # NEST -> TVB:
    # # Use S_e and S_i instead of r_e and r_i
    # # for transmitting to the TVB state variables directly
    # connections = OrderedDict()
    # #            TVB <- NEST
    # connections["R_e"] = ["E"]
    # connections["R_i"] = ["I"]
    # nest_to_tvb_interfaces = \
    #     [{"model": "spike_detector", "params": {},
    #       # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
    #       "weights": 1.0, "delays": 0.0,
    #       # ------------------------------------------------------------------------------------------------------------------
    #       "connections": connections, "nodes": None}]  # None means all here
    #
    # tvb_nest_builder.w_tvb_to_current = 1000 * tvb_nest_builder.tvb_model.J_N[0]  # (nA of TVB -> pA of NEST)
    # # WongWang model parameter r is in Hz, just like poisson_generator assumes in NEST:
    # tvb_nest_builder.w_tvb_to_spike_rate = 1.0
    # # We return from a NEST spike_detector the ratio number_of_population_spikes / number_of_population_neurons
    # # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
    # # as long as a neuron cannot fire twice during a TVB time step, i.e.,
    # # as long as the TVB time step (usually 0.001 to 0.1 ms)
    # # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
    # # For conversion to a rate, one has to do:
    # # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
    # # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
    # # given WongWang model parameter r is in Hz but tvb dt is in ms:
    # tvb_nest_builder.w_spikes_to_tvb = 1000.0 / tvb_nest_builder.tvb_dt
    #
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------

    tvb_nest_model = tvb_nest_builder.build_interface()

    # -----------------------------------5. Simulate and gather results------- -----------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=100.0)
    # Integrate NEST one more NEST time step so that multimeters get the last time point
    # unless you plan to continue simulation later
    simulator.run_spiking_simulator(simulator.tvb_nest_interface.nest_instance.GetKernelStatus("resolution"))
    # Clean-up NEST simulation
    if simulator.run_spiking_simulator == simulator.tvb_nest_interface.nest_instance.Run:
        simulator.tvb_nest_interface.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------6. Plot results--------------------------------------------------------

    plot_results(results, simulator, tvb_nest_model,
                 tvb_state_variable_type_label="Synaptic Gating Variable",
                 tvb_state_variables_labels=simulator.model.variables_of_interest,
                 plotter=plotter)
