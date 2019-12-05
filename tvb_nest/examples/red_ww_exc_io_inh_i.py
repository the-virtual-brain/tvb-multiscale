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
    nest_model_builder.populations_order = 100

    # or...
    #
    # # ----------------------------------------------------------------------------------------------------
    # # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------
    #
    # # Connection weights between the distinct populations:
    # # Inhibition to excitation feedback inhibition
    # # that could result from Feedback Inhibition Control
    # # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
    # nest_model_builder.w_ee = 1.4
    # nest_model_builder.J_i = 1.0
    #
    # # Common order of neurons' number per population:
    # nest_model_builder.populations_order = 100
    # # Populations' configurations
    # nest_model_builder.populations = [
    #     {"label": "E", "model": nest_model_builder.default_population["model"],
    #      "params": {}, "scale": 1, "nodes": None},  # None means "all"
    #     {"label": "I", "model": nest_model_builder.default_population["model"],
    #      "params": {}, "scale": 0.7, "nodes": None}  # None means "all"
    #                                  ]
    #
    # # Within region-node connections
    # nest_model_builder.populations_connections = [
    #     {"source": "E", "target": "E",  # # E -> E This is a self-connection for population "E"
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "params": nest_model_builder.default_populations_connection["params"],
    #      "weight": nest_model_builder.w_ee,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "E", "target": "I",  # E -> I
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "params": nest_model_builder.default_populations_connection["params"],
    #      "weight": 1.0,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "I", "target": "E",  # I -> E
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "params": nest_model_builder.default_populations_connection["params"],
    #      "weight": -nest_model_builder.J_i,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #     {"source": "I", "target": "I",  # I -> I This is a nmbd-connection for population "I"
    #      "model": nest_model_builder.default_populations_connection["model"],
    #      "params": nest_model_builder.default_populations_connection["params"],
    #      "weight": -1.0,
    #      "delay": nest_model_builder.default_populations_connection["delay"],
    #      "receptor_type": 0, "nodes": None},  # None means "all"
    #                                             ]
    #
    # # Among/Between region-node connections
    # # Given that only the AMPA population of one region-node couples to
    # # all populations of another region-node,
    # # we need only one connection type
    # nest_model_builder.nodes_connections = [
    #     {"source": "E", "target": ["E", "I"],
    #      "model": nest_model_builder.default_nodes_connection["model"],
    #      "params": nest_model_builder.default_nodes_connection["params"],
    #      "weight": 1.0,  # weight scaling the TVB connectivity weight
    #      # additional delay to the one of TVB connectivity:
    #      "delay": nest_model_builder.default_nodes_connection["delay"],
    #      # Each region emits spikes in its own port:
    #      "receptor_type": 0, "source_nodes": None, "target_nodes": None}  # None means "all"
    #                                         ]
    #
    # # Creating  devices to be able to observe NEST activity:
    # # Labels have to be different
    # nest_model_builder.output_devices = []
    # connections = OrderedDict({})
    # #          label <- target population
    # connections["E"] = "E"
    # connections["I"] = "I"
    # nest_model_builder.output_devices.append(
    #     {"model": "spike_detector", "params": {},
    #      "connections": connections, "nodes": None})  # None means all here
    # connections = OrderedDict({})
    # connections["Excitatory"] = "E"
    # connections["Inhibitory"] = "I"
    # params = dict(nest_model_builder.config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
    # params["interval"] = nest_model_builder.monitor_period
    # nest_model_builder.output_devices.append(
    #     {"model": "multimeter", "params": params,
    #      "connections": connections, "nodes": None})  # None means all here
    #
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------

    nest_network = nest_model_builder.build_nest_network()

    # -----------------------------------4. Build the TVB-NEST interface model -----------------------------------------

    # Build a TVB-NEST interface with all the appropriate connections between the
    # TVB and NEST modelled regions
    # Using all default parameters for this example
    tvb_nest_builder = \
        RedWWexcIOinhIBuilder(simulator, nest_network, nest_nodes_ids, exclusive_nodes=True)

    # or...

    # # ----------------------------------------------------------------------------------------------------------------
    # # ----Uncomment below to modify the builder by changing the default options:--------------------------------------
    # # ----------------------------------------------------------------------------------------------------------------

    # # TVB -> NEST:
    # # ------------Modifications to the default options of the builder---------------

    # # For directly setting an external current parameter in NEST neurons instantaneously:
    # tvb_nest_builder.tvb_to_nest_interfaces = [{"model": "current",  "parameter": "I_e",
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    #                                             "interface_weights": 1.0,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                                 TVB state var -> NEST population
    #                                             "connections": {"S_e": ["E", "I"]},
    #                                             "nodes": None}]  # None means all here

    # # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    # tvb_nest_builder.tvb_to_nest_interfaces =
    #     [{"model": "dc_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #       "interface_weights": 1.0,  # Applied outside NEST for each interface device
    #       "weights": 1.0,  # To multiply TVB connectivity weight
    # #             To add to TVB connectivity delay:
    #       "delays": nest_network.nodes_min_delay,
    # # ----------------------------------------------------------------------------------------------------------------
    # #             TVB state var -> NEST population
    #       "connections": {"S_e": ["E", "I"]},
    #       "source_nodes": None, "target_nodes": None}]  # None means all here

    #  # For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    # tvb_nest_builder.tvb_to_nest_interfaces = \
    #      [{"model": "poisson_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #        "interface_weights": 100.0,  # Applied outside NEST for each interface device
    #        "weights": 1.0,  # To multiply TVB connectivity weight
    # #          To add to TVB connectivity delay:
    #        "delays": nest_network.nodes_min_delay,
    #        "receptor_types": 0,
    # # ----------------------------------------------------------------------------------------------------------------
    # #    TVB state var or param -> NEST population
    #        "connections": {"R_e": ["E", "I"]},
    #        "source_nodes": None, "target_nodes": None}]  # None means all here

    # # NEST -> TVB:
    # # Use S_e and S_i instead of r_e and r_i
    # # for transmitting to the TVB state variables directly
    # connections = OrderedDict()
    # #            TVB <- NEST
    # connections["R_e"] = ["E"]
    # connections["R_i"] = ["I"]
    # tvb_nest_builder.nest_to_tvb_interfaces = \
    #     [{"model": "spike_detector", "params": {},
    # # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
    #       "weights": 1.0, "delays": 0.0,
    # # ------------------------------------------------------------------------------------------------------------------
    #       "connections": connections, "nodes": None}]  # None means all here

    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    tvb_nest_model = tvb_nest_builder.build_interface()

    # -----------------------------------5. Simulate and gather results------- -----------------------------------------

    # Configure the simulator with the TVB-NEST interface...
    simulator.configure(tvb_nest_interface=tvb_nest_model)
    # ...and simulate!
    t_start = time.time()
    results = simulator.run(simulation_length=100.0)
    # Integrate NEST one more NEST time step so that multimeters get the last time point
    # unless you plan to continue simulation later
    simulator.simulate_nest(simulator.tvb_nest_interface.nest_instance.GetKernelStatus("resolution"))
    # Clean-up NEST simulation
    if simulator.simulate_nest == simulator.tvb_nest_interface.nest_instance.Run:
        simulator.tvb_nest_interface.nest_instance.Cleanup()
    print("\nSimulated in %f secs!" % (time.time() - t_start))

    # -------------------------------------------6. Plot results--------------------------------------------------------

    plot_results(results, simulator, tvb_nest_model,
                 tvb_state_variable_type_label="Synaptic Gating Variable",
                 tvb_state_variables_labels=simulator.model.variables_of_interest,
                 plotter=plotter)
