"""
File: tuning_tvb_nest_2023.py
Author: Alice Geminiani 
Email: alice.geminiani@unipv.it
Date: oct 2023
Description: script for tuning TVB to NEST cosim interface
"""

from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_nest_script import *
from tvb_multiscale.core.plot.plotter import Plotter
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import *        #build_NEST_network, plot_nest_results

model_params = {'STIMULUS': 0.0, 'G': 6}        # Tuning is done at baseline

tuned_values_tvb_nest = [0.01]    #[0.05]          #[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

#tuned_values_nest_tvb = [100, 150]

#tuned_value_nest_tvb = 50   # 0.1
#tuned_value_tvb_nest = 0.65          # 0.8 

RMSEs = []
for tuned_value_tvb_nest in tuned_values_tvb_nest:
    # Get configuration
    config, plotter = configure(output_folder='nest_tvb_'+str(tuned_value_tvb_nest)+'_', verbose=2)
    print("config.NEST_PERIPHERY",config.NEST_PERIPHERY)
    config.model_params.update(model_params)
    config.SIMULATION_LENGTH = 30000 #20000 30000
    config.TRANSIENT_RATIO = 0.25
    config.MOSSY_MAX_RATE = 100
    config.w_TVB_to_NEST = tuned_value_tvb_nest
    
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)

    
    # Build TVB-NEST interfaces
    nest_network, nest_nodes_inds, neuron_models, neuron_number = build_NEST_network(config)
    simulator, nest_network = build_tvb_nest_interfaces(simulator, nest_network, nest_nodes_inds, config)
    # Simulate TVB-NEST model
    results, transient, simulator, nest_network = simulate_tvb_nest(simulator, nest_network, config)
            


    # Compute error
    
    # Get spike events from NEST spike recorders
    events = nest_network.output_devices['mossy_fibers']['Right Ansiform lobule'].get_events()

    # Compute approximate average rate of mossy fibers as:
    # number_of_spikes / (number_of_neurons * time_length_in_ms) * 1000 (to convert to spikes/sec)
    duration = config.SIMULATION_LENGTH - transient
    n_spikes = np.sum(events['times'] > events['times'][-1] - duration)
    print("Approximate mossy_fibers rate during the last %g ms = %g" % 
        (duration, n_spikes / 
                    (nest_network.output_devices['mossy_fibers']['Right Ansiform lobule'].number_of_neurons 
                    * duration) * 1000))
    rate = n_spikes / (nest_network.output_devices['mossy_fibers']['Right Ansiform lobule'].number_of_neurons * duration) * 1000
    RMSEs.append(abs(rate-3.9))
    
print("RMSE for TVB-NEST gains ", tuned_values_tvb_nest, " are ", RMSEs)