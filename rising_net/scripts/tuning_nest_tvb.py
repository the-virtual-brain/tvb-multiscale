"""
File: tuning_nest_tvb_2023.py
Author: Alice Geminiani 
Email: alice.geminiani@unipv.it
Date: oct 2023
Description: script for tuning NEST to TVB cosim interface
"""

from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_nest_script import *
from tvb_multiscale.core.plot.plotter import Plotter
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import *        #build_NEST_network, plot_nest_results
import math

model_params = {'STIMULUS': 0.0, 'G': 6}        # Tuning is done at baseline


factor_tvb_nest = 0.07*100          # product of w_tvb_nest*max_mossy as tuned for tvb to nest

tuned_values_mossy_max = [120]

regs = ['ansilob', 'cereb_nuclei', 'oliv'] 

#tuned_value_nest_tvb = 50   # 0.1
#tuned_value_tvb_nest = 0.65          # 0.8 
COMPUTE_REF = False          # True if you want to run TVB-only

source_ts_interface_reference = {'ansilob':-0.2888, 'cereb_nuclei':-0.4297, 'oliv':-0.4914}
source_ts_interface = {}

if COMPUTE_REF:
    # Get configuration
    config, plotter = configure(output_folder='nest_tvb_reference', verbose=2)
    print("config.NEST_PERIPHERY",config.NEST_PERIPHERY)
    config.model_params.update(model_params)
    config.SIMULATION_LENGTH = 30000 #20000 30000
    config.TRANSIENT_RATIO = 0.25
        
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    
    # Run simulation and get results for reference values
    results, transient = simulate(simulator, config)
    
    # Get reference
    source_ts = results[0]
    
    for reg in regs:
        source_ts_interface_reference[reg] = source_ts[1][-config.SIMULATION_LENGTH:,0, inds[reg][1]]
        #print("source ts shape ", source_ts_interface[reg])
        #print("source ts parts", source_ts_interface[reg].Time, source_ts_interface[reg].values)
        print("Avg baseline for ", reg, np.mean(source_ts_interface_reference[reg]))
else:
    RMSEs = []
    for tuned_value_mossy_max in tuned_values_mossy_max:
        # Get configuration
        config, plotter = configure(output_folder='nest_tvb_'+str(tuned_value_mossy_max)+'_', verbose=2)
        print("config.NEST_PERIPHERY",config.NEST_PERIPHERY)
        config.model_params.update(model_params)
        config.SIMULATION_LENGTH = 30000 #20000 30000
        config.TRANSIENT_RATIO = 0.25
        config.MOSSY_MAX_RATE = tuned_value_mossy_max
        config.w_TVB_to_NEST = factor_tvb_nest/tuned_value_mossy_max
        
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
        source_ts = results[0]
        #print(inds.keys(), inds['oliv'])
        sum_square_error = 0
        if source_ts is not None:
            for reg in regs:
                source_ts_interface[reg] = source_ts[1][-config.SIMULATION_LENGTH:,0, inds[reg][1]]
                #print("source ts shape ", source_ts_interface[reg])
                #print("source ts parts", source_ts_interface[reg].Time, source_ts_interface[reg].values)
                print("Avg baseline for ", reg, np.mean(source_ts_interface[reg]))
                sum_square_error = sum_square_error+pow(np.mean(source_ts_interface[reg])-np.mean(source_ts_interface_reference[reg]),2)
                print("sum_square_error: ", sum_square_error)
            
        
        
        RMSEs.append(math.sqrt(sum_square_error/3))
        
    print("RMSE for TVB-NEST gains ", tuned_values_mossy_max, " are ", RMSEs)