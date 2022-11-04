import sys
import numpy as np
import auxiliary_functions as aux
import nest
import os
import errno
import glob


nest.Install("cerebmodule")

''' SIMULATION PROPERTIES '''
CORES = int(sys.argv[1])

aux.tic()
''' VARIABLES INITIALIZATION '''
nest.set_verbosity('M_ERROR')
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': CORES,
                      'total_num_virtual_procs': CORES,
                      'resolution': 1.0,
                      'overwrite_files': True})
msd = 1000  # master seed
msdrange1 = range(msd, msd + CORES)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + CORES + 1, msd + 1 + 2 * CORES)
nest.SetKernelStatus({'grng_seed': msd + CORES,
                      'rng_seeds': msdrange2})

Pre_neurons = nest.Create("iaf_cond_exp", 1000)
Post_neurons = nest.Create("iaf_cond_exp", 1000)
PreGen = nest.Create("spike_generator", 1000)
PostGen = nest.Create("spike_generator", 1000)

recdict2 = {"to_memory": False,
            "to_file": True,
            "label": "iSTDP",
            "senders": Pre_neurons,
            "targets": Post_neurons
            }
WeightRec = nest.Create('weight_recorder', params=recdict2)

C = 5.0451
r_LTP_LTD = 0.552
Wmax_istdp = 15.0
Wmin_istdp = -15.0
tau_p_istdp = 125.0
tau_m_istdp = 195.6
nest.CopyModel('istdp_synapse', 'istdp',
               {'lambda': C, 'mu_plus': r_LTP_LTD, 'mu_minus': tau_m_istdp, 'Wmax': Wmax_istdp, 'Wmin': Wmin_istdp,
                'tau_plus': tau_p_istdp, "weight_recorder": WeightRec[0]})



recdict = {"to_memory": True, "to_file": True, "withgid": True, "withtime": True, "label": "Spike_Detector"}
spikedetector = nest.Create("spike_detector", 1, params=recdict)
nest.Connect(Pre_neurons, [spikedetector[0]])
nest.Connect(Post_neurons, [spikedetector[0]])


# Generate input activity on Pre and Post neurons
for i, InputGeni in enumerate(PreGen):
    nest.SetStatus([InputGeni], {'spike_times': [1.0, float(i)+9001.0, 20001.0]})

for i, InputGeni in enumerate(PostGen):
    nest.SetStatus([InputGeni], {'spike_times': [float(i)+1.0, 9001.0]})

# Connect the network
nest.Connect(PreGen, Pre_neurons, "one_to_one", {"model": "static_synapse",
                                                 "weight": 1000.0,
                                                 "delay": 1.0
                                                 })
nest.Connect(PostGen, Post_neurons, "one_to_one", {"model": "static_synapse",
                                                 "weight": 1000.0,
                                                 "delay": 1.0
                                                 })
nest.Connect(Pre_neurons, Post_neurons, "one_to_one", {"model": "istdp",
                                                   "weight": 0.0,
                                                   "delay": 1.0
                                                   })

aux.toc()

msd = 1000  # master seed
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd + n_vp)
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
nest.SetKernelStatus({'grng_seed': msd + n_vp,
                      'rng_seeds': msdrange2})

print("### SIMULATION STARTS ###")
aux.tic()
nest.Simulate(5000.0)
nest.SetStatus(nest.GetConnections(Pre_neurons,Post_neurons),{'weight':0.0})
nest.Simulate(15004.0)
aux.toc()

sys.exit(0)  # Everything went fine
