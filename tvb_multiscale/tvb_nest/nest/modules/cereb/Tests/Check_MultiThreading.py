import sys
import nest
import numpy as np
nest.Install("cerebmodule")

CORES = int(sys.argv[1])

nest.set_verbosity('M_WARNING')
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads' : CORES,
                      'total_num_virtual_procs' : CORES,
                      'resolution' : 1.0,
                      'overwrite_files' : True})
msd = 1000 # master seed
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd+n_vp )
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1, msd+1+2*n_vp)
nest.SetKernelStatus({'grng_seed' : msd+n_vp,
                      'rng_seeds' : msdrange2})


VT = nest.Create("volume_transmitter_alberto", 1)
PRE = nest.Create("parrot_neuron", 1)
POST = nest.Create("iaf_cond_exp", 1)
Modulator = nest.Create("parrot_neuron",1)
recdict = {"to_memory": False,
           "to_file":    True,
           "label":     "Weights_"+str(CORES),
           "senders":    PRE,
           "targets":    POST
           }
WeightRec= nest.Create('weight_recorder',params=recdict)

nest.SetDefaults('stdp_synapse_sinexp',{"A_minus":  -0.10,   # double - Amplitude of weight change for depression
                                        "A_plus":    0.01,   # double - Amplitude of weight change for facilitation 
                                        "Wmin":      0.0,    # double - Minimal synaptic weight 
                                        "Wmax":      4.0,    # double - Maximal synaptic weight
                                        "vt":        VT[0],
                                        "weight_recorder": WeightRec[0] })

conn_param1 = {"model":    'stdp_synapse_sinexp',
               "weight":   1.0,
               "delay":    1.0}

nest.Connect(PRE,POST,{'rule': 'one_to_one'},conn_param1)
A = nest.GetConnections(PRE,POST)
nest.SetStatus(A, {"vt_num" : 0.0})
SPIKES1 = [50.0, 100.0, 110.0, 120.0, 130.0, 210.0, 370.0, 490.0]
SPIKES2 = [1.0, 200.0, 220.0, 230.0]

PoissonGen1 = nest.Create('spike_generator',params={'spike_times': SPIKES1})
PoissonGen2 = nest.Create('spike_generator',params={'spike_times': SPIKES2})

nest.Connect(PoissonGen1,PRE,'one_to_one')
nest.Connect(PoissonGen2,Modulator,'one_to_one')
nest.Connect(Modulator,VT,'one_to_one')

nest.Simulate(500.0)
        
sys.exit(0) #Everything went fine
