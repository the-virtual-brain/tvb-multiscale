import sys
import numpy as np
import auxiliary_functions as aux
import nest
import os
import errno
import glob

nest.Install("cerebmodule")

# Cell numbers
GR_number = 65600
PC_number = 2
IO_number = 2

''' SIMULATION PROPERTIES '''
CORES = int(sys.argv[1])
RECORDING_CELLS = True
RECORDING_WEIGHTS = True
PLAST1 = True

''' PROTOCOL SETUP '''
LTP1 =  1.0e-3
LTD1 = -1.0e-2

Init_PFPC = 0.07

aux.tic()
''' VARIABLES INITIALIZATION '''
nest.set_verbosity('M_ERROR')
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads' : CORES,
                      'total_num_virtual_procs' : CORES,
                      'resolution' : 1.0,
                      'overwrite_files' : True})
msd = 1000 # master seed
msdrange1 = range(msd, msd+CORES )
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+CORES+1, msd+1+2*CORES)
nest.SetKernelStatus({'grng_seed' : msd+CORES,
                      'rng_seeds' : msdrange2})

# Define the new kinds of neuron
nest.CopyModel('parrot_neuron','granular_neuron')
nest.CopyModel('iaf_cond_exp','purkinje_neuron')
nest.CopyModel('parrot_neuron','olivary_neuron')

nest.SetDefaults('purkinje_neuron',{'t_ref' : 2.0,
                                    'C_m' : 400.0,
                                    'V_th' : -52.0,
                                    'V_reset' : -70.0,
                                    'g_L' : 16.0,
                                    'tau_syn_ex' : 0.5,
                                   #'tau_m' : 25.0,
                                    'tau_syn_in' : 1.6})

GR = nest.Create("granular_neuron", GR_number)
PC = nest.Create("purkinje_neuron", PC_number)
IO = nest.Create("olivary_neuron", IO_number)

InputGen = nest.Create("spike_generator", GR_number)
ErrorGen = nest.Create("spike_generator", IO_number)
if RECORDING_WEIGHTS:
    recdict2 = {"to_memory": False,
                "to_file":    True,
                "label":     "PFPC",
                "senders":    GR,
                "targets":    PC,
                "precision":  8
               }
    WeightPFPC = nest.Create('weight_recorder',params=recdict2)

if PLAST1:
    vt=nest.Create("volume_transmitter_alberto",PC_number)
    for n,vti in enumerate(vt):
        nest.SetStatus([vti],{"vt_num" : n})

# Connection 0
nest.Connect(InputGen,GR,"one_to_one",{"model": "static_synapse",
                                       "weight": 1.0,
                                       "delay": 1.0
                                       })

nest.Connect(ErrorGen,IO,"one_to_one",{"model": "static_synapse",
                                       "weight": 1.0,
                                       "delay": 1.0 
                                      })

# Connection 6
if PLAST1:
    if RECORDING_WEIGHTS:
        nest.SetDefaults('stdp_synapse_sinexp',{"A_minus":   LTD1,   # double - Amplitude of weight change for depression
                                                "A_plus":    LTP1,   # double - Amplitude of weight change for facilitation
                                                "Wmin":      0.0,    # double - Minimal synaptic weight
                                                "Wmax":      4.0,    # double - Maximal synaptic weight
                                                "weight_recorder": WeightPFPC[0],
                                                "vt": vt[0]
                                                })
    else:
        nest.SetDefaults('stdp_synapse_sinexp',{"A_minus":   LTD1,   # double - Amplitude of weight change for depression
                                                "A_plus":    LTP1,   # double - Amplitude of weight change for facilitation
                                                "Wmin":      0.0,    # double - Minimal synaptic weight
                                                "Wmax":      4.0,
                                                "vt":        vt[0]
                                                                                                })

    for n,PCi in enumerate(PC):
        PFPC_conn_param = {"model": 'stdp_synapse_sinexp',
                           "weight": Init_PFPC,
                           "delay": 1.0,
                           "vt_num": float(n)
                           }
        nest.Connect(GR, [PCi], {'rule': "fixed_indegree", 'indegree': 65600, "multapses": False, "autapses": False},
                     PFPC_conn_param)


else:
    PFPC_conn_param = {"model":  "static_synapse",
                       "weight": Init_PFPC,
                       "delay":  1.0,
                       }
    nest.Connect(GR,PC,{'rule': "fixed_indegree", 'indegree':65600, "multapses": False, "autapses": False},PFPC_conn_param)

# Connection 7
if PLAST1:
    # IO-PC teaching connections
    nest.Connect(IO,vt,'one_to_one',{"model": "static_synapse",
                                     "weight": 1.0,
                                     "delay": 1.0})
    IOPC_conn = nest.GetConnections(IO,vt)
    print("Number of IO-PC (volume_transmitter) synapses: " + str(len(IOPC_conn)))

if RECORDING_CELLS:
    # Create Auxiliary tools
    recdict = [{"to_memory": True, "to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_GR"},
               {"to_memory": True, "to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_PC"},
               {"to_memory": True, "to_file":True, "withgid":True, "withtime":  True, "label":"Spike_Detector_IO"}]
    spikedetector = nest.Create("spike_detector",3,params=recdict)
    nest.Connect(GR, [spikedetector[0]])
    nest.Connect(PC, [spikedetector[1]])
    nest.Connect(IO, [spikedetector[2]])

# Load input activity on GRs
GRinput_file = open("GR_65600.dat", 'r')
for InputGeni in InputGen:
    Spikes_s = GRinput_file.readline()
    Spikes_s = Spikes_s.split()
    Spikes_f = []
    for elements in Spikes_s:
        Spikes_f.append(float(elements))
    nest.SetStatus([InputGeni],{'spike_times' : Spikes_f})

nest.SetStatus([ErrorGen[0]],{'spike_times' : [8.0, 98.0, 298.0, 308.0, 318.0, 498.0, 598.0, 698.0, 798.0, 997.0]})
nest.SetStatus([ErrorGen[1]],{'spike_times' : [18.0, 198.0, 258.0, 358.0, 368.0, 458.0, 558.0, 658.0, 758.0, 957.0]})

aux.toc()

msd = 1000 # master seed
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
msdrange1 = range(msd, msd+n_vp )
pyrngs = [np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1, msd+1+2*n_vp)
nest.SetKernelStatus({'grng_seed' : msd+n_vp,
                      'rng_seeds' : msdrange2})

print("### SIMULATION STARTS ###")
aux.tic()
nest.Simulate(1001)
aux.toc()

sys.exit(0) #Everything went fine
