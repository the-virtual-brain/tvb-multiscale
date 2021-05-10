import nest
import sys

nest.Install("cerebmodule")

VT = nest.Create("volume_transmitter_alberto", 1)

PRE = nest.Create("iaf_cond_exp", 1)
POST = nest.Create("iaf_cond_exp", 1)

conn_param1 = {"model":    'stdp_synapse_sinexp',
               "A_minus": -0.01,   # double - Amplitude of weight change for depression
               "A_plus":   0.01,   # double - Amplitude of weight change for facilitation 
               "Wmin":     0.0,    # double - Minimal synaptic weight 
               "Wmax":     4.0,    # double - Maximal synaptic weight,              
               "weight":   1.0,
               "delay":    1.0}

nest.Connect(PRE,POST,{'rule': 'one_to_one'},conn_param1)
A=nest.GetConnections(PRE,POST)
nest.SetStatus(A,{'vt_num': 0.0})

conn_param2 = {"model":    'stdp_synapse_cosexp',
               "A_minus": -0.01,   # double - Amplitude of weight change for depression
               "A_plus":   0.01,   # double - Amplitude of weight change for facilitation 
               "Wmin":     0.0,    # double - Minimal synaptic weight 
               "Wmax":     4.0,    # double - Maximal synaptic weight,                 
               "weight":   1.0,
               "delay":    1.0}

nest.Connect(POST,PRE,{'rule': 'one_to_one'},conn_param2)
A=nest.GetConnections(POST,PRE)
nest.SetStatus(A,{'vt_num': 0.0})

sys.exit(0) #Everything went fine
