import sys
import numpy as np
import auxiliary_functions as aux
import nest
import os
import errno
import glob

#@profile
def main():
    nest.Install("cerebmodule")

    # Cell numbers
    MF_number = 100
    PC_number = 100
    DCN_number = 100

    ''' SIMULATION PROPERTIES '''
    CORES = int(sys.argv[1])

    ''' PROTOCOL SETUP '''
    LTP2 = 1.0e-5
    LTD2 = -1.0e-1

    Init_MFDCN = 1.0
    Init_PCDCN = -0.0000000001

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


    MF = nest.Create("iaf_cond_exp", MF_number)
    PC = nest.Create("iaf_cond_exp", PC_number)
    DCN = nest.Create("iaf_cond_exp", DCN_number)

    InputGen = nest.Create("spike_generator", MF_number)
    ErrorGen = nest.Create("spike_generator", PC_number)

    recdict2 = {"to_memory": False,
                "to_file": True,
                "label": "MFDCN",
                "senders": MF,
                "targets": DCN
                }
    WeightMFDCN = nest.Create('weight_recorder', params=recdict2)


    vt2 = nest.Create("volume_transmitter_alberto", DCN_number)
    for n, vti in enumerate(vt2):
        nest.SetStatus([vti], {"vt_num": n})

    nest.Connect(InputGen, MF, "one_to_one", {"model": "static_synapse",
                                              "weight": 1000.0,
                                              "delay": 1.0
                                              })

    nest.Connect(ErrorGen, PC, "one_to_one", {"model": "static_synapse",
                                              "weight": 1000.0,
                                              "delay": 1.0
                                              })

    PCDCN_conn_param = {"model": "static_synapse",
                        "weight": Init_PCDCN,
                        "delay": 1.0}

    nest.Connect(PC, DCN, "one_to_one", PCDCN_conn_param)


    nest.SetDefaults('stdp_synapse_cosexp', {"A_minus": LTD2,  # double - Amplitude of weight change for depression
                                             "A_plus": LTP2,  # double - Amplitude of weight change for facilitation
                                             "Wmin": 0.0,  # double - Minimal synaptic weight
                                             "Wmax": 1.0,  # double - Maximal synaptic weight
                                             "vt": vt2[0],
                                             "weight_recorder": WeightMFDCN[0]
                                             })

    nest.CopyModel('static_synapse', 'static_synapse_vt')
    for i, DCNind in enumerate(DCN):
        MFDCN_conn_param = {"model": 'stdp_synapse_cosexp',
                            "weight": Init_MFDCN,
                            "delay": 1.0,
                            "vt_num": float(i)}

        nest.Connect([MF[i]], [DCNind], "one_to_one", MFDCN_conn_param)

        B = nest.GetConnections(PC, [DCNind])
        B = np.array(B)
        if (len(B) > 0):
            source = B[:, 0]
            nest.Connect(source.tolist(), [vt2[i]], {"rule": "all_to_all"}, {'model': 'static_synapse_vt',
                                                                               'delay': 1.0,
                                                                               'weight': 1.0})



    # Create Auxiliary tools
    recdict = [{"to_memory": True, "to_file": True, "withgid": True, "withtime": True, "label": "Spike_Detector_MF"},
               {"to_memory": True, "to_file": True, "withgid": True, "withtime": True, "label": "Spike_Detector_PC"},
               {"to_memory": True, "to_file": True, "withgid": True, "withtime": True, "label": "Spike_Detector_DCN"}]
    spikedetector = nest.Create("spike_detector", 3, params=recdict)
    nest.Connect(MF, [spikedetector[0]])
    nest.Connect(PC, [spikedetector[1]])
    nest.Connect(DCN, [spikedetector[2]])

    # Load input activity on MFs and PCs
    for i, InputGeni in enumerate(InputGen):
        nest.SetStatus([InputGeni], {'spike_times': [1.0, 1001.0, 3001+float(i), 5001.0]})

    for i, InputGeni in enumerate(ErrorGen):
        nest.SetStatus([InputGeni], {'spike_times': [1001.0+float(i), 3000.0]})

    aux.toc()

    msd = 1000  # master seed
    n_vp = nest.GetKernelStatus('total_num_virtual_procs')
    msdrange1 = range(msd, msd + n_vp)
    pyrngs = [np.random.RandomState(s) for s in msdrange1]
    msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
    nest.SetKernelStatus({'grng_seed': msd + n_vp,
                          'rng_seeds': msdrange2})



    print( " - Number synapses: " + str(len(nest.GetConnections())))

    print("### SIMULATION STARTS ###")
    aux.tic()
    nest.Simulate(500.0)
    nest.SetStatus(nest.GetConnections(MF,DCN),{'weight': Init_MFDCN})
    nest.Simulate(1500.0)
    nest.SetStatus(nest.GetConnections(MF,DCN),{'weight': Init_MFDCN})
    nest.Simulate(5000.0)

    aux.toc()

    sys.exit(0)  # Everything went fine


if __name__ == '__main__':
    main()

    nest.GetStatus(nest.GetConnections(MF, DCN), {'weight': Init_MFDCN})
