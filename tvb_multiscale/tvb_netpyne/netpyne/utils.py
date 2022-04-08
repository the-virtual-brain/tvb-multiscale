import numpy as np

poisson_gen_rand_state = np.random.RandomState(5)

def generateSpikesForPopulation(numNeurons, rates, times):

    # this method generates random spike sequence for number of neurons, given the list of instantaneous rates changing in time.
    # however, due to performance reasons, instead of generating it in straightforward way, using inhomogeneous rate for each of `numNeurons` neuron,
    # this method "re-shapes" time domain into neurons domain, and for each value of rate in `rates` array it generates homogeneous spike sequence,
    # but using longer interval, namely - `numNeurons` * `dt`, that afterwards is re-shaped back to original domain,
    # by dividing obtained spikes into `numNeurons` equal time bins - one per each neuron.

    spikesPerNeuron = {}

    for iInterval in range(1, len(times)):

        rate = rates[iInterval]
        intervalStart = times[iInterval - 1]
        dt = times[iInterval] - intervalStart

        # prepare to generate spikes train once for all neurons
        totalDuration = numNeurons * dt

        # equal time bins, each of duration dt. Spikes that will fall into each of this bins will be treated as belonging to corresponding neuron
        binsPerNeuron = np.linspace(dt, totalDuration, numNeurons)

        spikeTimes = poisson_generator(rate, 0.0, totalDuration, poisson_gen_rand_state) # generate spikes

        # re-shape back: detect to which neuron these spikes belong, and which is proper time
        neuronInds = np.digitize(spikeTimes, binsPerNeuron)
        spikeTimes -= neuronInds * dt

        for neuron, spike in zip(neuronInds, spikeTimes):
            if neuron not in spikesPerNeuron:
                spikesPerNeuron[neuron] = []
            spikesPerNeuron[neuron].append(intervalStart + spike)

    return spikesPerNeuron

def poisson_generator(rate, t_start, t_stop, rand_state):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if 
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
    -------
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)
        array   - if True, a np array of sorted spikes is returned,
                    rather than a SpikeTrain object.

    Examples:
    --------
        >> gen.poisson_generator(50, 0, 1000)
        >> gen.poisson_generator(20, 5000, 10000, array=True)

    See also:
    --------
        inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
    """

    #number = int((t_stop-t_start)/1000.0*2.0*rate)

    # less wasteful than double length method above
    n = (t_stop-t_start)/1000.0*rate
    number = np.ceil(n+3*np.sqrt(n))
    if number<100:
        number = min(5+np.ceil(2*n),100)

    if number > 0:
        isi = rand_state.exponential(1.0/rate, int(number))*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes+=t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i==len(spikes):
        # ISI buf overrun
        
        t_last = spikes[-1] + rand_state.exponential(1.0/rate, 1)[0]*1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += rand_state.exponential(1.0/rate, 1)[0]*1000.0
        
        spikes = np.concatenate((spikes,extra_spikes))

    else:
        spikes = np.resize(spikes,(i,))

    return spikes