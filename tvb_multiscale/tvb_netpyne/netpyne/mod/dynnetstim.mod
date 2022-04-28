: $Id: nsloc.mod,v 1.7 2013/06/20  salvad $
: from nrn/src/nrnoc/netstim.mod
: modified to use allow for time-dependent intervals 

NEURON	{ 
  ARTIFICIAL_CELL DynamicNetStim
  RANGE checkInterval
  THREADSAFE : only true if every instance has its own distinct Random
  POINTER donotuse
}

PARAMETER {
    fflag           = 1             : don't change -- indicates that this is an artcell
    check_interval = 0.1 (ms) : time between checking if interval has changed
}

ASSIGNED {
    intervalEnd (ms)
    spikeOne (ms)
    spikeTwo (ms)
	spikeThree (ms)
	checkCounter
	donotuse
}

INITIAL {
    intervalEnd = -1
    spikeOne = -1
    spikeTwo = -1
	spikeThree = -1

    checkCounter = 0

    net_send(0, 1)
}	

PROCEDURE set_next_spikes(end, spkOne, spkTwo, spkThree) {
    intervalEnd = end
    spikeOne = spkOne
    spikeTwo = spkTwo
	spikeThree = spkThree
}

FUNCTION get_check_counter() {
    get_check_counter = checkCounter
}

NET_RECEIVE (w) {

    if (flag == 1) {

        if (intervalEnd == -1) {
            : waiting for interval initialization
            net_send(check_interval, 1)
        } else {
            if (spikeOne > -1) {
                if (spikeOne < t) {
                    spikeOne = t
                }
                net_event(spikeOne)
            }
            if (spikeTwo > -1) {
                if (spikeTwo < t) {
                    spikeTwo = t
                }
                net_event(spikeTwo)
            }
			if (spikeThree > -1) {
                if (spikeThree < t) {
                    spikeThree = t
                }
                net_event(spikeThree)
            }
            net_send(intervalEnd - t, 1)
            intervalEnd = -1
        }
        checkCounter = checkCounter + 1
    }
}

COMMENT
Presynaptic spike generator
---------------------------

This mechanism has been written to be able to use synapses in a single
neuron receiving various types of presynaptic trains.  This is a "fake"
presynaptic compartment containing a spike generator.  The trains
of spikes can be either periodic or noisy (Poisson-distributed)

Parameters;
   noise: 	between 0 (no noise-periodic) and 1 (fully noisy)
   interval: 	mean time between spikes (ms)
   number: 	number of spikes (independent of noise)

Written by Z. Mainen, modified by A. Destexhe, The Salk Institute

Modified by Michael Hines for use with CVode
The intrinsic bursting parameters have been removed since
generators can stimulate other generators to create complicated bursting
patterns with independent statistics (see below)

Modified by Michael Hines to use logical event style with NET_RECEIVE
This stimulator can also be triggered by an input event.
If the stimulator is in the on==0 state (no net_send events on queue)
 and receives a positive weight
event, then the stimulator changes to the on=1 state and goes through
its entire spike sequence before changing to the on=0 state. During
that time it ignores any positive weight events. If, in an on!=0 state,
the stimulator receives a negative weight event, the stimulator will
change to the on==0 state. In the on==0 state, it will ignore any ariving
net_send events. A change to the on==1 state immediately fires the first spike of
its sequence.

ENDCOMMENT