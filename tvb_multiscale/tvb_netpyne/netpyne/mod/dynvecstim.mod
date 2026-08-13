:  Vector stream of events

NEURON {
	THREADSAFE
	ARTIFICIAL_CELL DynamicVecStim
	POINTER ptr
}

ASSIGNED {
	index
	intervalEnd
	etime (ms)
	ptr
}


INITIAL {
	index = 0
	inferNextEvent()
	if (index > 0) {
		net_send(etime - t, 1) : schedule spike at time etime
	} else {
		: no spikes. wait for inetrval end to be able to initialize next interval
		net_send(intervalEnd - t, 3)
	}
}

NET_RECEIVE (w) {
	: flag 1 - emit a spike and prepare for the next one
	: flag 2 - interval just started, prepare for the first spike
	: flag 3 - interval ended, waiting for initialization of the next one
	if (flag == 1 || flag == 2) { 
		if (flag == 1) {
			net_event(t) : emit spike
		}
		inferNextEvent()

		if (index > 0) {
			if (etime < t) {
				: this may happen if the first spike in this interval is earlier than dt. Need to do this correction:
				etime = t
			}
			net_send(etime - t, 1) : schedule next spike
			
		} else {
			: no more spikes. wait for inetrval end to be able to initialize next interval
			LOCAL endTime
			endTime = intervalEnd - t
			intervalEnd = -1
			net_send(endTime, 3)
		}
	} else if (flag == 3) {
		if (intervalEnd == -1) {
			: still waiting for next interval initialization
			net_send(dt, 3)
		} else {
			: next interval is already initialized ( from play() )! Proceed to first spike in it
			net_send(0, 2)
		}
	}
}

DESTRUCTOR {
VERBATIM
	void* vv = (void*)(_p_ptr);  
        if (vv) {
		hoc_obj_unref(*vector_pobj(vv));
	}
ENDVERBATIM
}

PROCEDURE inferNextEvent() {
VERBATIM	
  { void* vv; int i, size; double* px;
	i = (int)index;
	if (i >= 0) {
		vv = (void*)(_p_ptr);
		if (vv) {
			size = vector_capacity(vv);
			px = vector_vec(vv);
			if (i < size) {
				etime = px[i];
				if (etime < intervalEnd) {
					index += 1.;
				} else {
					printf("WARNING in DynamicVecStim. Spike at %f will be skipped, as well as all later spikes in this interval, since they go after the interval end\n", etime);
					index = -1.;
				}
			}else{
				index = -1.;
			}
		}else{
			index = -1.;
		}
	}
  }
ENDVERBATIM
}

PROCEDURE play() {
	index = 0
VERBATIM
	void** pv;
	void* ptmp = NULL;
	if (ifarg(2)) {
		intervalEnd = *getarg(2);
	} else {
		printf("ERROR in DynamicVecStim! End of interval should be specified as second arg in play() !!");
		exit(1);
	}
	if (ifarg(1)) {
		ptmp = vector_arg(1);
		hoc_obj_ref(*vector_pobj(ptmp));
	}
	pv = (void**)(&_p_ptr);
	if (*pv) {
		hoc_obj_unref(*vector_pobj(*pv));
	}
	*pv = ptmp;
ENDVERBATIM
}
