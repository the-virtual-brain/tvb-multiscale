NEURON {
    POINT_PROCESS Izhi2003c
    RANGE a, b, c, d, C, Ie, n0, n1, n2, Vr, uInit, vInit, thresh, cellid, mu, noise, Iinj
    NONSPECIFIC_CURRENT i
}

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
}

PARAMETER {
    a       = 0.02 (/ms)
    b       = 0.2  (/ms)
    c       = -65  (mV)   : reset potential after a spike
    d       = 2    (mV/ms)
    C       = 1
    Ie      = 0    (nA)
    n0
    n1
    n2
    mu = 0
    noise = 0
    Iinj = 0
    Eampa = 0
    Egaba = -90
    Vr = 0
    tau = 10
    refrac = 10
    uInit = -18.55
    vInit = -70
    thresh = 30   (mV)   : spike threshold
    cellid = -1 : A parameter for storing the cell ID, if required (useful for diagnostic information)
}

ASSIGNED {
    i (nA)
    refracEnds
}

INITIAL {
  V = vInit
  u = uInit
  gampa = 0
  ggaba = 0
  refracEnds = -1e10
  net_send(0,1) : to start watching for threshold
}

STATE {
    u (mV/ms)
    V
    gampa
    ggaba
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
}

LOCAL clamp
DERIVATIVE states {
    gampa' = -gampa/tau
    ggaba' = -ggaba/tau
    
    if (t < refracEnds) {
        clamp = 0 : clamp u and V at reset values during refractory period (i.e. keep derivative at zero)
    } else {
        clamp = 1
    }
    u' = clamp * (a * (b * (V - Vr) - u))
    V' = clamp * (n2 * V * V + n1 * V + n0 - u / C -gampa * (V - Eampa) - ggaba * (V - Egaba) + Ie + Iinj + mu * noise)
}

NET_RECEIVE (w) {
    if (flag == 1) {
        WATCH (V > thresh) 2
    } else if (flag == 2) { : spike fired
        net_event(t)
        V = c
        u = u + d
        refracEnds = t + refrac
    } else { : spike received
        : by convention here, negative weights are inhibitory
        if (w > 0) {
            gampa = gampa + w
        } else {
            ggaba = ggaba - w
        }
    }
}