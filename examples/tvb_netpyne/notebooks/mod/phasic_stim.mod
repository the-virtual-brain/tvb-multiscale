COMMENT
Since this is an electrode current, positive values of i depolarize the cell
and in the presence of the extracellular mechanism there will be a change
in vext since i is not a transmembrane current but a current injected
directly to the inside of the cell.
ENDCOMMENT

NEURON {
        POINT_PROCESS PhasicStim
        RANGE amplitude, ampfactor, kappa, frequency, duration, mono
        ELECTRODE_CURRENT i
}

UNITS {
        (nA) = (nanoamp)
}

PARAMETER {
        amplitude = 20
        ampfactor = 10
        kappa = 1
        frequency = 130
        duration = 0.3
        mono = 1
        pi=3.14159265358979323846
}

ASSIGNED {
        i (nA)
}

LOCAL h1, h2, h4, i1, i2

BREAKPOINT {

        if (sin(2*pi*(frequency/1000)*t) > 0) {
                h1 = 1
        } else {
                h1 = 0
        }
        if (mono) {
                if (sin(2*pi*(frequency/1000)*(t+duration)) > 0) {
                        h2 = 1
                } else {
                        h2 = 0
                }
                i = amplitude*kappa*h1*(1-h2)
        } else {
                if (sin(2*pi*(frequency/1000)*(t-duration))) {
                        h2 = 1
                } else {
                        h2 = 0
                }
                if (sin(2*pi*(frequency/1000)*(t-((ampfactor+1)*duration))) > 0) {
                        h4 = 1
                } else {
                        h4 = 0
                }
                i1 = amplitude*(1+ 1/ampfactor)*kappa*h1*(1-h2)
                i2 = (-amplitude/ampfactor) *kappa *h1*(1-h4)
                i = i1 + i2
        }
}