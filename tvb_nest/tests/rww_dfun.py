import numpy

def compute_rww_dfun(sn, sg, coupling):

    taon = 100.0   # tau_e
    taog = 10.0    # tau_i
    gamma = 0.641  # gamma_e * 1000
    JN = 0.15      # J_N
    J = 1.0        # J_i
    I0 = 0.382     # I_o
    Jexte = 1.     # W_e
    Jexti = 0.7    # W_i
    w = 1.4        # w_p
    C = 2.0        # G

    # corresponding TVB variables:
    for x in numpy.nditer(sn):
        if x>1: x=1    # S_e
        if x<0: x=0    # S_e

    for x in numpy.nditer(sg):
        if x>1: x=1    # S_i
        if x<0: x=0

    xn = I0*Jexte + w*JN*sn + JN*C*coupling - J*sg # x_e
    xg = I0*Jexti + JN*sn - sg  # x_i

    rn = phie(xn)  # H_e
    rg = phii(xg)  # H_i

    dsn = -sn/taon + (1-sn)*gamma*rn/1000.  # dS_e
    dsg = -sg/taog + rg/1000.  # dS_i

    out = [dsn, dsg]    # rn, rg, xn, xg
    return out

def phie(x):
# corresponding TVB parameters:
    g=0.16  # d_e
    I=125.  # b_e
    c=310.  # a_e
    y=c*x-I
    #y1=x-I;
    result = y/(1-numpy.exp(-g*y))
    return result

def phii(x):
# corresponding TVB parameters:
    g=0.087 # d_i
    I=177.  # b_i
    c=615.  # a_i
    y=c*x-I
    result = y/(1-numpy.exp(-g*y))
    return result
