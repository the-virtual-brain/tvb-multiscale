import numpy as np
import os

def get_weights_and_probs(conn_mode, data_mode, fix_prob, data_path=None):
    if conn_mode == "average":
        if data_mode == "control":
            strengths_maith = np.array([1.93, 3.56, 1.46, 4.51, 3.52, 2.30, 2.34, 3.78, 1.98, 
                1.30, 1.82, 3.56, 3.02, 1.78, 1.36, 2.27, 4.13, 2.74, 3.27])*1e-3
            
            weights_maith = np.array([0.00907183, 0.00894054, 0.00698021, 0.01150882, 0.01260393, 0.00728473, 0.00722696, 0.01235836, 0.00706699, 0.00658632,
                0.0121719 , 0.00799819, 0.01397709, 0.00980494, 0.01037502, 0.01264787, 0.00972397, 0.01185354, 0.00950505])
            probs_maith = np.array([0.20180186, 0.40279948, 0.20737336, 0.39547699, 0.28712551, 0.31780591, 0.32295512, 0.30711735, 0.27485569, 0.17846736,
                0.1476936 , 0.4475118 , 0.21627755, 0.18081887, 0.1299808, 0.18558126, 0.42457663, 0.23344353, 0.34163213])
        else: # patient
            strengths_maith = np.array([3.27, 3.80, 2.65, 3.66, 3.06, 3.06, 3.25, 4.02, 3.32, 
                2.98, 3.45, 3.64, 2.50, 2.12, 2.86, 2.79, 3.96, 3.69, 3.87])*1e-3
            
            weights_maith = np.array([0.00967186, 0.01121781, 0.00945479, 0.01184126, 0.00911763, 0.01037674, 0.01020191, 0.01102725, 0.00910883, 0.00887558,
                0.01131445, 0.01073137, 0.01086344, 0.00902846, 0.01030135, 0.01004465, 0.01072881, 0.01075001, 0.01081699])
            probs_maith = np.array([0.32064033, 0.33880283, 0.28047213, 0.32021597, 0.34429632, 0.31140683, 0.32376521, 0.34524913, 0.36733546, 0.31642416,
                0.29528564, 0.33537892, 0.2323661 , 0.22588549, 0.28335612, 0.28254826, 0.36067394, 0.35109882, 0.35887993])
        num_conns = len(strengths_maith)

        if fix_prob:
            probs_maith = np.ones(num_conns) * fix_prob # fixed mean prob 
            weights_maith = strengths_maith / probs_maith
        else:

            # weights_maith = np.ones(num_conns) * 0.01 # fixed mean weight 
            # probs_maith = strengths_maith / weights_maith
            pass

    else: # "subject"
        fit_data_path = os.path.join(data_path, "ANNarchyFittedModels/dataFits_2020_02_05/databestfits", )

        if data_mode == "patient":
            subject_data = os.path.join(fit_data_path, "patientleft/OutputSim_Patient09.mat")
        else:
            subject_data = os.path.join(fit_data_path, "controlleft/OutputSim_Patient08.mat")
        import scipy.io as sio
        weights=sio.loadmat(subject_data) # weights start from index 19
        weights_maith = weights["X"][0, 19:] # these are indices 19 till 37
        probs_maith = weights["X"][0, :19] # these are indices 0 till 18
    return weights_maith, probs_maith


def lorentzian(x0, delta, N, deterministic=True):
    # # Lorentzian PDF
    # x = np.arange(0, 14, 0.001)
    # L = 1/(np.pi * delta * (1 + ((x-x0)/delta)**2))

    if not deterministic:
        # uniform
        # np.random.seed(1)
        u = np.random.normal(0, 1, N)

        # Inverse CDF (cumulative distribution function) for Lorenzian distr. is
        l = x0 + delta * np.tan(np.pi * (u - 0.5))
    else:
        # deterministic approximation
        k = np.arange(1, N+1)
        l = x0 + delta * np.tan(np.pi/2 * (2*k - N - 1) / (N + 1))

    return l

def test_lorentz():
    import matplotlib.pyplot as plt
    delta = 0.02
    E = lorentzian(7, delta, 600)
    I = lorentzian(0, delta, 150)
    plt.scatter(range(600), E, s=15, marker='o', label='E')
    plt.scatter(range(150), I, s=15, marker='o', label='I')
    plt.title(f'Deterministic Lorentzian with delta eta = {delta}')
    plt.xlabel('# neuron')
    plt.ylabel('Input current')
    plt.legend()
    plt.savefig(f'lorentzian-{delta}.png')