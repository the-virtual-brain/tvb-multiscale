# -*- coding: utf-8 -*-

from scipy.signal import welch
from scipy.interpolate import interp1d

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *


def load_connectome(config, plotter=None):
    import h5py
    connectome = {}
    f = h5py.File(config.BRAIN_CONN_FILE)
    connectome['region_labels'] = np.array(f["region_labels"][()]).astype("<U128")
    connectome['centres'] = np.array(f["centres"][()])
    # connectome['hemispheres'] = np.array(f["hemispheres"][()]),
    connectome['weights'] = np.array(f["weights"][()])
    connectome['tract_lengths'] = np.array(f["tract_lengths"][()])
    f.close()

    major_structs_labels = np.load(config.MAJOR_STRUCTS_LABELS_FILE)
    voxel_count = np.load(config.VOXEL_COUNT_FILE)
    inds = np.load(config.INDS_FILE, allow_pickle=True).item()
    if plotter:
        print("major_structs_labels:\n", np.unique(major_structs_labels))
        print("ROI inds:\n", inds)

    return connectome, major_structs_labels, voxel_count, inds


def construct_extra_inds_and_maps(connectome, inds):
    maps = {}
    region_labels = connectome['region_labels']
    inds["subcrtx"] = np.arange(len(region_labels)).astype('i')
    inds["subcrtx"] = np.delete(inds["subcrtx"], inds["crtx"])
    maps["is_subcortical"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_subcortical"][inds["subcrtx"]] = True
    maps["is_cortical"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_cortical"][inds["crtx"]] = True
    maps["is_thalamic"] = np.array([False] * region_labels.shape[0]).astype("bool")
    maps["is_thalamic"][inds["thalspec"]] = True
    maps["not_thalamic"] = np.logical_not(maps["is_thalamic"])
    maps["is_subcortical_not_thalspec"] = np.logical_and(maps["is_subcortical"], np.logical_not(maps["is_thalamic"]))
    inds["subcrtx_not_thalspec"] = np.where(maps["is_subcortical_not_thalspec"])[0]
    inds["not_subcrtx_not_thalspec"] = np.where(np.logical_not(maps['is_subcortical_not_thalspec']))[0]
    return inds, maps


def plot_norm_w_hist(w, wp, inds):
    h = w[wp].flatten()
    print('number of all connections > 0: %d' % h.size)
    h, bins = np.histogram(h, range=(1.0, 31), bins=100)

    w_within_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_from_sub = w[inds["not_subcrtx_not_thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_to_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["not_subcrtx_not_thalspec"][None, :]]
    h_sub = np.array(w_within_sub.flatten().tolist() +
                     w_from_sub.flatten().tolist() +
                     w_to_sub.flatten().tolist())
    h_sub = h_sub[h_sub > 0].flatten()
    print('number of h_sub > 0: %d' % h_sub.size)
    h_sub, bins_sub = np.histogram(h_sub, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_sub)

    h_crtx = np.array(w[inds["not_subcrtx_not_thalspec"][:, None],
                        inds["not_subcrtx_not_thalspec"][None, :]].flatten().tolist())
    h_crtx = h_crtx[h_crtx > 0]
    print('number of h_crtx > 0: %d' % h_crtx.size)
    h_crtx, bins_crtx = np.histogram(h_crtx, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_crtx)

    h2 = h_crtx + h_sub
    print('number of total > 0: %d' % np.sum(h2))

    x = bins[:-1] + np.diff(bins) / 2
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, h, 'b', label='All connections')
    plt.plot(x, h_crtx, 'g', label='Non Subcortical connections')
    plt.plot(x, h_sub, 'r', label='Subcortical connections')
    # plt.plot(x, h-h_sub, 'r--', label='All - Subcortical connections')
    # plt.plot(x, h-h_crtx, 'g--', label='All - Non Subcortical connections')
    # plt.plot(x, h2, 'k--', label='Total connections')
    plt.title("Histogram of logtransformed connectome weights")
    plt.legend()
    plt.ylim([0.0, h.max()])
    plt.tight_layout()
    return fig


def logprocess_weights(connectome, inds, print_flag=True, plotter=None):
    w = connectome['weights'].copy()
    w[np.isnan(w)] = 0.0  # zero nans
    w0 = w <= 0  # zero weights
    wp = w > 0  # positive weights
    w /= w[wp].min()  # divide by the minimum to have a minimum of 1.0
    w *= np.exp(1)  # multiply by e to have a minimum of e
    w[wp] = np.log(w[wp])  # log positive values
    w[w0] = 0.0  # zero zero values (redundant)
    connectome['weights'] = w
    if print_flag:
        print('\nnormalized weights [min, max] = \n', [w[wp].min(), w[wp].max()])
    if plotter:
        plot_norm_w_hist(w, wp, inds)
    return connectome


def build_connectivity(connectome, inds, config, print_flag=True, plotter=None):
    from tvb.datatypes.connectivity import Connectivity

    connectivity = Connectivity(**connectome)

    # Selectively scale up some connections, if required:
    for reg1, reg2, sc in config.BRAIN_CONNECTIONS_TO_SCALE:
        iR1 = np.where([reg in reg1 for reg in connectivity.region_labels])[0]
        iR2 = np.where([reg in reg2 for reg in connectivity.region_labels])[0]
        connectivity.weights[iR1, iR2] *= sc

    # Normalize connectivity weights
    connectivity.weights[np.logical_or(np.isnan(connectivity.weights), np.isinf(connectivity.weights))] = 0.0
    if config.CONN_SCALE:
        connectivity.weights = connectivity.scaled_weights(mode=config.CONN_SCALE)
    if config.CONN_NORM_PERCENTILE:
        connectivity.weights /= np.percentile(connectivity.weights, config.CONN_NORM_PERCENTILE)
    if config.CONN_CEIL:
        connectivity.weights[connectivity.weights > config.CONN_CEIL] = config.CONN_CEIL

    connectivity.speed = np.array([config.CONN_SPEED])
    connectivity.tract_lengths = np.maximum(connectivity.speed * config.DEFAULT_DT,
                                            connectivity.tract_lengths)

    connectivity.configure()

    if plotter:
        # Plot TVB connectome:
        plotter.plot_tvb_connectivity(connectivity);

    # Remove connections between specific thalami and the rest of the subcortex:
    connectivity.weights[inds["subcrtx_not_thalspec"][:, None], inds["thalspec"][None, :]] = 0.0
    # Retain connections
    # from spinal nucleus of the trigeminal to S1 barrel field:
    w_s1brlthal_trigeminal = connectivity.weights[inds["s1brlthal"], inds["trigeminal"]].copy()
    # from interposed nucleus to M1:
    w_m1thal_cerebnuclei = connectivity.weights[inds["m1thal"], inds["trigeminal"]].copy()
    connectivity.weights[inds["thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]] = 0.0
    connectivity.weights[inds["s1brlthal"], inds["trigeminal"]] = w_s1brlthal_trigeminal
    connectivity.weights[inds["m1thal"], inds["trigeminal"]] = w_m1thal_cerebnuclei

    # Homogenize crtx <-> subcrtx connnectivity
    # connectivity.weights[inds["crtx"][:, None], inds["subcrtx_not_thalspec"][None, :]] *= 0.0 # 0.0 # 0.02
    # connectivity.weights[inds["subcrtx_not_thalspec"][:, None], inds["crtx"][None, :]] *= 0.0 # 0.0 # 0.02

    # # Disconnect subcortex completely
    # connectivity.weights[inds["not_subcrtx_not_thalspec"][:, None],
    #                      inds["subcrtx_not_thalspec"][None, :]] *= 0.0 # 0.0 # 0.02
    # connectivity.weights[inds["subcrtx_not_thalspec"][:, None],
    #                     inds["not_subcrtx_not_thalspec"][None, :]] *= 0.0 # 0.0 # 0.02

    return connectivity


def build_model(number_of_regions, inds, maps, config):
    # We are not running dynamic FIC for fitting:
    # if config.FIC:
    #     from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import \
    #         WilsonCowanThalamoCorticalFIC as WilsonCowanThalamoCortical
    # else:
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical

    dummy = np.ones((number_of_regions,))

    STIMULUS = config.model_params.pop("STIMULUS", None)

    model_params = {}
    for p, pval in config.model_params.items():
        if pval is not None:
            pval = np.array([pval]).flatten()
            if p == 'G':
                # G normalized by the number of regions as in Griffiths et al paper
                # Geff = G /(number_of_regions - inds['thalspec'].size)
                pval = pval / (number_of_regions - inds['thalspec'].size)
            model_params[p] = pval

    if STIMULUS:
        # Stimulus to M1 and S1 barrel field
        # inds_stim = np.concatenate((inds["motor"][:2], inds["sens"][-2:])
        inds_stim = np.concatenate((inds["facial"], inds["trigeminal"]))
        # Stimuli:
        A_st = 0 * dummy.astype("f")
        f_st = 0 * dummy.astype("f")
        # Stimulus to M1
        A_st[inds_stim] = STIMULUS
        f_st[inds_stim] = config.STIMULUS_RATE  # Hz
        model_params.update({"A_st": A_st, "f_st": f_st})

    model = WilsonCowanThalamoCortical(is_cortical=maps['is_cortical'][:, np.newaxis],
                                       is_thalamic=maps['is_thalamic'][:, np.newaxis],
                                       **model_params)
    model.dt = config.DEFAULT_DT

    # Remove Specific thalamic relay -> nonspecific subcortical structures connections!
    w_se = model.w_se * dummy
    w_se[inds['subcrtx']] = 0.0  #  model.G[0]
    model.w_se = w_se
    # Remove specific thalamic relay -> inhibitory nonspecific subcortical structures connections
    w_si = model.w_si * dummy
    w_si[inds['subcrtx']] = 0.0  # * model.G[0]
    model.w_si = w_si

    # Long range connections to specific thalamic relay and reticular structures connections' weights:
    model.G = model.G * dummy
    model.G[inds["thalspec"]] = 0.0
    # Retain connections
    # from spinal nucleus of the trigeminal to S1 barrel field:
    model.G[inds["s1brlthal"]] = model.G[inds["crtx"][0]]
    # from interposed nucleus to M1:
    model.G[inds["m1thal"]] = model.G[inds["crtx"][0]]

    return model


# An approximate automatic FIC:

def fic(param, p_orig, weights, trg_inds=None, src_inds=None, FIC=1.0, dummy=None, subtitle="", plotter=None):
    number_of_regions = weights.shape[0]
    # This function will adjust inhibitory weights based on total indegree and some scaling
    if trg_inds is None:
        trg_inds = np.arange(number_of_regions).astype('i')

    if src_inds is None:
        src_inds = np.arange(number_of_regions).astype('i')

    # Scale w_ie to grow to greater negative values from the default w_ie = -2.0
    p_orig = np.array(p_orig)
    if p_orig.size == 1:
        if dummy is None:
            dummy = np.ones((number_of_regions,))
            p_orig = p_orig.item() * dummy
    p = p_orig.copy()
    pscalar = p_orig[trg_inds].mean().item()
    # Move them to have a maximum of p_orig:
    # FICindegree = (indegree - indegree_min) / indegree_std
    indegree = weights[trg_inds][:, src_inds].sum(axis=1)
    FICindegree = (indegree - indegree.min()) / np.std(indegree)
    # p_fic = p * (1 - FIC * FICindegree) = p * (1 + FIC * (indegree - indegree_min) / indegree_std)
    # assuming p < 0.0
    p[trg_inds] = pscalar * (1 + FIC * FICindegree)

    try:
        assert np.all(np.argsort(indegree) == np.argsort(-p[trg_inds]))  # the orderings should reverse
    except:
        plt.figure()
        plt.plot(indegree, p[trg_inds], "-o")
        plt.xlabel("%g*indegree" % FIC)
        plt.ylabel("%s scaled" % param)
        plt.title("Testing indegree and parameter anti-correlation")
        plt.tight_layout()

    # Plot and confirm:
    if plotter:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[1].hist(FICindegree, 30)
        axes[1].set_xlabel("Indegree Scaler values")
        axes[1].set_ylabel("Histogram of region counts")
        axes[1].set_title("Indegree scaler = %g*(indegree-min(indegree))/std(indegree)" % FIC)
        axes[0].hist(p[trg_inds], 30)
        axes[0].set_xlabel("Parameter values")
        axes[0].set_ylabel("Histogram of region counts")
        axes[0].set_title("FICed parameter %s%s = %g (1 + Indegree scaler))" % (param, subtitle, pscalar))
        fig.tight_layout()
    return p


def prepare_fic(simulator, inds, FIC, G, print_flag=True, plotter=None):
    # Optimize w_ie and w_rs according to total indegree and G
    if FIC and G > 0.0:

        # Indices of cortical and subcortical regions excluding specific thalami
        inds["non_thalamic"] = np.unique(inds['crtx'].tolist() + inds["subcrtx_not_thalspec"].tolist())

        # FIC for cortical w_ie against indegree for all incoming connections exluding the ones from specific thalami
        simulator.model.w_ie = fic("w_ie", simulator.model.w_ie, simulator.connectivity.weights,
                                   inds["crtx"], inds["non_thalamic"],
                                   FIC=FIC, dummy=None, subtitle=" for cortex", plotter=plotter)

        w_to_subcrtx = simulator.connectivity.weights[inds["subcrtx_not_thalspec"]].sum()
        if w_to_subcrtx:
            # FIC for subcortical w_ie against indegree for all incoming connections including the ones from specific thalami
            simulator.model.w_ie = fic("w_ie", simulator.model.w_ie, simulator.connectivity.weights,
                                       inds["subcrtx_not_thalspec"],
                                       src_inds=inds["non_thalamic"],  # after removal of subcrtx <-> specific thalamic
                                       FIC=FIC, dummy=None, subtitle=" for subcortex", plotter=plotter)

        # # !!!Not needed after removal of subcrtx <-> specific thalamic connections!!!
        # w_subcrtx_to_thal = simulator.connectivity.weights[inds["thalspec"]][:, inds["subcrtx_not_thalspec"]].sum()
        # if w_subcrtx_to_thal:
        #     # FIC for specific thalami w_rs against indegree for incoming connections from subcortical regions
        #     simulator.model.w_rs = fic("w_rs", simulator.model.w_rs, simulator.connectivity.weights,
        #                                inds["thalspec"], inds["subcrtx_not_thalspec"],
        #                                FIC=FICeff, dummy=None, subtitle=" for specific thalami", plotter=plotter)

        return simulator


def build_simulator(connectivity, model, inds, maps, config, print_flag=True, plotter=None):
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import SigmoidalPreThalamoCortical
    from tvb.simulator.monitors import Raw, Bold, TemporalAverage

    simulator = CoSimulatorSerial()

    simulator.model = model
    simulator.connectivity = connectivity

    dummy = np.ones((simulator.connectivity.number_of_regions,))

    # Variability to thalamocortical connections:
    if config.THAL_CRTX_FIX:

        if "w" in config.THAL_CRTX_FIX:
            # Fix structural connectivity (specific) thalamo-cortical weights to 1,
            # such that all thalamo-cortical weights are equal to the parameters
            # w_er, w_es, w_se, w_si
            simulator.connectivity.weights[inds["crtx"], inds["thalspec"]] = 1.0
            simulator.connectivity.weights[inds["thalspec"], inds["crtx"]] = 1.0

        if "d" in config.THAL_CRTX_FIX:
            # Fix structural connectivity (specific) thalamo-cortical tracts length to a value,
            # such that all thalamo-cortical delays are equal to the parameter tau_ct,
            # given connectivity's speed.
            ct_lengths = simulator.connectivity.speed * \
                         simulator.model.tau_ct * dummy[inds["crtx"]]
            simulator.connectivity.tract_lengths[inds["crtx"], inds["thalspec"]] = ct_lengths
            simulator.connectivity.tract_lengths[inds["thalspec"], inds["crtx"]] = ct_lengths

        simulator.connectivity.configure()

    if not config.THAL_CRTX_FIX or "d" not in config.THAL_CRTX_FIX:
        tau_ct = simulator.model.tau_ct * dummy
        tau_ct[inds['crtx']] = simulator.connectivity.delays[inds["thalspec"], inds["crtx"]]
        tau_ct[inds['thalspec']] = simulator.connectivity.delays[inds["crtx"], inds["thalspec"]]
        simulator.model.tau_ct = tau_ct

    # h_sub2 = np.array(simulator.connectivity.weights[inds["not_subcrtx_not_thalspec"][:, None],
    #                                                  inds["subcrtx_not_thalspec"][None, :]].flatten().tolist() +
    #                   simulator.connectivity.weights[inds["subcrtx_not_thalspec"][:, None],
    #                                                  inds["not_subcrtx_not_thalspec"][None, :]].flatten().tolist())
    # h_sub2, bins = np.histogram(h_sub2[h_sub2>0].flatten(), range=(0.0, 1.25), bins=100)
    # wp = simulator.connectivity.weights > 0
    # h2, bins = np.histogram(simulator.connectivity.weights[wp].flatten(), range=(0.0, 1.25), bins=100)

    # if plot_flag:
    # x = bins[:-1] + np.diff(bins)/2
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, h1, 'b', label='All connections before downscaling')
    # plt.plot(x, h2, 'b--', label='All connections after downscaling')
    # plt.plot(x, h_sub1, 'r', label='Subcortical connections before downscaling')
    # plt.plot(x, h_sub2, 'r--', label='Subcortical connections after downscaling')
    # plt.title("Histogram of logtransformed connectome weights\nwith downscaling connections")
    # plt.legend()
    # plt.ylim([0.0, h1.max()])
    # plt.tight_layout()

    simulator.coupling = SigmoidalPreThalamoCortical(
        is_thalamic=maps['is_thalamic'],
        is_subcortical=maps['is_subcortical'],
        sigma=np.array([1.0]),
        midpoint=simulator.model.sigma,
        cmin=np.array([0.0]),
        cmax=np.array([1.0]),
        a=simulator.model.beta)

    # Set integrator abnd noise
    simulator.integrator = EulerStochastic()
    simulator.integrator.dt = config.DEFAULT_DT
    simulator.integrator.noise.nsig = np.array(
        [config.DEFAULT_NSIG] * (simulator.model.nvar - 1) + [0.0])  # No Noise for state variabla A for BOLD monitor

    # Set initial conditions around zero
    simulator.initial_conditions = 0.1 * np.random.normal(size=(1000, simulator.model.nvar,
                                                                connectivity.number_of_regions, 1))

    if config.FIC:
        # We will modify the w_ie and w_rs parameters a bit based on indegree and G:
        simulator = prepare_fic(simulator, inds, config.FIC, simulator.model.G[0], print_flag, plotter)
        # We will not run FIC though when fitting...
        # simulator.initial_conditions[:, -1, maps['is_thalamic'], :] = simulator.model.w_rs[
        #     None, maps['is_thalamic'], None]
        # simulator.initial_conditions[:, -1, maps['not_thalamic'], :] = simulator.model.w_ie[
        #     None, maps['not_thalamic'], None]
        # simulator.model.eta = np.array([-0.05])

    # Set monitors:
    if config.RAW_PERIOD > config.DEFAULT_DT:
        mon_raw = TemporalAverage(period=config.RAW_PERIOD)  # ms
    else:
        mon_raw = Raw()
    if config.BOLD_PERIOD:
        bold = Bold(period=config.BOLD_PERIOD,
                    variables_of_interest=np.array([2]))  # !!! Set a proper (1-2 sec??) TR time in ms !!!
        simulator.monitors = (mon_raw, bold)
    else:
        simulator.monitors = (mon_raw,)

    simulator.configure()

    simulator.integrate_next_step = simulator.integrator.integrate_with_update

    if print_flag:
        simulator.print_summary_info_details(recursive=1)

    # Serializing TVB cosimulator is necessary for parallel cosimulation:
    from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = serialize_tvb_cosimulator(simulator)

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    return simulator


def configure_simulation_length_with_transient(config):
    # Compute transient as a percentage of the total simulation length, and add it to the simulation length:
    simulation_length = float(config.SIMULATION_LENGTH)
    transient = config.TRANSIENT_RATIO * simulation_length
    simulation_length += transient
    return simulation_length, transient


def simulate(simulator, config, print_flag=True):
    simulator.simulation_length, transient = configure_simulation_length_with_transient(config)
    # Simulate and return results
    tic = time.time()
    results = simulator.run()
    if print_flag:
        print("\nSimulated in %f secs!" % (time.time() - tic))
    return results, transient


def compute_target_PSDs(config, write_files=True, plotter=None):
    # Load Popa 2013 files:
    psd_m1 = np.load(os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_M1.npy"))
    psd_s1 = np.load(os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_S1.npy"))

    # Interpolate to the desired frequency range:
    f = config.TARGET_FREQS
    psd_m1_target = np.interp(f, psd_m1[:, 0], psd_m1[:, 1])
    psd_s1_target = np.interp(f, psd_s1[:, 0], psd_s1[:, 1])

    # Normalize to generate a PSD:
    psd_m1_target = psd_m1_target / psd_m1_target.sum()
    psd_s1_target = psd_s1_target / psd_s1_target.sum()

    PSD_target = {"f": f, "PSD_M1_target": psd_m1_target, "PSD_S1_target": psd_s1_target}
    if write_files:
        np.save(config.PSD_TARGET_PATH, PSD_target)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(f, psd_m1_target, "b", label='M1 PS')
        axes[0].plot(f, psd_s1_target, "g", label='S1 PS')
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(f, psd_m1_target, "b", label='M1 PS')
        axes[1].semilogy(f, psd_s1_target, "g", label='S1 PS')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')

    return PSD_target


def compute_data_PSDs(raw_results, PSD_target, inds, transient=None, write_files=True, plotter=None):
    # Time and frequency
    dt = np.mean(np.diff(raw_results[0]))
    fs = 1000.0 / dt  # sampling frequency in sec
    if transient is None:
        transient = raw_results[0][0]
    transient = int(np.ceil(transient / dt))  # in data points

    # data
    data = raw_results[1][transient:, 0, inds['m1s1brl'], 0].squeeze().T

    # Window:
    # NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    ftarg = PSD_target["f"]
    # fmin = ftarg[0]  # The minimum frequency of the PSD_target...
    # win_len = int(np.ceil(1000.0 / fmin / dt))  # ...will determine the length of the sliding window....
    nperseg = int(np.ceil(2048 / dt))  # NPERSEG[np.argmin(np.abs(NPERSEG - win_len))]

    # Compute Power Spectrum
    f, Pxx_den = welch(data, fs, nperseg=nperseg)

    print(Pxx_den.shape)

    # Compute spectrum interpolation...
    interp = interp1d(f, Pxx_den, kind='linear', axis=1,
                      copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)

    # ...to the target frequencies:
    Pxx_den = interp(PSD_target["f"])

    # Normalize to get a density summing to 1.0:
    for ii in range(4):
        Pxx_den[ii] = Pxx_den[ii] / np.sum(Pxx_den[ii])

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 PS target')
        axes[0].plot(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 PS target')
        axes[0].plot(ftarg, Pxx_den[0], "b--", label='M1 PS right')
        axes[0].plot(ftarg, Pxx_den[1], "b-.", label='M1 PS left')
        axes[0].plot(ftarg, Pxx_den[2], "g--", label='S1 PS right')
        axes[0].plot(ftarg, Pxx_den[3], "g-.", label='S1 PS left')
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 PS')
        axes[1].semilogy(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 PS')
        axes[1].semilogy(ftarg, Pxx_den[0], "b--", label='M1 PS right')
        axes[1].semilogy(ftarg, Pxx_den[1], "b-.", label='M1 PS left')
        axes[1].semilogy(ftarg, Pxx_den[2], "g--", label='S1 PS right')
        axes[1].semilogy(ftarg, Pxx_den[3], "g-.", label='S1 PS left')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')

    if write_files:
        np.save
    return Pxx_den.flatten()


# def plot_tvb(results, simulation_length, transient, config=None):


#def tvb_res_to_time_series()

def run_workflow(G=5.0, STIMULUS=0.25,
                 I_E=-0.25, I_S=0.25,
                 W_IE=-3.0, W_RS=-2.0,
                 #TAU_E=10/0.9, TAU_I=10/0.9, TAU_S=10/0.25, TAU_R=10/0.25,
                 PSD_target=None, plot_flag=True, output_folder=None):
    # Get configuration
    config, plotter = configure(G, STIMULUS, I_E, I_S, W_IE, W_RS,
                                #TAU_E, TAU_I, TAU_S, TAU_R,
                                plot_flag=plot_flag, output_folder=output_folder)
    # Load connectome and other structural files
    connectome, major_structs_labels, voxel_count, inds = load_connectome(config, plotter=plotter)
    # Construct some more indices and maps
    inds, maps = construct_extra_inds_and_maps(connectome, inds)
    # Logprocess connectome
    connectome = logprocess_weights(connectome, inds, print_flag=True, plotter=plotter)
    # Prepare connectivity with all possible normalizations
    connectivity = build_connectivity(connectome, inds, config, print_flag=True, plotter=plotter)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, print_flag=True, plotter=plotter)
    # Run simulation and get results
    results, transient = simulate(simulator, config, print_flag=True)

    if PSD_target is None:
        # This is the PSD target we are trying to fit:
        PSD_target = compute_target_PSDs(config, write_files=True, plotter=plotter)
    # This is the PSD computed from our simulation results.
    PSD = compute_data_PSDs(results[0], PSD_target, inds, transient, plotter=plotter)

    if plot_flag:
        output_config = deepcopy(config)
        output_config.inds = inds
        output_config.maps = maps
        output_config.transient = transient
        return PSD, results, simulator, output_config
    else:
        return PSD, results
