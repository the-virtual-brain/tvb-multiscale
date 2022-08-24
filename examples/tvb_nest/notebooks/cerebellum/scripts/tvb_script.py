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
    if config.VERBOSE > 1:
        print("major_structs_labels:\n", np.unique(major_structs_labels))
        print("ROI inds:\n", inds)

    return connectome, major_structs_labels, voxel_count, inds


def construct_extra_inds_and_maps(connectome, inds):
    maps = {}
    region_labels = connectome['region_labels']
    inds["subcrtx"] = np.arange(len(region_labels)).astype('i')
    inds["subcrtx"] = np.delete(inds["subcrtx"], inds["crtx"])
    inds['crtx_and_subcrtx'] = np.sort(np.concatenate([inds['crtx'], inds["subcrtx"]]))
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


def plot_norm_w_hist(w, wp, inds, plotter_config, title_string=""):
    h = w[wp].flatten()
    # print('number of all connections > 0: %d' % h.size)
    h, bins = np.histogram(h, range=(1.0, 31), bins=100)

    w_within_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_from_sub = w[inds["not_subcrtx_not_thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]]
    w_to_sub = w[inds["subcrtx_not_thalspec"][:, None], inds["not_subcrtx_not_thalspec"][None, :]]
    h_sub = np.array(w_within_sub.flatten().tolist() +
                     w_from_sub.flatten().tolist() +
                     w_to_sub.flatten().tolist())
    h_sub = h_sub[h_sub > 0].flatten()
    # print('number of h_sub > 0: %d' % h_sub.size)
    h_sub, bins_sub = np.histogram(h_sub, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_sub)

    h_crtx = np.array(w[inds["not_subcrtx_not_thalspec"][:, None],
                        inds["not_subcrtx_not_thalspec"][None, :]].flatten().tolist())
    h_crtx = h_crtx[h_crtx > 0]
    # print('number of h_crtx > 0: %d' % h_crtx.size)
    h_crtx, bins_crtx = np.histogram(h_crtx, range=(1.0, 31), bins=100)
    assert np.all(bins == bins_crtx)

    h2 = h_crtx + h_sub
    # print('number of total > 0: %d' % np.sum(h2))

    x = bins[:-1] + np.diff(bins) / 2
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, h, 'b', label='All connections')
    plt.plot(x, h_crtx, 'g', label='Non Subcortical connections')
    plt.plot(x, h_sub, 'r', label='Subcortical connections')
    # plt.plot(x, h-h_sub, 'r--', label='All - Subcortical connections')
    # plt.plot(x, h-h_crtx, 'g--', label='All - Non Subcortical connections')
    # plt.plot(x, h2, 'k--', label='Total connections')
    plt.title("Histogram of %s connectome weights" % title_string)
    plt.legend()
    plt.ylim([0.0, h.max()])
    plt.tight_layout()
    plt.savefig(os.path.join(plotter_config.FOLDER_FIGURES, "%sWeightsHistogram.png" % title_string))
    fig.show()
    return fig


def logprocess_weights(connectome, inds, verbose=1, plotter=None):
    w = connectome['weights'].copy()
    w[np.isnan(w)] = 0.0  # zero nans
    w0 = w <= 0  # zero weights
    wp = w > 0  # positive weights
    if plotter:
        print("\nPlotting weights' histogram...")
        plot_norm_w_hist(w, wp, inds, plotter.config)
    w /= w[wp].min()  # divide by the minimum to have a minimum of 1.0
    w *= np.exp(1)  # multiply by e to have a minimum of e
    w[wp] = np.log(w[wp])  # log positive values
    w[w0] = 0.0  # zero zero values (redundant)
    connectome['weights'] = w
    if verbose > 1:
        print('\nnormalized weights [min, max] = \n', [w[wp].min(), w[wp].max()])
    if plotter:
        print("\nPlotting logtransformed weights' histogram...")
        plot_norm_w_hist(w, wp, inds, plotter.config, title_string="logtransformed ")
    return connectome


def prepare_connectome(config, plotter=None):
    # Load connectome and other structural files
    connectome, major_structs_labels, voxel_count, inds = load_connectome(config, plotter=plotter)
    # Construct some more indices and maps
    inds, maps = construct_extra_inds_and_maps(connectome, inds)
    if config.CONN_LOG:
        if config.VERBOSE:
            print("Logtransforming connectivity weights!")
        # Logprocess connectome
        connectome = logprocess_weights(connectome, inds, verbose=config.VERBOSE, plotter=plotter)
    # Prepare connectivity with all possible normalizations
    return connectome, major_structs_labels, voxel_count, inds, maps


def build_connectivity(connectome, inds, config):
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
        if config.VERBOSE:
            print("Scaling connectivity weights with %s!" % config.CONN_SCALE)
        connectivity.weights = connectivity.scaled_weights(mode=config.CONN_SCALE)
    if config.CONN_NORM_PERCENTILE:
        if config.VERBOSE:
            print("Normalizing connectivity weights with %g percentile!" % config.CONN_NORM_PERCENTILE)
        connectivity.weights /= np.percentile(connectivity.weights, config.CONN_NORM_PERCENTILE)
    if config.CONN_CEIL:
        if config.VERBOSE:
            print("Ceiling connectivity to %g!" % config.CONN_CEIL)
        connectivity.weights[connectivity.weights > config.CONN_CEIL] = config.CONN_CEIL

    connectivity.speed = np.array([config.CONN_SPEED])
    connectivity.tract_lengths = np.maximum(connectivity.speed * config.DEFAULT_DT,
                                            connectivity.tract_lengths)

    connectivity.configure()

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

    if config.VERBOSE:
        print("Configuring model with parameters:\n%s" % str(config.model_params))

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
        if config.NEST_PERIPHERY:
            inds_stim = np.array(inds["facial"])
        else:
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


def prepare_fic(simulator, inds, FIC, G, plotter=None):
    # Optimize w_ie and w_rs according to total indegree and G
    if FIC and G > 0.0:
        # Indices of cortical and subcortical regions excluding specific thalami
        inds["non_thalamic"] = np.unique(inds['crtx'].tolist() + inds["subcrtx_not_thalspec"].tolist())

        # FIC for cortical w_ie against indegree for all incoming connections excluding the ones from specific thalami
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


def build_simulator(connectivity, model, inds, maps, config, plotter=None):
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
            if config.VERBOSE:
                print("Fixing thalamocortical weights!")
            # Fix structural connectivity (specific) thalamo-cortical weights to 1,
            # such that all thalamo-cortical weights are equal to the parameters
            # w_er, w_es, w_se, w_si
            simulator.connectivity.weights[inds["crtx"], inds["thalspec"]] = 1.0
            simulator.connectivity.weights[inds["thalspec"], inds["crtx"]] = 1.0

        if "d" in config.THAL_CRTX_FIX:
            if config.VERBOSE:
                print("Fixing thalamocortical delays!")
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

    # Set initial conditions around baseline currents of each kind of population for a shorter transient:
    simulator.initial_conditions = np.zeros((1000, simulator.model.nvar, connectivity.number_of_regions, 1))
    n_crtx_subcrtx = len(inds['crtx_and_subcrtx'])
    simulator.initial_conditions[:, [[0]], inds['crtx_and_subcrtx']] =\
        simulator.model.I_e.mean().item() + 0.1 * np.random.normal(size=(1000, 1, n_crtx_subcrtx, 1))
    simulator.initial_conditions[:, [[1]], inds['crtx_and_subcrtx']] = \
        simulator.model.I_i.mean().item() + 0.1 * np.random.normal(size=(1000, 1, n_crtx_subcrtx, 1))
    n_thalspec = len(inds['thalspec'])
    simulator.initial_conditions[:, [[0]], inds['thalspec']] = \
        simulator.model.I_s.mean().item() + 0.1 * np.random.normal(size=(1000, 1, n_thalspec, 1))
    simulator.initial_conditions[:, [[1]], inds['thalspec']] = \
        simulator.model.I_r.mean().item() + 0.1 * np.random.normal(size=(1000, 1, n_thalspec, 1))

    if config.FIC:
        if config.VERBOSE:
            print("Applying FIC!")
        # We will modify the w_ie and w_rs parameters a bit based on indegree and G:
        simulator = prepare_fic(simulator, inds, config.FIC, simulator.model.G[0], plotter)
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

    if config.VERBOSE > 1:
        simulator.print_summary_info_details(recursive=config.VERBOSE)

    # Serializing TVB cosimulator is necessary for parallel cosimulation:
    from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    sim_serial = serialize_tvb_cosimulator(simulator)

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    if plotter:
        # Plot TVB connectome:
        plotter.plot_tvb_connectivity(simulator.connectivity);

    return simulator


def configure_simulation_length_with_transient(config):
    # Compute transient as a percentage of the total simulation length, and add it to the simulation length:
    simulation_length = float(config.SIMULATION_LENGTH)
    transient = config.TRANSIENT_RATIO * simulation_length
    simulation_length += transient
    return simulation_length, transient


def simulate(simulator, config):
    simulator.simulation_length, transient = configure_simulation_length_with_transient(config)
    # Simulate and return results
    tic = time.time()
    results = simulator.run()
    if config.VERBOSE:
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

    # print(Pxx_den.shape)

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

    # if write_files:
    #     np.save
    return Pxx_den.flatten()


def tvb_res_to_time_series(results, simulator, config=None, write_files=True):

    config = assert_config(config, return_plotter=False)

    writer = False
    # if write_files:
    #     # If you want to see what the function above does, take the steps, one by one
    #     try:
    #         # We need framework_tvb for writing and reading from HDF5 files
    #         from tvb_multiscale.core.tvb.io.h5_writer import H5Writer
    #         writer = H5Writer()
    #     except:
    #         warning("H5Writer cannot be imported! Probably you haven't installed tvb_framework.")

    from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
    from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

    # Put the results in a Timeseries instance
    from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion

    source_ts = None
    bold_ts = None

    outputs = []
    if results is not None:
        source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
            data=results[0][1], time=results[0][0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.integrator.dt)

        source_ts.configure()

        outputs.append(source_ts)

        # t = source_ts.time

        # Write to file
        if writer:
            writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                            connectivity=source_ts.connectivity),
                                   os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
        # print("Raw ts:\n%s" % str(source_ts))

        if len(results) > 1:
            bold_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
                data=results[1][1], time=results[1][0],
                connectivity=simulator.connectivity,
                labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                labels_dimensions={"State Variable": ["BOLD"],
                                   "Region": simulator.connectivity.region_labels.tolist()})
            bold_ts.configure()

            outputs.append(bold_ts)

            # bold_t = source_ts.time

            # Write to file
            if writer:
                writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(bold_ts._data,
                                                                                connectivity=bold_ts.connectivity),
                                       os.path.join(config.out.FOLDER_RES, bold_ts.title) + ".h5")
            # print("BOLD ts:\n%s" % str(bold_ts))

    return tuple(outputs)


def plot_tvb(transient, inds,
             results=None, source_ts=None, bold_ts=None, PSD_target=None, PSD=None,
             simulator=None, plotter=None, config=None, write_files=True):
    config, plotter = assert_config(config, return_plotter=True)
    MAX_VARS_IN_COLS = 2
    MAX_REGIONS_IN_ROWS = 10
    MIN_REGIONS_FOR_RASTER_PLOT = 9
    transient += 0.5
    FIGSIZE = config.figures.DEFAULT_SIZE

    if source_ts is None:
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
        source_ts = results[0]
        if len(results) > 1:
            bold_ts = results[1]

    # Plot TVB time series
    if source_ts is not None:
        source_ts[:, :, :, :].plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if source_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                                              figsize=FIGSIZE);
    # Focus on the m1 and s1 barrel field nodes:
    if source_ts is not None:
        source_ts_m1s1brl = source_ts[-10000:, :, inds["m1s1brl"]]
        source_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                          hue="Region" if source_ts_m1s1brl.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                          per_variable=source_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                          figsize=FIGSIZE, figname="M1 and S1 barrel field nodes TVB Time Series");
    # Focus on the the motor pathway:
    if source_ts is not None:
        source_ts_motor = source_ts[-10000:, :, inds["motor"]]
        source_ts_motor.plot_timeseries(plotter_config=plotter.config,
                                        hue="Region" if source_ts_motor.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                        per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                        figsize=FIGSIZE, figname="Motor pathway TVB Time Series");
    # Focus on the motor pathway: raster plot
    if source_ts_motor is not None and source_ts_motor.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
        source_ts_motor.plot_raster(plotter_config=plotter.config,
                                    per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                    figsize=FIGSIZE, figname="Motor pathway TVB Time Series Raster");
    # Focus on the sensory pathway:
    if source_ts is not None:
        source_ts_sens = source_ts[-10000:, :, inds["sens"]]
        source_ts_sens.plot_timeseries(plotter_config=plotter.config,
                                       hue="Region" if source_ts_sens.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                       per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                       figsize=FIGSIZE, figname="Sensory pathway TVB Time Series");
    # Focus on the sensory pathway: raster plot
    if source_ts is not None and source_ts_sens.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
        source_ts_sens.plot_raster(plotter_config=plotter.config,
                                   per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                   figsize=FIGSIZE, figname="Sensory pathway TVB Time Series Raster");
    # bold_ts TVB time series
    if bold_ts is not None:
        bold_ts.plot_timeseries(plotter_config=plotter.config,
                                hue="Region" if bold_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                per_variable=bold_ts.shape[1] > MAX_VARS_IN_COLS,
                                figsize=FIGSIZE);

    # PSD results versus target plot:
    if PSD_target is None:
        PSD_target = compute_target_PSDs(config, write_files=False, plotter=None)
        PSD = compute_data_PSDs([source_ts.time, source_ts.data], PSD_target, inds, transient,
                                write_files=False, plotter=plotter)

    from examples.tvb_nest.notebooks.cerebellum.utils import compute_plot_selected_spectra_coherence #, compute_plot_ica

    # Further spectra and coherence plots:

    NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    NPERSEG = NPERSEG[np.argmin(np.abs(NPERSEG - (source_ts.shape[0] - transient / config.DEFAULT_DT)))]

    # Power Spectra and Coherence for M1 - S1 barrel field
    CxyR, fR, fL, CxyL = compute_plot_selected_spectra_coherence(source_ts, inds["m1s1brl"],
                                                                 transient=transient, nperseg=NPERSEG, fmin=0.0,
                                                                 fmax=100.0)
    if write_files:
        import pickle
        with open('coherence_MF_cerebON_2sec.pickle', 'wb') as handle:
            pickle.dump([CxyR, fR, fL, CxyL], handle)

    # Power Spectra and Coherence along the sensory pathway:
    # for Medulla SPV, Sensory PONS
    compute_plot_selected_spectra_coherence(source_ts, inds["sens"],
                                            transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0)

    # Better summary figure:
    import matplotlib.pyplot as plt

    data = source_ts.data
    time = source_ts.time

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for iT, regs in enumerate(["crtx", "subcrtx_not_thalspec", "thalspec"]):
        transient_in_points = int((transient + 0.5) / simulator.monitors[0].period)
        dat = data[transient_in_points:, 0, inds[regs]].squeeze()
        axes[iT].plot(time[transient_in_points:], dat, alpha=0.25)
        if iT == 0:
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, inds["m1"]].squeeze(),
                          'b--', linewidth=3, label='M1')
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, inds["s1brl"]].squeeze(),
                          'g--', linewidth=3, label='S1 barrel field')
        elif iT == 1:
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, inds["facial"]].squeeze(),
                          'b--', linewidth=3, label='Facial motor nucleus')
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, inds["trigeminal"]].squeeze(),
                          'g--', linewidth=3, label='Spinal trigeminal nuclei')
        else:
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, [44, 166]].squeeze(),
                          'b--', linewidth=3, label='M1 specific thalami')
            axes[iT].plot(time[transient_in_points:], data[transient_in_points:, 0, [47, 169]].squeeze(),
                          'g--', linewidth=3, label='S1 barrel field specific thalami')
            axes[iT].set_xlabel('Time (ms)')
        axes[iT].plot(time[transient_in_points:], dat.mean(axis=1), 'k--', linewidth=3, label='Total mean')
        axes[iT].legend()
        axes[iT].set_title("%s range=[%g, %g, %g, %g, %g] " %
                           (regs, dat.min(), np.percentile(dat, 5), dat.mean(), np.percentile(dat, 95), dat.max()))
    fig.tight_layout()
    if config.figures.SAVE_FLAG:
        plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, "SummaryTimeSeries." + config.figures.FIG_FORMAT))


def run_workflow(PSD_target=None, model_params={}, config=None, **config_args):
    # Get configuration
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)
    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Run simulation and get results
    results, transient = simulate(simulator, config)
    if PSD_target is None:
        # This is the PSD target we are trying to fit:
        PSD_target = compute_target_PSDs(config, write_files=True, plotter=plotter)
    # This is the PSD computed from our simulation results.
    PSD = compute_data_PSDs(results[0], PSD_target, inds, transient, plotter=plotter)
    if config_args.get('plot_flag', True):
        plot_tvb(transient, inds, results=results,
                 source_ts=None, bold_ts=None, PSD_target=PSD_target, PSD=PSD,
                 simulator=simulator, plotter=plotter, config=config, write_files=True)
        return PSD, results, simulator, config
    else:
        return PSD, results


if __name__ == "__main__":
    parser = args_parser("tvb_script")
    args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
    verbose = args.get('verbose', DEFAULT_ARGS['verbose'])
    if verbose:
        print("Running %s with arguments:\n" % parser.description)
        print(args, "\n")
    run_workflow(**args)
