# -*- coding: utf-8 -*-

import warnings
from scipy.signal import welch
from scipy.interpolate import interp1d

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.utils import get_regions_indices
from tvb_multiscale.core.utils.file_utils import dump_pickled_dict


def load_connectome(config):
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
    plt.plot(x, h_crtx, 'g', label='Isocortical connections')
    plt.plot(x, h_sub, 'r', label='Non-isocortical connections')
    # plt.plot(x, h-h_sub, 'r--', label='All - Subcortical connections')
    # plt.plot(x, h-h_crtx, 'g--', label='All - Non Subcortical connections')
    # plt.plot(x, h2, 'k--', label='Total connections')
    plt.title("Histogram of %s connectome weights" % title_string)
    plt.legend()
    plt.ylim([0.0, h.max()])
    plt.tight_layout()
    if plotter_config.SAVE_FLAG:
        plt.savefig(os.path.join(plotter_config.FOLDER_FIGURES, "%sWeightsHistogram.png" % title_string))
    if plotter_config.SHOW_FLAG:
        fig.show()
    else:
        plt.close(fig)
    return fig


def logprocess_weights(connectome, inds, verbose=1, plotter=None):
    w = connectome['weights'].copy()
    w[np.isnan(w)] = 0.0  # zero nans
    w0 = w <= 0  # zero weights
    wp = w > 0  # positive weights
    if plotter:
        plot_norm_w_hist(w, wp, inds, plotter.config)
    w /= w[wp].min()  # divide by the minimum to have a minimum of 1.0
    w *= np.exp(1)  # multiply by e to have a minimum of e
    w[wp] = np.log(w[wp])  # log positive values
    w[w0] = 0.0  # zero zero values (redundant)
    connectome['weights'] = w
    if verbose > 1:
        print('\nnormalized weights [min, max] = \n', [w[wp].min(), w[wp].max()])
    if plotter:
        plot_norm_w_hist(w, wp, inds, plotter.config, title_string="logtransformed ")
    return connectome


def prepare_connectome(config, plotter=None):
    # Load connectome and other structural files
    connectome, major_structs_labels, voxel_count, inds = load_connectome(config)
    # Construct some more indices and maps
    inds, maps = construct_extra_inds_and_maps(connectome, inds)
    if config.CONN_LOG:
        if config.VERBOSE:
            print("Logtransforming connectivity weights!")
        # Logprocess connectome
        connectome = logprocess_weights(connectome, inds, verbose=config.VERBOSE, plotter=plotter)
    # Prepare connectivity with all possible normalizations
    return connectome, major_structs_labels, voxel_count, inds, maps


def scale_connections(connectivity, brain_connections_to_scale):
    # Selectively scale some connections, if required:
    for reg1, reg2, sc in brain_connections_to_scale:
        connectivity.weights[get_regions_indices(reg1, connectivity.region_labels),
                             get_regions_indices(reg2, connectivity.region_labels)] *= sc
    return connectivity


def build_connectivity(connectome, inds, config):
    from tvb.datatypes.connectivity import Connectivity

    connectivity = Connectivity(**connectome)

    # Selectively scale some connections, if required:
    if config.BRAIN_CONNECTIONS_TO_SCALE:
        connectivity = scale_connections(connectivity, config.BRAIN_CONNECTIONS_TO_SCALE)

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
    # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
    w_s1brlthal_trigeminal = connectivity.weights[inds["s1brlthal"], inds["trigeminal"]].copy()
    # from merged Cerebellar Nuclei to M1:
    w_m1thal_cerebnuclei = connectivity.weights[inds["m1thal"], inds["cereb_nuclei"]].copy()
    connectivity.weights[inds["thalspec"][:, None], inds["subcrtx_not_thalspec"][None, :]] = 0.0
    connectivity.weights[inds["s1brlthal"], inds["trigeminal"]] = w_s1brlthal_trigeminal
    connectivity.weights[inds["m1thal"], inds["cereb_nuclei"]] = w_m1thal_cerebnuclei

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
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import WilsonCowanThalamoCortical

    dummy = np.ones((number_of_regions,))

    if config.VERBOSE:
        print("Configuring model with parameters:\n%s" % str(config.model_params))

    STIMULUS = config.model_params.get("STIMULUS", None)

    model_params = {}
    for p, pval in config.model_params.items():
        if p != "STIMULUS":
            if pval is not None:
                pval = np.array([pval]).flatten()
                if p == 'G':
                    # G normalized by the number of regions as in Griffiths et al paper
                    # Geff = G /(number_of_regions - inds['thalspec'].size)
                    pval = pval / (number_of_regions - inds['thalspec'].size)
                model_params[p] = pval

    if STIMULUS:
        if model_params.get("G", WilsonCowanThalamoCortical.G.default)[0].item() > 0.0:
            # Stimulus to M1 and S1 barrel field
            # inds_stim = np.concatenate((inds["motor"][:2], inds["sens"][-2:])
            if config.NEST_PERIPHERY:
                inds_stim = np.array(inds["facial"])
            else:
                inds_stim = np.concatenate((inds["facial"], inds["trigeminal"]))
        else:
            # Stimulus directly to all specific thalami:
            inds_stim = inds['thalspec']
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
    # from spinal nucleus of the trigeminal to S1 barrel field specific thalamus:
    model.G[inds["s1brlthal"]] = model.G[inds["crtx"][0]]
    # from Cerebellar Nuclei to M1:
    model.G[inds["m1thal"]] = model.G[inds["crtx"][0]]

    return model


# An approximate automatic FIC:

def fic(param, p_orig, weights, trg_inds=None, src_inds=None, FIC=1.0, G=None, dummy=None, subtitle="", plotter=None):
    number_of_regions = weights.shape[0]
    # This function will adjust inhibitory weights based on total indegree and some scaling
    if trg_inds is None:
        trg_inds = np.arange(number_of_regions).astype('i')

    if src_inds is None:
        src_inds = np.arange(number_of_regions).astype('i')

    # Scale w_ie or I_e to grow to greater negative values from the defaults
    p_orig = np.array(p_orig)
    if p_orig.size == 1:
        if dummy is None:
            dummy = np.ones((number_of_regions,))
            p_orig = p_orig.item() * dummy
    p = p_orig.copy()
    pscalar = p_orig[trg_inds].mean().item()
    # Move them to have a maximum of p_orig:
    # FICindegree = (indegree - indegree_min) / indegree_max
    indegree = weights[trg_inds][:, src_inds].sum(axis=1)
    FICindegree = (indegree - indegree.min()) / (indegree.max() - indegree.min())
    # p_fic = p * (1 + FIC * FICindegree) = p * (1 + FIC * (indegree - indegree_min) / (indegree_max - indegree_min))
    # assuming p < 0.0, and FIC >= 0.0
    if G is None:
        p[trg_inds] = pscalar * (1.0 + FIC * FICindegree)
    else:
        p[trg_inds] = pscalar * (1.0 + G * FIC * FICindegree)

    try:
        assert np.all(np.argsort(indegree) == np.argsort(-p[trg_inds]))  # the orderings should reverse
    except Exception as e:
        fig = plt.figure()
        plt.plot(indegree, p[trg_inds], "-o")
        if G is None:
            plt.xlabel("%g*indegree" % FIC)
        else:
            plt.xlabel("%g*%g*indegree" % (G, FIC))
        plt.ylabel("%s scaled" % param)
        plt.title("Testing indegree and parameter anti-correlation")
        plt.tight_layout()
        warnings.warn(str(e))
        # raise e

    # Plot and confirm:
    if plotter:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[1].hist(FICindegree, 30)
        axes[1].set_xlabel("Indegree Scaler values")
        axes[1].set_ylabel("Histogram of region counts")
        if G is None:
            axes[1].set_title("Indegree scaler = %g*(indegree - min(indegree)) / (max(indegree) - min(indegree))" % FIC)
        else:
            axes[1].set_title("Indegree scaler = %g*%g*(indegree - min(indegree)) / (max(indegree) - min(indegree))" % (G, FIC))
        axes[0].hist(p[trg_inds], 30)
        axes[0].set_xlabel("Parameter values")
        axes[0].set_ylabel("Histogram of region counts")
        axes[0].set_title("FICed parameter %s%s = %g * (1 + Indegree scaler)" % (param, subtitle, pscalar))
        fig.tight_layout()
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "FIC.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return p


def apply_fic(simulator, inds, FIC, G=None, param='w_ie', plotter=None):
    
    # Indices of cortical and subcortical regions excluding specific thalami
    inds["non_thalamic"] = np.unique(inds['crtx'].tolist() + inds["subcrtx_not_thalspec"].tolist())

    # FIC for all non-specific thalamic, cortical and subcortical, w_ie, 
    # against indegree for all incoming connections excluding the ones from specific thalami
    setattr(simulator.model, param, 
            fic(param, getattr(simulator.model, param), simulator.connectivity.weights,
                inds["non_thalamic"], inds["non_thalamic"], FIC=FIC, G=G, dummy=None, subtitle="", plotter=plotter))

    # # FIC for cortical w_ie against indegree for all incoming connections excluding the ones from specific thalami
    # setattr(simulator.model, param, 
    #         fic(param, getattr(simulator.model, param), simulator.connectivity.weights,
    #             inds["crtx"], inds["non_thalamic"], FIC=FIC, dummy=None, subtitle=" for cortex", plotter=plotter))

    # w_to_subcrtx = simulator.connectivity.weights[inds["subcrtx_not_thalspec"]].sum()
    # if w_to_subcrtx:
    #     # FIC for subcortical w_ie against indegree for all incoming connections excluding the ones from specific thalami
    #     setattr(simulator.model, param,
    #             fic(param, getattr(simulator.model, param), simulator.connectivity.weights,
    #                 inds["subcrtx_not_thalspec"],
    #                 src_inds=inds["non_thalamic"],  # after removal of subcrtx <-> specific thalamic
    #                 FIC=FIC, dummy=None, subtitle=" for subcortex", plotter=plotter))

    return simulator


def build_simulator(connectivity, model, inds, maps, config, plotter=None):
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
    from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb import SigmoidalPreThalamoCortical
    from tvb.simulator.monitors import Raw, Bold, TemporalAverage, AfferentCoupling, AfferentCouplingTemporalAverage

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

    # if plotter is not None:
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

    if config.FIC and simulator.model.G[0].item():
        n_non_thalamic_regions = (simulator.connectivity.weights.shape[0] - inds['thalspec'].size)
        G = simulator.model.G[0].item() * n_non_thalamic_regions
        if config.FIC_SPLIT is not None:
            FICsplit = [config.FIC_SPLIT, 1.0 - config.FIC_SPLIT]
        else:
            FICsplit = [1.0] * len(config.FIC_PARAMS)
        for fp, fv, split_string in zip(config.FIC_PARAMS, FICsplit, ["FIC_SPLIT", "(1.0-FIC_SPLIT)"]):
            ficsplit = config.FIC * fv
            fic = G * ficsplit
            if config.VERBOSE:
                print("Applying FIC for parameter %s: G * FIC * %s = %g * %g * %g = %g!" % (fp, split_string, G, config.FIC, fv, fic))
            # We will modify the w_ie and w_rs parameters a bit based on indegree:
            simulator = apply_fic(simulator, inds, ficsplit, G, fp, plotter)

    # Set monitors:
    if config.RAW_PERIOD > config.DEFAULT_DT:
        mon_raw = TemporalAverage(period=config.RAW_PERIOD)  # ms
        afferent = AfferentCouplingTemporalAverage(period=config.RAW_PERIOD, variables_of_interest=np.array([0, 1]))
    else:
        mon_raw = Raw()
        afferent = AfferentCoupling(variables_of_interest=np.array([0, 1]))
    if config.BOLD_PERIOD:
        bold = Bold(period=config.BOLD_PERIOD,
                    variables_of_interest=np.array([2]))
        simulator.monitors = (mon_raw, bold, afferent)
    else:
        simulator.monitors = (mon_raw, afferent)


    simulator.configure()

    simulator.integrate_next_step = simulator.integrator.integrate_with_update

    if config.VERBOSE > 1:
        simulator.print_summary_info_details(recursive=config.VERBOSE)

    # Serializing TVB cosimulator is necessary for parallel cosimulation:
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
    if config.RAW_PERIOD > config.DEFAULT_DT:
        transient = (transient // config.RAW_PERIOD) * config.RAW_PERIOD + config.RAW_PERIOD/2
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


def compute_target_PSDs(config):
    # Load Popa 2013 files:
    psd_m1 = np.load(os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_M1.npy"))
    psd_s1 = np.load(os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_S1.npy"))

    # Interpolate to the desired frequency range:
    psd_m1_target = np.interp(config.TARGET_FREQS, psd_m1[:, 0], psd_m1[:, 1])
    psd_s1_target = np.interp(config.TARGET_FREQS, psd_s1[:, 0], psd_s1[:, 1])

    # Normalize to generate a PSD:
    psd_m1_target = psd_m1_target / psd_m1_target.sum()
    psd_s1_target = psd_s1_target / psd_s1_target.sum()

    return psd_m1_target, psd_s1_target


def compute_target_PSDs_1D(config, write_files=True, plotter=None):
    # Load, interpolate and normalize Popa 2013 m1 and s1 power spectra:
    psd_m1_target, psd_s1_target = compute_target_PSDs(config)

    psd_target = (psd_m1_target + psd_s1_target)/2

    PSD_target = {"f": config.TARGET_FREQS, "PSD_target": psd_target}
    if write_files:
        np.save(config.PSD_TARGET_PATH, PSD_target)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(config.TARGET_FREQS, psd_target, "k")
        axes[0].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[0].set_ylabel('PS')
        axes[0].set_title('Target average of M1 and S1 PS')
        axes[1].semilogy(config.TARGET_FREQS, psd_target, "k")
        axes[1].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "TargetPSD1D.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig) 
    return PSD_target


def compute_target_PSDs_m1s1brl(config, write_files=True, plotter=None):
    # Load, interpolate and normalize Popa 2013 m1 and s1 power spectra:
    psd_m1_target, psd_s1_target = compute_target_PSDs(config)

    PSD_target = {"f": config.TARGET_FREQS, "PSD_M1_target": psd_m1_target, "PSD_S1_target": psd_s1_target}
    if write_files:
        np.save(config.PSD_TARGET_PATH, PSD_target)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(config.TARGET_FREQS, psd_m1_target, "b", label='M1')
        axes[0].plot(config.TARGET_FREQS, psd_s1_target, "g", label='S1')
        axes[0].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[0].set_ylabel('PS')
        axes[0].set_title('Target M1 and S1 PS')
        axes[0].legend()
        axes[1].semilogy(config.TARGET_FREQS, psd_m1_target, "b", label='M1')
        axes[1].semilogy(config.TARGET_FREQS, psd_s1_target, "g", label='S1')
        axes[1].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "TargetPSDm1s1brl.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return PSD_target


def compute_data_PSDs(data, dt, ftarg, transient=None, average_region_ps=False):
    # Time and frequency
    fs = 1000.0 / dt  # sampling frequency in sec
    if transient is None:
        transient = 0
    else:
        transient = int(np.ceil(transient / dt))  # in data points
    # Remove possible transient and transpose time and signals:
    data = data[transient:].T

    # Window:
    # NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    # fmin = ftarg[0]  # The minimum frequency of the PSD_target...
    # win_len = int(np.ceil(1000.0 / fmin / dt))  # ...will determine the length of the sliding window....
    nperseg = 512  # int(np.ceil(2048 / dt))  # NPERSEG[np.argmin(np.abs(NPERSEG - win_len))]

    # Compute Power Spectrum
    f, Pxx_den = welch(data, fs, nperseg=nperseg)

    if average_region_ps:
        # Average power spectra across regions for the case of 1D computations
        Pxx_den = Pxx_den.mean(axis=0, keepdims=True)

    # Compute spectrum interpolation...
    interp = interp1d(f, Pxx_den, kind='linear', axis=1,
                      copy=True, bounds_error=None, fill_value=0.0, assume_sorted=True)
    # ...to the target frequencies:
    Pxx_den = interp(ftarg)

    # Normalize to get a density summing to 1.0:
    for ii in range(Pxx_den.shape[0]):
        Pxx_den[ii] = Pxx_den[ii] / np.sum(Pxx_den[ii])

    return Pxx_den


def compute_data_PSDs_from_raw(raw_results, ftarg, inds, transient=None, average_region_ps=False):
    return compute_data_PSDs(raw_results[1][:, 0, inds, 0].squeeze(),
                             np.mean(np.diff(raw_results[0])),
                             ftarg,
                             transient=transient, average_region_ps=average_region_ps)


def compute_data_PSDs_1D(raw_results, PSD_target, inds, transient=None, write_files=True, plotter=None):

    # Select regions' data, compute PSDs, average them across region,
    # interpolate them to the target frequencies, and normalize them to sum up to 1.0:
    ftarg = PSD_target['f']
    Pxx_den = compute_data_PSDs_from_raw(raw_results, ftarg, inds['crtx'],
                                         transient=transient, average_region_ps=True)
    Pxx_den = Pxx_den.flatten()

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(ftarg, PSD_target['PSD_target'], "k", label='Target')
        axes[0].plot(ftarg, Pxx_den, "r", label='Cortical average')
        axes[0].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(ftarg, PSD_target['PSD_target'], "k", label='Target')
        axes[1].semilogy(ftarg, Pxx_den, "r", label='Cortical average')
        axes[1].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "DataVSTargetPSD1D.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    # if write_files:
    #     np.save
    return Pxx_den


def compute_data_PSDs_m1s1brl(raw_results, PSD_target, inds, transient=None, write_files=True, plotter=None):

    # Select regions' data, compute PSDs, interpolate them to the target frequencies, 
    # and normalize them to sum up to 1.0:
    ftarg = PSD_target['f']
    Pxx_den = compute_data_PSDs_from_raw(raw_results, ftarg, inds['m1s1brl'],
                                         transient=transient, average_region_ps=False)

    if plotter:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 target')
        axes[0].plot(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 target')
        axes[0].plot(ftarg, Pxx_den[0], "b--", label='M1 right')
        axes[0].plot(ftarg, Pxx_den[1], "b-.", label='M1 left')
        axes[0].plot(ftarg, Pxx_den[2], "g--", label='S1 right')
        axes[0].plot(ftarg, Pxx_den[3], "g-.", label='S1 left')
        axes[0].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[0].set_ylabel('PS')
        axes[0].legend()
        axes[1].semilogy(ftarg, PSD_target['PSD_M1_target'], "b", label='M1 target')
        axes[1].semilogy(ftarg, PSD_target['PSD_S1_target'], "g", label='S1 target')
        axes[1].semilogy(ftarg, Pxx_den[0], "b--", label='M1 right')
        axes[1].semilogy(ftarg, Pxx_den[1], "b-.", label='M1 left')
        axes[1].semilogy(ftarg, Pxx_den[2], "g--", label='S1 right')
        axes[1].semilogy(ftarg, Pxx_den[3], "g-.", label='S1 left')
        axes[1].set_xticks([6.0, 8.0, 10.0, 20.0, 30.0, 40.0])
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('log(PS)')
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "DataVSTargetPSDm1s1brl.png"))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)

    return Pxx_den.flatten()


def dump_picked_time_series(time_series, filepath):
    dump_pickled_dict({"time_series": time_series.data[:, :, :, 0],
                       "dimensions_labels": np.array(time_series.labels_ordering)[:-1],
                       "time": time_series.time, "time_unit": time_series.time_unit,
                       "sampling_period": time_series.sample_period,
                       "state_variables": np.array(time_series.variables_labels),
                       "region_labels": np.array(time_series.space_labels)},
                      filepath)
    return filepath


def tvb_res_to_time_series(results, simulator, config=None, write_files=True):

    config = assert_config(config, return_plotter=False)

    writer = False
    if write_files:
        # If you want to see what the function above does, take the steps, one by one
        try:
            # We need framework_tvb for writing and reading from HDF5 files
            from tvb_multiscale.core.tvb.io.h5_writer import H5Writer
            from examples.plot_write_results import write_RegionTimeSeriesXarray_to_h5
            writer = H5Writer()
        except:
            warnings.warn("H5Writer cannot be imported! Probably you haven't installed tvb_framework.")

    # Put the results in a Timeseries instance
    from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray

    source_ts = None
    bold_ts = None
    afferent_ts = None

    outputs = []
    if results is not None:
        source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion for TVB like functionality
            data=results[0][1], time=results[0][0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.integrator.dt)

        source_ts.configure()

        afferent_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
            data=results[-1][1], time=results[-1][0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": ["cortical coupling","subcortical coupling"],
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.integrator.dt)

        afferent_ts.configure()

        if write_files:
            if config.VERBOSE:
                print("Pickle-dumping source_ts to %s!" % config.SOURCE_TS_PATH)
            dump_picked_time_series(source_ts, config.SOURCE_TS_PATH)
            dump_picked_time_series(afferent_ts, config.AFFERENT_TS_PATH)
            # import pickle
            # with open(config.SOURCE_TS_PATH, 'wb') as handle:
            #     pickle.dump(source_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(config.AFFERENT_TS_PATH, 'wb') as handle:
            #     pickle.dump(afferent_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Write to file
        if writer:
            try:
                write_RegionTimeSeriesXarray_to_h5(source_ts, writer,
                                                   os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
            except Exception as e:
                    warnings.warn("Failed to to write source time series to file with error!:\n%s" % str(e))

        if config.VERBOSE > 1:
            print("Raw ts:\n%s" % str(source_ts))

        outputs.append(source_ts)

        # t = source_ts.time

        if len(results) > 2:
            print("results [1] inside tvb_script ", results[1])
            bold_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
                data=results[1][1], time=results[1][0],
                connectivity=simulator.connectivity,
                labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                labels_dimensions={"State Variable": ["BOLD"],
                                   "Region": simulator.connectivity.region_labels.tolist()},
                sample_period=simulator.monitors[1].period)
            bold_ts.configure()

            outputs.append(bold_ts)

            if write_files:
                if config.VERBOSE:
                    print("Pickle-dumping bold_ts to %s!" % config.BOLD_TS_PATH)
                dump_picked_time_series(bold_ts, config.BOLD_TS_PATH)

            # bold_t = source_ts.time

            # Write to file
            if writer:
                try:
                    write_RegionTimeSeriesXarray_to_h5(bold_ts._data, writer,
                                                       os.path.join(config.out.FOLDER_RES, bold_ts.title) + ".h5")
                except Exception as e:
                    warnings.warn("Failed to to write BOLD time series to file with error!:\n%s" % str(e))
            
            if config.VERBOSE > 1:
                print("BOLD ts:\n%s" % str(bold_ts))

        outputs.append(afferent_ts)
    return tuple(outputs)


def plot_tvb(transient, inds,
             results=None, source_ts=None, bold_ts=None, afferent_ts=None,
             simulator=None, plotter=None, config=None, write_files=True):
    if plotter is None:
        config, plotter = assert_config(config, return_plotter=True)
    else:
        config = assert_config(config, return_plotter=False)
    MAX_VARS_IN_COLS = 2
    MAX_REGIONS_IN_ROWS = 10
    MIN_REGIONS_FOR_RASTER_PLOT = 9
    FIGSIZE = config.figures.DEFAULT_SIZE

    outputs = ()
    if source_ts is None:
        results = tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
        outputs = results
        source_ts = results[0]
        if len(results) > 1:
            bold_ts = results[1]
            afferent_ts = results[-1]

    # Plot TVB time series
    if source_ts is not None:
        source_ts[:, :, :, :].plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if source_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                                              figsize=FIGSIZE)
    # Focus on the m1 and s1 barrel field nodes:
    if source_ts is not None:
        source_ts_m1s1brl = source_ts[-10000:, :, inds["m1s1brl"]]
        source_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                          hue="Region" if source_ts_m1s1brl.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                          per_variable=source_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                          figsize=FIGSIZE, figname="M1 and S1 barrel field nodes TVB Time Series")
    # Focus on the motor pathway:
    if source_ts is not None:
        source_ts_motor = source_ts[-10000:, :, inds["motor"]]
        source_ts_motor.plot_timeseries(plotter_config=plotter.config,
                                        hue="Region" if source_ts_motor.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                        per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                        figsize=FIGSIZE, figname="Motor pathway TVB Time Series")

    # Focus on the sensory pathway:
    if source_ts is not None:
        source_ts_sens = source_ts[-10000:, :, inds["sens"]]
        source_ts_sens.plot_timeseries(plotter_config=plotter.config,
                                       hue="Region" if source_ts_sens.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                       per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                       figsize=FIGSIZE, figname="Sensory pathway TVB Time Series")

    # Focus on regions potentially modelled in NEST (ansiform lobule, Cerebellar Nuclei, inferior olive):
    if source_ts is not None:
        source_ts_cereb = source_ts[-10000:, :, inds["cereb"]]
        source_ts_cereb.plot_timeseries(plotter_config=plotter.config,
                                        hue="Region" if source_ts_cereb.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                        per_variable=source_ts_cereb.shape[1] > MAX_VARS_IN_COLS,
                                        figsize=FIGSIZE, figname="Cerebellum TVB Time Series")

    # Focus on the s1 barrel field nodes:
    if afferent_ts is not None:
        afferent_ts_m1s1brl = afferent_ts[-10000:, :, inds["s1brlthal"]]
        afferent_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                           hue="Region" if afferent_ts_m1s1brl.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                           per_variable=afferent_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                           figsize=FIGSIZE, figname="S1 barrel field nodes TVB Time Series")
   # Focus on regions potentially modelled in NEST (ansiform lobule, interposed nucleus, inferior olive):
    if afferent_ts is not None:
        afferent_ts_cereb = afferent_ts[-10000:, :, inds["ansilob"]]
        afferent_ts_cereb.plot_timeseries(plotter_config=plotter.config,
                                          hue="Region" if afferent_ts_cereb.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                          per_variable=afferent_ts_cereb.shape[1] > MAX_VARS_IN_COLS,
                                          figsize=FIGSIZE, figname="Ansiform Lobule TVB Afferent Time Series")


    # bold_ts TVB time series
    if bold_ts is not None:
        bold_ts.plot_timeseries(plotter_config=plotter.config,
                                hue="Region" if bold_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                per_variable=bold_ts.shape[1] > MAX_VARS_IN_COLS,
                                figsize=FIGSIZE);

    from examples.tvb_nest.notebooks.cerebellum.utils import compute_plot_selected_spectra_coherence #, compute_plot_ica

    # Further spectra and coherence plots:

    # NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    # dt = source_ts.time[1] - source_ts.time[0]
    # NPERSEG = NPERSEG[np.argmin(np.abs(NPERSEG - (source_ts.shape[0] - transient / dt)/10))]

    NPERSEG = 512

    # Power Spectra and Coherence for M1 - S1 barrel field
    Pxx_den, f, CxyR, fR, CxyL, fL = \
        compute_plot_selected_spectra_coherence(source_ts, inds["m1s1brl"],
                                                transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0, 
                                                figures_path=config.figures.FOLDER_FIGURES,
                                                figname="M1_S1brl", figformat="png",
                                                show_flag=plotter.config.SHOW_FLAG, save_flag=plotter.config.SAVE_FLAG)
    if write_files:
        import pickle
        with open('coherence_MF_cerebON_2sec.pickle', 'wb') as handle:
            pickle.dump([CxyR, fR, fL, CxyL], handle)

    # Power Spectra and Coherence along the sensory pathway:
    # for Medulla SPV, Sensory PONS
    compute_plot_selected_spectra_coherence(source_ts, inds["sens"],
                                            transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0, 
                                            figures_path=config.figures.FOLDER_FIGURES,
                                            figname="SPV_PonsSens", figformat="png",
                                            show_flag=plotter.config.SHOW_FLAG, save_flag=plotter.config.SAVE_FLAG)
    print("psd input cereb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Power Spectra and Coherence at cerebellar input - ansiform lobule:
    print("inds ansilob",inds["ansilob"])
    print("Ansiform lobule source_ts PSD, with compute_plot_selected_spectra_coherence")
    compute_plot_selected_spectra_coherence(source_ts, inds["ansilob"],
                                            transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0,
                                            figures_path=config.figures.FOLDER_FIGURES, figname="AnsiLob", figformat="png",
                                            show_flag=plotter.config.SHOW_FLAG, save_flag=plotter.config.SAVE_FLAG)

    # Power Spectra and Coherence of cerebellar input - afferent to ansiform lobule:
    print("Ansiform lobule afferent PSD, with compute_plot_selected_spectra_coherence")
    Pxx_den_ansilob = []
    for iC in range(0, 2):
        CxyR, fR, fL, CxyL, Pxx_den_ansilob_temp = \
            compute_plot_selected_spectra_coherence(
                afferent_ts[:, iC], inds["ansilob"],
                transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0,
                figures_path=config.figures.FOLDER_FIGURES, figname="AnsiLob_afferent", figformat="png",
                show_flag=plotter.config.SHOW_FLAG, save_flag=plotter.config.SAVE_FLAG)
        Pxx_den_ansilob.append(Pxx_den_ansilob_temp)

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
    if config.figures.SHOW_FLAG:
        plt.show()
    else:
        plt.close(fig)

    return outputs, CxyR, fR, fL, CxyL


def ansilob_affrerent_coupling_psd_rmse(ref_mossy_firing, afferent_ts, ftarg=None, transient=None):
    if ftarg is None:
        # TODO: confirm that we like this ftarg!
        ftarg = np.arange(2.0, 51.0, 1.0)
    # Adding the time vector to ref_mossy_firing - for a sim duration of 10s and 2.5-ms time bins
    # TODO: Confirm that dt = 5.0 ms!
    Pxx_den_ref = compute_data_PSDs(ref_mossy_firing, 5.0, ftarg,
                                    transient=None, average_region_ps=False)
    # First sum up the (non)isocortical afferent couplings!
    #                                       iscortical                        non-isocortical
    total_afferent_ts_ansilob = afferent_ts[1][:, 0, inds["ansilob"]] + afferent_ts[1][:, 1, inds["ansilob"]]
    Pxx_den_ansilob = compute_data_PSDs(total_afferent_ts_ansilob.squeeze(),
                                        np.mean(np.diff(afferent_ts[0])), ftarg,
                                       transient=transient, average_region_ps=False)
    MSE = np.square(np.subtract(Pxx_den_ansilob, Pxx_den_ref)).mean()
    RMSE = math.sqrt(MSE)
    print("RMSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE with pathway gain = ", pathway_gain,
          " is ", RMSE)
    return RMSE


def run_workflow(PSD_target=None, model_params={}, config=None, write_files=True,
                 switch_off_cereb=False, **config_args):
    tic = time.time()
    # Get configuration
    plot_flag = config_args.get('plot_flag', DEFAULT_ARGS.get('plot_flag'))
    config, plotter = assert_config(config, return_plotter=True, **config_args)
    config.model_params.update(model_params)
    if config.VERBOSE:
        print("\n\n------------------------------------------------\n\n"+
              "Running TVB workflow for plot_flag=%s, write_files=%s,\nand model_params=\n%s...\n" 
              % (str(plot_flag), str(write_files), str(config.model_params)))
    # Load and prepare connectome and connectivity with all possible normalizations:
    connectome, major_structs_labels, voxel_count, inds, maps = prepare_connectome(config, plotter=plotter)
    connectivity = build_connectivity(connectome, inds, config)

    if switch_off_cereb:
        ## switch cereb OFF
        reg1='Left Cerebellar Cortex'
        reg2='Left Cerebellar Nuclei'
        reg3='Left Ansiform lobule'
        reg4='Left Interposed nucleus'
        reg5='Right Cerebellar Cortex'
        reg6='Right Cerebellar Nuclei'
        reg7='Right Ansiform lobule'
        reg8='Right Interposed nucleus'
        # find the indices in region labels of these strings
        iR1 = np.where([reg1 in reg for reg in connectivity.region_labels])[0]
        iR2 = np.where([reg2 in reg for reg in connectivity.region_labels])[0]
        iR3 = np.where([reg3 in reg for reg in connectivity.region_labels])[0]
        iR4 = np.where([reg4 in reg for reg in connectivity.region_labels])[0]
        iR5 = np.where([reg5 in reg for reg in connectivity.region_labels])[0]
        iR6 = np.where([reg6 in reg for reg in connectivity.region_labels])[0]
        iR7 = np.where([reg7 in reg for reg in connectivity.region_labels])[0]
        iR8 = np.where([reg8 in reg for reg in connectivity.region_labels])[0]
        for i in [iR1, iR2, iR3, iR4, iR5, iR6, iR7, iR8]:
           connectivity.weights[i, :]=0
           connectivity.weights[:, i]=0
        ######## end of cereb switch off

    # Prepare model
    model = build_model(connectivity.number_of_regions, inds, maps, config)
    # Prepare simulator
    simulator = build_simulator(connectivity, model, inds, maps, config, plotter=plotter)
    # Run simulation and get results
    results, transient = simulate(simulator, config)
    if PSD_target is None:
        # This is the PSD target we are trying to fit...
        if config.model_params['G']:
            # ...for a connected brain, i.e., PS of bilateral M1 and S1:
            PSD_target = compute_target_PSDs_m1s1brl(config, write_files=True, plotter=plotter)
        else:
            # ...for a disconnected brain, average PS of all regions:
            PSD_target = compute_target_PSDs_1D(config, write_files=True, plotter=plotter)
    # This is the PSD computed from our simulation results...
    if config.model_params['G']:
        # ...for a connected brain, i.e., PS of bilateral M1 and S1:
        PSD = compute_data_PSDs_m1s1brl(results[0], PSD_target, inds, transient, plotter=plotter)
    else:
        # ...for a disconnected brain, average PS of all regions:
        PSD = compute_data_PSDs_1D(results[0], PSD_target, inds, transient, plotter=plotter)
    outputs = (PSD, results, transient, simulator, config)
    if plotter is not None:
        outputs = outputs + plot_tvb(transient, inds, results=results, source_ts=None, bold_ts=None, afferent_ts=None,
                                     simulator=simulator, plotter=plotter, config=config, write_files=write_files)
    else:
        if write_files:
            outputs = outputs + tvb_res_to_time_series(results, simulator, config=config, write_files=write_files)
    if config.VERBOSE:
        print("\nFinished TVB workflow in %g sec!\n" % (time.time() - tic))
    return outputs


if __name__ == "__main__":
    parser = args_parser("tvb_script")
    args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
    verbose = args.get('verbose', DEFAULT_ARGS['verbose'])
    if verbose:
        print("Running %s with arguments:\n" % parser.description)
        print(args, "\n")
    run_workflow(**args)
