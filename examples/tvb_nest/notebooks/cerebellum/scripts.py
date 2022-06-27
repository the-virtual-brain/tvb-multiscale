# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from collections import OrderedDict
import time

import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
import random

import torch
from sbi.inference.base import infer, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.simulator.integrators import EulerStochastic

from tvb_multiscale.tvb_nest.config import *

from tvb_multiscale.core.plot.plotter import Plotter
from matplotlib import pyplot as plt


def configure(G=2.0, STIMULUS=0.5,
              I_E=-0.25, I_S=0.25,
              W_IE=-3.0, W_RS=-2.0,
              # TAU_E=10/0.9, TAU_I=10/0.9, TAU_S=10/0.25, TAU_R=10/0.25,
              plot_flag=True):

    # -----------------------------------------------

    # Flags that affect the result's path:
    # Files:
    BRAIN_CONN_FILE = "Connectivity_SummedSubcortical_Thals.h5"
    MAJOR_STRUCTS_LABELS_FILE = "major_structs_labels_SummedSubcortical_Thals.npy"  # "major_structs_labels_Thals.npy" # "major_structs_labels_SummedSubcortical_Thals.npy"
    VOXEL_COUNT_FILE = "voxel_count_SummedSubcortical_Thals.npy"  # "voxel_count_Thals.npy" # "voxel_count_SummedSubcortical_Thals.npy"
    INDS_FILE = "inds_SummedSubcortical_Thals.npy"  # "inds_Thals.npy" # "inds_SummedSubcortical_Thals.npy"

    # For connectivity
    THAL_CRTX_FIX = "wd"  # "wd", "w", "d" or False, in order to fix values of thalamocortical Weights, Delays, or both, to the Griffiths et al values, or not

    # For fitting:
    FIC = 1.0  # 0.2  # 0.185 # 0.19 # 0.2 # 0.15

    # Construct configuration
    work_path = os.getcwd()
    data_path = os.path.join(work_path.split("tvb_nest")[0], "data", "cerebellum")
    tvb_conn_filepath = os.path.join(data_path, BRAIN_CONN_FILE)
    major_structs_labels_filepath = os.path.join(data_path, MAJOR_STRUCTS_LABELS_FILE)
    voxel_count_filepath = os.path.join(data_path, VOXEL_COUNT_FILE)
    inds_filepath = os.path.join(data_path, INDS_FILE)
    popa_freqs_path = os.path.join(data_path, 'PS_popa2013')
    outputs_path = os.path.join(work_path, "outputs/cereb_wilson_cowan")
    # outputs_path += '_G%g' % G
    # if STIMULUS:
    #     outputs_path += "_Stim%g" % STIMULUS
    # outputs_path += '_Is%g' % I_S
    # outputs_path += '_Ie%g' % I_E
    outputs_path += "_TVBonly"
    outputs_path += "_%s" % (BRAIN_CONN_FILE.split("Connectivity_")[-1].split(".h5")[0])
    if FIC:
        if FIC == "SIM":
            outputs_path += "_FICSIM"
        else:
            outputs_path += "_FIC%g" % FIC
    if THAL_CRTX_FIX:
        outputs_path += "THAL_CRTX_FIX%s" % THAL_CRTX_FIX.upper()

    print("Outputs' path: %s" % outputs_path)

    config = Config(output_base=outputs_path)

    if plot_flag:
        plotter = Plotter(config.figures)
        config.figures.SHOW_FLAG = True
        config.figures.SAVE_FLAG = True
        config.figures.FIG_FORMAT = 'png'
        config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE
    else:
        plotter = None

    # ----------- Simulation options ----------------

    # Integration
    config.DEFAULT_DT = 0.1
    config.DEFAULT_NSIG = 1e-4  # NOISE strength
    config.DEFAULT_STOCHASTIC_INTEGRATOR = EulerStochastic
    config.DEFAULT_INTEGRATOR = config.DEFAULT_STOCHASTIC_INTEGRATOR

    # Simulation...
    config.SIMULATION_LENGTH = 400.0  # 4000.0
    config.TRANSIENT_RATIO = 0.1

    # Connectivity
    config.CONN_SPEED = 3.0
    config.BRAIN_CONN_FILE = tvb_conn_filepath
    config.MAJOR_STRUCTS_LABELS_FILE = major_structs_labels_filepath
    config.VOXEL_COUNT_FILE = voxel_count_filepath
    config.INDS_FILE = inds_filepath
    config.THAL_CRTX_FIX = THAL_CRTX_FIX
    config.BRAIN_CONNECTIONS_TO_SCALE = []  # e.g., [["Region 1", ["Region 2", "Region 3"], scaling_factor]]
    config.CONN_SCALE = None  # "region"
    config.CONN_NORM_PERCENTILE = 99
    config.CONN_CEIL = False

    # Model parameters
    config.model_params = OrderedDict()
    config.model_params['G'] = G
    config.model_params['STIMULUS'] = STIMULUS
    config.model_params['I_e'] = I_E
    config.model_params['I_s'] = I_S
    config.model_params['w_ie'] = W_IE
    config.model_params['w_rs'] = W_RS
    # config.model_params['tau_e'] = TAU_E
    # config.model_params['tau_i'] = TAU_I
    # config.model_params['tau_s'] = TAU_S
    # config.model_params['tau_r'] = TAU_R

    # Monitors:
    config.RAW_PERIOD = 1.0
    config.BOLD_PERIOD = None  # 1024.0 or None, If None, BOLD will not be computed

    # ...and fitting
    config.FIC = FIC
    config.SBI_NUM_WORKERS = 1
    config.SBI_METHOD = 'SNPE'
    config.TARGET_PSD_POPA_PATH = popa_freqs_path
    config.PSD_TARGET_PATH = os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_target.npy")
    config.TARGET_FREQS = np.arange(5.0, 48.0, 1.0)
    config.FIC = FIC
    config.SAMPLES_GS_PATH = os.path.join(config.out.FOLDER_RES, "samples_fit_Gs.npy")
    config.N_RUNS = 4  # 3 - 10
    config.N_SIMULATIONS = 100  # 500 - 1000
    config.N_SIM_BATCHES = 10
    config.SPLIT_RUN_SAMPLES = 2
    config.N_SAMPLES_PER_RUN = 1000
    config.BATCH_PRIORS_SAMPLES_FILE = "bps.pt"
    config.BATCH_SIM_RES_FILE = "bsr.npy"
    config.Gs = np.array([0.0, 1.0, 10.0])  # np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0])
    config.PRIORS_MODE = "normal"  # "normal" or "uniform"
    config.PRIORS_PARAMS_NAMES = ['STIMULUS', 'I_E', 'I_S', 'W_IE', 'W_RS']  # , 'TAU_E', 'TAU_I', 'TAU_S', 'TAU_R']
    #                    0.       1.     2.     3.      4.       5.    6.       7.        8.
    #                 STIMULUS,  I_e,   I_s,  w_ie,   w_rs,   tau_e,  tau_i,   tau_s,   tau_r
    # Uniform priors:
    config.prior_min = [0.1, -1.0, 0.0, -10.0, -5.0]  # ,    1.0,    1.0,    1.0,     1.0]
    config.prior_max = [0.5, 0.0, 1.0, 0.0, 0.0]  # ,   20.0,   20.0,   80.0,     80.0]
    # Normal priors:
    config.prior_loc = [0.25, -0.5, 0.5, -5.0, -2.5]  # ,  10/0.9,  10/0.9, 10/0.25, 10/0.25]
    config.prior_sc = [0.1, 0.25, 0.25, 2.5, 1.25]  # ,    2.0,     2.0,    4.0,      4.0]
    config.n_priors = len(config.prior_min)

    return config, plotter


def assert_config(config=None):
    if config is None:
        # Create a configuration if one is not given
        config = configure(plot_flag=False)[0]
    return config


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
        f_st[inds_stim] = 6.0  # Hz
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


def simulate(simulator, config, print_flag=True):
    # Compute transient as a percentage of the total simulation length, and add it to the simulation length:
    simulation_length = float(config.SIMULATION_LENGTH)
    transient = config.TRANSIENT_RATIO * simulation_length
    simulation_length += transient
    simulator.simulation_length = simulation_length

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
    NPERSEG = np.array([256, 512, 1024, 2048, 4096])
    ftarg = PSD_target["f"]
    fmin = ftarg[0]  # The minimum frequency of the PSD_target...
    win_len = int(np.ceil(1000.0 / fmin / dt))  # ...will determine the length of the sliding window....
    nperseg = NPERSEG[np.argmin(np.abs(NPERSEG - win_len))]

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


def run_workflow(G=5.0, STIMULUS=0.25,
                 I_E=-0.25, I_S=0.25,
                 W_IE=-3.0, W_RS=-2.0,
                 #TAU_E=10/0.9, TAU_I=10/0.9, TAU_S=10/0.25, TAU_R=10/0.25,
                 PSD_target=None, plot_flag=True):
    print()
    # Get configuration
    config, plotter = configure(G, STIMULUS, I_E, I_S, W_IE, W_RS,
                                #TAU_E, TAU_I, TAU_S, TAU_R,
                                plot_flag)
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


def build_priors(config):
    if config.PRIORS_MODE.lower() == "uniform":
        priors = utils.torchutils.BoxUniform(low=torch.as_tensor(config.prior_min),
                                             high=torch.as_tensor(config.prior_max))
    else:
        priors_normal = torch.distributions.Normal(loc=torch.as_tensor(config.prior_loc),
                                                   scale=torch.as_tensor(config.prior_sc))
        #     priors = torch.distributions.MultivariateNormal(loc=torch.as_tensor(config.prior_loc),
        #                                                     scale_tril=torch.diag(torch.as_tensor(config.prior_sc)))
        priors = torch.distributions.Independent(priors_normal, 1)
    return priors


def sample_priors_for_sbi(config=None):
    config = assert_config(config)
    dummy_sim = lambda priors: priors
    priors = build_priors(config)
    simulator, priors = prepare_for_sbi(dummy_sim, priors)
    priors_samples, sim_res = simulate_for_sbi(dummy_sim, proposal=priors,
                                               num_simulations=config.N_SIMULATIONS,
                                               num_workers=config.SBI_NUM_WORKERS)
    return priors_samples, sim_res


def priors_samples_per_batch(priors_samples=None, config=None, write_to_files=True):
    config = assert_config(config)
    if priors_samples is None:
        priors_samples = sample_priors_for_sbi(config)[0]
    batch_samples = []
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_PRIORS_SAMPLES_FILE))
    for iR in range(config.N_SIM_BATCHES):
        batch_samples.append(priors_samples[iR::config.N_SIM_BATCHES])
        if write_to_files:
            torch.save(batch_samples[-1], "%s_%03d%s" % (filepath, iR, extension))
    return batch_samples


def priors_samples_per_batch_for_iG(priors_samples=None, iG=None, config=None, write_to_files=True):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_PRIORS_SAMPLES_FILE:
        BATCH_PRIORS_SAMPLES_FILE, extension = os.path.splitext(config.BATCH_PRIORS_SAMPLES_FILE)
        config.BATCH_PRIORS_SAMPLES_FILE = "%s_iG%02d%s" % (BATCH_PRIORS_SAMPLES_FILE, iG, extension)
    return priors_samples_per_batch(priors_samples, config, write_to_files)


def load_priors_samples_per_batch(iB, config=None):
    config = assert_config(config)
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_PRIORS_SAMPLES_FILE))
    return torch.load("%s_%03d%s" % (filepath, iB, extension))


def load_priors_samples_per_batch_per_iG(iB, iG=None, config=None):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_PRIORS_SAMPLES_FILE:
        BATCH_PRIORS_SAMPLES_FILE, extension = os.path.splitext(config.BATCH_PRIORS_SAMPLES_FILE)
        config.BATCH_PRIORS_SAMPLES_FILE = "%s_iG%02d%s" % (BATCH_PRIORS_SAMPLES_FILE, iG, extension)
    return load_priors_samples_per_batch(iB, config)


def write_batch_sim_res_to_file(sim_res, iB, config=None):
    config = assert_config(config)
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_SIM_RES_FILE))
    np.save("%s_%03d%s" % (filepath, iB, extension), sim_res, allow_pickle=True)


def write_batch_sim_res_to_file_per_iG(sim_res, iB, iG=None, config=None):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_SIM_RES_FILE:
        BATCH_SIM_RES_FILE, extension = os.path.splitext(config.BATCH_SIM_RES_FILE)
        config.BATCH_SIM_RES_FILE = "%s_iG%02d%s" % (BATCH_SIM_RES_FILE, iG, extension)
    return write_batch_sim_res_to_file(sim_res, iB, config)


def simulate_TVB_for_sbi_batch(iB, iG=None, config=None, write_to_file=True):
    config = assert_config(config)
    # Get the default values for the parameter except for G
    params = OrderedDict()
    for pname, pval in zip(config.PRIORS_PARAMS_NAMES, config.model_params.values()):
        params[pname] = pval
    batch_samples = load_priors_samples_per_batch_per_iG(iB, iG, config)
    n_simulations = batch_samples.shape[0]
    sim_res = []
    for iS in range(n_simulations):
        priors_params = params.copy()
        if iG is not None:
            priors_params["G"] = config.Gs[iG]
        for prior, prior_name in zip(batch_samples[iS], config.PRIORS_PARAMS_NAMES):
            try:
                numpy_prior = prior.numpy()
            except:
                numpy_prior = prior
            priors_params[prior_name] = numpy_prior
        print("\n\nSimulating for parameters:\n%s\n" % str(priors_params))
        sim_res.append(run_workflow(**priors_params, plot_flag=False)[0])
    if write_to_file:
        write_batch_sim_res_to_file_per_iG(sim_res, iB, iG, config)
    return sim_res


# def simulate_TVB_for_sbi(priors, priors_params_names, **params):
#     priors_params = params.copy()
#     # Convert all tensor parameters to numpy arrays
#     for prior, prior_name in zip(priors, priors_params_names):
#         try:
#             numpy_prior = prior.numpy()
#         except:
#             numpy_prior = prior
#         priors_params[prior_name] = numpy_prior
#     # Run the simulation and return only the PSD output to be fit:
#     return run_workflow(**priors_params, plot_flag=False)[0]
#
#
# def sbi_fit(iG, config=None):
#
#     tic = time.time()
#
#     if config is None:
#         # Create a configuration if one is not given
#         config = configure(plot_flag=False)[0]
#
#     # Get the default values for the parameter except for G
#     params = OrderedDict()
#     for pname, pval in zip(config.PRIORS_PARAMS_NAMES, config.model_params.values()):
#         params[pname] = pval
#     print("params =\n", params)
#
#     # Load the target
#     PSD_target = np.load(config.PSD_TARGET_PATH, allow_pickle=True).item()
#     # Duplicate the target for the two M1 regions (right, left) and the two S1 barrel field regions (right, left)
#     #                                        right                       left
#     psd_targ_conc = np.concatenate([PSD_target["PSD_M1_target"], PSD_target["PSD_M1_target"],
#                                     PSD_target["PSD_S1_target"], PSD_target["PSD_S1_target"]])
#
#     # Get G for this run:
#     G = config.Gs[iG]
#
#     # Define the simulation function for sbi for this G
#     simulate_for_sbi_for_g = lambda priors: simulate_TVB_for_sbi(priors,
#                                                                  priors_params_names=config.PRIORS_PARAMS_NAMES,
#                                                                  G=G, **params)
#
#     # Build the priors
#     priors = build_priors(config)
#
#     print("\n\nFitting for G = %g!\n" % G)
#     tic = time.time()
#     for iR in range(config.N_RUNS):
#         print("\nFitting run %d " % iR)
#
#         # Train the neural network to approximate the posterior:
#         posterior = infer(simulate_for_sbi_for_g, priors,
#                           method=config.SBI_METHOD,
#                           num_simulations=config.N_SIMULATIONS,
#                           num_workers=config.SBI_NUM_WORKERS)
#
#         print("\nSampling posterior...")
#         if iR:
#             samples_fit = torch.cat((samples_fit, posterior.sample((config.N_SAMPLES_PER_RUN,), x=psd_targ_conc)), 0)
#         else:
#             samples_fit = posterior.sample((config.N_SAMPLES_PER_RUN,), x=psd_targ_conc)
#
#     print("Done in %g sec!" % (time.time() - tic))
#
#     # Compute the sample mean, add to the results dictionary and write to file:
#     if os.path.isfile(config.SAMPLES_GS_PATH):
#         samples_fit_Gs = np.load(config.SAMPLES_GS_PATH, allow_pickle=True).item()
#     else:
#         samples_fit_Gs = {}
#     samples_fit_Gs[G] = {}
#     samples_fit_Gs[G]['samples'] = samples_fit.numpy()
#     samples_fit_Gs[G]['mean'] = samples_fit.mean(axis=0).numpy()
#     np.save(config.SAMPLES_GS_PATH, samples_fit_Gs, allow_pickle=True)
#
#     # Plot posterior:
#     print("\nPlotting posterior...")
#     limits = []
#     for pmin, pmax in zip(config.prior_min, config.prior_max):
#         limits.append([pmin, pmax])
#     fig, axes = analysis.pairplot(samples_fit,
#                                   limits=limits,
#                                   ticks=limits,
#                                   figsize=(10, 10),
#                                   points=np.array(list(params.values())),
#                                   points_offdiag={'markersize': 6},
#                                   points_colors=['r'] * config.n_priors)
#     plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, 'sbi_pairplot_%g.png' % G))
#
#     # Run one simulation with the posterior means:
#     print("\nSimulating with posterior means...")
#     params.update(dict(zip(config.PRIORS_PARAMS_NAMES, samples_fit_Gs[G]['mean'])))
#     PSD, results, simulator, output_config = run_workflow(PSD_target=PSD_target, plot_flag=True, G=G, **params)
#
#     duration = time.time() - tic
#     print("\n\nFinished after %g sec!" % duration)
#     print("\n\nFind results in %s!" % config.out.FOLDER_RES)
#
#     return samples_fit_Gs, results, fig, simulator, output_config


if __name__ == "__main__":
    import sys

    # samples_fit_Gs, results, fig, simulator, output_config = sbi_fit(int(sys.argv[-1]))
    config = configure()[0]
    if len(sys.argv) == 1:
        iB = int(sys.argv[-1])
        iG = None
    else:
        iB = int(sys.argv[-1])
        iG = int(sys.argv[-2])

    sim_res = simulate_TVB_for_sbi_batch(iB, iG, config=config, write_to_file=True)