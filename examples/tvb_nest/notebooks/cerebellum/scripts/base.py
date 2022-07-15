# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from collections import OrderedDict
import time

import numpy as np
import random

from matplotlib import pyplot as plt

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.simulator.integrators import EulerStochastic


def configure(G=2.0, STIMULUS=0.5,
              I_E=-0.25, I_S=0.25,
              W_IE=-3.0, W_RS=-2.0,
              # TAU_E=10/0.9, TAU_I=10/0.9, TAU_S=10/0.25, TAU_R=10/0.25,
              output_folder=None,
              plot_flag=True):

    from tvb_multiscale.tvb_nest.config import Config

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
    cereb_scaffold_path = os.path.join(data_path, 'balanced_DCN_IO.hdf5')
    outputs_path = os.path.join(work_path, "outputs/cwc")
    if output_folder is not None:
        outputs_path = os.path.join(outputs_path, output_folder)
    # # outputs_path += '_G%g' % G
    # # if STIMULUS:
    # #     outputs_path += "_Stim%g" % STIMULUS
    # # outputs_path += '_Is%g' % I_S
    # # outputs_path += '_Ie%g' % I_E
    # outputs_path += "_TVBonly"
    # outputs_path += "_%s" % (BRAIN_CONN_FILE.split("Connectivity_")[-1].split(".h5")[0])
    # if FIC:
    #     if FIC == "SIM":
    #         outputs_path += "_FICSIM"
    #     else:
    #         outputs_path += "_FIC%g" % FIC
    # if THAL_CRTX_FIX:
    #     outputs_path += "THAL_CRTX_FIX%s" % THAL_CRTX_FIX.upper()

    print("Outputs' path: %s" % outputs_path)

    config = Config(output_base=outputs_path)

    if plot_flag:
        from tvb_multiscale.core.plot.plotter import Plotter
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
    config.SIMULATION_LENGTH = 4000.0
    config.TRANSIENT_RATIO = 0.1
    config.NEST_PERIPHERY = False
    config.INVERSE_SIGMOIDAL_NEST_TO_TVB = True

    # Connectivity
    config.CONN_SPEED = 3.0
    config.BRAIN_CONN_FILE = tvb_conn_filepath
    config.MAJOR_STRUCTS_LABELS_FILE = major_structs_labels_filepath
    config.VOXEL_COUNT_FILE = voxel_count_filepath
    config.INDS_FILE = inds_filepath
    config.CEREB_SCAFFOLD_PATH = cereb_scaffold_path
    config.THAL_CRTX_FIX = THAL_CRTX_FIX
    config.BRAIN_CONNECTIONS_TO_SCALE = []  # e.g., [["Region 1", ["Region 2", "Region 3"], scaling_factor]]
    config.CONN_SCALE = None  # "region"
    config.CONN_NORM_PERCENTILE = 99
    config.CONN_CEIL = False

    # Model parameters

    config.STIMULUS_RATE = 6.0  # Hz

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

    # NEST model parameters:
    config.NEST_STIMULUS = 15.0  # Hz
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
    config.POSTERIOR_SAMPLES_PATH = os.path.join(config.out.FOLDER_RES, "samples_fit.npy")
    config.N_FIT_RUNS = 10  # 3 - 10
    config.N_SIMULATIONS = 1000  # 500 - 1000
    config.N_SIM_BATCHES = 10
    config.SPLIT_RUN_SAMPLES = 2
    config.N_SAMPLES_PER_RUN = 1000
    config.BATCH_FILE_FORMAT = "%s_%03d%s"
    config.BATCH_FILE_FORMAT_G = "%s_iG%02d_%03d%s"
    config.BATCH_PRIORS_SAMPLES_FILE = "bps.pt"  # bps_iG01_iB010.pt
    config.BATCH_SIM_RES_FILE = "bsr.npy"  # bsr_iG01_iB010.npy
    config.Gs = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    config.PRIORS_MODE = "normal"  # "normal" or "uniform"
    config.PRIORS_PARAMS_NAMES = ['STIMULUS', 'I_E', 'I_S', 'W_IE', 'W_RS']  # , 'TAU_E', 'TAU_I', 'TAU_S', 'TAU_R']
    #                    0.       1.     2.     3.      4.       5.    6.       7.        8.
    #                 STIMULUS,  I_e,   I_s,  w_ie,   w_rs,   tau_e,  tau_i,   tau_s,   tau_r
    # Uniform priors:
    config.prior_min = [0.1,     -1.0,  0.0, -10.0,   -5.0]  # ,    1.0,    1.0,    1.0,     1.0]
    config.prior_max = [1.0,      0.0,  1.0,  0.0,     0.0]  # ,   20.0,   20.0,   80.0,     80.0]
    # Normal priors:
    config.prior_loc = [0.25,    -0.5,  0.5,  -5.0,  -2.5]  # ,  10/0.9,  10/0.9, 10/0.25, 10/0.25]
    config.prior_sc = [0.1,      0.25,  0.25,  2.5,  1.25]  # ,    2.0,     2.0,    4.0,      4.0]
    config.n_priors = len(config.prior_min)

    return config, plotter


def assert_config(config=None):
    if config is None:
        # Create a configuration if one is not given
        config = configure(plot_flag=False)[0]
    return config


