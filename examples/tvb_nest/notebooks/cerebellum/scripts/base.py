# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from collections import OrderedDict
import time
import dill

import argparse

import numpy as np
import random

from matplotlib import pyplot as plt

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.simulator.integrators import EulerStochastic


DEFAULT_ARGS = {'G': 2.0, 'STIMULUS': 0.5,
                'I_e': -0.25, 'I_s': 0.25,
                'w_ie': -3.0, 'w_rs': -2.0,
                'CONN_LOG': True, 'FIC': 'fit', 'PRIORS_DIST': 'uniform',
                'output_folder': '', 'verbose': 1, 'return_plotter': True}
# tau_e=10/0.9, tau_i=10/0.9, tau_s=10/0.25, tau_r=10/0.25,


def create_plotter(config):
    from tvb_multiscale.core.plot.plotter import Plotter
    plotter = Plotter(config.figures)
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE
    return config, plotter


def configure(**ARGS):
    
    args = deepcopy(DEFAULT_ARGS)
    args.update(**ARGS)

    from tvb_multiscale.tvb_nest.config import Config

    # -----------------------------------------------

    # Flags that affect the result's path:
    # Files:
    BRAIN_CONN_FILE = "Connectivity_wavCntrs_TLwav_SummedSubcortical_Thals.h5"
    MAJOR_STRUCTS_LABELS_FILE = "major_structs_labels_SummedSubcortical_Thals.npy"  # "major_structs_labels_Thals.npy" # "major_structs_labels_SummedSubcortical_Thals.npy"
    VOXEL_COUNT_FILE = "voxel_count_SummedSubcortical_Thals.npy"  # "voxel_count_Thals.npy" # "voxel_count_SummedSubcortical_Thals.npy"
    INDS_FILE = "inds_SummedSubcortical_Thals.npy"  # "inds_Thals.npy" # "inds_SummedSubcortical_Thals.npy"

    # For connectivity
    THAL_CRTX_FIX = "wd"  # "wd", "w", "d" or False, in order to fix values of thalamocortical Weights, Delays, or both, to the Griffiths et al values, or not

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
    if len(args['output_folder']):
        outputs_path = os.path.join(outputs_path, args['output_folder'])
    # # outputs_path += '_G%g' % G
    # # if STIMULUS:
    # #     outputs_path += "_Stim%g" % STIMULUS
    # # outputs_path += '_Is%g' % I_s
    # # outputs_path += '_Ie%g' % I_e
    # outputs_path += "_TVBonly"
    # outputs_path += "_%s" % (BRAIN_CONN_FILE.split("Connectivity_")[-1].split(".h5")[0])
    # if args['CONN_LOG']:
    #     outputs_path += "CONN_LOG"
    if args['FIC']:
        outputs_path += "_FIC"
    # outputs_path += "_PRIORS%s" % args['PRIORS_DIST']
    # if THAL_CRTX_FIX:
    #     outputs_path += "THAL_CRTX_FIX%s" % THAL_CRTX_FIX.upper()

    if args['verbose']:
        print("Outputs' path: %s" % outputs_path)

    config = Config(output_base=outputs_path)

    config.VERBOSE = args['verbose']

    if args['return_plotter']:
        config, plotter = create_plotter(config)
    else:
        plotter = None

    # ------.----- Simulation options ----------------

    # Integration
    config.DEFAULT_DT = 0.1
    config.DEFAULT_NSIG = 1e-4  # NOISE strength
    config.DEFAULT_STOCHASTIC_INTEGRATOR = EulerStochastic
    config.DEFAULT_INTEGRATOR = config.DEFAULT_STOCHASTIC_INTEGRATOR

    # Simulation...
    config.SIMULATION_LENGTH = 4000.0
    config.TRANSIENT_RATIO = 1.0
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
    config.CONN_LOG = args['CONN_LOG']
    config.CONN_SCALE = None  # "region"
    config.CONN_NORM_PERCENTILE = 99
    config.CONN_CEIL = False

    # Model parameters

    config.STIMULUS_RATE = 6.0  # Hz

    config.model_params = OrderedDict()
    config.model_params['G'] = args['G']
    config.model_params['STIMULUS'] = args['STIMULUS']
    config.model_params['I_e'] = args['I_e']
    config.model_params['I_s'] = args['I_s']
    config.model_params['w_ie'] = args['w_ie']
    config.model_params['w_rs'] = args['w_rs']
    # config.model_params['tau_e'] = args['tau_e']
    # config.model_params['tau_i'] = args['tau_i']
    # config.model_params['tau_s'] = args['tau_s']
    # config.model_params['tau_r'] = args['tau_r']

    # NEST model parameters:
    config.NEST_STIMULUS = 15.0  # Hz
    # Monitors:
    config.RAW_PERIOD = 1.0
    config.BOLD_PERIOD = None  # 1024.0 or None, If None, BOLD will not be computed

    # ...and fitting
    config.FIC = args['FIC']
    config.SBI_NUM_WORKERS = 1
    config.SBI_METHOD = 'SNPE'
    config.TARGET_PSD_POPA_PATH = popa_freqs_path
    config.PSD_TARGET_PATH = os.path.join(config.TARGET_PSD_POPA_PATH, "PSD_target.npy")
    config.TARGET_FREQS = np.arange(5.0, 48.0, 1.0)
    config.POSTERIOR_SAMPLES_PATH = os.path.join(config.out.FOLDER_RES, "samples_fit.npy")
    config.N_FIT_RUNS = 10  # 3 - 10
    config.N_SIMULATIONS = 1000
    config.N_SIM_BATCHES = 10
    config.SPLIT_RUN_SAMPLES = 2
    config.N_SAMPLES_PER_RUN = 1000
    config.BATCH_FILE_FORMAT = "%s_%03d%s"
    config.BATCH_FILE_FORMAT_G = "%s_iG%02d_%03d%s"
    config.BATCH_PRIORS_SAMPLES_FILE = "bps.pt"  # bps_iG01_iB010.pt
    config.BATCH_SIM_RES_FILE = "bsr.npy"  # bsr_iG01_iB010.npy
    config.Gs = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
    config.PRIORS_DIST = args['PRIORS_DIST']  # "normal" or "uniform"
    config.PRIORS_PARAMS_NAMES = ['STIMULUS', 'I_e', 'I_s', 'w_ie', 'w_rs']  # 'FIC', 'tau_e', 'tau_i', 'tau_s', 'tau_r']
    #                    0.       1.     2.     3.      4.       5.        6.       7.        8.    9.0
    #                 STIMULUS,  I_e,   I_s,  w_ie,   w_rs,      FIC     tau_e,  tau_i,   tau_s,   tau_r
    # Uniform priors:
    config.prior_min = [0.0,     -1.0,  -0.5, -10.0,   -5.0]  #   0.0,    1.0,    1.0,    1.0,     1.0]
    config.prior_max = [1.0,      0.0,  0.5,    0.0,    0.0]  #   5.0,   20.0,   20.0,   80.0,     80.0]
    # Normal priors:
    config.prior_loc = [0.25,    -0.5,  0.25,  -5.0,  -2.5]  # ,  2.0,    10/0.9,  10/0.9, 10/0.25, 10/0.25]
    config.prior_sc = [0.1,      0.25,  0.25,   2.5,  1.25]  # ,  0.5,     2.0,     2.0,    4.0,      4.0]
    if config.FIC == "fit":
        config.FIC = 1.0
        config.PRIORS_PARAMS_NAMES.append("FIC")
        config.prior_min.append(0.0)
        config.prior_max.append(5.0)
        config.prior_loc.append(2.0)
        config.prior_sc.append(0.5)
    config.n_priors = len(config.prior_min)

    if config.VERBOSE:
        print(config)

    with open(os.path.join(config.out.FOLDER_RES, 'config.pkl'), 'wb') as file:
        dill.dump(config, file, recurse=1)

    return config, plotter


def assert_config(config=None, return_plotter=False, **config_args):
    if config is None:
        if return_plotter:
            # Create a configuration if one is not given
            return configure(plot_flag=True, **config_args)
        else:
            return configure(plot_flag=False, **config_args)[0]
    else:
        if return_plotter:
            return create_plotter(config)
        else:
            return config


def args_parser(funname, args=DEFAULT_ARGS):

    def FICtype(FIC):
        if FIC == 'fit':
            return FIC
        return float(FIC)

    arguments = {'G': ['g', float, 'Global connectivity scaling'],
                 'STIMULUS': ['st', float, 'Whisking stimulus amplitude'],
                 'I_e': ['ie', float, 'Cortical excitatory population baseline current'],
                 'I_s': ['is', float, 'Thalamic relay excitatory population baseline current'],
                 'w_ie': ['wie', float, 'Inhibitory local cortical coupling weight'],
                 'w_rs': ['wrs', float, 'Inhibitory local thalamic coupling weight'],
                 'CONN_LOG': ['cl', bool, 'Boolean flag to logtransform connectivity weights or not'],
                 'FIC': ['fic', FICtype, 'Indegree FIC weight'],
                 'PRIORS_DIST': ['pd', str, "Priors' distribution ('uniform' (default) or 'normal')"],
                 'output_folder': ['o', str, 'Output folder name'],
                 'verbose': ['v', int,
                             'Integer flag to print output messages (when > 0) or not (when == 0). Default = 1.0'],
                 'return_plotter': ['plot', bool, 'Boolean flag to plot or not']
                 }
    parser = argparse.ArgumentParser(description='%s.py' % funname)
    for arg, vals in arguments.items():
        parser.add_argument('--%s' % arg,
                            '-%s' % vals[0],
                            dest=arg, metavar=arg,
                            type=vals[1],
                            default=args[arg], required=False,  # nargs=1,
                            help=vals[2])
    return parser


def parse_args(parser, def_args=DEFAULT_ARGS):
    args = deepcopy(def_args)
    parser_args = parser.parse_args()
    for arg, val in def_args.items():
        args[arg] = getattr(parser_args, arg)
    return args, parser_args, parser
