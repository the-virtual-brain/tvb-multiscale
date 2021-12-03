# -*- coding: utf-8 -*-
import os
import time
import h5py
from collections import OrderedDict

import numpy as np
from scipy.io import savemat

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import Config
from tvb_multiscale.core.tvb.cosimulator.models.wc_thalamocortical_cereb \
    import WilsonCowanThalamoCortical, GriffithsThalamoCortical, SigmoidalPre, SigmoidalPreThalamoCortical
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
from tvb_multiscale.core.plot.plotter import Plotter

# from examples.plot_write_results import plot_write_results
from examples.tvb_nest.notebooks.cerebellum.utils import \
    print_conn, compute_plot_selected_spectra_coherence, compute_plot_ica

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import EulerStochastic  # HeunStochastic
from tvb.simulator.monitors import Raw, Bold  #, EEG

from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesXarray


def results_path_fun_config(tvb_sim_model, sim_constants, files, model_params, config=None):
    if config is None:
        work_path = os.getcwd()
        outputs_path = os.path.join(work_path, "outputs/cereb_wilson_cowan", tvb_sim_model.__name__)
        outputs_path += '_G%g' % model_params["G"]
        outputs_path += "_TVBonly"
        outputs_path += "_%s" % (files["BRAIN_CONN"].split("Connectivity_")[-1].split(".h5")[0])
        if sim_constants["THAL_CRTX_FIX"]:
            outputs_path += "_THAL_CRTX_FIX%s" % sim_constants["THAL_CRTX_FIX"].upper()
        if sim_constants["THAL_CRTX_VAR"]:
            outputs_path += "_THAL_CRTX_VAR%s" % sim_constants["THAL_CRTX_VAR"].upper()
        if sim_constants["STIMULUS"]:
            outputs_path += "_Stim"
        if sim_constants["DEMO"]:
            outputs_path += "_demo"
        config = Config(output_base=outputs_path)
        return config
    else:
        return config.out.FOLDER_RES


def configure_connectivity(connectivity, sim_constants, model_params):
    # Determine connectivity file(name)s
    files = {}
    if not isinstance(connectivity, Connectivity):
        files["BRAIN_CONN"] = model_params.pop("BRAIN_CONN", connectivity)

    else:
        if sim_constants["model_mode"]:
            files["BRAIN_CONN"] = model_params.pop("BRAIN_CONN", "Connectivity_SummedSubcortical_Thals.h5")
        else:
            files["BRAIN_CONN"] = model_params.pop("BRAIN_CONN", "Connectivity_SummedSubcortical.h5")
    brain_conn_file_str = files["BRAIN_CONN"].split("Connectivity")[-1].split("h5")[0] + "npy"
    files["MAJOR_STRUCTS_LABELS"] = model_params.pop("MAJOR_STRUCTS_LABELS",
                                                 "major_structs_labels" + brain_conn_file_str)
    files["VOXEL_COUNT"] = model_params.pop("VOXEL_COUNT", "voxel_count" + brain_conn_file_str)
    files["INDS"] = model_params.pop("INDS", "inds" + brain_conn_file_str)

    return files, model_params


def configure(tvb_sim_model, config, connectivity, model_params):

    sim_constants = {}

    if tvb_sim_model == WilsonCowanThalamoCortical:
        sim_constants["model_mode"] = 1
    else:
        sim_constants["model_mode"] = 0

    # Set the simulation mode:
    sim_constants["STIMULUS"] = model_params.pop("STIMULUS", True)
    sim_constants["THAL_CRTX_FIX"] = model_params.pop("THAL_CRTX_FIX", False)
    sim_constants["THAL_CRTX_VAR"] = model_params.pop("THAL_CRTX_VAR", "wd")
    sim_constants["DEMO"] = model_params.pop("DEMO", False)
    model_params["G"] = model_params.pop("G", 5.0)
    if sim_constants["DEMO"]:
        sim_constants["STIMULUS"] = False
        if sim_constants["model_mode"]:
            sim_constants["THAL_CRTX_FIX"] = "wd"
            sim_constants["THAL_CRTX_VAR"] = False
        else:
            sim_constants["THAL_CRTX_FIX"] = False
            sim_constants["THAL_CRTX_VAR"] = False
        model_params["G"] = 0

    files, model_params = configure_connectivity(connectivity, sim_constants, model_params)

    if config is None:
        config = results_path_fun_config(tvb_sim_model, sim_constants, files, model_params, config)

    return config, sim_constants, files, model_params


def create_inds_flags(connectivity, data_path, sim_constants, files, model_params):

    major_structs_labels = np.load(os.path.join(data_path, files["MAJOR_STRUCTS_LABELS"]))
    voxel_count = np.load(os.path.join(data_path, files["VOXEL_COUNT"]))
    inds = np.load(os.path.join(data_path, files["INDS"]), allow_pickle=True).item()
    print(inds)

    flags = {}
    # Find thalamic and subcortical indices index:
    flags["is_cortical"] = np.array([False] * connectivity.region_labels.shape[0]).astype("bool")
    flags["is_cortical"][inds["crtx"]] = True
    inds["subcrtx"] = np.arange(len(connectivity.region_labels)).astype('i')
    inds["subcrtx"] = np.delete(inds["subcrtx"], inds["crtx"])
    flags["is_subcortical"] = np.array([False] * connectivity.region_labels.shape[0]).astype("bool")
    flags["is_subcortical"][inds["subcrtx"]] = True
    if sim_constants["model_mode"]:
        flags["is_thalamic"] = np.array([False] * connectivity.region_labels.shape[0]).astype("bool")
        flags["is_thalamic"][inds["thalspec"]] = True
        flags["is_subcortical_not_thalamic"] = np.logical_and(flags["is_subcortical"],
                                                              np.logical_not(flags["is_thalamic"]))

    return major_structs_labels, voxel_count, inds, flags, model_params


def create_connectivity(connectivity, sim_constants, files, model_params):
    # First, create the Connectivity:
    if not isinstance(connectivity, Connectivity):
        work_path = os.getcwd()
        data_path = os.path.join(work_path.split("examples")[0], "examples/data", "cerebellum")
        tvb_conn_filepath = os.path.join(data_path, files["BRAIN_CONN"])
        f = h5py.File(tvb_conn_filepath)
        centres = np.array(f["centres"][()])  # hemispheres=np.array(f["hemispheres"][()]),
        region_labels = np.array(f["region_labels"][()]).astype("<U128")
        weights = np.array(f["weights"][()])
        tract_lengths = np.array(f["tract_lengths"][()])
        f.close()
        # Create connectivity
        connectivity = Connectivity(centres=centres, region_labels=region_labels,
                                    weights=weights, tract_lengths=tract_lengths)

    # Selectively scale up some connections, if required:
    for reg1, reg2, sc in model_params.pop("BRAIN_CONNECTIONS_TO_SCALE", []):
        iR1 = np.where([reg in reg1 for reg in connectivity.region_labels])[0]
        iR2 = np.where([reg in reg2 for reg in connectivity.region_labels])[0]
        connectivity.weights[iR1, iR2] *= sc

        # Normalize connectivity weights
    connectivity.weights[np.isnan(connectivity.weights)] = 0.0
    connectivity.weights = connectivity.scaled_weights(mode="region")
    connectivity.weights /= np.percentile(connectivity.weights, 99)
    # connectivity.weights[connectivity.weights > 1.0] = 1.0
    connectivity.tract_lengths = np.maximum(connectivity.speed * sim_constants["dt"],
                                            connectivity.tract_lengths)
    connectivity.configure()

    major_structs_labels, voxel_count, inds, flags, model_params = \
        create_inds_flags(connectivity, data_path, sim_constants, files, model_params)
    return connectivity, major_structs_labels, voxel_count, inds, flags, model_params


def build_simulator(connectivity, sim_constants, inds, flags, model_params, config):

    simulator = CoSimulatorSerial()
    simulator.connectivity = connectivity
    if sim_constants["model_mode"]:
        model_params.update({"G": np.array([model_params["G"]]) /
                                  (simulator.connectivity.number_of_regions - inds['thalspec'].size)})
    else:
        model_params.update({"G": np.array([model_params["G"]]) / simulator.connectivity.number_of_regions})

    NOISE = model_params.pop("NOISE", config.DEFAULT_NSIG)
    if sim_constants["DEMO"]:
        model_params.update({'I_o': np.array([0.5]),
                             'w_ee': np.array([1.4]),
                             'w_ei': np.array([1.4]),
                             'w_ie': np.array([-3.0]),
                             'tau_e': np.array([10.0 / 0.9]),
                             'tau_i': np.array([10.0 / 0.9]),
                             'tau_s': np.array([10.0 / 0.25]),
                             'tau_r': np.array([10.0 / 0.25])
                             })
        NOISE = 0.0

    dummy_shape = (simulator.connectivity.number_of_regions,)

    if sim_constants["STIMULUS"]:
        if sim_constants["model_mode"]:
            # Stimuli:
            A_st = np.zeros(dummy_shape).astype("f")
            f_st = np.zeros(dummy_shape).astype("f")
            # Sensory to Medulla SPV
            A_st[inds["trigeminal"]] = 1.0
            f_st[inds["trigeminal"]] = 6.0  # Hz
            # Motor to Facial nucleus
            A_st[inds["facial"]] = 1.0
            f_st[inds["facial"]] = 6.0  # Hz
            model_params.update({"A_st": A_st, "f_st": f_st})
        else:
            # Stimuli:
            M_e = np.zeros(dummy_shape).astype("f")
            f_e = np.zeros(dummy_shape).astype("f")
            M_s = np.zeros(dummy_shape).astype("f")
            f_s = np.zeros(dummy_shape).astype("f")
            # Sensory to Medulla SPV
            M_e[inds["trigeminal"]] = 1.0
            f_e[inds["trigeminal"]] = 6.0  # Hz
            # Motor to Facial nucleus
            M_e[inds["facial"]] = 1.0
            f_e[inds["facial"]] = 6.0  # Hz
            model_params.update({"M_e": M_e, "M_s": M_s, "f_e": f_e, "f_s": f_s})

    if sim_constants["model_mode"]:
        simulator.model = WilsonCowanThalamoCortical(is_thalamic=flags["is_thalamic"][:, np.newaxis],
                                                     **model_params)

        # Specific thalamic relay -> nonspecific subcortical structures connections' weights:
        w_se = simulator.model.w_se * np.ones(dummy_shape)
        w_se[inds['subcrtx']] = simulator.model.G[0]
        simulator.model.w_se = w_se
        w_si = simulator.model.w_si * np.ones(dummy_shape)
        w_si[inds['subcrtx']] = 0.0
        simulator.model.w_si = w_si

        # Fixing thalamocortical connections:
        if sim_constants["THAL_CRTX_FIX"]:
            if "w" in sim_constants["THAL_CRTX_FIX"]:
                # Fix structural connectivity (specific) thalamo-cortical weights to 1,
                # such that all thalamo-cortical weights are equal to the parameters
                # w_er, w_es, w_se, w_si
                simulator.connectivity.weights[inds["crtx"], inds["thalspec"]] = 1.0
                simulator.connectivity.weights[inds["thalspec"], inds["crtx"]] = 1.0
            if "d" in sim_constants["THAL_CRTX_FIX"]:
                # Fix structural connectivity (specific) thalamo-cortical tracts length to a value,
                # such that all thalamo-cortical delays are equal to the parameter tau_ct,
                # given connectivity's speed.
                ct_lengths = simulator.connectivity.speed * \
                             simulator.model.tau_ct * np.ones((simulator.connectivity.number_of_regions,))[inds["crtx"]]
                simulator.connectivity.tract_lengths[inds["crtx"], inds["thalspec"]] = ct_lengths
                simulator.connectivity.tract_lengths[inds["thalspec"], inds["crtx"]] = ct_lengths

            simulator.connectivity.configure()

        simulator.coupling = SigmoidalPreThalamoCortical(
            is_thalamic=flags["is_thalamic"],
            is_subcortical=flags["is_subcortical"],
            sigma=np.array([1.0]),
            midpoint=simulator.model.sigma,
            cmin=np.array([0.0]),
            cmax=np.array([1.0]),
            a=simulator.model.beta)

    else:
        simulator.model = GriffithsThalamoCortical(is_cortical=flags["is_cortical"][:, np.newaxis], **model_params)

        # Variability to thalamocortical connections:
        if sim_constants["THAL_CRTX_VAR"]:

            from scipy.stats import zscore
            n_crtx2 = int(len(inds["crtx"]) / 2)

            def scaler(x, var=1.0):
                assert var <= 1.0
                y = zscore(x, axis=0, ddof=0, nan_policy='omit')
                y /= np.abs(y).max()
                y[np.isnan(y)] = 0.0
                return var * y

            def scale(x, sc):
                return (1 + sc) * x

            # weights:
            if "w" in sim_constants["THAL_CRTX_VAR"]:
                w_thal_to_crtx_right_zs = scaler(
                    simulator.connectivity.weights[inds["crtx"][:n_crtx2], inds["thal"][0]])
                w_thal_to_crtx_left_zs = scaler(simulator.connectivity.weights[inds["crtx"][n_crtx2:], inds["thal"][1]])
                w_crtx_to_thal_right_zs = scaler(
                    simulator.connectivity.weights[inds["thal"][0], inds["crtx"][:n_crtx2]])
                w_crtx_to_thal_left_zs = scaler(simulator.connectivity.weights[inds["thal"][1], inds["crtx"][n_crtx2:]])
                simulator.model.w_se = simulator.model.w_se * np.ones(dummy_shape)
                simulator.model.w_si = simulator.model.w_si * np.ones(dummy_shape)
                simulator.model.w_er = simulator.model.w_er * np.ones(dummy_shape)
                simulator.model.w_es = simulator.model.w_es * np.ones(dummy_shape)
                simulator.model.w_se[inds["crtx"][:n_crtx2]] = scale(simulator.model.w_se[inds["crtx"][:n_crtx2]],
                                                                     w_thal_to_crtx_right_zs)
                simulator.model.w_si[inds["crtx"][:n_crtx2]] = scale(simulator.model.w_si[inds["crtx"][:n_crtx2]],
                                                                     w_thal_to_crtx_right_zs)
                simulator.model.w_se[inds["crtx"][n_crtx2:]] = scale(simulator.model.w_se[inds["crtx"][n_crtx2:]],
                                                                     w_thal_to_crtx_left_zs)
                simulator.model.w_si[inds["crtx"][n_crtx2:]] = scale(simulator.model.w_si[inds["crtx"][n_crtx2:]],
                                                                     w_thal_to_crtx_left_zs)
                simulator.model.w_er[inds["crtx"][:n_crtx2]] = scale(simulator.model.w_er[inds["crtx"][:n_crtx2]],
                                                                     w_crtx_to_thal_right_zs)
                simulator.model.w_es[inds["crtx"][:n_crtx2]] = scale(simulator.model.w_es[inds["crtx"][:n_crtx2]],
                                                                     w_crtx_to_thal_right_zs)
                simulator.model.w_er[inds["crtx"][n_crtx2:]] = scale(simulator.model.w_er[inds["crtx"][n_crtx2:]],
                                                                     w_crtx_to_thal_left_zs)
                simulator.model.w_es[inds["crtx"][n_crtx2:]] = scale(simulator.model.w_es[inds["crtx"][n_crtx2:]],
                                                                     w_crtx_to_thal_left_zs)
            # delays:
            if "d" in sim_constants["THAL_CRTX_VAR"]:
                d_thal_to_crtx_right_zs = scaler(
                    simulator.connectivity.tract_lengths[inds["crtx"][:n_crtx2], inds["thal"][0]] /
                    simulator.connectivity.speed)
                d_thal_to_crtx_left_zs = scaler(
                    simulator.connectivity.tract_lengths[inds["crtx"][n_crtx2:], inds["thal"][1]] /
                    simulator.connectivity.speed)
                simulator.model.tau_ct = simulator.model.tau_ct * np.ones(dummy_shape)
                simulator.model.tau_ct[inds["crtx"][:n_crtx2]] = scale(simulator.model.tau_ct[inds["crtx"][:n_crtx2]],
                                                                       d_thal_to_crtx_right_zs)
                simulator.model.tau_ct[inds["crtx"][n_crtx2:]] = scale(simulator.model.tau_ct[inds["crtx"][n_crtx2:]],
                                                                       d_thal_to_crtx_left_zs)
                simulator.model.tau_ct[inds["crtx"]] = np.round(simulator.model.tau_ct[inds["crtx"]]
                                                                / simulator.model.dt) * simulator.model.dt

            d = OrderedDict()
            d[("Region", 55)] = ["%d.%s" % (iR, reg) for iR, reg in
                                 enumerate(connectivity.region_labels[inds["crtx"]])]
            d[("w_se(%g)" % simulator.model.w_se[-1], 12)] = simulator.model.w_se[inds["crtx"]]
            d[("w_si(%g)" % simulator.model.w_si[-1], 12)] = simulator.model.w_si[inds["crtx"]]
            d[("w_er(%g)" % simulator.model.w_er[-1], 12)] = simulator.model.w_er[inds["crtx"]]
            d[("w_es(%g)" % simulator.model.w_es[-1], 12)] = simulator.model.w_es[inds["crtx"]]
            d[("tau_ct(%g)" % simulator.model.tau_ct[-1], 12)] = simulator.model.tau_ct[inds["crtx"]]

            print_conn(d,
                       prnt="", printit=True);

        simulator.coupling = SigmoidalPre(sigma=np.array([1.0]),
                                          midpoint=simulator.model.sigma,
                                          cmin=np.array([0.0]),
                                          cmax=np.array([1.0]),
                                          a=simulator.model.beta)

    simulator.model.dt = sim_constants["dt"]

    simulator.integrator = EulerStochastic()  # HeunStochastic()
    simulator.integrator.dt = sim_constants["dt"]
    simulator.integrator.noise.nsig = np.array([NOISE] * (simulator.model.nvar - 1) + [0.0])

    simulator.initial_conditions = np.zeros((1, simulator.model.nvar, connectivity.number_of_regions, 1))

    mon_raw = Raw(period=1.0)  # ms
    bold = Bold(period=128.0,  # !!! Set a proper (1-2 sec??) TR time in ms !!!
                variables_of_interest=np.array([np.where(sim_constants["model_mode"], 2, 4)]))
    simulator.monitors = (mon_raw, bold)

    return simulator


def plot_write_analyse_results(results, simulator, inds, config, transient=0.0):

    MAX_VARS_IN_COLS = 3
    MAX_REGIONS_IN_ROWS = 10
    MIN_REGIONS_FOR_RASTER_PLOT = 9
    FIGSIZE = config.figures.DEFAULT_SIZE

    try:
        # We need framework_tvb for writing and reading from HDF5 files
        from tvb_multiscale.tvb.core.io.h5_writer import H5Writer
        writer = H5Writer()
    except:
        writer = False

    source_ts = None
    bold_ts = None

    if results is not None:

        dt = simulator.integrator.dt

        plotter = Plotter(config)

        # Put the results in a Timeseries instance

        source_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
            data=results[0][1], time=results[0][0],
            connectivity=simulator.connectivity,
            labels_ordering=["Time", "State Variable", "Region", "Neurons"],
            labels_dimensions={"State Variable": list(simulator.model.variables_of_interest),
                               "Region": simulator.connectivity.region_labels.tolist()},
            sample_period=simulator.integrator.dt)
        source_ts.configure()

        t = source_ts.time

        # Write to file
        if writer:
            writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(source_ts._data,
                                                                            connectivity=source_ts.connectivity),
                                   os.path.join(config.out.FOLDER_RES, source_ts.title) + ".h5")
        print("Raw ts:\n%s" % str(source_ts))

        if len(results) > 1:
            bold_ts = TimeSeriesXarray(  # substitute with TimeSeriesRegion fot TVB like functionality
                data=results[1][1], time=results[1][0],
                connectivity=simulator.connectivity,
                labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                labels_dimensions={"State Variable": ["BOLD"],
                                   "Region": simulator.connectivity.region_labels.tolist()})
            bold_ts.configure()

            bold_t = source_ts.time

            # Write to file
            if writer:
                writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(bold_ts._data,
                                                                                connectivity=bold_ts.connectivity),
                                       os.path.join(config.out.FOLDER_RES, bold_ts.title) + ".h5")
            print("BOLD ts:\n%s" % str(bold_ts))

        # Plot TVB time series
        if source_ts is not None:
            source_ts[-10000:].plot_timeseries(plotter_config=plotter.config,
                                               hue="Region" if source_ts.shape[
                                                                   2] > MAX_REGIONS_IN_ROWS else None,
                                               per_variable=source_ts.shape[1] > MAX_VARS_IN_COLS,
                                               figsize=FIGSIZE);

            source_ts_m1s1brl = source_ts[-10000:, :, inds["m1s1brl"]]
            source_ts_m1s1brl.plot_timeseries(plotter_config=plotter.config,
                                              hue="Region" if source_ts_m1s1brl.shape[
                                                                  2] > MAX_REGIONS_IN_ROWS else None,
                                              per_variable=source_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                              figsize=FIGSIZE,
                                              figname="M1 and S1 barrel field nodes TVB Time Series");

            if source_ts_m1s1brl.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
                source_ts_m1s1brl.plot_raster(plotter_config=plotter.config,
                                              per_variable=source_ts_m1s1brl.shape[1] > MAX_VARS_IN_COLS,
                                              figsize=FIGSIZE,
                                              figname="M1 and S1 barrel field nodes TVB Time Series Raster");

            source_ts_motor = source_ts[-10000:, :, inds["motor"]]
            source_ts_motor.plot_timeseries(plotter_config=plotter.config,
                                            hue="Region" if source_ts_motor.shape[
                                                                2] > MAX_REGIONS_IN_ROWS else None,
                                            per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                            figsize=FIGSIZE, figname="Motor pathway TVB Time Series");
            if source_ts_motor.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
                source_ts_motor.plot_raster(plotter_config=plotter.config,
                                            per_variable=source_ts_motor.shape[1] > MAX_VARS_IN_COLS,
                                            figsize=FIGSIZE, figname="Motor pathway TVB Time Series Raster");

            source_ts_sens = source_ts[-10000:, :, inds["sens"]]
            source_ts_sens.plot_timeseries(plotter_config=plotter.config,
                                           hue="Region" if source_ts_sens.shape[
                                                               2] > MAX_REGIONS_IN_ROWS else None,
                                           per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                           figsize=FIGSIZE, figname="Sensory pathway TVB Time Series");
            if source_ts_sens.number_of_labels > MIN_REGIONS_FOR_RASTER_PLOT:
                source_ts_sens.plot_raster(plotter_config=plotter.config,
                                           per_variable=source_ts_sens.shape[1] > MAX_VARS_IN_COLS,
                                           figsize=FIGSIZE, figname="Sensory pathway TVB Time Series Raster");

            NPERSEG = np.array([256, 512, 1024, 2048, 4096])
            NPERSEG = NPERSEG[np.argmin(np.abs(NPERSEG - (source_ts.shape[0] - transient / dt)))]

            # Power Spectra and Coherence for M1 - S1 barrel field
            compute_plot_selected_spectra_coherence(source_ts, inds["m1s1brl"],
                                                    transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0)

            # Power Spectra and Coherence along the sensory pathway:
            # for Medulla SPV, Sensory PONS
            compute_plot_selected_spectra_coherence(source_ts, inds["sens"],
                                                    transient=transient, nperseg=NPERSEG, fmin=0.0, fmax=100.0)

            if source_ts is not None:
                # LFP (E) ICA:
                lfp_ica_comps, lfp_ics_ts, lfp_ica = compute_plot_ica(
                    source_ts[transient:].data[:, 0, :, 0].squeeze(),
                    source_ts[transient:].time,
                    variable="LFP", n_components=10, plotter=plotter)
                savemat(os.path.join(config.out.FOLDER_RES, "LFP_ICA") + ".mat",
                        {"lfp_ica_comps": lfp_ica_comps,
                         "lfp_ics_ts": lfp_ics_ts.data})

        # bold_ts TVB time series
        if bold_ts is not None:
            bold_ts.plot_timeseries(plotter_config=plotter.config,
                                    hue="Region" if bold_ts.shape[2] > MAX_REGIONS_IN_ROWS else None,
                                    per_variable=bold_ts.shape[1] > MAX_VARS_IN_COLS,
                                    figsize=FIGSIZE);

            # BOLD ICA:
            bold_ica_comps, bold_ics_ts, bold_ica = compute_plot_ica(bold_ts.data[:, 0, :, 0].squeeze(),
                                                                     bold_ts.time,
                                                                     variable="BOLD", n_components=10,
                                                                     plotter=plotter)
            savemat(os.path.join(config.out.FOLDER_RES, "BOLD_ICA") + ".mat",
                    {"bold_ica_comps": bold_ica_comps,
                     "bold_ics_ts": bold_ics_ts.data})


def wc_thalamocortical_example(tvb_sim_model=WilsonCowanThalamoCortical,
                               connectivity="Connectivity_SummedSubcortical_Thals.h5",
                               simulation_length=1124.0, transient=100.0,
                               config=None, plot_write=True, **model_params):

    config, sim_constants, files, model_params = configure(tvb_sim_model, config, connectivity, model_params)

    sim_constants["dt"] = config.DEFAULT_DT

    # ----------------------1. Define a TVB simulator (model, integrator, monitors...)----------------------------------

    # First, create the Connectivity:
    connectivity, major_structs_labels, voxel_count, inds, flags, model_params = \
        create_connectivity(connectivity, sim_constants, files, model_params)


    # -----------------------------------Build cosimulator manually--------------------------------
    simulator = build_simulator(connectivity, sim_constants, inds, flags, model_params, config)

    # -----------------------------------2. Simulate and gather results-------------------------------------------------
    # Configure the simulator
    simulator.configure()
    simulator.integrate_next_step = simulator.integrator.integrate_with_update

    # ...and simulate!
    print("\n\nSimulating...")
    t_start = time.time()
    results = simulator.run(simulation_length=simulation_length)
    print("\nSimulated in %f secs!\n" % (time.time() - t_start))

    # -------------------------------------------3. Plot results--------------------------------------------------------
    if plot_write:
        # TODO: add "config": config.__dict__ when I have it serialized!
        savemat(os.path.join(config.out.FOLDER_RES, "sim_params") + ".mat",
                {"sim_constants": sim_constants,
                 "files": files,
                 "inds": inds,
                 "flags": flags,
                 "simulator": serialize_tvb_cosimulator(simulator)})
        #try:
        plot_write_analyse_results(results, simulator, inds, config, transient)
        # plot_write_results(results, simulator, transient=transient,
        #                    tvb_state_variables_labels=simulator.model.variables_of_interest,
        #                    # populations_sizes=populations_sizes,
        #                    plotter=Plotter(config), config=config)

        #except Exception as e:
        #    print("Error in analyzing, plotting or writing to files!:\n%s" % str(e))

    return results, simulator


if __name__ == "__main__":

    import sys

    if sys.argv[-1] == "1":
        wc_thalamocortical_example(GriffithsThalamoCortical, DEMO=True)
    else:
        wc_thalamocortical_example(WilsonCowanThalamoCortical, DEMO=True)
