# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesX
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, is_float


def print_lbl(lbl, siz, prnt=""):
    prnt += lbl
    prnt += "." * (siz - len(lbl))
    return prnt


def print_row(vals, sizes, prnt=""):
    prnt += "\n"
    for iV, (val, siz) in enumerate(zip(vals, sizes)):
        if is_integer(val):
            prnt = print_lbl("%d" % val, siz, prnt)
        elif is_float(val):
            prnt = print_lbl("%g" % val, siz, prnt)  # %.3f
        else:
            prnt = print_lbl(str(val), siz, prnt)
    return prnt


def print_conn(d={}, prnt="", maxrow=200, printit=True):
    sizes = []
    values = []
    for col, val in d.items():
        sizes.append(col[1])
        values.append(np.array(val))
        prnt = print_lbl(col[0], sizes[-1], prnt)
    prnt += "\n" + "-" * np.sum(sizes)
    for iV, vals in enumerate(zip(*values)):
        if iV == maxrow:
            break
        prnt = print_row(vals, sizes, prnt)
    if printit:
        print(prnt)
    return prnt


def compute_plot_selected_spectra_coherence(source_ts, inds, transient=0.0, conn=None, nperseg=256):
    n_regions = int(len(inds) / 2)
    data = source_ts[transient:, 0, inds, 0].squeeze().T
    if conn is None:
        conn = source_ts.connectivity
    fs = 1000/source_ts.sample_period
    f, Pxx_den = signal.welch(data, fs, nperseg=nperseg)
    print(np.mean(np.diff(f)))
    fig, axes = plt.subplots(n_regions, 2, figsize=(15, 5*n_regions))
    for ii in range(n_regions):
        iR = ii*2
        iL = ii*2 + 1
        axes[ii, 0].plot(f, Pxx_den[0], label="%.1fHz, %s" % (f[np.argmax(Pxx_den[0])], conn.region_labels[inds[iR]]))
        axes[ii, 0].plot(f, Pxx_den[1], label="%.1fHz, %s" % (f[np.argmax(Pxx_den[0])], conn.region_labels[inds[iL]]))
        axes[ii, 0].set_xlim([0, 100])
        axes[ii, 0].set_xlabel('frequency [Hz]')
        axes[ii, 0].set_ylabel('PSD [V**2/Hz]')
        axes[ii, 0].legend()
        axes[ii, 1].semilogy(f, Pxx_den[0], label=conn.region_labels[inds[iR]])
        axes[ii, 1].semilogy(f, Pxx_den[1], label=conn.region_labels[inds[iL]])
        axes[ii, 1].set_xlim([0, 100])
        axes[ii, 1].set_xlabel('frequency [Hz]')
        axes[ii, 1].set_ylabel('PSD [log(V**2/Hz)]')
        axes[ii, 1].legend()
    # plt.ylim([1e-7, 1e2])
    plt.show()

    n_regions2 = int(n_regions * (n_regions - 1)/2)
    fig, axes = plt.subplots(n_regions2, 2, figsize=(15, 5*n_regions2))
    if len(axes.shape) < 2:
        axes = axes[np.newaxis, :]
    ii = 0
    for i1 in range(0, n_regions-1):
        iR1 = 2*i1
        iL1 = 2*i1 + 1
        for i2 in range(i1+1, n_regions):
            iR2 = 2*i2
            iL2 = 2*i2 + 1
            fR, CxyR = signal.coherence(data[iR1], data[iR2], fs, nperseg=nperseg)
            fL, CxyL = signal.coherence(data[iL1], data[iL2], fs, nperseg=nperseg)
            axes[ii, 0].plot(fR, CxyR.T,
                             label="%s - %s" % (conn.region_labels[inds[iR1]], conn.region_labels[inds[iR2]]))
            axes[ii, 0].plot(fL, CxyL.T,
                             label="%s - %s" % (conn.region_labels[inds[iL1]], conn.region_labels[inds[iL2]]))
            axes[ii, 0].set_xlim([0, 100])
            axes[ii, 0].set_xlabel('frequency [Hz]')
            axes[ii, 0].set_ylabel('Coherence')
            axes[ii, 0].legend()
            axes[ii, 1].semilogy(fR, CxyR.T,
                                 label="%s - %s" % (conn.region_labels[inds[iR1]], conn.region_labels[inds[iR2]]))
            axes[ii, 1].semilogy(fL, CxyL.T,
                                 label="%s - %s" % (conn.region_labels[inds[iL1]], conn.region_labels[inds[iL2]]))
            axes[ii, 1].set_xlim([0, 100])
            axes[ii, 1].set_xlabel('frequency [Hz]')
            axes[ii, 1].set_ylabel('log10(Coherence)')
            axes[ii, 1].legend()
            ii += 1
    plt.show()


def compute_plot_ica(data, time, variable="BOLD", n_components=10, plotter=None):
    ica = FastICA(n_components=n_components)
    ics_ts = ica.fit_transform(data)
    ics_ts = TimeSeriesX(
        data=ics_ts[:, np.newaxis, :, np.newaxis], time=time,
        labels_ordering=["Time", "State Variable", "ICA", "Modes"],
        labels_dimensions={"State Variable": [variable],
                           "ICA": np.arange(ics_ts.shape[1])})
    ics_ts.configure()

    if plotter:
        ics_ts.plot_timeseries(plotter_config=plotter.config,
                               hue="ICA" if ics_ts.shape[2] > plotter.config.MAX_REGIONS_IN_ROWS else None,
                               per_variable=ics_ts.shape[1] > plotter.config.MAX_VARS_IN_COLS,
                               figsize=plotter.config.FIGSIZE, figname="%s ICA components Time Series" % variable)

        plt.figure(figsize=(plotter.config.FIGSIZE[0], 5))
        plt.imshow(ica.components_)
        plt.xlabel("Region")
        plt.ylabel("ICA component")
        plt.title("ICA components")
        plt.colorbar()
        plt.tight_layout()
    return ica.components_, ics_ts, ica  # (ICA components, ICA components time series, ICA class instance)
