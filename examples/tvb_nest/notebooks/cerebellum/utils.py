# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesX
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, is_float, ensure_list


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


def get_region_indice(reg, labels):
    if isinstance(reg, str):
        return np.where(labels==reg)[0].item()
    if is_integer(reg):
        return reg
    raise ValueError("reg should be either a region label or integer indice, but it is %s!" % str(reg))


def get_regions_indices(regs, labels):
    if regs is None:
        return slice(None)
    iR = []
    for reg in ensure_list(regs):
        iR.append(get_region_indice(reg, labels))
    return iR


def compute_plot_selected_spectra_coherence(source_ts, inds,
                                            transient=0.0, conn=None, nperseg=256, fmin=0.0, fmax=100.0,
                                            figsize=(15, 5), figures_path="", figname="", figformat="png", 
                                            show_flag=True, save_flag=True):
    n_regions = int(len(inds) / 2)
    data = source_ts[transient:, 0, inds].squeeze().T
    if conn is None:
        conn = source_ts.connectivity
    fs = 1000/source_ts.sample_period
    f, Pxx_den = signal.welch(data, fs, nperseg=nperseg)
    fig, axes = plt.subplots(n_regions, 2, figsize=(figsize[0], figsize[1]*n_regions))
    if axes.ndim == 1:
        axes = np.array([axes])
    for ii in range(n_regions):
        iR = ii*2
        iL = ii*2 + 1
        axes[ii, 0].plot(f, Pxx_den[iR],
                         label="%.1fHz, %s" % (f[np.argmax(Pxx_den[iR])], conn.region_labels[inds[iR]]))
        axes[ii, 0].plot(f, Pxx_den[iL],
                         label="%.1fHz, %s" % (f[np.argmax(Pxx_den[iL])], conn.region_labels[inds[iL]]))
        axes[ii, 0].set_xlim([fmin, fmax])
        axes[ii, 0].set_xlabel('frequency [Hz]')
        axes[ii, 0].set_ylabel('PSD [V**2/Hz]')
        axes[ii, 0].legend()
        axes[ii, 1].semilogy(f, Pxx_den[iR], label=conn.region_labels[inds[iR]])
        axes[ii, 1].semilogy(f, Pxx_den[iL], label=conn.region_labels[inds[iL]])
        axes[ii, 1].set_xlim([fmin, fmax])
        axes[ii, 1].set_xlabel('frequency [Hz]')
        axes[ii, 1].set_ylabel('PSD [log(V**2/Hz)]')
        axes[ii, 1].legend()
    # plt.ylim([1e-7, 1e2])
    if save_flag and len(figures_path) + len(figname):
        plt.savefig(os.path.join(figures_path, figname + "_PSD.%s" % figformat))
    if show_flag:
        plt.show()
    else:
        plt.close(fig)

    CxyR = []
    fR = []
    fL = []
    CxyL = []
    n_regions2 = int(n_regions * (n_regions - 1)/2)
    if nregions2:
        fig, axes = plt.subplots(n_regions2, 2, figsize=(figsize[0], figsize[1]*n_regions))
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
                axes[ii, 0].set_xlim([fmin, fmax])
                axes[ii, 0].set_ylim([fmin, 0.45])
                axes[ii, 0].set_xlabel('frequency [Hz]')
                axes[ii, 0].set_ylabel('Coherence')
                axes[ii, 0].legend()
                axes[ii, 1].semilogy(fR, CxyR.T,
                                     label="%s - %s" % (conn.region_labels[inds[iR1]], conn.region_labels[inds[iR2]]))
                axes[ii, 1].semilogy(fL, CxyL.T,
                                     label="%s - %s" % (conn.region_labels[inds[iL1]], conn.region_labels[inds[iL2]]))
                axes[ii, 1].set_xlim([fmin, fmax])
                axes[ii, 1].set_xlabel('frequency [Hz]')
                axes[ii, 1].set_ylabel('log10(Coherence)')
                axes[ii, 1].legend()
                ii += 1
        if save_flag and len(figures_path) + len(figname):
            plt.savefig(os.path.join(figures_path, figname + "_COH.%s" % figformat))
        if show_flag:
            plt.show()
        else:
            plt.close(fig)
    return Pxx_den, f, CxyR, fR, CxyL, fL


def only_plot_selected_spectra_coherence_and_diff(freq, avg_coherence, color, fmin=0.0, fmax=50.0, 
                                                  figsize=(15, 5), figures_path="", figformat="png",
                                                  show_flag=True, save_flag=True):
    import numpy as np
    yranges = [[0,0.35], [-0.2, 0.2]]    # Ranges for coherence and diff plot respectively
    ylabel = ['Spectral coherence','Diff in spectral coherence']
    # avg_coherence is a dictionary with average coherence between L and R M1-S1 for each simulation test
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]*2))
    for test in avg_coherence.keys():
        # Plot coherence
        axes[0].plot(freq, avg_coherence[test], color=color[test])
    # Plot coherence diff cosim vs MF cereb-OFF
    axes[1].plot(freq,np.subtract(avg_coherence['MF_cerebOFF'],avg_coherence['cosim']), color=color['cosim'])
    # Plot coherence diff MF cereb-ON vs MF cereb-OFF
    axes[1].plot(freq,np.subtract(avg_coherence['MF_cerebOFF'],avg_coherence['MF_cerebON']), color=color['MF_cerebON'])

    for ii in range(len(axes)):
        axes[ii].set_xlim([fmin, fmax])
        axes[ii].set_xlabel('frequency [Hz]')
        axes[ii].set_ylabel(ylabel[ii])
        axes[ii].vlines(25, yranges[ii][0], yranges[ii][1])
        axes[ii].vlines(45, yranges[ii][0], yranges[ii][1])
        
        
    axes[0].set_ylim(yranges[0])
    axes[0].set_title('M1-S1 coherence spectra during virtual whisking')
    axes[0].legend(avg_coherence.keys())
    axes[1].set_ylim(yranges[1])
    axes[1].set_title('change in M1-S1 coherence after virtual cerebellar inactivation')
    axes[1].legend(['OFF-ON cosim','OFF-ON MF'])

    if show_flag:
        plt.show()
    else:
        plt.close(fig)
    
    if save_flag and len(figures_path):
        plt.savefig(os.path.join(figures_path, "COHselectDiff.%s" % figformat))


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
                               figsize=plotter.config.DEFAULT_SIZE, figname="%s ICA components Time Series" % variable)

        fig = plt.figure(figsize=(plotter.config.DEFAULT_SIZE[0], 5))
        plt.imshow(ica.components_)
        plt.xlabel("Region")
        plt.ylabel("ICA component")
        plt.title("ICA components")
        plt.colorbar()
        plt.tight_layout()
        if plotter.config.SAVE_FLAG:
            plt.savefig(os.path.join(plotter.config.FOLDER_FIGURES, "ICA.%s" % plotter.config.FIG_FORMAT))
        if plotter.config.SHOW_FLAG:
            plt.show()
        else:
            plt.close(fig)
    return ica.components_, ics_ts, ica  # (ICA components, ICA components time series, ICA class instance)

# Example about how ICA works:
# from sklearn.decomposition import FastICA
# np.random.seed(0)
# n_samples = 2000
# time = np.linspace(0, 8, n_samples)
# s1 = np.sin(2 * time)
# s2 = np.sign(np.sin(3 * time))
# s3 = signal.sawtooth(2 * np.pi * time)
# S = np.c_[s1, s2, s3]
# S += 0.2 * np.random.normal(size=S.shape)
# S /= S.std(axis=0)
# A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
# X = np.dot(S, A.T)
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(X)
# fig = plt.figure()
# models = [X, S, S_]
# names = ['mixtures', 'real sources', 'predicted sources']
# colors = ['red', 'blue', 'orange']
# for i, (name, model) in enumerate(zip(names, models)):
#     plt.subplot(4, 1, i+1)
#     plt.title(name)
#     for sig, color in zip (model.T, colors):
#         plt.plot(sig, color=color)

# fig.tight_layout()
# plt.show()
