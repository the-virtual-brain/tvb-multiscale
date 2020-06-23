# -*- coding: utf-8 -*-

import os
import time
from collections import OrderedDict
import gc

import numpy as np
from xarray import DataArray

from matplotlib import pyplot as pl

from tvb.datatypes.connectivity import Connectivity
from tvb_multiscale.config import CONFIGURED, Config
try:
    from tvb_multiscale.io.h5_writer import H5Writer
    WRITER = H5Writer()
except:
    WRITER = None
from tvb_multiscale.io.h5_reader import H5Reader
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message

from tvb.contrib.scripts.utils.file_utils import safe_makedirs


def print_PSE(pse_params):
    msg = ""
    for p, val in pse_params.items():
        msg += p + "=%g " % val
    print(msg)


def plot_result(PSE_params, result, name, path):
    arr = DataArray(data=result, dims=list(PSE_params.keys()), coords=dict(PSE_params), name=name)
    if arr.shape[1] > 1:
        fig, axes = pl.subplots(nrows=2, ncols=2, gridspec_kw={"width_ratios": [1, 0.1]}, figsize=(10, 10))
        axes[1, 1].remove()
        im = arr.plot(x=arr.dims[0], y=arr.dims[1], ax=axes[0, 0], add_colorbar=False)
        axes[0, 0].set_xlabel("")
        fig.colorbar(im, cax=axes[0, 1], orientation="vertical", label=arr.name)
        arr.plot.line(x=arr.dims[0], hue=arr.dims[1], ax=axes[1, 0])
    else:
        fig = pl.figure(figsize=(10, 5))
        arr.plot.line(x=arr.dims[0])
    pl.savefig(path)


def symmetric_connectivity(SC, Nreg):
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.weights = np.zeros((Nreg, Nreg))
    SC = np.array(SC)
    SCsize = np.sqrt(np.sum(SC ** 2))
    inds = np.tril_indices(Nreg, -1)
    connectivity.weights[inds[0], inds[1]] = SC
    triu_inds = np.triu_indices(Nreg, 1)
    connectivity.weights[triu_inds[0], triu_inds[1]] = SC
    return connectivity, SC, SCsize, triu_inds


class PSEWorkflowBase(object):
    name = "PSEWorkflow"
    config = Config(separate_by_run=False)
    PSE = {"params": OrderedDict(), "n_params": OrderedDict(), "results": OrderedDict()}
    writer = WRITER
    reader = H5Reader()
    print_progression_message = False

    def configure_paths(self, **kwargs):
        self.folder_res = self.config.out._folder_res.replace("res", self.name)
        safe_makedirs(self.folder_res)
        self.res_path = os.path.join(self.folder_res, self.name)
        self.folder_figs = os.path.join(self.folder_res, "figs")
        for key, val in kwargs.items():
            addstring = "_%s%g" % (key, val)
            self.res_path = self.res_path + addstring
            self.folder_figs = self.folder_figs + addstring
        self.res_path = self.res_path + ".h5"
        safe_makedirs(self.folder_figs)

    def print_PSE(self, pse_params):
        print_PSE(pse_params)

    def configure_PSE(self):
        self.pse_shape = []
        for p in list(self.PSE["params"].keys()):
            np = "n_"+p
            self.PSE["n_params"][np] = len(self.PSE["params"][p])
            self.pse_shape.append(self.PSE["n_params"][np])
        self.pse_shape = tuple(self.pse_shape)

    def pse_to_model_params(self, pse_params):
        pass

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        pass

    def write_PSE(self):
        if self.writer:
            self.writer.write_dictionary(self.PSE, path=self.res_path)

    def plot_PSE(self):
        params = self.PSE["params"]
        PSE = self.PSE["results"]
        for res in self._plot_results:
            for pop in list(PSE[res].keys()):
                if res in self._corr_results:
                    name = "%s %s Corr" % (res, pop)
                else:
                    name = "%s %s" % (res, pop)
                if res == "rate":
                    name += " spikes/sec"
                filename = name.replace(" spikes/sec", "").replace(" ", "_") + ".png"
                try:
                    plot_result(params, PSE[res][pop], name,
                                os.path.join(self.folder_figs, filename))
                except:
                    continue

    def load_PSE(self):
        self.PSE = self.reader.read_dictionary(path=self.res_path, close_file=True)

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset(pse_params)
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                gc.collect()
                print_toc_message(tic)
        self.workflow = None
        self.write_PSE()
        self.plot_PSE()
