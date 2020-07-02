# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
import gc

import numpy as np
from xarray import DataArray

from matplotlib import pyplot as pl

from tvb_nest.config import Config
from tvb_nest.examples.paperwork.workflow import Workflow
from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity, PSEWorkflowBase

from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def plot_result(PSE_params, result, name, path):
    arr = DataArray(data=result, dims=list(PSE_params.keys()), coords=dict(PSE_params), name=name)
    fig, axes = pl.subplots(arr.shape[1], arr.shape[2], figsize=(5 * arr.shape[1], 5 * arr.shape[2]))
    axes = np.array(axes)
    while axes.ndim < 2:
        axes = np.expand_dims(axes, -1)
    for i_w1 in range(arr.shape[1]):
        for i_w2 in range(arr.shape[2]):
            line = arr[:, i_w1, i_w2].plot.line(x=arr.dims[0],
                                                add_legend=False, ax=axes[i_w1, i_w2])
            line[-1].set_color("k")
            # line[-1].set_linestyle("-")
    fig.canvas.draw()
    pl.savefig(path)


class PSETVBNESTWorkflow(PSEWorkflowBase):
    name = "PSETVBNESTWorkflow"

    def __init__(self, tvb_nodes=[0], nest_nodes=[1], SC=None,
                 branch="low", fast=False, output_base=None, **kwargs):
        super(PSETVBNESTWorkflow, self).__init__()
        self.w = "w"
        self.tvb_nodes = tvb_nodes
        self.nest_nodes = nest_nodes
        self.branch = branch
        self.name = self.branch + self.name
        self.config = Config(separate_by_run=False, output_base=output_base)
        self.wTVB = np.array([0.9, ])  # np.arange(0.8, 1.01, 0.05)
        self.wNEST = np.array([1.3, ])  # np.sort(np.arange(1.0, 1.41, 0.1).tolist() + [1.55])
        self.fast = fast
        kwargs = self.configure_PSE(**kwargs)
        self.configure_paths(**kwargs)
        self.n_tvb_nodes = len(self.tvb_nodes)
        self.n_nest_nodes = len(self.nest_nodes)
        self.n_nodes = self.n_tvb_nodes + self.n_nest_nodes
        self._SC = SC
        self.create_workflow()
        self.prepare_PSE_results()

    def configure_PSE(self, **kwargs):
        step = 10.0
        if self.fast:
            step *= 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 305.0, step)
        for n in ["TVB", "NEST"]:
            wn = "w%s" % n
            wnp = "w+%s" % n
            wval = ensure_list(kwargs.pop(wn, getattr(self, wn)))
            if len(wval) == 1:
                kwargs[wnp] = wval[0]
            self.PSE["params"][wnp] = np.sort(wval)
        super(PSETVBNESTWorkflow, self).configure_PSE()
        return kwargs

    def create_workflow(self):
        self.workflow = Workflow()
        self.workflow.config = self.config
        self.workflow.name = self.name
        self.workflow.tvb_model = ReducedWongWangExcIO
        # # ------Uncomment below to change model to ReducedWongWangExcIOInhI, Deco et al 2014------------------
        # self.workflow.tvb_model = ReducedWongWangExcIOInhI
        # self.workflow.tvb_rate_vars = ["R_e", "R_i"]
        # self.w = "w_p"
        # self.I_o = 0.3
        # # ------Uncomment above to change model to ReducedWongWangExcIOInhI, Deco et al 2014------------------
        self.workflow.tvb_to_nest_interface = "rate"
        self.workflow.nest_to_tvb_interface = True
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.force_dims = self.n_nodes
        self.workflow.nest_nodes_ids = self.nest_nodes
        if self._SC:
            connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, self.n_nodes)
            self.workflow.connectivity_path = "None"
            self.workflow.connectivity = connectivity
        self.workflow.dt = 0.1
        self.workflow.simulation_length = 3000.0
        self.workflow.transient = 1000.0
        if self.fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.nest_stimulus_rate = 2018.0
        self.workflow.nest_stimulus_times = [0.1]
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        if self.branch == "high":
            if self.n_tvb_nodes:
                self.workflow.tvb_init_cond[0, 0, self.tvb_nodes, 0] = 1.0
            if self.n_nest_nodes:
                self.workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
                self.workflow.simulation_length += self.workflow.transient
                self.workflow.transient *= 2
                self.workflow.nest_stimulus_times += [self.workflow.transient/2]
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = True
        self.workflow.writer = True
        self.workflow.write_time_series = False
        self.workflow.print_progression_message = self.print_progression_message

    def prepare_PSE_results(self):
        self._plot_results = []
        self._rate_results = []
        self._corr_results = []
        self._tvb_results = ["rate", "Pearson", "Spearman"]
        self._nest_results = ["spike rate", "spike Pearson", "spike Spearman", "spike train"]
        if self.n_tvb_nodes:
            self._plot_results.append("rate")
            self._rate_results.append("rate")
            if self.n_nodes > 1:
                self._plot_results += ["Pearson", "Spearman"]
                self._corr_results += ["Pearson", "Spearman"]
                self._triu_inds = np.triu_indices(self.n_nodes, 1)
        if self.n_nest_nodes:
            self._plot_results.append("spike rate")
            self._rate_results.append("spike rate")
            if self.n_nest_nodes > 1:
                self._plot_results += ["spike Pearson", "spike Spearman", "spike train"]
                self._corr_results += ["spike Pearson", "spike Spearman", "spike train"]
                self._nest_triu_inds = np.triu_indices(self.n_nest_nodes, 1)
        Ntvb_reg_shape = (self.n_nodes,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.empty(Ntvb_reg_shape) * np.nan,
                                                # "I": np.empty(Ntvb_reg_shape) * np.nan   # uncomment for ExcIOInhI model
                                                }
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan}
        Nnest_reg_shape = (self.n_nest_nodes,) + self.pse_shape
        self.PSE["results"]["spike rate per node"] = {"E": np.empty(Nnest_reg_shape) * np.nan,
                                                      "I": np.empty(Nnest_reg_shape) * np.nan}
        self.PSE["results"]["spike rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                             "I": np.empty(self.pse_shape) * np.nan}
        if self.n_nodes > 1:
            self.corr_size = int(self.n_nodes * (self.n_nodes - 1) / 2)
            self.nest_corr_size = int(self.n_nest_nodes * (self.n_nest_nodes - 1) / 2)
            for corr in self._corr_results:
                if corr in self._nest_results:
                    if self.nest_corr_size == 0:
                        continue
                    else:
                        corr_size = (self.nest_corr_size,) + self.pse_shape
                else:
                    corr_size = (self.corr_size,) + self.pse_shape
                self.PSE["results"][corr] = OrderedDict()
                self.PSE["results"][corr]["EE"] = np.empty(corr_size) * np.nan
                if corr_size[0] > 1:
                    self.PSE["results"][corr]["FC-SC"] = np.empty(self.pse_shape) * np.nan

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ]), self.w: np.array([pse_params["w+TVB"], ])}
        model_params["NEST"]["E"] = {"w_E": pse_params["w+NEST"]}
        return model_params

    def results_to_PSE(self, i_g, i_w1, i_w2, rates, corrs):
        PSE = self.PSE["results"]
        for rate in self._rate_results:
            if rate.find("spike") > -1:
                n = "NEST"
            else:
                n = "TVB"
            for i_pop, pop in enumerate(PSE[rate].keys()):
                PSE["%s per node" % rate][pop][:, i_g, i_w1, i_w2] = rates[n][i_pop].values.squeeze()
                PSE[rate][pop][i_g, i_w1, i_w2] = np.nanmean(PSE["%s per node" % rate][pop][:, i_g, i_w1, i_w2])
        for corr in self._corr_results:
            if corr in self._tvb_results:
                corr_name = corr.replace(" ", "_")
                PSE[corr]["EE"][:, i_g, i_w1, i_w2] = \
                    corrs["TVB"][corr_name][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
                if PSE[corr]["EE"][:, i_g, i_w1, i_w2].size > 1:
                    PSE[corr]["FC-SC"][i_g, i_w1, i_w2] = \
                        (np.dot(PSE[corr]["EE"][:, i_g, i_w1, i_w2], self.SC)) / \
                        (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w1, i_w2] ** 2)) * self.SCsize)
            elif self.n_nest_nodes > 1:
                corr_name = corr.replace("spike ", "").replace(" ", "_")
                PSE[corr]["EE"][:, i_g, i_w1, i_w2] = \
                    corrs["NEST"][corr_name][0, 0].values[self._nest_triu_inds[0], self._nest_triu_inds[1]]
                if PSE[corr]["EE"][:, i_g, i_w1, i_w2].size > 1:
                    PSE[corr]["FC-SC"][i_g, i_w1, i_w2] = \
                        (np.dot(PSE[corr]["EE"][:, i_g, i_w1, i_w2], self.SC_nest)) / \
                        (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w1, i_w2] ** 2)) * self.SCsize_nest)

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        compute_SC = (self.n_nodes > 2)
        for i_g, g in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = g
            for i_w1, w1 in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w1
                for i_w2, w2 in enumerate(self.PSE["params"][params[2]]):
                    pse_params[params[2]] = w2
                    tic = time.time()
                    try:
                        self.print_PSE(pse_params)
                        self.workflow.reset(pse_params)
                        self.workflow.configure()
                        self.workflow.model_params = self.pse_to_model_params(pse_params)
                        rates, corrs = self.workflow.run()
                        if compute_SC:
                            compute_SC = False
                            self.SC = self.workflow.simulator.connectivity.weights[self._triu_inds]
                            self.SCsize = np.sqrt(np.sum(self.SC ** 2))
                            if self.n_nest_nodes > 2:
                                self.SC_nest = self.workflow.simulator.connectivity.weights[self.nest_nodes][:,
                                               self.nest_nodes]
                                self.SC_nest = self.SC_nest[self._nest_triu_inds]
                                self.SCsize_nest = np.sqrt(np.sum(self.SC_nest ** 2))
                        self.results_to_PSE(i_g, i_w1, i_w2, rates, corrs)
                        gc.collect()
                    except:
                        pass
                    print_toc_message(tic)
        self.workflow = None
        self.write_PSE()
        self.plot_PSE()

    def plot_PSE(self):
        params = self.PSE["params"]
        PSE = self.PSE["results"]
        for res in self._plot_results:
            for pop in list(PSE[res].keys()):
                if res in self._corr_results:
                    name = "%s %s Corr" % (res, pop)
                else:
                    name = "%s %s" % (res, pop)
                if res.find("rate") > -1:
                    name += " spikes/sec"
                filename = name.replace(" spikes/sec", "").replace(" ", "_") + ".png"
                try:
                    plot_result(params, PSE[res][pop], name,
                                os.path.join(self.folder_figs, filename))
                except:
                    continue

    def load_PSE_2D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_w1, wTVB in enumerate(self.PSE["params"]["w+TVB"]):
            for i_w2, wNEST in enumerate(self.PSE["params"]["w+NEST"]):
                try:
                    PSE = self.reader.read_dictionary(path=self.res_path, close_file=True)
                except:
                    print("Failed to load file %s!" % self.res_path)
                    continue
                for rate in self._rate_results:
                   for i_pop, pop in enumerate(self.PSE["results"][rate].keys()):
                        self.PSE["results"]["%s per node" % rate][pop][:, :, i_w1, i_w2] = \
                            PSE["results"]["%s per node" % rate][pop].squeeze()
                        self.PSE["results"][rate][pop][:, i_w1, i_w2] = PSE["results"][rate][pop].squeeze()
                for corr in self._corr_results:
                    self.PSE["results"][corr]["EE"][:, :, i_w1, i_w2] = PSE["results"][corr]["EE"].squeeze()
                    try:
                        self.PSE["results"][corr]["FC-SC"][:, i_w1, i_w2] = PSE["results"][corr]["FC-SC"].squeeze()
                    except:
                        pass
