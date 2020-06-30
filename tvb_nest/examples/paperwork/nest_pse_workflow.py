# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np

from tvb_nest.config import Config
from tvb_nest.examples.paperwork.workflow import Workflow
from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity, PSEWorkflowBase

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class PSENESTWorkflowBase(PSEWorkflowBase):
    name = "PSENESTWorkflow"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSENESTWorkflowBase, self).__init__()
        self.branch = branch
        self.name = self.branch + self.name
        self.config = Config(separate_by_run=False, output_base=output_base)
        self._plot_results = ["rate", "Pearson", "Spearman", "spike train"]
        self._corr_results = ["Pearson", "Spearman", "spike train"]
        self.workflow = Workflow()
        self.workflow.config = self.config
        self.workflow.name = self.name
        self.workflow.tvb_to_nest_interface = None
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.dt = 0.1
        self.workflow.simulation_length = 3000.0
        self.workflow.transient = 1000.0
        if fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.nest_stimulus_rate = 2018.0
        self.workflow.nest_stimulus_times = [0.1]
        if self.branch == "high":
            self.workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
            self.workflow.simulation_length += self.workflow.transient
            self.workflow.transient *= 2
            self.workflow.nest_stimulus_times += [self.workflow.transient/2]
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = True
        self.workflow.writer = True
        self.workflow.write_time_series = False
        self.workflow.print_progression_message = self.print_progression_message
        kwargs = {}
        if w is not None:
            w = ensure_list(w)
            if len(w) == 1:
                kwargs = {"w+": w[0]}
        self.configure_paths(**kwargs)

    def configure_PSE(self, w=None, fast=False):
        if w is None:
            step = 0.1
            if fast:
                step = 0.3
            w = np.sort(np.arange(0.9, 1.6, step).tolist() + [1.55])
        else:
            w = np.sort(ensure_list(w))
        self.PSE["params"]["w+"] = w
        super(PSENESTWorkflowBase, self).configure_PSE()


class PSE_1_NESTnodeStW(PSENESTWorkflowBase):
    name = "PSE_1_NESTnodeStW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_1_NESTnodeStW, self).__init__(w, branch, fast, output_base)
        step = 1.0
        if fast:
            step *= 12.5
        self.PSE["params"]["Stimulus"] = np.arange(0.0, 25.1, step)
        self.configure_PSE(w, fast)
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self._plot_results = ["rate"]
        self.workflow.force_dims = [34]
        self.workflow.nest_nodes_ids = [0]  # the indices of fine scale regions modeled with NEST

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["NEST"]["E"] = {"w_E": pse_params["w+"].item()}
        self.workflow.nest_stimulus_rate = 2018.0 * (1 + pse_params["Stimulus"].item())
        if self.branch == "high":
            self.workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
            self.PSE["results"]["rate"][pop][i1, i2] = rates["NEST"][i_pop].values.item()

    def load_PSE_1D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_w, w in enumerate(self.PSE["params"]["w+"]):
            path = self.res_path.replace(".h5", "_w+%g.h5" % w)
            try:
                PSE = self.reader.read_dictionary(path=path, close_file=True)
            except:
                print("Failed to load file %s!" % path)
                continue
            for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                self.PSE["results"]["rate"][pop][:, i_w] = PSE["results"]["rate"][pop].squeeze()

    def load_PSE_2D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_s, s in enumerate(self.PSE["params"]["Stimulus"]):
            for i_w, w in enumerate(self.PSE["params"]["w+"]):
                path = self.res_path.replace(".h5", "_Stimulus%g_w+%g.h5" % (s, w))
                try:
                    PSE = self.reader.read_dictionary(path=path, close_file=True)
                except:
                    print("Failed to load file %s!" % path)
                    continue
                for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                    self.PSE["results"]["rate"][pop][i_s, i_w] = PSE["results"]["rate"][pop].item()


class PSE_2_NESTnodesGW(PSENESTWorkflowBase):
    name = "PSE_2_NESTnodesGW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_NESTnodesGW, self).__init__(w, branch, fast, output_base)
        step = 10.0
        if fast:
            step *= 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 205.0, step)
        self.configure_PSE(w, fast)
        Nreg = 2
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.empty(Nreg_shape) * np.nan,
                                                "I": np.empty(Nreg_shape) * np.nan}
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self.PSE["results"]["rate % diff"] = {"E": np.empty(self.pse_shape) * np.nan,
                                              "I": np.empty(self.pse_shape) * np.nan}
        for corr in self._corr_results:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.empty(self.pse_shape) * np.nan
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman", "spike train"]
        self.workflow.force_dims = [34, 35]
        self.workflow.nest_nodes_ids = [0, 1]  # the indices of fine scale regions modeled with NEST

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ])}
        model_params["NEST"]["E"] = {"w_E": pse_params["w+"]}
        return model_params

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        for i_pop, pop in enumerate(PSE["rate"].keys()):
            PSE["rate per node"][pop][:, i_g, i_w] = rates["NEST"][i_pop].values.squeeze()
            PSE["rate"][pop][i_g, i_w] = np.nanmean(PSE["rate per node"][pop][:, i_g, i_w])
            PSE["rate % diff"][pop][i_g, i_w] = \
                100 * np.abs(np.diff(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            corr_name = corr.replace(" ", "_")
            PSE[corr]["EE"][i_g, i_w] = corrs["NEST"][corr_name][0, 0, 0, 1].values.item()

    def load_PSE_1D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_w, w in enumerate(self.PSE["params"]["w+"]):
            path = self.res_path.replace(".h5", "_w+%g.h5" % w)
            try:
                PSE = self.reader.read_dictionary(path=path, close_file=True)
            except:
                print("Failed to load file %s!" % path)
                continue
            for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                self.PSE["results"]["rate per node"][pop][:, :, i_w] = PSE["results"]["rate per node"][pop].squeeze()
                self.PSE["results"]["rate"][pop][:, i_w] = PSE["results"]["rate"][pop].squeeze()
                self.PSE["results"]["rate % diff"][pop][:, i_w] = PSE["results"]["rate % diff"][pop].squeeze()
            for corr in self._corr_results:
                self.PSE["results"][corr]["EE"][:, i_w] = PSE["results"][corr]["EE"].squeeze()

    def load_PSE_2D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_g, g in enumerate(self.PSE["params"]["G"]):
            for i_w, w in enumerate(self.PSE["params"]["w+"]):
                path = self.res_path.replace(".h5", "_G%g_w+%g.h5" % (g, w))
                try:
                    PSE = self.reader.read_dictionary(path=path, close_file=True)
                except:
                    print("Failed to load file %s!" % path)
                    continue
                for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                    self.PSE["results"]["rate per node"][pop][:, i_g, i_w] = PSE["results"]["rate per node"][pop].squeeze()
                    self.PSE["results"]["rate"][pop][i_g, i_w] = PSE["results"]["rate"][pop].item()
                    self.PSE["results"]["rate % diff"][pop][i_g, i_w] = PSE["results"]["rate % diff"][pop].item()
                for corr in self._corr_results:
                    self.PSE["results"][corr]["EE"][i_g, i_w] = PSE["results"][corr]["EE"].item()


class PSE_3_NESTnodesGW(PSE_2_NESTnodesGW):
    name = "PSE_3_NESTnodesGW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_NESTnodesGW, self).__init__(w, branch, fast, output_base)
        step = 10.0
        if fast:
            step *= 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 205.0, step)
        self.configure_PSE(w, fast)
        Nreg = 3
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.empty(Nreg_shape) * np.nan,
                                                "I": np.empty(Nreg_shape) * np.nan}
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self.PSE["results"]["rate % zscore"] = {"E": np.empty(self.pse_shape) * np.nan,
                                                "I": np.empty(self.pse_shape) * np.nan}
        for corr in self._corr_results:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.empty(Nreg_shape) * np.nan
            self.PSE["results"][corr]["FC-SC"] = np.empty(self.pse_shape) * np.nan
        self._plot_results = ["rate", "rate % zscore", "Pearson", "Spearman", "spike train"]
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, 3)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.workflow.force_dims = Nreg
        self.workflow.nest_nodes_ids = [0, 1, 2]  # the indices of fine scale regions modeled with NEST

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        for i_pop, pop in enumerate(PSE["rate"].keys()):
            PSE["rate per node"][pop][:, i_g, i_w] = rates["NEST"][i_pop].values.squeeze()
            PSE["rate"][pop][i_g, i_w] = np.nanmean(PSE["rate per node"][pop][:, i_g, i_w])
            PSE["rate % zscore"][pop][i_g, i_w] = \
                100 * np.abs(np.nanstd(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            corr_name = corr.replace(" ", "_")
            PSE[corr]["EE"][:, i_g, i_w] = corrs["NEST"][corr_name][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)

    def load_PSE_1D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_w, w in enumerate(self.PSE["params"]["w+"]):
            path = self.res_path.replace(".h5", "_w+%g.h5" % w)
            try:
                PSE = self.reader.read_dictionary(path=path, close_file=True)
            except:
                print("Failed to load file %s!" % path)
                continue
            for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                self.PSE["results"]["rate per node"][pop][:, :, i_w] = PSE["results"]["rate per node"][pop].squeeze()
                self.PSE["results"]["rate"][pop][:, i_w] = PSE["results"]["rate"][pop].squeeze()
                self.PSE["results"]["rate % zscore"][pop][:, i_w] = PSE["results"]["rate % zscore"][pop].squeeze()
            for corr in self._corr_results:
                self.PSE["results"][corr]["EE"][:, :, i_w] = PSE["results"][corr]["EE"].squeeze()
                self.PSE["results"][corr]["FC-SC"][:, i_w] = PSE["results"][corr]["FC-SC"].squeeze()

    def load_PSE_2D(self, **kwargs):
        self.update_pse_params(**kwargs)
        for i_g, g in enumerate(self.PSE["params"]["G"]):
            for i_w, w in enumerate(self.PSE["params"]["w+"]):
                path = self.res_path.replace(".h5", "_G%g_w+%g.h5" % (g, w))
                try:
                    PSE = self.reader.read_dictionary(path=path, close_file=True)
                except:
                    print("Failed to load file %s!" % path)
                    continue
                for i_pop, pop in enumerate(self.PSE["results"]["rate"].keys()):
                    self.PSE["results"]["rate per node"][pop][:, i_g, i_w] = PSE["results"]["rate per node"][pop].squeeze()
                    self.PSE["results"]["rate"][pop][i_g, i_w] = PSE["results"]["rate"][pop].item()
                    self.PSE["results"]["rate % zscore"][pop][i_g, i_w] = PSE["results"]["rate % zscore"][pop].item()
                for corr in self._corr_results:
                    self.PSE["results"][corr]["EE"][:, i_g, i_w] = PSE["results"][corr]["EE"].squeeze()
                    self.PSE["results"][corr]["FC-SC"][i_g, i_w] = PSE["results"][corr]["FC-SC"].item()
