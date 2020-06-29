# -*- coding: utf-8 -*-
import time
from copy import deepcopy
from collections import OrderedDict
import gc
from itertools import cycle
import numpy as np

from tvb_multiscale.config import Config
from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity, PSEWorkflowBase
from tvb_multiscale.examples.paperwork.workflow import Workflow
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message

from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def prepare_spike_stimulus(rate, dt, time_length, number_of_regions, number_of_neurons):
    from tvb_elephant.spike_stimulus_builder import SpikeStimulusBuilder
    rate = ensure_list(rate)
    time_length = ensure_list(time_length)
    if len(rate) > 1:
        rates = []
        for r, t in zip(rate, cycle(time_length)):
            rates += list(r * np.ones((t, )))
        stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=np.array(rates), A=None,
                                   # A=None for uncorrelated spikes
                                   number_of_regions=number_of_regions, number_of_neurons=number_of_neurons,
                                   target_regions=None, t_start=0.0, dt=dt, time_length=np.sum(time_length),
                                   sparse=False)
        stimulus = stb.build_inhomogeneous_poisson_process()
    else:
        stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=rate[0], A=None,
                                  # A=None for uncorrelated spikes
                                  number_of_regions=number_of_regions, number_of_neurons=number_of_neurons,
                                  target_regions=None, t_start=0.0, dt=dt, time_length=time_length[0],
                                  sparse=False)
        stimulus = stb.build_compound_poisson_process()
    return stimulus


class PSEWorkflowSpiking(PSEWorkflowBase):
    name = "PSEWorkflowSpiking"
    _plot_results = ["rate", "Pearson", "Spearman"]
    _corr_results = ["Pearson", "Spearman"]

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        self.config = Config(separate_by_run=False, output_base=output_base)
        self.branch = branch
        if self.branch == "low":
            self.noise = 0.01
        else:
            self.noise = 0.0
        self.name = branch + self.name
        self.workflow = Workflow()
        self.workflow.config = self.config
        self.workflow.tvb_model = SpikingWongWangExcIOInhI
        self.workflow.name = self.name
        self.workflow.populations_sizes = [100, 100]
        self.n_neurons = np.sum(self.workflow.populations_sizes).item()
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        self.workflow.tvb_init_cond[:, 5, :, :] = -70.0  # Setting V_m to V_rest
        self.workflow.dt = 0.025
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        self.stimulus_rate = 2018.0
        if fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.stim_time_length = int(np.ceil(self.workflow.simulation_length / self.workflow.dt))
        if self.branch == "high":
            self.stimulus_rate *= np.array([2.0, 1.0])
            self.workflow.simulation_length += self.workflow.transient
            self.stim_time_length = [ int(np.ceil(self.workflow.transient / self.workflow.dt)), self.stim_time_length]
            self.workflow.transient *= 2
        self.workflow.tvb_noise_strength = self.noise  # 0.0001 / 2
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
        super(PSEWorkflowSpiking, self).configure_PSE()


class PSE_1_TVBspikingNodeStW(PSEWorkflowSpiking):
    name = "PSE_1_TVBspikingNodeStW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_1_TVBspikingNodeStW, self).__init__(w, branch, fast, output_base)
        step = 1.0
        if fast:
            step *= 12.5
        self.PSE["params"]["Stimulus"] = np.arange(0.0, 25.1, step)
        self.configure_PSE(w, fast)
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self._plot_results = ["rate"]
        self.workflow.force_dims = 1

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([0.0, ]),
                               "w_EE": np.array([pse_params["w+"], ])}
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        for i_pop, pop in enumerate(["E", "I"]):
            self.PSE["results"]["rate"][pop][i1, i2] = rates["TVB"][i_pop].values.item()

    def prepare_stimulus(self, s):
        return prepare_spike_stimulus(self.stimulus_rate * (1 + s),
                                      self.workflow.dt, self.stim_time_length, 1, self.n_neurons)

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            stimulus = self.prepare_stimulus(s)
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset(pse_params)
                self.workflow.tvb_spike_stimulus = stimulus
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                gc.collect()
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()

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


class PSE_2_TVBspikingNodesGW(PSEWorkflowSpiking):
    name = "PSE_2_TVBspikingNodesGW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_TVBspikingNodesGW, self).__init__(w, branch, fast, output_base)
        step = 10.0
        if fast:
            step *= 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, step)
        self.configure_PSE(w, fast)
        Nreg = 2
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.empty(Nreg_shape) * np.nan,
                                                "I": np.empty(Nreg_shape) * np.nan}
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self.PSE["results"]["rate % diff"] = {"E": np.empty(self.pse_shape) * np.nan,
                                              "I": np.empty(self.pse_shape) * np.nan}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.empty(self.pse_shape) * np.nan
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman"]
        self.workflow.force_dims = Nreg
        self.stimulus = \
            prepare_spike_stimulus(self.stimulus_rate, self.workflow.dt, self.stim_time_length, Nreg, self.n_neurons)

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ]),
                               "w_EE": np.array([pse_params["w+"], ])}
        return model_params

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                try:
                    self.print_PSE(pse_params)
                    self.workflow.reset(pse_params)
                    self.workflow.tvb_spike_stimulus = self.stimulus
                    self.workflow.configure()
                    self.workflow.model_params = self.pse_to_model_params(pse_params)
                    rates, corrs = self.workflow.run()
                    self.results_to_PSE(i_s, i_w, rates, corrs)
                    gc.collect()
                except:
                    pass
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        for i_pop, pop in enumerate(["E", "I"]):
            PSE["rate per node"][pop][:, i_g, i_w] = rates["TVB"][i_pop].values.squeeze()
            PSE["rate"][pop][i_g, i_w] = np.nanmean(PSE["rate per node"][pop][:, i_g, i_w])
            PSE["rate % diff"][pop][i_g, i_w] = \
                100 * np.abs(np.diff(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            PSE[corr]["EE"][i_g, i_w] = corrs["TVB"][corr][0, 0, 0, 1].values.item()

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


class PSE_3_TVBspikingNodesGW(PSE_2_TVBspikingNodesGW):
    name = "PSE_3_TVBspikingNodesGW"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_TVBspikingNodesGW, self).__init__(w, branch, fast, output_base)
        step = 10.0
        if fast:
            step *= 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, step)
        self.configure_PSE(w, fast)
        Nreg = 3
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.empty(Nreg_shape) * np.nan,
                                                "I": np.empty(Nreg_shape) * np.nan}
        self.PSE["results"]["rate"] = {"E": np.empty(self.pse_shape) * np.nan,
                                       "I": np.empty(self.pse_shape) * np.nan}
        self.PSE["results"]["rate % zscore"] = {"E": np.empty(self.pse_shape) * np.nan,
                                                "I": np.empty(self.pse_shape) * np.nan}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.empty(Nreg_shape) * np.nan
            self.PSE["results"][corr]["FC-SC"] = np.empty(self.pse_shape) * np.nan
        self._plot_results = ["rate", "rate % zscore", "Pearson", "Spearman"]
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, Nreg)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.stimulus = \
            prepare_spike_stimulus(self.stimulus_rate, self.workflow.dt, self.stim_time_length, Nreg, self.n_neurons)

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        for i_pop, pop in enumerate(["E", "I"]):
            PSE["rate per node"][pop][:, i_g, i_w] = rates["TVB"][i_pop].values.squeeze()
            PSE["rate"][pop][i_g, i_w] = np.nanmean(PSE["rate per node"][pop][:, i_g, i_w])
            PSE["rate % zscore"][pop][i_g, i_w] = \
                100 * np.abs(np.std(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            PSE[corr]["EE"][:, i_g, i_w] = corrs["TVB"][corr][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
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
