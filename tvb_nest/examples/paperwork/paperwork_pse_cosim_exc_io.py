# -*- coding: utf-8 -*-
import os
import time
from copy import deepcopy

import numpy as np
from xarray import DataArray

from matplotlib import pyplot as pl
from matplotlib import cm

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb_nest.examples.paperwork.tvb_nest_pse_workflow import PSETVBNESTWorkflow

from tvb.contrib.scripts.utils.log_error_utils import print_toc_message
from tvb.contrib.scripts.utils.file_utils import safe_makedirs


def run_PSE(pse_class, todo="run", **kwargs):
    pse_workflow = pse_class(**kwargs)
    if todo == "run":
        pse_workflow.run()
    else:
        if todo.find("1D") > -1:
            pse_workflow.load_PSE_1D(**kwargs)
        elif todo.find("2D") > -1:
            pse_workflow.load_PSE_2D(**kwargs)
        else:
            pse_workflow.load_PSE()
        if todo.find("plot") > -1:
            pse_workflow.plot_PSE()
    print(pse_workflow.PSE)
    PSE = deepcopy(pse_workflow.PSE)
    del pse_workflow
    return PSE


def two_nest_nodes_PSE(todo="run", wTVB=0.9, wNEST=1.3, branch="low", fast=False, output_base=None):
    return run_PSE(PSETVBNESTWorkflow, todo, branch=branch, fast=fast, output_base=output_base,
                   tvb_nodes=[], nest_nodes=[0, 1], wTVB=wTVB, wNEST=wNEST, SC=None)


def two_tvb_nodes_PSE(todo="run", wTVB=0.9, wNEST=1.3, branch="low", fast=False, output_base=None):
    return run_PSE(PSETVBNESTWorkflow, todo, branch=branch, fast=fast, output_base=output_base,
                   tvb_nodes=[0, 1], nest_nodes=[], wTVB=wTVB, wNEST=wNEST, SC=None)


def one_tvb_node_one_nest_node_PSE(todo="run", wTVB=0.9, wNEST=1.3, branch="low", fast=False, output_base=None):
    return run_PSE(PSETVBNESTWorkflow, todo, branch=branch, fast=fast, output_base=output_base,
                   tvb_nodes=[0], nest_nodes=[1], wTVB=wTVB, wNEST=wNEST, SC=None)


def two_tvb_nodes_one_nest_node_PSE(todo="run", wTVB=0.9, wNEST=1.3, branch="low", fast=False, output_base=None):
    return run_PSE(PSETVBNESTWorkflow, todo, branch=branch, fast=fast, output_base=output_base,
                   tvb_nodes=[0, 1], nest_nodes=[2], wTVB=wTVB, wNEST=wNEST, SC=[0.9, 0.5, 0.1])


def one_tvb_node_two_nest_nodes_PSE(todo="run", wTVB=0.9, wNEST=1.3, branch="low", fast=False, output_base=None):
    return run_PSE(PSETVBNESTWorkflow, todo, branch=branch, fast=fast, output_base=output_base,
                   tvb_nodes=[0], nest_nodes=[1, 2], wTVB=wTVB, wNEST=wNEST, SC=[0.9, 0.5, 0.1])


def plot_result(PSE_params, result, name, path):
    dims = ["branch"] + list(PSE_params.keys())
    coords = dict(PSE_params)
    coords["branch"] = ["low", "high"]
    arr = DataArray(data=np.array(result), dims=dims, coords=coords, name=name)
    fig, axes = pl.subplots(arr.shape[2], arr.shape[3], figsize=(5*arr.shape[2], 5*arr.shape[3]))
    axes = np.array(axes)
    while axes.ndim < 2:
        axes = np.expand_dims(axes, -1)
    for i_w1, w1 in enumerate(coords[dims[2]]):
        for i_w2, w2 in enumerate(coords[dims[2]]):
            lines = arr[:, :, i_w1, i_w2].plot.line(x=arr.dims[1], hue=arr.dims[0],
                                                    add_legend=False, ax=axes[i_w1, i_w2])
        for i_line, (style, branch) in enumerate(zip(["--", "-"], coords["branch"])):
            lines[i_line].set_color("k")
            lines[i_line].set_linestyle(style)
            # lines[i_line].set_label("%s, %s" % (w, branch))
    fig.canvas.draw()
    # pl.legend()
    pl.savefig(path)


def plot_results(PSElow, PSEhigh, name, results, pops, names, folder_figs=None):
    if folder_figs is None:
        folder_figs = CONFIGURED.out.FOLDER_RES.replace("res", name)
    safe_makedirs(folder_figs)
    for res, nam in zip(results, names):
        for pop in pops:
            try:
                this_name = "%s %s" % (pop, nam)
                path = os.path.join(folder_figs, this_name+".png").replace(" (spikes/sec)", "").replace(" ", "_")
                plot_result(PSElow["params"],
                            [PSElow["results"][res][pop],
                             PSEhigh["results"][res][pop]],
                            this_name, path)
            except:
                pass


if __name__ == "__main__":

    tic = time.time()

    output_base_base ='/Users/dionperd/Software/TVB/tvb-multiscale/tvb_nest/examples/paperwork/outputs/PSE1TVB1NESTnodesG'
    FAST = False

    # try:
    name = "PSE1TVB1NESTnodesG"
    output_base = os.path.join(output_base_base, name)
    PSElow = deepcopy(one_tvb_node_one_nest_node_PSE(todo="run", output_base=output_base, fast=FAST))
    PSEhigh = deepcopy(one_tvb_node_one_nest_node_PSE(todo="run", branch="high", output_base=output_base, fast=FAST))
    results = ["rate", "spike rate", "Pearson", "Spearman", "spike Pearson", "spike Spearman", "spike train"]
    pops = ["E", "I", "EE", "FC-SC"]
    names = ["Rate (spikes/sec)", "Spike rate (spikes/sec)",
             "Pearson Corr", "Spearman Corr", "Spike Pearson Corr", "Spike Spearman Corr", "Spike train Corr"]
    plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # except:
    #     pass

    # # try:
    # name = "PSE1TVB2NESTnodesG"
    # output_base = os.path.join(output_base_base, name)
    # PSElow = deepcopy(one_tvb_node_two_nest_nodes_PSE(todo="run", output_base=output_base, fast=FAST))
    # PSEhigh = deepcopy(one_tvb_node_two_nest_nodes_PSE(todo="run", branch="high", output_base=output_base, fast=FAST))
    # results = ["rate", "spike rate", "Pearson", "Spearman", "spike Pearson", "spike Spearman", "spike train"]
    # pops = ["E", "I", "EE", "FC-SC"]
    # names = ["Rate (spikes/sec)", "Spike rate (spikes/sec)",
    #          "Pearson Corr", "Spearman Corr", "Spike Pearson Corr", "Spike Spearman Corr", "Spike train Corr"]
    # plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # # except:
    # #     pass
    #
    # # try:
    # name = "PSE1TVB2NESTnodesG"
    # output_base = os.path.join(output_base_base, name)
    # PSElow = deepcopy(two_tvb_nodes_one_nest_node_PSE(todo="run", output_base=output_base, fast=FAST))
    # PSEhigh = deepcopy(two_tvb_nodes_one_nest_node_PSE(todo="run", branch="high", output_base=output_base, fast=FAST))
    # results = ["rate", "spike rate", "Pearson", "Spearman", "spike Pearson", "spike Spearman", "spike train"]
    # pops = ["E", "I", "EE", "FC-SC"]
    # names = ["Rate (spikes/sec)", "Spike rate (spikes/sec)",
    #          "Pearson Corr", "Spearman Corr", "Spike Pearson Corr", "Spike Spearman Corr", "Spike train Corr"]
    # plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # # except:
    # #     pass

    print_toc_message(tic)
