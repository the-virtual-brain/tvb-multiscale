# -*- coding: utf-8 -*-
import time
import os
from copy import deepcopy

import numpy as np
from xarray import DataArray

from matplotlib import pyplot as pl
from matplotlib import cm

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.config import CONFIGURED
from tvb_utils.utils import print_toc_message


def run_PSE(pse_class, todo="run", **kwargs):
    pse_workflow = pse_class(**kwargs)
    if todo == "run":
        pse_workflow.run()
    else:
        pse_workflow.load_PSE()
        if todo == "plot":
            pse_workflow.plot_PSE()
    print(pse_workflow.PSE)
    return pse_workflow.PSE


def single_mf_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_1_TVBmfNodeStW
    return run_PSE(PSE_1_TVBmfNodeStW, todo, w=w, branch=branch, fast=fast)


def two_symmetric_mf_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_2_TVBmfNodesGW
    return run_PSE(PSE_2_TVBmfNodesGW, todo, w=w, branch=branch, fast=fast)


def three_symmetric_mf_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_3_TVBmfNodesGW
    return run_PSE(PSE_3_TVBmfNodesGW, todo, w=w, branch=branch, fast=fast)


def single_spiking_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_1_TVBspikingNodeStW
    return run_PSE(PSE_1_TVBspikingNodeStW, todo, w=w, branch=branch, fast=fast)


def two_symmetric_spiking_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_2_TVBspikingNodesGW
    return run_PSE(PSE_2_TVBspikingNodesGW, todo, w=w, branch=branch, fast=fast)


def three_symmetric_spiking_PSE(todo="run", w=None, branch="low", fast=False):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_3_TVBspikingNodesGW
    return run_PSE(PSE_3_TVBspikingNodesGW, todo, w=w, branch=branch, fast=fast)


def plot_result(PSE_params, result, name, path):
    dims = ["branch"] + list(PSE_params.keys())
    coords = dict(PSE_params)
    coords["branch"] = ["low", "high"]
    arr = DataArray(data=np.array(result), dims=dims, coords=coords, name=name)
    fig = pl.figure()
    arr = arr.stack({"%sbranch" % dims[2]: [dims[2], "branch"]})
    lines = arr.plot.line(x=arr.dims[0], hue=arr.dims[1], add_legend=False)
    cmap = cm.get_cmap('jet')
    nw = np.maximum(1.0, np.ceil(arr.shape[1]/2) - 1.0)
    i_line = 0
    for i_color, w in enumerate(coords[dims[2]]):
        color = cmap(i_color/nw)
        for style, branch in zip(["--", "-"], coords["branch"]):
            lines[i_line].set_color(color)
            lines[i_line].set_linestyle(style)
            lines[i_line].set_label("%s, %s" % (w, branch))
            i_line += 1
    fig.canvas.draw()
    pl.legend()
    pl.savefig(path)


def plot_results(name, results, pops, names):
    folder_figs = CONFIGURED.out.FOLDER_RES.replace("res", name)
    if not os.path.isdir(folder_figs):
        os.makedirs(folder_figs)
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
    # PSElow = deepcopy(single_mf_PSE())
    # PSEhigh = deepcopy(single_mf_PSE(branch="high"))
    # PSElow = deepcopy(two_symmetric_mf_PSE())
    # PSEhigh = deepcopy(two_symmetric_mf_PSE(branch="high"))
    # PSElow = deepcopy(three_symmetric_mf_PSE())
    # PSEhigh = deepcopy(three_symmetric_mf_PSE(branch="high"))

    # try:
    # PSElow = deepcopy(single_mf_PSE(w=0.9, fast=True))
    # PSEhigh = deepcopy(single_mf_PSE(w=0.9, branch="high", fast=True))
    #
    # name = "PSE_1_TVBmfNodeStW"
    # results = ["rate", "Pearson", "Spearman"]
    # pops = ["E", "I", "EE", "FC-SC"]
    # names = ["Rate (spikes/sec)", "Pearson Corr", "Spearman Corr"]
    # plot_results(name, results, pops, names)
#
# except:
#     pass

# try:
    PSElow = deepcopy(two_symmetric_mf_PSE(w=0.9, fast=True))
    PSEhigh = deepcopy(two_symmetric_mf_PSE(w=0.9, branch="high", fast=True))
    name = "PSE_2_TVBmfNodesGW"
    results = ["rate", "Pearson", "Spearman"]
    pops = ["E", "I", "EE", "FC-SC"]
    names = ["Rate (spikes/sec)", "Pearson Corr", "Spearman Corr"]
    plot_results(name, results, pops, names)
    #
    # except:
    #     pass

    # try:
    PSElow = deepcopy(three_symmetric_mf_PSE(w=0.9, fast=True))
    PSEhigh = deepcopy(three_symmetric_mf_PSE(w=0.9, branch="high", fast=True))

    name = "PSE_3_TVBmfNodesGW"
    results = ["rate", "Pearson", "Spearman"]
    pops = ["E", "I", "EE", "FC-SC"]
    names = ["Rate (spikes/sec)", "Pearson Corr", "Spearman Corr"]
    plot_results(name, results, pops, names)
    # except:
    #     pass

    print_toc_message(tic)

