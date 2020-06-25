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


def single_mf_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_1_TVBmfNodeStW
    return run_PSE(PSE_1_TVBmfNodeStW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def two_symmetric_mf_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_2_TVBmfNodesGW
    return run_PSE(PSE_2_TVBmfNodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def three_symmetric_mf_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_mean_field import PSE_3_TVBmfNodesGW
    return run_PSE(PSE_3_TVBmfNodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def single_spiking_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_1_TVBspikingNodeStW
    return run_PSE(PSE_1_TVBspikingNodeStW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def two_symmetric_spiking_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_2_TVBspikingNodesGW
    return run_PSE(PSE_2_TVBspikingNodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def three_symmetric_spiking_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_multiscale.examples.paperwork.pse_workflow_spiking import PSE_3_TVBspikingNodesGW
    return run_PSE(PSE_3_TVBspikingNodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def plot_result(PSE_params, result, name, path):
    dims = ["branch"] + list(PSE_params.keys())
    coords = dict(PSE_params)
    coords["branch"] = ["low", "high"]
    arr = DataArray(data=np.array(result), dims=dims, coords=coords, name=name)
    fig = pl.figure(figsize=(10, 10))
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

    output_base_base = "/Users/dionperd/Software/TVB/tvb-multiscale/tvb_multiscale/examples/paperwork/outputs/PSE_TVBmfNodes2"
    FAST = False

    # try:
    name = "PSE_1_TVBmfNodeStW"
    output_base = os.path.join(output_base_base, name)
    PSElow = deepcopy(single_mf_PSE(todo="plot1D", output_base=output_base, fast=FAST))
    PSEhigh = deepcopy(single_mf_PSE(todo="plot1D", branch="high", output_base=output_base, fast=FAST))
    results = ["rate"]
    pops = ["E", "I"]
    names = ["Rate (spikes/sec)"]
    plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # except:
    #     pass

    # # try:
    # name = "PSE_2_TVBmfNodesGW"
    # output_base = os.path.join(output_base_base, name)
    # PSElow = deepcopy(two_symmetric_mf_PSE(todo="plot1D", output_base=output_base, fast=FAST))
    # PSEhigh = deepcopy(two_symmetric_mf_PSE(todo="plot1D", branch="high", output_base=output_base, fast=FAST))
    # results = ["rate", "Pearson", "Spearman"]
    # pops = ["E", "I", "EE"]
    # names = ["Rate (spikes/sec)", "Pearson Corr", "Spearman Corr"]
    # plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # # except:
    # #     pass

    # # try:
    # name = "PSE_3_TVBmfNodesGW"
    # output_base = os.path.join(output_base_base, name)
    # PSElow = deepcopy(three_symmetric_mf_PSE(todo="plot1D", output_base=output_base, fast=FAST))
    # PSEhigh = deepcopy(three_symmetric_mf_PSE(todo="plot1D", branch="high", output_base=output_base, fast=FAST))
    # results = ["rate", "Pearson", "Spearman"]
    # pops = ["E", "I", "EE", "FC-SC"]
    # names = ["Rate (spikes/sec)", "Pearson Corr", "Spearman Corr"]
    # plot_results(PSElow, PSEhigh, name, results, pops, names, output_base)
    # # except:
    # #    pass

    print_toc_message(tic)

