# -*- coding: utf-8 -*-

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_script import *
from examples.tvb_nest.notebooks.cerebellum.scripts.nest_script import *
from examples.tvb_nest.notebooks.cerebellum.scripts.sbi_script import *
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb_nest_script import *


if __name__ == "__main__":
    import sys

    # samples_fit_Gs, results, fig, simulator, output_config = sbi_fit(int(sys.argv[-1]))
    config = configure()[0]
    if len(sys.argv) == 1:
        iB = int(sys.argv[-1])
        iG = None
    else:
        iB = int(sys.argv[-1])
        iG = int(sys.argv[-2])

    sim_res = simulate_TVB_for_sbi_batch(iB, iG, config=config, write_to_file=True)
