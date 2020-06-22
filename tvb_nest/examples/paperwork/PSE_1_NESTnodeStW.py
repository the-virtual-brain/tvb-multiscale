# -*- coding: utf-8 -*-
import sys
import time
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import single_nest_PSE
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import print_toc_message


args = sys.argv

tic = time.time()

try:
    fast = args[3] == "fast"
except:
    fast = False

single_nest_PSE(w=float(args[1]), branch=args[2], fast=fast)

print_toc_message(tic)
