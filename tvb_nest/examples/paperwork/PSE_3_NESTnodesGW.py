# -*- coding: utf-8 -*-
import sys
import time
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import three_nest_nodes_PSE
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import print_toc_message


args = sys.argv

tic = time.time()

try:
    fast = args[3] == "True"
except:
    fast = False

three_nest_nodes_PSE(w=float(args[1]), branch=args[2], fast=fast)

print_toc_message(tic)
