# -*- coding: utf-8 -*-
import sys
import time
from tvb_nest.examples.paperwork.paperwork_pse_cosim_exc_io import one_tvb_node_two_nest_nodes_PSE
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message

args = sys.argv

tic = time.time()

try:
    fast = args[5] == "fast"
except:
    fast = False

print("fast=%s" % str(fast))

# Run PSE for default St values:
one_tvb_node_two_nest_nodes_PSE(wTVB=float(args[1]), wNEST=float(args[2]), branch=args[3],
                                fast=fast, output_base=args[4])

print_toc_message(tic)
