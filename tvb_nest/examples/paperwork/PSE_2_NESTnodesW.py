# -*- coding: utf-8 -*-
import sys
import time
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import three_nest_nodes_PSE
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message

args = sys.argv

tic = time.time()

try:
    fast = args[4] == "fast"
except:
    fast = False

print("fast=%s" % str(fast))

# Run PSE for default G values:
three_nest_nodes_PSE(w=float(args[1]), branch=args[2], fast=fast, output_base=args[3])

print_toc_message(tic)
