# -*- coding: utf-8 -*-
import sys
import time
from tvb_multiscale.examples.paperwork.paperwork_pse_exc_io import three_symmetric_mf_PSE
from tvb_multiscale.examples.paperwork.paperwork_pse_exc_io import print_toc_message


args = sys.argv

tic = time.time()

try:
    fast = args[3] == "fast"
except:
    fast = False

three_symmetric_mf_PSE(w=float(args[1]), branch=args[2], fast=fast)

print_toc_message(tic)
