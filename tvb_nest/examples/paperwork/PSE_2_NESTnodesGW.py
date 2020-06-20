# -*- coding: utf-8 -*-
import sys
import time
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import two_nest_nodes_PSE
from tvb_nest.examples.paperwork.paperwork_pse_exc_io import print_toc_message


args = sys.argv
print(args)

tic = time.time()

two_nest_nodes_PSE(w=float(args[1]), branch=args[2])

print_toc_message()
