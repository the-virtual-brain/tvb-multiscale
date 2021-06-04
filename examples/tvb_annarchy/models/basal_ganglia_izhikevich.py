# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.annarchy_models.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_annarchy.interfaces.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder

from examples.tvb_annarchy.example import main_example
from examples.models.basal_ganglia_izhiikevich import \
    basal_ganglia_izhikevich_example as basal_ganglia_izhikevich_example_base


def basal_ganglia_izhikevich_example(**kwargs):
    return main_example(basal_ganglia_izhikevich_example_base,
                        BasalGangliaIzhikevichBuilder(), BasalGangliaIzhikevichTVBANNarchyInterfaceBuilder(),
                        **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "2":
        basal_ganglia_izhikevich_example(model="CURRENT")
    elif sys.argv[-1] == "1":
        basal_ganglia_izhikevich_example(model="SPIKES")
    else:
        basal_ganglia_izhikevich_example(model="RATE")
