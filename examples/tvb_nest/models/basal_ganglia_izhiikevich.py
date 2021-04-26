# -*- coding: utf-8 -*-

from examples.tvb_nest.example import main_example
from examples.models.basal_ganglia_izhiikevich import \
    basal_ganglia_izhikevich_example as basal_ganglia_izhikevich_example_base

from tvb_multiscale.tvb_nest.nest_models.models.basal_ganglia_izhikevich import BasalGangliaIzhikevichBuilder
from tvb_multiscale.tvb_nest.interfaces.models.builders.red_ww_basal_ganglia_izhikevich \
    import RedWWexcIOBuilder as BasalGangliaIzhikevichTVBNESTInterfaceBuilder


def basal_ganglia_izhikevich_example(**kwargs):
    return main_example(basal_ganglia_izhikevich_example_base,
                        BasalGangliaIzhikevichBuilder, BasalGangliaIzhikevichTVBNESTInterfaceBuilder,
                        **kwargs)


if __name__ == "__main__":
    basal_ganglia_izhikevich_example()
