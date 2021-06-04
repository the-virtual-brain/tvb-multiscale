# -*- coding: utf-8 -*-


from tvb_multiscale.tvb_nest.nest_models.models.cereb import CerebBuilder
from tvb_multiscale.tvb_nest.interfaces.models.cereb import CerebTVBNESTInterfaceBuilder

from examples.tvb_nest.example import main_example
from examples.models.cereb import \
    cereb_example as cereb_example_base


def cereb_example(**kwargs):
    return main_example(cereb_example_base,
                        CerebBuilder(), CerebTVBNESTInterfaceBuilder(),
                        **kwargs)


if __name__ == "__main__":
    cereb_example()
