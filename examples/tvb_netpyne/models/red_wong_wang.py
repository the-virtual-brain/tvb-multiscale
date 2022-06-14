# from tvb_multiscale.tvb_nest.interfaces.models.red_wong_wang import \
    # RedWongWangExcIOTVBNESTInterfaceBuilder, RedWongWangExcIOInhITVBNESTInterfaceBuilder
# from tvb_multiscale.tvb_nest.nest_models.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder
from tvb_multiscale.tvb_netpyne.interfaces.models.default import RedWongWangExcIOInhITVBNetpyneInterfaceBuilder

from examples.tvb_netpyne.example import main_example
from examples.models.red_wong_wang import red_wong_wang_excio_inhi_example as excio_inhi_example_base

def excio_inhi_example(**kwargs):
    # model_params = {"model_params": {"lamda": 0.5}}
    # kwargs.update(model_params)

    tvb_netpyne_model = DefaultExcIOInhIBuilder()
    tvb_netpyne_interface_builder = RedWongWangExcIOInhITVBNetpyneInterfaceBuilder()
    return main_example(excio_inhi_example_base, tvb_netpyne_model, tvb_netpyne_interface_builder, **kwargs)

if __name__ == "__main__":
    excio_inhi_example()
