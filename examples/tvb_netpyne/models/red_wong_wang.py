from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder
from tvb_multiscale.tvb_netpyne.interfaces.models.default import RedWongWangExcIOInhITVBNetpyneInterfaceBuilder

from examples.tvb_netpyne.example import main_example
from examples.models.red_wong_wang import red_wong_wang_excio_inhi_example as excio_inhi_example_base

def excio_inhi_example(**kwargs):
    params = {
        "simulation_length": 500,
        # "model_params": {"lamda": 0.5}
    }

    netpyne_network_builder = DefaultExcIOInhIBuilder()
    params["spiking_proxy_inds"] = [0, 1]

    kwargs.update(params)

    tvb_netpyne_interface_builder = RedWongWangExcIOInhITVBNetpyneInterfaceBuilder()
    return main_example(excio_inhi_example_base, netpyne_network_builder, tvb_netpyne_interface_builder, **kwargs)

if __name__ == "__main__":
    excio_inhi_example()
