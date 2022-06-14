from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import Config, initialize_logger
from tvb_multiscale.tvb_netpyne.orchestrators import TVBNetpyneSerialOrchestrator
from tvb_multiscale.tvb_netpyne.interfaces.models.default import RedWongWangExcIOInhITVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

from examples.example import default_example as default_example_base

def main_example(model_fun_to_run, netpyne_model_builder, tvb_netpyne_interface_builder, **kwargs):

    some_args = {"population_order": 100,
                 "spiking_proxy_inds": [60, 61], # superiortemporal L and R
                 "simulation_length": 220}
    kwargs.update(some_args)

    return model_fun_to_run(netpyne_model_builder, tvb_netpyne_interface_builder, TVBNetpyneSerialOrchestrator,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):
    return main_example(default_example_base, DefaultExcIOInhIBuilder(), RedWongWangExcIOInhITVBNetpyneInterfaceBuilder(), **kwargs)


if __name__ == "__main__":
    default_example()