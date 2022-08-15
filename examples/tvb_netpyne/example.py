from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import Config, initialize_logger
from tvb_multiscale.tvb_netpyne.orchestrators import TVBNetpyneSerialOrchestrator
from tvb_multiscale.tvb_netpyne.interfaces.models.default import RedWongWangExcIOInhITVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder

def main_example(model_fun_to_run, netpyne_model_builder, tvb_netpyne_interface_builder, **kwargs):

    some_args = {"population_order": 100,
                 "spiking_proxy_inds": [60, 61], # superiortemporal L and R
                 "simulation_length": 220}
    kwargs.update(some_args)

    return model_fun_to_run(netpyne_model_builder, tvb_netpyne_interface_builder, TVBNetpyneSerialOrchestrator,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):
    print("Using Reduced Wong-Wang ExcIO InhI as a default model")
    from examples.models.red_wong_wang import red_wong_wang_excio_inhi_example as base_model
    return main_example(base_model, DefaultExcIOInhIBuilder(), RedWongWangExcIOInhITVBNetpyneInterfaceBuilder(), **kwargs)


if __name__ == "__main__":
    default_example()