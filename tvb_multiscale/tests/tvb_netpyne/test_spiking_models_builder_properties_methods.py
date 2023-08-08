from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_netpyne.config import CONFIGURED, Config
from tvb_multiscale.tvb_netpyne.netpyne_models.models.default_exc_io_inh_i import DefaultExcIOInhIBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.wilson_cowan import WilsonCowanBuilder

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb.simulator.monitors import Raw

def test( dt = 0.1, duration = 100, config=CONFIGURED ):
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()

    nodes = [0, 1] # the indices of fine scale regions modeled with NetPyNE

    simulator = Simulator()
    simulator.integrator.dt = dt
    simulator.model = Linear()

    simulator.connectivity = connectivity
    mon_raw = Raw(period = simulator.integrator.dt )
    simulator.monitors = (mon_raw,)

    netpyne_model_builder = DefaultExcIOInhIBuilder(simulator, nodes, config=config)
    netpyne_model_builder.configure()
    print(netpyne_model_builder.info())

if __name__ == "__main__":
    test()
