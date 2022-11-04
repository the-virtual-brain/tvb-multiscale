import os
import shutil

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import CONFIGURED, Config
from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest
from tvb_multiscale.tvb_nest.nest_models.models.default import DefaultExcIOBuilder

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def test(dt=0.1, noise_strength=0.001, config=CONFIGURED):
    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = [0, 1]  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.integrator.dt = dt
    # simulator.integrator.noise.nsig = np.array([noise_strength])
    simulator.model = Linear()

    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # Build a NEST network model with the corresponding builder
    # Using all default parameters for this example
    nest_model_builder = DefaultExcIOBuilder(simulator, nest_nodes_ids,
                                             nest_instance=load_nest(config), config=config)
    nest_model_builder.configure()
    print(nest_model_builder.info())


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    test()
