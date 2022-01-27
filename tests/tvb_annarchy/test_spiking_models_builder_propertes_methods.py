import os
import shutil

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, Config
from tvb_multiscale.tvb_annarchy.annarchy_models.models.default import DefaultExcIOBuilder

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb_multiscale.core.tvb.cosimulator.models.linear import Linear
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def test(dt=0.1, noise_strength=0.001, config=CONFIGURED):
    # Select the regions for the fine scale modeling with ANNarchy spiking networks
    anarchy_nodes_ids = list(range(10))  # the indices of fine scale regions modeled with ANNarchy
    # In this example, we model parahippocampal cortices (left and right) with ANNarchy
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

    # Build a ANNarchy network model with the corresponding builder
    # Using all default parameters for this example
    anarchy_model_builder = DefaultExcIOBuilder(simulator, anarchy_nodes_ids, config=config)
    anarchy_model_builder.configure()
    print(anarchy_model_builder.info())


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    test()
