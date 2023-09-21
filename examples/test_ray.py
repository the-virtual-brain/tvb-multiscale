# -*- coding: utf-8 -*-

import ray
from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.server_client.ray import RayNESTServer
from tvb_multiscale.tvb_nest.nest_models.server_client.ray import RayNESTClient


if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)
    RayNESTServer = ray.remote(RayNESTServer)
    nest_server = \
            RayNESTServer.options(name="nest_server",
                                  num_cpus=CONFIGURED.DEFAULT_LOCAL_NUM_THREADS).remote(config=CONFIGURED)
    # except:
    #     nest_server = ray.get_actor("nest_server")
    # nest = RayNESTClient(nest_server)

    remoteRayNESTClient = ray.remote(RayNESTClient).remote(nest_server)

    gids = ray.get(remoteRayNESTClient.Create.remote("iaf_cond_alpha", 1))
