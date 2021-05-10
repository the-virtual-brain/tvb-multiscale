# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import compile_modules


if __name__ == "__main__":
    # import sys
    #
    # if sys.argv[-1] == "1":
    #     compile_modules("izhikevich_hamker", recompile=True)
    # elif sys.argv[-1] == "2":
    #     compile_modules("cereb", recompile=True)
    # else:
    #     compile_modules("iaf_cond_ww_deco", recompile=True)
    compile_modules("cereb", recompile=True)

