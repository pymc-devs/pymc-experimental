from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)


def make_default_coords(ss_mod):
    coords = {
        ALL_STATE_DIM: ss_mod.state_names,
        ALL_STATE_AUX_DIM: ss_mod.state_names,
        OBS_STATE_DIM: ss_mod.observed_states,
        OBS_STATE_AUX_DIM: ss_mod.observed_states,
        SHOCK_DIM: ss_mod.shock_names,
        SHOCK_AUX_DIM: ss_mod.shock_names,
    }

    return coords
