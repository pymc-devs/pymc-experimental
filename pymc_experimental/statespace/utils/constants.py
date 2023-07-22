ALL_STATE_DIM = "states"
ALL_STATE_AUX_DIM = "states_aux"
OBS_STATE_DIM = "observed_states"
OBS_STATE_AUX_DIM = "observed_states_aux"
SHOCK_DIM = "shocks"
TIME_DIM = "time"
EXTENDED_TIME_DIM = "extended_time"


MATRIX_NAMES = ["x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
FILTER_OUTPUT_NAMES = [
    "filtered_states",
    "predicted_states",
    "filtered_covariances",
    "predicted_covariances",
]

SMOOTHER_OUTPUT_NAMES = ["smoothed_states", "smoothed_covariances"]
OBSERVED_OUTPUT_NAMES = ["observed_states", "observed_covariances"]

MATRIX_DIMS = {
    "x0": (ALL_STATE_DIM,),
    "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "c": (ALL_STATE_DIM,),
    "d": (OBS_STATE_DIM,),
    "T": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "Z": (OBS_STATE_DIM, ALL_STATE_DIM),
    "R": (ALL_STATE_DIM, SHOCK_DIM),
    "H": (OBS_STATE_DIM, OBS_STATE_AUX_DIM),
    "Q": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
}

FILTER_OUTPUT_DIMS = {
    "filtered_states": (TIME_DIM, ALL_STATE_DIM),
    "smoothed_states": (TIME_DIM, ALL_STATE_DIM),
    "predicted_states": (EXTENDED_TIME_DIM, ALL_STATE_DIM),
    "observed_states": (TIME_DIM, OBS_STATE_DIM),
    "filtered_covariances": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "smoothed_covariances": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_covariances": (EXTENDED_TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "observed_covariances": (TIME_DIM, OBS_STATE_DIM, OBS_STATE_AUX_DIM),
    "obs": (TIME_DIM, OBS_STATE_DIM),
}
