ALL_STATE_DIM = "state"
ALL_STATE_AUX_DIM = "state_aux"
OBS_STATE_DIM = "observed_state"
OBS_STATE_AUX_DIM = "observed_state_aux"
SHOCK_DIM = "shock"
SHOCK_AUX_DIM = "shock_aux"
TIME_DIM = "time"
EXTENDED_TIME_DIM = "extended_time"
AR_PARAM_DIM = "ar_lag"
MA_PARAM_DIM = "ma_lag"

MISSING_FILL = -9999.0

MATRIX_NAMES = ["x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]

SHORT_NAME_TO_LONG = {
    "x0": "initial_state",
    "P0": "initial_state_cov",
    "c": "state_intercept",
    "d": "obs_intercept",
    "T": "transition",
    "Z": "design",
    "R": "selection",
    "H": "obs_cov",
    "Q": "state_cov",
}

FILTER_OUTPUT_NAMES = [
    "filtered_state",
    "predicted_state",
    "filtered_covariance",
    "predicted_covariance",
]

SMOOTHER_OUTPUT_NAMES = ["smoothed_state", "smoothed_covariance"]
OBSERVED_OUTPUT_NAMES = ["observed_state", "observed_covariance"]

MATRIX_DIMS = {
    "x0": (ALL_STATE_DIM,),
    "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "c": (ALL_STATE_DIM,),
    "d": (OBS_STATE_DIM,),
    "T": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "Z": (OBS_STATE_DIM, ALL_STATE_DIM),
    "R": (ALL_STATE_DIM, SHOCK_DIM),
    "H": (OBS_STATE_DIM, OBS_STATE_AUX_DIM),
    "Q": (SHOCK_DIM, SHOCK_AUX_DIM),
}

FILTER_OUTPUT_DIMS = {
    "filtered_state": (TIME_DIM, ALL_STATE_DIM),
    "smoothed_state": (TIME_DIM, ALL_STATE_DIM),
    "predicted_state": (EXTENDED_TIME_DIM, ALL_STATE_DIM),
    "filtered_covariance": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "smoothed_covariance": (TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "predicted_covariance": (EXTENDED_TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM),
    "obs": (TIME_DIM, OBS_STATE_DIM),
}
