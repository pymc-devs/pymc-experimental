import pytensor.tensor as pt

from pymc_experimental.statespace.core.representation import (
    NEVER_TIME_VARYING,
    VECTOR_VALUED,
)
from pymc_experimental.statespace.utils.constants import JITTER_DEFAULT


def decide_if_x_time_varies(x, name):
    if name in NEVER_TIME_VARYING:
        return False

    ndim = x.ndim

    if name in VECTOR_VALUED:
        if ndim not in [1, 2]:
            raise ValueError(
                f"Vector {name} has {ndim} dimensions; it should have either 1 (static),"
                f" or 2 (time varying )"
            )

        return ndim == 2

    if ndim not in [2, 3]:
        raise ValueError(
            f"Matrix {name} has {ndim} dimensions; it should have either"
            f" 2 (static), or 3 (time varying)."
        )

    return ndim == 3


def split_vars_into_seq_and_nonseq(params, param_names):
    """
    Split inputs into those that are time varying and those that are not. This division is required by scan.
    """
    sequences, non_sequences = [], []
    seq_names, non_seq_names = [], []

    for param, name in zip(params, param_names):

        if decide_if_x_time_varies(param, name):
            sequences.append(param)
            seq_names.append(name)
        else:
            non_sequences.append(param)
            non_seq_names.append(name)

    return sequences, non_sequences, seq_names, non_seq_names


def stabilize(cov, jitter=None):
    if jitter is None:
        jitter = JITTER_DEFAULT

    # Ensure diagonal is non-zero
    cov += pt.identity_like(cov) * jitter

    # Ensure matrix is symmetric
    cov = 0.5 * (cov + cov.T)

    return cov
