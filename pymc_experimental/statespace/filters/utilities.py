def split_vars_into_seq_and_nonseq(params, param_names):
    """
    Split inputs into those that are time varying and those that are not. This division is required by scan.
    """
    sequences, non_sequences = [], []
    seq_names, non_seq_names = [], []

    for param, name in zip(params, param_names):
        if param.ndim == 2:
            non_sequences.append(param)
            non_seq_names.append(name)
        elif param.ndim == 3:
            sequences.append(param)
            seq_names.append(name)
        else:
            raise ValueError(
                f"Matrix {name} has {param.ndim}, it should either 2 (static) or 3 (time varying)."
            )

    return sequences, non_sequences, seq_names, non_seq_names
