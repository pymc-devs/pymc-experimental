def get_slice_and_move_cursor(cursor, param_count, last_slice=False):
    param_slice = slice(cursor, None if last_slice else cursor + param_count)
    cursor += param_count

    return param_slice, cursor
