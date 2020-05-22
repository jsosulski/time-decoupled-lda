import numpy as np


def get_diagonal_blocks_as_matrix(in_mat, n_channels, n_times):
    out_mat = np.zeros_like(in_mat)
    for i in range(n_times):
        idx_start = i * n_channels
        idx_end = idx_start + n_channels
        out_mat[idx_start:idx_end, idx_start:idx_end] = np.copy(in_mat[idx_start:idx_end, idx_start:idx_end])
    return out_mat


def get_diagonal_blocks_as_list(in_mat, n_channels, n_times):
    out_list = list()
    for i in range(n_times):
        idx_start = i * n_channels
        idx_end = idx_start + n_channels
        out_list.append(np.copy(in_mat[idx_start:idx_end, idx_start:idx_end]))
    return out_list


def get_offdiagonal_blocks_as_matrix(in_mat, n_channels, n_times):
    out_mat = np.copy(in_mat)
    for i in range(n_times):
        idx_start = i * n_channels
        idx_end = idx_start + n_channels
        out_mat[idx_start:idx_end, idx_start:idx_end] = np.zeros((n_channels, n_channels))
    return out_mat


def get_offdiagonal_blocks_as_list(in_mat, n_channels, n_times):
    raise NotImplementedError("This function is non-trivial to implement.")


def get_row_as_list(in_mat, n_channels, n_times, row_index):
    out_list = list()
    out_mat_main = None
    for i in range(n_times):
        idx_start = i * n_channels
        idx_end = idx_start + n_channels
        row_start = row_index * n_channels
        row_end = row_start + n_channels
        if i == row_index:
            out_mat_main = np.copy(in_mat[idx_start:idx_end, idx_start:idx_end])
        out_list.append(np.copy(in_mat[row_start:row_end, idx_start:idx_end]))
    return out_list, out_mat_main


def get_arbitrary_block(in_mat, n_channels, row_index, col_index):
    row_start = row_index * n_channels
    row_end = row_start + n_channels
    col_start = col_index * n_channels
    col_end = col_start + n_channels
    return np.copy(in_mat[col_start:col_end, row_start:row_end])
