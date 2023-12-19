import numpy as np
# from numba import njit
from scipy.fft import dct, idct


EPS = 1e-6

# @njit
def unpack_axis_angle(axis_angle):
    angle = np.sqrt(np.sum(axis_angle**2))
    axis = axis_angle / (angle + EPS)
    return axis, angle

# @njit
def angle_continuity(angle, last_angle):
    angle, last_angle = angle / np.pi, last_angle/ np.pi
    diff_angle = angle - last_angle
    diff_angle = min([diff_angle % 2, diff_angle % 2 -2], key=lambda x: np.abs(x))
    angle = (last_angle + diff_angle) * np.pi
    return angle

# @njit
def axis_angle_continuity(axis_angle):
    # axis_angle: (seq_len, 3)
    last_axis, last_angle = unpack_axis_angle(axis_angle[0])
    for i in range(1, axis_angle.shape[0]):
        axis, angle = unpack_axis_angle(axis_angle[i])
        is_right_direction = np.sum((axis - last_axis)**2) < np.sum((-axis - last_axis)**2)
        if is_right_direction:
            axis = axis
            angle = angle
        else:
            axis = -axis
            angle = -angle
        angle = angle_continuity(angle, last_angle)
        axis_angle[i] = axis * angle
        last_axis, last_angle = axis, angle
    return axis_angle

def filter_power_top(arr, ratio=0.9):
    # arr: (batch_len, seq_len)
    batch_len, seq_len = arr.shape

    power_arr = np.abs(arr) ** 2
    idx_sorted = np.argsort(-power_arr, axis=-1)
    power_arr_sorted = np.take_along_axis(power_arr, idx_sorted, axis=-1)
    cumulative_sum = np.cumsum(power_arr_sorted, axis=-1)
    total_sum = np.sum(power_arr, axis=-1, keepdims=True)
    threshold = (1 - ratio) * total_sum

    num_idx = np.sum((cumulative_sum >= threshold), axis=-1, keepdims=True)
    filter_mask = (np.ones_like(arr, dtype=np.int64).cumsum(axis=-1) >= num_idx)
    filter_mask = np.flip(filter_mask, axis=[-1])
    idx_sorted = idx_sorted + (np.arange(batch_len) * seq_len).reshape(arr.shape[:-1])[..., None]
    filter_idx = idx_sorted[filter_mask]
    arr = arr.flatten()
    arr[filter_idx.flatten()] = 0

    arr = arr.reshape(batch_len, seq_len)
    return arr

def denoise_curves(curves, ratio=0.9):
    # curves: (batch_len, seq_len)
    assert curves.shape[1] > 1, "curves should be at least 2 points"
    curves_dct = dct(curves, axis=-1)
    curves_dct[:, 1:] = filter_power_top(curves_dct[:, 1:], ratio)
    curves = idct(curves_dct, axis=-1)
    return curves
