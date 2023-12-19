"""
The implementation of so3_exp_map and so3_log_map is sourced from the
pytorch3d.transforms module, which is a part of the PyTorch3D library.
For in-depth documentation and additional information, you can refer to
the PyTorch3D documentation at the following URL:
https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html#so3_exp_map
"""


from typing import Tuple
import numpy as np
import torch
import math
from scipy.interpolate import CubicSpline

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    return (-1.0) / math.sqrt(1.0 - x * x)


def hat(v: torch.Tensor) -> torch.Tensor:
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def hat_inv(h: torch.Tensor) -> torch.Tensor:
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    return _so3_exp_map(log_rot, eps=eps)[0]


def so3_log_map(
    R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> torch.Tensor:
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot


def slerp(axisangle_left, axisangle_right, t):
    """Spherical linear interpolation."""
    # https://en.wikipedia.org/wiki/Slerp
    # t: (time - timeleft / (timeright - timeleft)) (0, 1)
    assert (
        axisangle_left.shape == axisangle_right.shape
    ), "axisangle_left and axisangle_right must have the same shape"
    assert (
        axisangle_left.shape[-1] == 3
    ), "axisangle_left and axisangle_right must be axis-angle representations"
    assert (
        t.shape[:-1] == axisangle_left.shape[:-1]
    ), "t must have the same shape as axisangle_left and axisangle_right"

    main_shape = axisangle_left.shape[:-1]
    axisangle_left = axisangle_left.reshape(-1, 3)
    axisangle_right = axisangle_right.reshape(-1, 3)
    t = t.reshape(-1, 1)
    delta_rotation = so3_exp_map(
        so3_log_map(so3_exp_map(-axisangle_left) @ so3_exp_map(axisangle_right)) * t
    )

    return so3_log_map(so3_exp_map(axisangle_left) @ delta_rotation).reshape(
        *main_shape, 3
    )


def slerp_interpolate(
    frames: dict, 
) -> torch.Tensor:
    """
    Interpolate between two axis-angle representations.
    """
    frames = dict(sorted(frames.items(), key=lambda x: x[0]))
    timepoints = list(frames.keys())
    min_t, max_t = min(timepoints), max(timepoints)
    if len(timepoints) == max_t + 1 - min_t:
        return frames
    
    timeline = list(range(min_t, max_t + 1))

    timepoints_fill, time_ratio = [], []
    axisangle_left, axisangle_right = [], []
    for t in timeline:
        if t not in timepoints:
            timepoints_fill.append(t)
            time_ratio.append((t - left_t)/(right_t - left_t + 1e-5))
            axisangle_left.append(frames[left_t])
            axisangle_right.append(frames[right_t])
        else:
            left_t = t
            left_t_idx = timepoints.index(left_t)
            right_t_idx = min([left_t_idx + 1, len(timepoints)-1])
            right_t = timepoints[right_t_idx]

    time_ratio = torch.tensor(time_ratio)
    axisangle_left = torch.cat(axisangle_left, dim=0)
    axisangle_right = torch.cat(axisangle_right, dim=0)

    shape = axisangle_left.shape
    slerp_shape = (*shape[:-1], shape[-1]//3, 3)
    axisangle_left = axisangle_left.reshape(slerp_shape)
    axisangle_right = axisangle_right.reshape(slerp_shape)
    time_ratio = time_ratio[...,  None, None].expand(*axisangle_left.shape[:-1], 1)
    axisangle_fill = slerp(axisangle_left, axisangle_right, time_ratio)
    axisangle_fill = axisangle_fill.reshape(shape).numpy()

    axisangle_fill = np.split(axisangle_fill, axisangle_fill.shape[0], axis=0)

    frames.update(zip(timepoints_fill, axisangle_fill))
    frames = dict(sorted(frames.items(), key=lambda x: x[0]))
    return frames

def cubic_interpolation(frames):
    frames = dict(sorted(frames.items(), key=lambda x: x[0]))
    timepoints = list(frames.keys())
    min_t, max_t = min(timepoints), max(timepoints)
    if len(timepoints) == max_t + 1 - min_t:
        return frames
    
    motion = torch.cat(list(frames.values()), dim=0)
    cs = CubicSpline(np.array(timepoints), motion)

    timeline = list(range(min(timepoints), max(timepoints) + 1))
    new_motion = [m[None, ...] for m in cs(timeline)]
    frames = dict(zip(timeline, new_motion))
    frames = dict(sorted(frames.items(), key=lambda x: x[0]))
    return frames
