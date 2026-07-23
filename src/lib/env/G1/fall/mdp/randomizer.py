from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg

from lib.domain_randomizer.randomizer import push_by_setting_velocity


def push_and_log(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply the standard push and record the realized root-velocity delta.

    Wraps :func:`lib.domain_randomizer.randomizer.push_by_setting_velocity` without
    changing its sampling or semantics, and measures the applied delta by diffing the
    root velocity around the call. The delta is exposed through ``env.extras`` so that
    a collection script can read which disturbance each environment received.

    Only the first push of an episode is applied: one environment carries exactly one
    disturbance condition. The guard is structural so that it does not depend on the
    interval/episode-length arithmetic staying in sync.

    Requires ``env.extras["disturbance"]`` [num_envs, 6] and
    ``env.extras["disturbance_applied"]`` [num_envs] to exist.
    """
    env_ids = env_ids[~env.extras["disturbance_applied"][env_ids]]
    if env_ids.numel() == 0:
        return

    asset = env.scene[asset_cfg.name]

    before = asset.data.root_vel_w[env_ids].clone()
    # sample random velocities in body frame
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    delta_b = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], before.shape, device=asset.device)

    # frame transformation (Body -> World)
    root_quat = asset.data.root_quat_w[env_ids]
    delta_w_lin = math_utils.quat_apply(root_quat, delta_b[:, :3])
    delta_w_ang = math_utils.quat_apply(root_quat, delta_b[:, 3:])

    # set the velocities into the physics simulation
    vel_w = before + torch.cat([delta_w_lin, delta_w_ang], dim=-1)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    env.extras["disturbance"][env_ids] = delta_b
    env.extras["disturbance_applied"][env_ids] = True
