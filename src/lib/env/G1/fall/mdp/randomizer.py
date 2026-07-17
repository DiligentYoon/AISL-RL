from __future__ import annotations

import torch

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
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    delta = asset.data.root_vel_w[env_ids] - before

    env.extras["disturbance"][env_ids] = delta
    env.extras["disturbance_applied"][env_ids] = True
