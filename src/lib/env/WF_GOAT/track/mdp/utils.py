"""Utility functions for the GOAT joint tracking task.

Contains a lightweight, batched forward-kinematics (FK) helper that maps a leg joint
configuration (hip, thigh, knee) to the wheel-center position expressed in the base frame.

All constants are transcribed directly from the robot description URDF
``src/lib/assets/robots/GOAT/WF_GOAT/urdf/WF_GOAT.urdf``. Every leg joint origin in that
URDF has ``rpy="0 0 0"`` (no fixed rotation offset) and a principal rotation axis, so the
FK reduces to a chain of pure axis-angle rotations with constant translation offsets.
"""

from __future__ import annotations

import torch


def axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix about a fixed unit axis (Rodrigues' formula).

    Args:
        axis: Unit rotation axis. Shape (3,).
        angle: Rotation angle(s) in radians. Shape (...,).

    Returns:
        Rotation matrices. Shape (..., 3, 3).
    """
    kx, ky, kz = axis[0], axis[1], axis[2]
    zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
    # skew-symmetric matrix K of the unit axis (constant for a fixed axis)
    K = torch.stack([
        torch.stack([zero, -kz, ky]),
        torch.stack([kz, zero, -kx]),
        torch.stack([-ky, kx, zero]),
    ])  # (3, 3)
    K2 = K @ K  # (3, 3)

    sin = torch.sin(angle)[..., None, None]   # (..., 1, 1)
    cos = torch.cos(angle)[..., None, None]   # (..., 1, 1)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)

    return eye + sin * K + (1.0 - cos) * K2   # (..., 3, 3)


def _matvec(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrices ``R`` (..., 3, 3) to vectors ``p`` ((3,) or (..., 3))."""
    return torch.matmul(R, p.unsqueeze(-1)).squeeze(-1)


class GOATLegFK:
    """Forward kinematics for the GOAT legs (hip -> thigh -> knee -> wheel center).

    The wheel-center position only depends on the hip/thigh/knee angles; the wheel joint
    rotation does not move the wheel center, so it is ignored.

    Constants (meters / unit axes) extracted from WF_GOAT.urdf joint origins:
        Left : hip_L_Joint, thigh_L_Joint, knee_L_Joint, wheel_L_Joint
        Right: hip_R_Joint, thigh_R_Joint, knee_R_Joint, wheel_R_Joint
    """

    def __init__(self, device: torch.device | str):
        self.device = device

        def vec(x):
            return torch.tensor(x, device=device, dtype=torch.float32)

        # --- Left leg ---
        self.left = {
            "t_hip":   vec([0.038386,  0.094233, -0.152597]),
            "a_hip":   vec([-1.0, 0.0, 0.0]),
            "t_thigh": vec([-0.054, -0.017, 0.0]),
            "a_thigh": vec([0.0, 1.0, 0.0]),
            "t_knee":  vec([0.0, 0.018, -0.205]),
            "a_knee":  vec([0.0, -1.0, 0.0]),
            "t_wheel": vec([0.0, 0.016555, -0.200]),
        }
        # --- Right leg (y-mirrored origins, sign-flipped thigh/knee axes) ---
        self.right = {
            "t_hip":   vec([0.038386, -0.094233, -0.152597]),
            "a_hip":   vec([-1.0, 0.0, 0.0]),
            "t_thigh": vec([-0.054, 0.017, 0.0]),
            "a_thigh": vec([0.0, -1.0, 0.0]),
            "t_knee":  vec([0.0, -0.018, -0.205]),
            "a_knee":  vec([0.0, 1.0, 0.0]),
            "t_wheel": vec([0.0, -0.016555, -0.200]),
        }

    def _fk(self, q: torch.Tensor, c: dict) -> torch.Tensor:
        """Wheel center in base frame for one leg.

        Args:
            q: Joint angles [hip, thigh, knee]. Shape (..., 3).
            c: Per-leg constant dict (``self.left`` or ``self.right``).

        Returns:
            Wheel center position in the base frame. Shape (..., 3).
        """
        q_hip, q_thigh, q_knee = q[..., 0], q[..., 1], q[..., 2]

        p = c["t_wheel"]                                                   # in calf frame
        p = c["t_knee"] + _matvec(axis_angle_rotation(c["a_knee"], q_knee), p)    # -> thigh
        p = c["t_thigh"] + _matvec(axis_angle_rotation(c["a_thigh"], q_thigh), p) # -> hip
        p = c["t_hip"] + _matvec(axis_angle_rotation(c["a_hip"], q_hip), p)       # -> base
        return p

    def both_wheel_centers_b(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left/right wheel centers (base frame) from the left-leg command.

        Args:
            command: Left-leg joint command [hip, thigh, knee]. Shape (..., 3).
                The right leg uses the sign-flipped command (``q_R = -command``),
                matching the commander's mirroring convention.

        Returns:
            Tuple ``(p_left, p_right)``, each of shape (..., 3) in the base frame.
        """
        p_left = self._fk(command, self.left)
        p_right = self._fk(-command, self.right)
        return p_left, p_right
