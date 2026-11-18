"""R2: fractional integration of action chunks, plus an explicit TCP adaptation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class IncrementPath:
    """Paper path in action units: H rows of [dx,dy,dz,drx,dry,drz,g].

    Clipping/unnormalization belongs to the policy adapter, before construction.
    Rotations here are *additive*, exactly as in the paper's relative OSC interface.
    """

    def __init__(self, actions):
        actions = np.array(actions, dtype=np.float64, copy=True)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
            raise ValueError("actions must have shape (H, 7), H > 0")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        self.actions = actions
        self.horizon = len(actions)
        self.cumulative = np.vstack((np.zeros(6), np.cumsum(actions[:, :6], axis=0)))

    def _position(self, p: float) -> float:
        if not np.isfinite(p) or p < -1e-9 or p > self.horizon + 1e-9:
            raise ValueError(f"path position {p} outside [0, {self.horizon}]")
        return float(np.clip(p, 0, self.horizon))

    def phi(self, p: float) -> np.ndarray:
        p = self._position(p)
        index = min(int(p), self.horizon - 1)
        return self.cumulative[index] + (p - index) * self.actions[index, :6]

    def delta(self, start: float, end: float) -> np.ndarray:
        if end < start:
            raise ValueError("path traversal must be forward")
        return self.phi(end) - self.phi(start)

    def gripper(self, end: float) -> float:
        # Each action owns (i, i+1]; at integer end=1, execute action[0].
        end = self._position(end)
        index = max(0, min(int(np.ceil(end - 1e-10)) - 1, self.horizon - 1))
        return float(self.actions[index, 6])

    def breakpoints(self, start: float, end: float) -> np.ndarray:
        """All slope changes and endpoints; needed to reject disconnected feasible rates."""
        self._position(start)
        self._position(end)
        return np.unique(np.r_[start, np.arange(np.floor(start) + 1, np.ceil(end)), end])

    def angular_travel(self, start: float, end: float) -> float:
        points = self.breakpoints(start, end)
        return float(sum(np.linalg.norm(self.delta(a, b)[3:]) for a, b in zip(points[:-1], points[1:])))


class AbsolutePosePath(IncrementPath):
    """Real-robot adaptation: absolute xyz/xyzw targets, with the observation pose at p=0.

    Translation follows Eq. 3. Orientation follows the piecewise SLERP path on
    SO(3), NOT a sum of absolute rotation vectors. Each returned rotation is a
    world-frame left-multiplied relative rotation. This is an adaptation, not
    the paper's small-angle additive OSC rotation approximation.
    """

    def __init__(self, positions, quaternions, grippers, *, anchor_position, anchor_quaternion):
        positions = np.asarray(positions, dtype=np.float64)
        quaternions = np.asarray(quaternions, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        anchor_position = np.asarray(anchor_position, dtype=np.float64)
        anchor_quaternion = np.asarray(anchor_quaternion, dtype=np.float64)
        h = len(positions)
        if positions.shape != (h, 3) or quaternions.shape != (h, 4) or grippers.shape != (h,):
            raise ValueError("expected positions (H,3), xyzw quaternions (H,4), grippers (H,)")
        if anchor_position.shape != (3,) or anchor_quaternion.shape != (4,):
            raise ValueError("invalid anchor pose")
        self.positions = np.vstack((anchor_position, positions))
        all_quats = np.vstack((anchor_quaternion, quaternions))
        if not np.all(np.isfinite(all_quats)) or np.any(np.linalg.norm(all_quats, axis=1) < 1e-12):
            raise ValueError("quaternions must be finite and nonzero")
        self.rotations = Rotation.from_quat(all_quats)
        dr = (self.rotations[1:] * self.rotations[:-1].inv()).as_rotvec()
        super().__init__(np.column_stack((np.diff(self.positions, axis=0), dr, grippers)))
        self._slerp = Slerp(np.arange(h + 1), self.rotations)

    def orientation(self, p: float) -> Rotation:
        return self._slerp(self._position(p))

    def delta(self, start: float, end: float) -> np.ndarray:
        out = super().delta(start, end)
        out[3:] = (self.orientation(end) * self.orientation(start).inv()).as_rotvec()
        return out

