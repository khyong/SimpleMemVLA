"""UnsupBG proprio channel: gripper events + Panda forward kinematics (no simulator state).

Inputs are only what the LeRobot data / the rollout gives: ``observation.state`` = 7 arm joints + 1 finger joint
(0 = closed, 0.04 = open) at 20 Hz.

Events (``gripper_events``):
    grasp     the finger settles in (W_EMPTY, W_OPEN) after closing  -> an object is held
    release   the finger opens again after a grasp
    close     the finger settles at ~0 (closed on nothing: button press, push with closed gripper)
Each event carries the frame, the end-effector (TCP) position from FK (world frame of RoboMME, base at x=-0.615)
and the held width. Stick robots (PatternLock, RouteStick) have no gripper: ``tcp_dwell_events`` returns low-z
dwell points instead.

FK: Franka Panda modified-DH parameters (equivalent to the panda URDF joint origins) + flange 0.107 + TCP 0.1034.
``tests/bg/test_unsup_proprio.py`` checks it against the simulator's panda_hand_tcp link from the replay.
"""
from __future__ import annotations

import numpy as np

# (a, d, alpha) per joint, modified DH (Craig)
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
       (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE = 0.107
TCP = 0.1034
STICK_TCP = 0.15  # panda_stick: tip 0.15 m from the flange (fit to the simulator, residual < 9 mm)
BASE_XYZ = np.array([-0.615, 0.0, 0.0])  # RoboMME _load_agent pose
W_OPEN, W_EMPTY = 0.035, 0.004


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def panda_fk(q7: np.ndarray, tcp: bool = True, stick: bool = False) -> np.ndarray:
    """4x4 pose of the TCP (or flange) in the robot base frame for 7 joint angles."""
    T = np.eye(4)
    for (a, d, al), th in zip(_DH, q7):
        T = T @ _mdh(a, d, al, th)
    T = T @ _mdh(0.0, FLANGE, 0.0, 0.0)
    if tcp:
        T = T @ _mdh(0.0, STICK_TCP if stick else TCP, 0.0, -np.pi / 4)
    return T


def tcp_world(q7: np.ndarray, base_xyz: np.ndarray = BASE_XYZ, stick: bool = False) -> np.ndarray:
    return panda_fk(q7, stick=stick)[:3, 3] + base_xyz


def gripper_events(state: np.ndarray, settle: int = 6) -> list[dict]:
    """state (T, 8). Returns grasp / release / close events in time order."""
    w = state[:, 7]
    T = len(w)
    ev = []
    held = False
    closed_empty = False
    i = 1
    while i < T:
        closing = w[i] < w[i - 1] - 1e-4
        opening = w[i] > w[i - 1] + 1e-4
        if closing:
            j = i
            while j + 1 < T and abs(w[j + 1] - w[j]) > 1e-5:  # until the finger stops
                j += 1
            k = min(T - 1, j + settle)
            wk = float(np.median(w[j:k + 1]))
            if W_EMPTY < wk < W_OPEN and not held:
                ev.append({"type": "grasp", "frame": j, "width": wk, "tcp": tcp_world(state[j, :7]).tolist()})
                held, closed_empty = True, False
            elif wk <= W_EMPTY and not closed_empty and not held:
                ev.append({"type": "close", "frame": j, "width": wk, "tcp": tcp_world(state[j, :7]).tolist()})
                closed_empty = True
            elif wk <= W_EMPTY and held:  # object slipped out while closed
                ev.append({"type": "release", "frame": j, "width": wk, "tcp": tcp_world(state[j, :7]).tolist(),
                           "slip": True})
                held, closed_empty = False, True
            i = j + 1
            continue
        if opening and w[i] > W_EMPTY:
            if held:
                ev.append({"type": "release", "frame": i, "width": float(w[i]), "tcp": tcp_world(state[i, :7]).tolist()})
                held = False
            closed_empty = False
        i += 1
    return ev


def tcp_dwell_events(state: np.ndarray, z_max: float = 0.12, min_frames: int = 4, v_max: float = 0.004) -> list[dict]:
    """Stick robots: frames where the TCP stops low over the table (touch / waypoint)."""
    P = np.stack([tcp_world(q[:7], stick=True) for q in state])
    v = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    low = (P[:, 2] < z_max) & (v < v_max)
    ev, i = [], 0
    while i < len(P):
        if low[i]:
            j = i
            while j + 1 < len(P) and low[j + 1]:
                j += 1
            if j - i + 1 >= min_frames:
                m = (i + j) // 2
                ev.append({"type": "dwell", "frame": m, "tcp": P[m].tolist(), "span": [i, j]})
            i = j + 1
        else:
            i += 1
    return ev
