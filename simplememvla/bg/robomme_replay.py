"""Privileged-state replay of RoboMME training episodes (simulator side of the BG-Oracle).

The H5 / LeRobot data store no object states, but every RoboMME episode is reproducible from its metadata seed:
``env.reset()`` regenerates the conditioning video (the scripted demonstration) and the recorded joint actions
replay the robot part. ``ReplayRecorder`` wraps the DemonstrationWrapper's ``_step_batch`` so that every frame
the wrapper would record (demo frames and execution frames, "NO RECORD" frames dropped exactly like the wrapper
does) yields one privileged snapshot:

    frame          index in the episode (aligned with LeRobot frame_index after ``align``)
    is_demo        frame produced during reset() (the conditioning video)
    task_index     env.timestep (index of the current sub-goal in env.task_list; increments on completion)
    subgoal        online sub-goal name
    qpos           robot joint positions (7 arm + gripper fingers)
    poses          {actor name: xyz}
    vis_px         {actor name: front-camera segmentation pixel count}
    grasped        actor names the agent is grasping
    attrs          task attributes that changed since the previous snapshot (ints/floats/strings/actor names)

Run in an env with the RoboMME simulator (ManiSkill fork 07be6fbc); rendering on CPU (llvmpipe) by default.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_SKIP = ("table", "ground", "panda", "camera", "goal_site")
_ATTR_SKIP = {"spec", "scene", "agent", "_hidden_objects", "task_list", "single_observation_space",
              "observation_space", "action_space", "single_action_space"}


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def setup_cpu_render() -> None:
    """SAPIEN render on Mesa llvmpipe (no GPU): ManiSkill cannot express the 'pci:0' llvmpipe device, wrap it."""
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
    import sapien
    import mani_skill.envs.sapien_env as SE

    if getattr(SE.parse_sim_and_render_backend, "_bg_cpu", False):
        return
    orig = SE.parse_sim_and_render_backend

    def _cpu(sim_backend, render_backend):
        info = orig("physx_cpu", "none")
        info.render_device = sapien.Device(os.environ.get("BG_SAPIEN_DEVICE", "pci:0"))
        info.render_backend = "sapien_cpu"
        return info

    _cpu._bg_cpu = True
    SE.parse_sim_and_render_backend = _cpu


def _vendored_robomme_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "robomme_sim")


class ReplayRecorder:
    def __init__(self, task: str, split: str = "train", cpu_render: bool = True):
        if cpu_render:
            setup_cpu_render()
        p = _vendored_robomme_path()
        if p not in sys.path:
            sys.path.insert(0, p)
        import robomme.robomme_env  # noqa: F401  (registers envs)
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self.task = task
        self.builder = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="joint_angle", max_steps=3000)
        self.snaps: list[dict[str, Any]] = []
        self._prev_attrs: dict[str, Any] = {}
        self._depth = 0
        self._in_reset = False

    # ------------------------------------------------------------------ snapshot
    def _actors(self, u) -> dict[str, Any]:
        out = {}
        for name, a in getattr(u.scene, "actors", {}).items():
            if not any(s in name for s in _SKIP):
                out[name] = a
        return out

    def _attrs(self, u, actor_names: dict[int, str]) -> dict[str, Any]:
        out = {}
        for k, v in vars(u).items():
            if k in _ATTR_SKIP or k.startswith("__"):
                continue
            if isinstance(v, (bool, int, float, str)) or v is None:
                out[k] = v
            elif hasattr(v, "detach") and getattr(v, "numel", lambda: 99)() == 1:
                out[k] = _np(v).reshape(-1)[0].item()
            elif id(v) in actor_names:
                out[k] = "@" + actor_names[id(v)]
            elif isinstance(v, (list, tuple)) and 0 < len(v) <= 16:
                if all(isinstance(e, (bool, int, float, str)) for e in v):
                    out[k] = list(v)
                elif all(id(e) in actor_names for e in v):
                    out[k] = ["@" + actor_names[id(e)] for e in v]
            elif isinstance(v, dict) and 0 < len(v) <= 16 and all(isinstance(kk, (str, int)) for kk in v):
                vals = {}
                for kk, e in v.items():
                    if isinstance(e, (bool, int, float, str)) or e is None:
                        vals[str(kk)] = e
                    elif id(e) in actor_names:
                        vals[str(kk)] = "@" + actor_names[id(e)]
                if vals:
                    out[k] = vals
        return out

    def _snapshot(self, wrapper, maniskill_obs) -> dict[str, Any]:
        u = wrapper.unwrapped
        actors = self._actors(u)
        names_by_id = {id(a): n for n, a in actors.items()}
        # articulation links that tasks refer to (buttons: cap_link, cap_links[...])
        for k, v in vars(u).items():
            if hasattr(v, "pose") and id(v) not in names_by_id and hasattr(v, "name"):
                names_by_id[id(v)] = str(v.name)
        poses = {}
        for n, a in actors.items():
            poses[n] = _np(a.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
        def _add_pose(v):
            if hasattr(v, "pose") and hasattr(v, "name") and str(v.name) not in poses:
                try:
                    poses[str(v.name)] = _np(v.pose.p).reshape(-1)[:3].astype(np.float32).tolist()
                except Exception:
                    pass

        for k, v in vars(u).items():  # links / articulations the task refers to (button caps, peg ends, ...)
            if isinstance(v, dict):
                for e in v.values():
                    _add_pose(e)
            elif isinstance(v, (list, tuple)) and len(v) <= 32:
                for e in v:
                    _add_pose(e)
            else:
                _add_pose(v)
        vis = {}
        if maniskill_obs is not None:
            seg = _np(maniskill_obs["sensor_data"]["base_camera"]["segmentation"])[0, ..., 0]
            ids, cnt = np.unique(seg, return_counts=True)
            counts = dict(zip(ids.tolist(), cnt.tolist()))
            for sid, obj in u.segmentation_id_map.items():
                n = str(getattr(obj, "name", ""))
                if n in poses:
                    vis[n] = int(counts.get(int(sid), 0))
        cam = None
        if maniskill_obs is not None and not getattr(self, "_cam_done", False):
            sp = maniskill_obs["sensor_param"]["base_camera"]
            cam = {"K": _np(sp["intrinsic_cv"])[0].tolist(), "extrinsic": _np(sp["extrinsic_cv"])[0].tolist()}
            self._cam_done = True
        grasped = []
        for n, a in actors.items():
            try:
                if bool(_np(u.agent.is_grasping(a)).reshape(-1)[0]):
                    grasped.append(n)
            except Exception:
                pass
        attrs = self._attrs(u, names_by_id)
        changed = {k: v for k, v in attrs.items() if self._prev_attrs.get(k, "__none__") != v}
        self._prev_attrs = attrs
        return {
            "is_demo": bool(self._in_reset),
            "task_index": int(getattr(u, "timestep", 0) or 0),
            "subgoal": str(getattr(wrapper, "current_task_name", getattr(u, "current_task_name", ""))),
            "subgoal_online": str(getattr(u, "current_task_name_online", "")),
            "qpos": _np(u.agent.robot.qpos).reshape(-1).astype(np.float32).tolist(),
            "poses": poses,
            "vis_px": vis,
            "grasped": grasped,
            "attrs": changed,
            "elapsed": int(_np(getattr(u, "elapsed_steps", 0)).reshape(-1)[0]) if hasattr(u, "elapsed_steps") else -1,
            **({"cam": cam} if cam is not None else {}),
        }

    def _install_hook(self, env):
        w = env
        while not hasattr(w, "_step_batch"):
            w = w.env
        orig = w._step_batch
        rec = self

        def hooked(action):
            rec._depth += 1
            try:
                batch = orig(action)
            finally:
                rec._depth -= 1
            if rec._depth == 0:
                obs_b, _, _, _, info_b = batch
                ms = obs_b.get("maniskill_obs", [None])[-1]
                snap = rec._snapshot(w, ms)
                snap["subgoal_rec"] = str(info_b.get("simple_subgoal_online", [""])[-1])
                rec.snaps.append(snap)
            return batch

        w._step_batch = hooked
        return w

    # ------------------------------------------------------------------ episode
    def reset(self, episode: int, keep_no_record: bool = True):
        """keep_no_record=True matches the training data (H5 / LeRobot keep the "NO RECORD" reset frames between
        the conditioning video and execution); False matches what evaluation's reset() returns."""
        env = self.builder.make_env_for_episode(int(episode), include_maniskill_obs=True)
        self.env = env
        self.wrapper = self._install_hook(env)
        self.snaps, self._prev_attrs, self._cam_done = [], {}, False
        self._in_reset = True
        obs, info = env.reset()
        self._in_reset = False
        if not keep_no_record:  # evaluation: the wrapper drops "NO RECORD" frames from what reset() returns
            self.snaps = [s for s in self.snaps if s["subgoal_rec"].strip() != "NO RECORD"]
        # the last reset frame is the initial (non-demo) step taken inside reset()
        if self.snaps:
            self.snaps[-1]["is_demo"] = False
            self.snaps[-1]["init_step"] = True
        self.task_list = [dict(name=t.get("name"), demonstration=bool(t.get("demonstration", False)),
                               segment=self._seg_name(t.get("segment")))
                          for t in self.wrapper.unwrapped.task_list]
        return obs, info

    def _seg_name(self, seg):
        if seg is None:
            return None
        if isinstance(seg, (list, tuple)):
            return [self._seg_name(s) for s in seg]
        return str(getattr(seg, "name", seg))

    def step(self, action: np.ndarray):
        obs, _r, term, trunc, info = self.env.step(np.asarray(action, dtype=np.float64)[:8])
        return obs, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def frame_qpos8(snap: dict) -> np.ndarray:
    """LeRobot observation.state layout: 7 arm joints + first gripper finger."""
    q = np.asarray(snap["qpos"], dtype=np.float32)
    return np.concatenate([q[:7], q[7:8] if q.size > 7 else np.zeros(1, np.float32)])  # stick robot: gripper 0
