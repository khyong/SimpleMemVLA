"""UnsupBG visual channel: open-vocabulary detection (OWLv2) + SAM2 video tracking on the front camera.

Only data the policy also has: LeRobot front video frames (256x256 RGB). No depth, no segmentation, no poses.

Frame schedule (``schedule``): 2 fps base, 10 fps within +-1 s of every proprio event, and 10 fps inside
"motion windows" (frame-difference bursts while the gripper holds nothing, e.g. swaps / video demos).

``Perception.run`` returns tracks: persistent object ids are SAM2 object ids created at the first confident
detection; new detections that do not overlap an existing track start new tracks. Tracking runs forward only
(causal); the detector re-checks every ``redetect_every`` sampled frames.

Output per track: category, first_frame, and per sampled frame (frame index, visible, area, cx, cy, bbox).
Everything is cached per episode (npz/json) by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUERY = {  # spec category -> detector text query (RoboMME renderings: prompt check in reports/03)
    "red cube": "a small red cube", "green cube": "a small green cube", "blue cube": "a small blue cube",
    "cube": "a small cube", "container": "a white cube", "bin": "a square hole",
    "button": "a gray button", "target": "a purple and white circular target", "peg": "a thin stick",
    "box": "a wooden box", "stick": "a thin vertical stick", "highlight": "a white disk",
}
# per-category score thresholds (OWLv2-base on 256x256 RoboMME frames, CPU check of 6 frames: cubes 0.5-0.7,
# targets ~0.5, containers 0.1-0.2, white highlight disk ~0.3; the BinFill box only as "a square hole" ~0.35, which
# also fires on the button; thin pegs / sticks / white grid dots of the imitation tasks were not detected)
THRESH = {"container": 0.08, "highlight": 0.25}
ROBOT_BAND_PX = 30  # detections whose box lies entirely in the top band are the robot base, not objects


def schedule(n_frames: int, event_frames: list[int], motion: np.ndarray | None = None, base_fps: float = 2.0,
             dense_fps: float = 10.0, fps: float = 20.0, window_s: float = 1.0, causal: bool = False) -> list[int]:
    """causal=True: dense sampling only *after* an event / motion onset (no look-ahead, D10 fix); with
    event_frames=[] and motion=None it is a uniform base_fps schedule."""
    base = set(range(0, n_frames, int(round(fps / base_fps))))
    step = int(round(fps / dense_fps))
    w = int(round(window_s * fps))
    for f in event_frames:
        lo = f if causal else max(0, f - w)
        base.update(range(lo, min(n_frames, f + w + 1), step))
    if motion is not None:
        for f in np.nonzero(motion)[0]:
            base.update(range(f if causal else max(0, f - step), min(n_frames, f + step + 1), step))
    base.add(n_frames - 1)
    return sorted(base)


def motion_windows(frames: np.ndarray, thr: float = 6.0) -> np.ndarray:
    """Per-frame flag: mean absolute frame difference above thr (uint8 scale)."""
    d = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    return np.r_[False, d > thr]


COLOR_HUE = {"red": [(0, 12), (168, 180)], "green": [(40, 85)], "blue": [(100, 130)]}  # OpenCV-style hue 0..180


def color_pixels(img: np.ndarray, mask: np.ndarray, category: str) -> int:
    """Mask pixels whose colour matches a coloured category ('red cube' -> red); area for other categories.
    A coloured cube covered by a container keeps a SAM2 mask on the container surface, but loses its colour."""
    col = category.split()[0] if " " in category else ""
    if col not in COLOR_HUE:
        return int(mask.sum())
    px = img[mask]
    if len(px) == 0:
        return 0
    rgb = px.astype(np.float32) / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 30.0  # 0..180
    ok = np.zeros(len(px), bool)
    for lo, hi in COLOR_HUE[col]:
        ok |= (h >= lo) & (h <= hi)
    ok &= (sat > 0.45) & (mx > 0.2)
    return int(ok.sum())


@dataclass
class Track:
    oid: int
    category: str
    first_frame: int
    rows: list = field(default_factory=list)  # (frame, visible, area, cx, cy, x0, y0, x1, y1)


class Perception:
    def __init__(self, owl_path: str, sam2_path: str, device: str = "cuda", det_thr: float = 0.3,
                 redetect_every: int = 4, min_area: int = 12):
        import torch
        from transformers import (Owlv2ForObjectDetection, Owlv2Processor, Sam2VideoModel,
                                  Sam2VideoProcessor)

        self.torch = torch
        self.device = device
        self.owl_proc = Owlv2Processor.from_pretrained(owl_path)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_path).to(device).eval()
        self.sam_proc = Sam2VideoProcessor.from_pretrained(sam2_path)
        self.sam = Sam2VideoModel.from_pretrained(sam2_path).to(device).eval()
        self.det_thr = det_thr
        self.redetect_every = redetect_every
        self.min_area = min_area

    def detect(self, img: np.ndarray, categories: list[str]) -> list[dict]:
        torch = self.torch
        queries = [QUERY.get(c, c) for c in categories]
        inp = self.owl_proc(text=[queries], images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.owl(**inp)
        H, W = img.shape[:2]
        res = self.owl_proc.post_process_grounded_object_detection(
            outputs=out, target_sizes=torch.tensor([[max(H, W), max(H, W)]]), threshold=min(THRESH.values()),
            text_labels=[queries])[0]
        dets = []
        for box, score, lab in zip(res["boxes"], res["scores"], res["labels"]):
            x0, y0, x1, y1 = [float(v) for v in box]
            cat = categories[int(lab)]
            if float(score) < THRESH.get(cat, self.det_thr) or y1 <= ROBOT_BAND_PX:
                continue
            dets.append({"box": [max(0, x0), max(0, y0), min(W, x1), min(H, y1)], "score": float(score),
                         "category": cat})
        return _nms(dets, 0.5)

    def run(self, frames: np.ndarray, sample: list[int], categories: list[str],
            tcp_uv: dict[int, tuple[float, float]] | None = None, late_thr: float = 0.45,
            late_after: int = 2) -> list[Track]:
        """frames: (T, H, W, 3) uint8 of the whole episode; sample: frame indices to process (sorted).
        tcp_uv: projected end-effector per frame; detections within 20 px of it are not new objects (gripper or a
        held object already tracked). Detections after the first ``late_after`` detector passes need
        score >= late_thr (objects entering later are rare; low-score late boxes are mostly re-detections)."""
        torch = self.torch
        clip = [frames[i] for i in sample]
        # on a GPU the frames and the per-frame tracking state stay in host memory (only placement changes, not the
        # computation): a long episode's float32 state exceeded 24 GB on an RTX 4090
        off = {} if str(self.device) == "cpu" else {"video_storage_device": "cpu", "inference_state_device": "cpu"}
        sess = self.sam_proc.init_video_session(video=clip, inference_device=self.device,
                                                dtype=torch.float32, **off)
        tracks: dict[int, Track] = {}
        next_id = 1
        H, W = frames.shape[1:3]
        last_masks: dict[int, np.ndarray] = {}
        for k, fidx in enumerate(sample):
            if k % self.redetect_every == 0:
                new = []
                late = (k // self.redetect_every) >= late_after
                for d in self.detect(clip[k], categories):
                    if late and d["score"] < THRESH.get(d["category"], self.det_thr) * 1.25:
                        continue
                    # same category: any overlap is the same object; another category may sit on/over it
                    # (a container dropped over a cube), so only a near-identical box counts as a duplicate
                    if any(_box_mask_iou(d["box"], m) > (0.1 if tracks[o].category == d["category"] else 0.6)
                           for o, m in last_masks.items() if o in tracks):
                        continue
                    if any(_iou(d["box"], n["box"]) > 0.3 for n in new):
                        continue
                    if tcp_uv is not None and fidx in tcp_uv and tcp_uv[fidx] is not None:
                        cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
                        if (cx - tcp_uv[fidx][0]) ** 2 + (cy - tcp_uv[fidx][1]) ** 2 < 20.0 ** 2:
                            continue
                    new.append(d)
                if new:
                    ids = list(range(next_id, next_id + len(new)))
                    next_id += len(new)
                    self.sam_proc.add_inputs_to_inference_session(
                        inference_session=sess, frame_idx=k, obj_ids=ids,
                        input_boxes=[[d["box"] for d in new]])
                    for oid, d in zip(ids, new):
                        tracks[oid] = Track(oid, d["category"], fidx)
            if not tracks:
                continue
            with torch.no_grad():
                out = self.sam(inference_session=sess, frame_idx=k)
            masks = self.sam_proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]],
                                                     binarize=True)[0]
            for j, oid in enumerate(sess.obj_ids):
                m = masks[j, 0].cpu().numpy().astype(bool)
                cat = tracks[oid].category if oid in tracks else ""
                area = color_pixels(clip[k], m, cat)
                vis = area >= self.min_area
                if vis:
                    ys, xs = np.nonzero(m)
                    row = (fidx, True, area, float(xs.mean()), float(ys.mean()), int(xs.min()), int(ys.min()),
                           int(xs.max()), int(ys.max()))
                    last_masks[oid] = m
                else:
                    row = (fidx, False, area, np.nan, np.nan, -1, -1, -1, -1)
                    if int(m.sum()) < self.min_area:
                        last_masks.pop(oid, None)
                    else:
                        last_masks[oid] = m  # covered coloured object: keep its region from re-detection
                if oid in tracks:
                    tracks[oid].rows.append(row)
        return list(tracks.values())


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets, thr):
    out = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(_iou(d["box"], o["box"]) < thr for o in out):
            out.append(d)
    return out


def _box_mask_iou(box, mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    return _iou(box, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
