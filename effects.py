"""AutoMatte effect nodes: find what an object does to the scene around it (shadow, caustic, reflection glow) so a
removal takes those too, without touching anything else.

- AutoMatteEffectZone  where an object's effects can be: the object plus a generous area around where it meets
                       the surface (its base), sized from the object, steady over time. Use it as the area a
                       generative fill may rebuild.
- AutoMatteEffectMask  after the fill: the parts of the zone where the rebuilt clip differs from the original by a
                       smooth change of light or colour (a shadow, a caustic, a glow) connected to the object, plus
                       the object itself, feathered. Paste the fill back only there; everything else stays original,
                       even inside the zone.

Only torch; no model weights.
"""

import json
import math

import torch
import torch.nn.functional as F

from .nodes import _device, _resize_frames, CHUNK
from .layers import srgb_to_ok, _fit_mask, _box


def _bbox_track(mask, thr=0.5):
    """Per-frame bbox (x0, y0, x1, y1) of mask > thr, or None; smoothed over +-4 frames where present."""
    n = mask.shape[0]
    boxes = []
    for i in range(n):
        ys, xs = torch.nonzero(mask[i] > thr, as_tuple=True)
        boxes.append(None if ys.numel() < 16 else [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()])
    out = []
    for i in range(n):
        near = [boxes[j] for j in range(max(0, i - 4), min(n, i + 5)) if boxes[j] is not None]
        out.append(None if boxes[i] is None or not near else [sum(b[k] for b in near) / len(near) for k in range(4)])
    return out


def _dilate(m, r):
    """Square dilation of (N,H,W), separable (rows then columns)."""
    if r <= 0:
        return m
    k = 2 * r + 1
    t = F.max_pool2d(m[:, None].float(), (1, k), stride=1, padding=(0, r))
    return F.max_pool2d(t, (k, 1), stride=1, padding=(r, 0))[:, 0]


class AutoMatteEffectZone:
    """Where an object's shadow, caustic or reflection can fall, so a generative fill may rebuild it: the object plus
    an ellipse around where it meets the surface (the bottom of its mask), wide_factor x its width across and
    depth_factor x its height deep, never above the lower part of the object. Sized per frame and smoothed over
    +-4 frames; anything in `protect` (grown a little) stays out. `enabled` off = the object only (e.g. when the object casts no visible effect: a bigger rebuild area
    only invites the fill to invent things). `images` sets the output size and frame count (e.g. frames padded for a
    model: the mask's last frame is repeated)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("MASK", {"tooltip": "The object's mask (white = object)."}),
                "images": ("IMAGE", {"tooltip": "Frames that set the output size and frame count."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = the object only."}),
                "protect": ("MASK", {"lazy": True, "tooltip": "Other things the fill must not rebuild (e.g. objects next to this one); only computed when enabled."}),
                "wide_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 6.0, "step": 0.05, "tooltip": "Half-width of the area around the base, in object widths."}),
                "depth_factor": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Half-height of the area around the base, in object heights."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("zone",)
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def check_lazy_status(self, **kw):
        if not kw.get("enabled", True):
            return []
        return ["protect"] if "protect" in kw and kw["protect"] is None else []

    def run(self, subject, images, enabled=True, protect=None, wide_factor=1.0, depth_factor=0.45):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        m = _fit_mask(subject, n, h, w)
        if not enabled:
            return ((m > 0.5).float().contiguous(),)
        keep_out = None
        if protect is not None:
            pr = _fit_mask(protect, n, h, w)
            keep_out = _dilate((pr > 0.5).float(), max(2, round(0.015 * min(h, w)))) > 0.5
        boxes = _bbox_track(m)
        yy = torch.arange(h, dtype=torch.float32)[:, None]
        xx = torch.arange(w, dtype=torch.float32)[None, :]
        zone = torch.zeros(n, h, w)
        for i, b in enumerate(boxes):
            if b is None:
                continue
            x0, y0, x1, y1 = b
            wo, ho = x1 - x0 + 1, y1 - y0 + 1
            a = wide_factor * wo + 0.02 * w
            bb = depth_factor * ho + 0.02 * h
            cx = (x0 + x1) / 2
            e = (((xx - cx) / a) ** 2 + ((yy - y1) / bb) ** 2) <= 1
            zi = e & (yy >= y1 - 0.4 * ho)
            if keep_out is not None:
                zi = zi & ~keep_out[i]
            zone[i] = zi.float()
        return (torch.maximum(zone, (m > 0.5).float()).contiguous(),)


class AutoMatteEffectMask:
    """Paste-back mask after a generative fill of an effect zone: the object (dilated by `object_grow`, a fraction of
    the shorter side) plus the parts of the zone where the fill differs from the original by a smooth change of light
    or colour connected to the object (its shadow, caustic, reflection glow), held over +-2 frames and feathered.
    Differences are measured low-pass in OKLab against the fill's own noise level (outer part of the zone), so new
    texture or small shifts of other things in the zone don't count. Connection is followed at most `reach`
    object widths from the object. Writes a JSON report (effect area, frames with effects)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The original frames."}),
                "generated": ("IMAGE", {"tooltip": "The filled frames (same size and frame count)."}),
                "subject": ("MASK", {"tooltip": "The removed object's mask."}),
                "zone": ("MASK", {"tooltip": "The area the fill rebuilt (e.g. from Effect Zone)."}),
            },
            "optional": {
                "object_grow": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "How far around the object is always pasted, as a fraction of the shorter side."}),
                "reach": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 6.0, "step": 0.1, "tooltip": "How far an effect may extend from the object, in object widths."}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "tooltip": "Above 1 finds fainter effects; below 1 only strong ones."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("paste_mask", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, generated, subject, zone, object_grow=0.03, reach=0.8, sensitivity=1.0):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if generated.shape[1:3] != (h, w):
            generated = _resize_frames(generated[..., :3], h, w, "bilinear")
        ng = generated.shape[0]
        if ng < n:
            generated = torch.cat([generated, generated[-1:].repeat(n - ng, 1, 1, 1)], 0)
        subj = _fit_mask(subject, n, h, w)
        zone = _fit_mask(zone, n, h, w)
        dev = _device()
        short = min(h, w)
        r_lp = max(3, round(0.012 * short))
        r1 = max(1, round(0.002 * short))
        r_obj = max(2, round(object_grow * short))
        boxes = _bbox_track(subj)
        s = 4  # connection is followed on a 1/4 grid
        effect = torch.zeros(n, h, w)
        obj = torch.zeros(n, h, w)
        thr_l, thr_c, frames_fx = [], [], 0
        for i in range(0, n, CHUNK):
            j = min(n, i + CHUNK)
            orr = images[i:j, ..., :3].to(dev).float()
            gr = generated[i:j, ..., :3].to(dev).float()
            o = srgb_to_ok(orr)
            g = srgb_to_ok(gr)
            d = _box(_box(g - o, r_lp), r_lp)
            dl, dc = d[..., 0].abs(), torch.hypot(d[..., 1], d[..., 2])
            sm = subj[i:j].to(dev)
            zm = zone[i:j].to(dev) > 0.5
            od = _dilate((sm > 0.5).float(), r_obj) > 0.5
            obj[i:j] = od.float().cpu()
            for k in range(j - i):
                f = i + k
                b = boxes[f]
                if b is None:
                    continue
                cand_zone = zm[k] & ~od[k]
                if int(cand_zone.sum()) < 64:
                    continue
                # noise level of the fill: the outer half of the zone (far from the object)
                far = cand_zone & (_dilate(od[k:k + 1].float(), max(4, round(0.25 * (b[2] - b[0])))) [0] < 0.5)
                ref = far if int(far.sum()) > 256 else cand_zone
                vl, vc = dl[k][ref], dc[k][ref]
                tl = max(2.5, 3.0 * 1.4826 * float((vl - vl.median()).abs().median()) + float(vl.median())) / sensitivity
                tc = max(2.5, 3.0 * 1.4826 * float((vc - vc.median()).abs().median()) + float(vc.median())) / sensitivity
                thr_l.append(tl)
                thr_c.append(tc)
                cand = cand_zone & ((dl[k] > tl) | (dc[k] > tc))
                # only a change of light: the fill may be lighter with the same detail (a shadow taken away) or
                # simpler (a caustic or glint taken away); new or changed things carry their own detail
                Lo, Lg = o[k:k + 1, ..., 0], g[k:k + 1, ..., 0]
                so_ = _box((_box(Lo, r1) - _box(Lo, r_lp)).abs(), r_lp)[0]
                sg_ = _box((_box(Lg, r1) - _box(Lg, r_lp)).abs(), r_lp)[0]
                gain = _box(gr[k:k + 1], r_lp) / _box(orr[k:k + 1], r_lp).clamp_min(1e-3)
                res = _box((gr[k:k + 1] - orr[k:k + 1] * gain).abs().mean(-1), r_lp)[0]
                ref_res = float(res[ref].median()) if bool(ref.any()) else 0.01
                lighter = d[k, ..., 0] > 0
                shadow_like = (res < max(0.012, 1.8 * ref_res)) & lighter
                simpler = sg_ < 0.75 * so_
                cand = cand & (sg_ <= so_) & (shadow_like | simpler)
                # keep what connects to the object, within `reach` object widths
                a = F.max_pool2d(cand.float()[None, None], s)[0, 0]
                seed = F.max_pool2d(_dilate(od[k:k + 1].float(), 2)[None], s)[0, 0]
                cur = seed.clone()
                steps = max(1, int(reach * (b[2] - b[0] + 1) / s))
                for _ in range(steps):
                    nxt = torch.maximum(F.max_pool2d(cur[None, None], 3, 1, 1)[0, 0] * a, seed)
                    if torch.equal(nxt, cur):
                        break
                    cur = nxt
                fx = F.interpolate((cur * a)[None, None], size=(h, w), mode="nearest")[0, 0] * cand.float()
                if float(fx.sum()) > 0.0005 * h * w:
                    frames_fx += 1
                effect[f] = fx.cpu()
        # hold over +-2 frames, grow a little, feather
        eff = F.max_pool1d(effect.permute(1, 2, 0).reshape(1, h * w, n), 5, 1, 2).reshape(h, w, n).permute(2, 0, 1)
        eff = _dilate(eff, max(2, round(0.006 * short)))
        paste = torch.maximum(eff, obj)
        r_f = max(2, round(0.01 * short))
        paste = _box(_box(paste, r_f), r_f).clamp(0, 1)
        paste = torch.maximum(paste, (subj > 0.5).float())  # the object itself is always fully replaced
        obj_area = float(obj.sum()) or 1.0
        report = {"effect_frames": frames_fx, "frames": n,
                  "effect_area_vs_object": round(float((eff > 0.5).sum()) / obj_area, 3),
                  "effect_threshold_L": round(sum(thr_l) / len(thr_l), 2) if thr_l else None,
                  "effect_threshold_chroma": round(sum(thr_c) / len(thr_c), 2) if thr_c else None,
                  "notes": [], "review_reasons": []}
        if frames_fx:
            report["notes"].append(f"Effects of the object (shadow, light or reflection) were found in {frames_fx} of "
                                   f"{n} frames and removed with it")
        if report["effect_area_vs_object"] > 3:
            report["review_reasons"].append("A large area around the object changed: check that nothing else was "
                                            "rebuilt")
        return (paste.contiguous(), json.dumps(report))


NODE_CLASS_MAPPINGS = {
    "AutoMatteEffectZone": AutoMatteEffectZone,
    "AutoMatteEffectMask": AutoMatteEffectMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteEffectZone": "AutoMatte: Effect Zone",
    "AutoMatteEffectMask": "AutoMatte: Effect Mask",
}
