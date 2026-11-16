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

from .nodes import _device, _resize_frames, chunk_frames, make_room
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
    +-4 frames; anything in `protect` (grown a little) stays out; `add` is always in. `enabled` off = the object only (e.g. when the object casts no visible effect: a bigger rebuild area
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
                "add": ("MASK", {"tooltip": "Always part of the zone, enabled or not (e.g. light the object gives off, from Light Zone)."}),
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

    def run(self, subject, images, enabled=True, protect=None, wide_factor=1.0, depth_factor=0.45, add=None):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        m = _fit_mask(subject, n, h, w)
        if add is not None:
            m = torch.maximum(m, _fit_mask(add, n, h, w))
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
    object widths from the object. `add` (e.g. Light Zone's light) is always pasted. Writes a JSON report (effect
    area, frames with effects)."""

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
                "add": ("MASK", {"tooltip": "Always pasted, like the object (e.g. light the object gives off, from Light Zone)."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("paste_mask", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, generated, subject, zone, object_grow=0.03, reach=0.8, sensitivity=1.0, add=None):
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
        ck = chunk_frames(h, w)
        make_room(dev, ck * h * w)
        for i in range(0, n, ck):
            j = min(n, i + ck)
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
        if add is not None:
            paste = torch.maximum(paste, (_fit_mask(add, n, h, w) > 0.5).float())
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


def _erode(m, r):
    """Square erosion of a (N,H,W) 0/1 mask (the image border doesn't erode)."""
    return 1.0 - _dilate(1.0 - m.float(), r)


def _hold(m, r=2):
    """Max of a (N,H,W) mask over +-r frames."""
    n, h, w = m.shape
    return F.max_pool1d(m.permute(1, 2, 0).reshape(1, h * w, n), 2 * r + 1, 1, r).reshape(h, w, n).permute(2, 0, 1)


def _resize_one(img, h, w):
    """(H,W,C) -> (h,w,C), area-like (bilinear with antialias)."""
    t = img.permute(2, 0, 1)[None].float()
    return F.interpolate(t, size=(h, w), mode="bilinear", antialias=True, align_corners=False)[0].permute(1, 2, 0)


class AutoMatteLightZone:
    """Light an object gives off (a projector beam, a lit screen's glow, a lamp's glow), so a removal takes it out
    with the object: otherwise a generative fill sees light that starts nowhere and invents a source for it (a bowl,
    a lamp). `light` is a mask of light in the clip (e.g. SAM 3.1 following "light beam"); only the parts connected
    to the object (touching it, grown by `touch` of the shorter side) are kept, followed up to `reach` object widths
    away, held over +-2 frames, grown and feathered a little. `enabled` off (e.g. a vision model says the object gives
    off no light) = an empty mask and `light` is never computed (lazy). Writes a JSON report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("MASK", {"tooltip": "The object's mask (white = object)."}),
                "images": ("IMAGE", {"tooltip": "Frames that set the output size and frame count."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = no light zone (light is never computed)."}),
                "light": ("MASK", {"lazy": True, "tooltip": "Light in the clip (e.g. SAM following 'light beam'); only computed when enabled."}),
                "touch": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.2, "step": 0.005, "tooltip": "How close to the object light must start, as a fraction of the shorter side."}),
                "reach": ("FLOAT", {"default": 12.0, "min": 0.5, "max": 40.0, "step": 0.5, "tooltip": "How far the light is followed from the object, in object widths."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("light_zone", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def check_lazy_status(self, **kw):
        if not kw.get("enabled", True):
            return []
        return ["light"] if "light" in kw and kw["light"] is None else []

    def run(self, subject, images, enabled=True, light=None, touch=0.03, reach=12.0):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        rep = {"light_from_subject": bool(enabled), "light_frames": 0, "light_area_vs_object": 0.0,
               "notes": [], "review_reasons": []}
        if not enabled or light is None:
            return (torch.zeros(n, h, w), json.dumps(rep))
        m = _fit_mask(subject, n, h, w) > 0.5
        lm = _fit_mask(light, n, h, w) > 0.5
        short = min(h, w)
        s = 4
        boxes = _bbox_track(m.float())
        seed_full = _dilate(m.float(), max(2, round(touch * short))) > 0.5
        out = torch.zeros(n, h, w)
        frames = 0
        for i in range(n):
            b = boxes[i]
            if b is None or not bool(lm[i].any()):
                continue
            a = F.max_pool2d(lm[i].float()[None, None], s)[0, 0]
            seed = F.max_pool2d(seed_full[i].float()[None, None], s)[0, 0] * a
            if not bool(seed.any()):
                continue
            cur = seed.clone()
            for _ in range(max(1, int(reach * (b[2] - b[0] + 1) / s))):
                nxt = torch.maximum(F.max_pool2d(cur[None, None], 3, 1, 1)[0, 0] * a, seed)
                if torch.equal(nxt, cur):
                    break
                cur = nxt
            z = F.interpolate(cur[None, None], size=(h, w), mode="nearest")[0, 0] * lm[i].float()
            if float(z.sum()) > 0.0005 * h * w:
                frames += 1
                out[i] = z
        out = _hold(out)
        out = _dilate(out, max(2, round(0.01 * short)))
        r_f = max(2, round(0.008 * short))
        out = _box(_box(out, r_f), r_f).clamp(0, 1)
        obj_area = float(m.float().sum()) or 1.0
        rep["light_frames"] = frames
        rep["light_area_vs_object"] = round(float((out > 0.5).sum()) / obj_area, 3)
        if frames:
            rep["notes"].append(f"The subject gives off light (a beam or a glow): it was removed with it in {frames} of "
                                f"{n} frames")
        else:
            rep["notes"].append("The subject seems to give off light, but none was found around it")
        return (out.contiguous(), json.dumps(rep))


class AutoMatteRingMatch:
    """Before pasting a generative fill back: match its colour to the original around the paste mask, so it doesn't
    show as a pale or dark patch (e.g. the object's contact shadow lies just outside the mask and the fill has none).
    Per frame and colour channel: gain = median of the original on a band just outside the mask / median of the fill
    on a band just inside it, both below the object's centre line (where both show the surface it stood on), clipped
    to [min_gain, max_gain] and smoothed over +-4 frames; the fill is multiplied by it. Frames without enough band
    pixels take the nearest frame's gain. `report` (optional JSON) is passed through with the gains added."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The original frames."}),
                "generated": ("IMAGE", {"tooltip": "The filled frames (same size and frame count)."}),
                "paste_mask": ("MASK", {"tooltip": "Where the fill will be pasted (e.g. Effect Mask)."}),
                "subject": ("MASK", {"tooltip": "The removed object's mask (sets the centre line)."}),
            },
            "optional": {
                "band": ("FLOAT", {"default": 0.035, "min": 0.005, "max": 0.2, "step": 0.005, "tooltip": "Band width on each side of the mask edge, as a fraction of the shorter side."}),
                "min_gain": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_gain": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 4.0, "step": 0.05}),
                "report": ("STRING", {"forceInput": True, "tooltip": "JSON report to pass through, with the gains added."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, generated, paste_mask, subject, band=0.035, min_gain=0.6, max_gain=1.6, report=None):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        gen = generated[..., :3]
        if gen.shape[1:3] != (h, w):
            gen = _resize_frames(gen, h, w, "bilinear")
        ng = gen.shape[0]
        if ng < n:
            gen = torch.cat([gen, gen[-1:].repeat(n - ng, 1, 1, 1)], 0)
        gen = gen[:n].float()
        pm = _fit_mask(paste_mask, n, h, w)
        subj = _fit_mask(subject, n, h, w)
        short = min(h, w)
        r_gap = max(2, round(0.004 * short))
        r_band = max(4, round(band * short))
        min_px = max(150, round(0.0002 * h * w))
        boxes = _bbox_track(subj)
        rows = torch.arange(h, dtype=torch.float32)[:, None].expand(h, w)
        gains = [None] * n
        ck = chunk_frames(h, w)
        for i in range(0, n, ck):
            j = min(n, i + ck)
            hard = (pm[i:j] > 0.5).float()
            touched = (pm[i:j] > 0.02).float()
            outer = (_dilate(touched, r_gap + r_band) > 0.5) & ~(_dilate(touched, r_gap) > 0.5)
            inner = (hard > 0.5) & ~(_erode(hard, r_band) > 0.5)
            for k in range(j - i):
                f = i + k
                b = boxes[f]
                if b is None:
                    ys = torch.nonzero(hard[k] > 0.5, as_tuple=True)[0]
                    if ys.numel() < 16:
                        continue
                    cy = (ys.min().item() + ys.max().item()) / 2
                else:
                    cy = (b[1] + b[3]) / 2
                below = rows > cy
                o_sel, i_sel = outer[k] & below, inner[k] & below
                if int(o_sel.sum()) < min_px or int(i_sel.sum()) < min_px:
                    continue
                mo = images[f, ..., :3][o_sel].float().median(0).values
                mi = gen[f][i_sel].median(0).values
                gains[f] = (mo / mi.clamp_min(1e-3)).clamp(min_gain, max_gain)
        have = [i for i in range(n) if gains[i] is not None]
        g = torch.ones(n, 3)
        if have:
            for i in range(n):
                near = gains[i] if gains[i] is not None else gains[min(have, key=lambda x: abs(x - i))]
                g[i] = near
            sm = g.clone()
            for i in range(n):
                sm[i] = g[max(0, i - 4):min(n, i + 5)].median(0).values
            g = sm
        out = (gen * g[:, None, None, :]).clamp(0, 1)
        try:
            rep = json.loads(report) if report else {}
        except Exception:
            rep = {"fill_report": report}
        if not isinstance(rep, dict):
            rep = {"fill_report": rep}
        med = g.median(0).values
        rep["edge_gain_median"] = [round(float(v), 3) for v in med]
        rep["edge_gain_range"] = [round(float(g.min()), 3), round(float(g.max()), 3)]
        rep["edge_gain_frames"] = len(have)
        if have and float((med - 1).abs().max()) > 0.08:
            rep.setdefault("notes", []).append(
                f"The fill was {'lighter' if float(med.mean()) < 1 else 'darker'} than the surface around it and was "
                f"matched to it (gain {float(med.mean()):.2f})")
        return (out.contiguous(), json.dumps(rep))


class AutoMatteMaskCrops:
    """Crops around a mask for a vision-model check of a fill, e.g. "is there an object in this spot now?": three
    frames (15%, 50%, 85% of the frames where the mask is present), each cropped around the mask with `margin` of
    its size on every side (the same crop size in all three, from the largest), scaled to about `megapixels`.
    Outputs the original crops and the filled crops at the same places."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The original frames."}),
                "filled": ("IMAGE", {"tooltip": "The frames after the fill (same size and frame count)."}),
                "mask": ("MASK", {"tooltip": "Where to look (e.g. the removed object's mask)."}),
            },
            "optional": {
                "margin": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "megapixels": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("original_crops", "filled_crops")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, filled, mask, margin=0.3, megapixels=0.3):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        fl = filled[..., :3]
        if fl.shape[1:3] != (h, w):
            fl = _resize_frames(fl, h, w, "bilinear")
        if fl.shape[0] < n:
            fl = torch.cat([fl, fl[-1:].repeat(n - fl.shape[0], 1, 1, 1)], 0)
        m = _fit_mask(mask, n, h, w) > 0.5
        present = [i for i in range(n) if int(m[i].sum()) >= 16]
        if present:
            picks = [present[(len(present) - 1) * p // 100] for p in (15, 50, 85)]
        else:
            picks = [(n - 1) * p // 100 for p in (15, 50, 85)]
        boxes = []
        for i in picks:
            ys, xs = torch.nonzero(m[i], as_tuple=True)
            if ys.numel() < 16:
                boxes.append((w / 2, h / 2, w, h))
                continue
            x0, x1, y0, y1 = xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            pad = 0.02 * min(h, w)
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, bw + 2 * max(pad, margin * bw), bh + 2 * max(pad, margin * bh)))
        cw = int(min(w, max(b[2] for b in boxes)))
        ch = int(min(h, max(b[3] for b in boxes)))
        sc = min(1.0, math.sqrt(megapixels * 1e6 / (cw * ch)))
        ow, oh = max(16, round(cw * sc)), max(16, round(ch * sc))
        oc, fc = [], []
        for i, (cx, cy, _, _) in zip(picks, boxes):
            x0 = int(min(max(0, round(cx - cw / 2)), w - cw))
            y0 = int(min(max(0, round(cy - ch / 2)), h - ch))
            oc.append(_resize_one(images[i, y0:y0 + ch, x0:x0 + cw, :3], oh, ow))
            fc.append(_resize_one(fl[i, y0:y0 + ch, x0:x0 + cw], oh, ow))
        return (torch.stack(oc).clamp(0, 1), torch.stack(fc).clamp(0, 1))


class AutoMatteRemovalReport:
    """One JSON report for an object removal: the matte report (where the mask came from), the fill report (effects,
    edge colour match), the light report, and the vision-model check after the fill ("is there an object in the
    subject's place?": `object_answer`; when it said yes the fill was redone with another seed and `retry_answer`
    is the same check on the redo). needs_review is set when any part asks for a review, or when the check found
    an object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"object_answer": ("STRING", {"forceInput": True})},
            "optional": {
                "matte_report": ("STRING", {"forceInput": True}),
                "fill_report": ("STRING", {"forceInput": True}),
                "light_report": ("STRING", {"forceInput": True}),
                "retry_answer": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    @staticmethod
    def _yes(a):
        return "yes" in (a or "").strip().lower()[:12]

    def run(self, object_answer, matte_report=None, fill_report=None, light_report=None, retry_answer=None):
        rep = {}
        for part in (matte_report, fill_report, light_report):
            try:
                d = json.loads(part) if part else {}
            except Exception:
                d = {"extra": part}
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if k in ("notes", "review_reasons") and isinstance(v, list):
                    rep[k] = list(rep.get(k) or []) + v
                elif k != "needs_review" or v:
                    rep[k] = v if k != "needs_review" else True
        rep.setdefault("notes", [])
        rep.setdefault("review_reasons", [])
        first = self._yes(object_answer)
        rep["object_in_fill"] = first
        if first:
            rep["fill_redone"] = True
            again = self._yes(retry_answer)
            rep["object_in_redone_fill"] = again
            if again:
                rep["review_reasons"].append("Something new appeared where the subject was, and again after a redo "
                                             "with another seed: check the clip; a Prompt describing the empty "
                                             "place can help")
            else:
                rep["review_reasons"].append("Something new appeared where the subject was; it was redone with "
                                             "another seed and the check found nothing there: check the clip")
        rep["needs_review"] = bool(rep.get("needs_review")) or bool(rep["review_reasons"])
        return (json.dumps(rep),)


NODE_CLASS_MAPPINGS = {
    "AutoMatteEffectZone": AutoMatteEffectZone,
    "AutoMatteEffectMask": AutoMatteEffectMask,
    "AutoMatteLightZone": AutoMatteLightZone,
    "AutoMatteRingMatch": AutoMatteRingMatch,
    "AutoMatteMaskCrops": AutoMatteMaskCrops,
    "AutoMatteRemovalReport": AutoMatteRemovalReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteEffectZone": "AutoMatte: Effect Zone",
    "AutoMatteEffectMask": "AutoMatte: Effect Mask",
    "AutoMatteLightZone": "AutoMatte: Light Zone",
    "AutoMatteRingMatch": "AutoMatte: Edge Colour Match",
    "AutoMatteMaskCrops": "AutoMatte: Mask Crops",
    "AutoMatteRemovalReport": "AutoMatte: Removal Report",
}
