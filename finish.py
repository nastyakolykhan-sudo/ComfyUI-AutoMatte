"""AutoMatte finishing nodes on a layer (a matte): grade only the subject or only the background, and hide things
(clearance: faces without a release, third-party logos, number plates) by blur or pixelation that follows them.

- AutoMatteLayerGrade  exposure (stops, in linear light), contrast (around the layer's own mid-tone), saturation
                       and warmth (OKLab), applied to the subject or to the background through the soft matte.
- AutoMatteObscure     blur or pixelate inside a matte, sized from each frame's object (so a near face and a far
                       face get the same look), feathered; or a solid fill.

Only torch; no model weights. Pixels outside the edited layer are returned unchanged.
"""

import json
import math

import torch
import torch.nn.functional as F

from .nodes import _device, CHUNK
from .layers import srgb_to_ok, ok_to_srgb, _fit_mask, _box


def _lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(0, 1) + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055).clamp(0, 1)


class AutoMatteLayerGrade:
    """Grade one layer of a clip: `target` Subject (inside the matte) or Background (outside it).
    exposure: stops in linear light (+1 = twice the light). contrast: -1..1 around the layer's own median lightness
    (whole clip, so nothing flickers). saturation: -1..1 (-1 = grey, +1 = twice the chroma). warmth: -1..1 (cooler ..
    warmer, a white-balance-like shift). All 0 = unchanged. Soft matte edges blend the grade."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK",),
                "target": (["Subject", "Background"], {"default": "Subject"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, matte, target="Subject", exposure=0.0, contrast=0.0, saturation=0.0, warmth=0.0):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        m = _fit_mask(matte, n, h, w)
        if str(target).lower().startswith("back"):
            m = 1 - m
        dev = _device()
        # the layer's median lightness over the clip (contrast pivot)
        samples = []
        for i in range(0, n, CHUNK):
            mm = m[i:i + CHUNK].to(dev)
            if bool((mm > 0.5).any()):
                L = srgb_to_ok(images[i:i + CHUNK, ..., :3].to(dev).float())[..., 0][mm > 0.5]
                samples.append(L[torch.linspace(0, L.numel() - 1, min(L.numel(), 50000), device=dev).long()].cpu())
        pivot = float(torch.cat(samples).median()) if samples else 50.0
        gain = 2.0 ** exposure
        k_con = 1.0 + contrast
        k_sat = max(0.0, 1.0 + saturation)
        out = images.clone()
        lsum = [0.0, 0.0, 0]
        for i in range(0, n, CHUNK):
            mm = m[i:i + CHUNK].to(dev)[..., None]
            if not bool((mm > 0).any()):
                continue
            rgb = images[i:i + CHUNK, ..., :3].to(dev).float()
            x = _srgb(_lin(rgb) * gain) if exposure else rgb
            ok = srgb_to_ok(x)
            L = pivot + (ok[..., 0] - pivot) * k_con
            a = ok[..., 1] * k_sat + 2.0 * warmth
            b = ok[..., 2] * k_sat + 10.0 * warmth
            new = ok_to_srgb(torch.stack([L.clamp(0, 100), a, b], -1))
            res = torch.where(mm > 0, rgb + (new - rgb) * mm, rgb)
            out[i:i + CHUNK, ..., :3] = res.to(out.dtype).cpu()
            sel = mm[..., 0] > 0.5
            if bool(sel.any()):
                lsum[0] += float(srgb_to_ok(rgb)[..., 0][sel].mean())
                lsum[1] += float(srgb_to_ok(res)[..., 0][sel].mean())
                lsum[2] += 1
        rep = {"edit": "layer_grade", "target": target, "exposure_stops": exposure, "contrast": contrast,
               "saturation": saturation, "warmth": warmth, "contrast_pivot_L": round(pivot, 1)}
        if lsum[2]:
            rep.update(layer_lightness_before=round(lsum[0] / lsum[2], 1), layer_lightness_after=round(lsum[1] / lsum[2], 1))
        return (out, json.dumps(rep))


class AutoMatteObscure:
    """Hide what a matte covers: Blur (a strong blur), Pixelate (square blocks) or Solid (a flat colour), sized per
    frame from each object's size so near and far objects look alike (`strength` 1 = unreadable for faces, logos and
    plates; higher = stronger). The matte is grown a little and feathered so edges and fast motion are covered."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK", {"tooltip": "What to hide (white), e.g. SAM on 'face, logo, license plate'."}),
                "style": (["Blur", "Pixelate", "Solid"], {"default": "Blur"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "colour": ("STRING", {"default": "#000000", "tooltip": "Solid only."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def run(self, images, matte, style="Blur", strength=1.0, colour="#000000"):
        from .layers import parse_colour
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        m = _fit_mask(matte, n, h, w)
        dev = _device()
        short = min(h, w)
        grow = max(2, round(0.006 * short))
        out = images.clone()
        hidden = 0
        sizes = []
        solid = torch.tensor([v / 255 for v in parse_colour(colour)[0]], device=dev) if style == "Solid" else None
        for i in range(n):
            mi = m[i].to(dev)
            if not bool((mi > 0.5).any()):
                continue
            hidden += 1
            ys, xs = torch.nonzero(mi > 0.5, as_tuple=True)
            size = max(8.0, math.sqrt(float((mi > 0.5).sum())))
            size = min(size, float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
            sizes.append(size)
            k = 2 * grow + 1
            mm = F.max_pool2d(mi[None, None], (1, k), 1, (0, grow))
            mm = F.max_pool2d(mm, (k, 1), 1, (grow, 0))[0, 0]
            mm = _box(mm[None], grow)[0].clamp(0, 1)[..., None]
            rgb = images[i, ..., :3].to(dev).float()
            if style == "Pixelate":
                block = max(4, int(size / 10 * strength))
                small = F.avg_pool2d(rgb.permute(2, 0, 1)[None], block, ceil_mode=True)
                fx = F.interpolate(small, scale_factor=block, mode="nearest")[0, :, :h, :w].permute(1, 2, 0)
            elif style == "Solid":
                fx = solid.expand_as(rgb)
            else:
                r = max(3, int(size / 6 * strength))
                fx = _box(_box(_box(rgb[None], r), r), r)[0]
            out[i, ..., :3] = (rgb + (fx - rgb) * mm).to(out.dtype).cpu()
        rep = {"edit": "obscure", "style": style, "strength": strength, "frames_hidden": hidden, "frames": n,
               "typical_object_px": round(sum(sizes) / len(sizes)) if sizes else None, "notes": [], "review_reasons": []}
        if not hidden:
            rep["review_reasons"].append("Nothing to hide was found: check the subject words")
        rep["needs_review"] = bool(rep["review_reasons"])
        return (out, json.dumps(rep))


NODE_CLASS_MAPPINGS = {
    "AutoMatteLayerGrade": AutoMatteLayerGrade,
    "AutoMatteObscure": AutoMatteObscure,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteLayerGrade": "AutoMatte: Layer Grade",
    "AutoMatteObscure": "AutoMatte: Obscure (blur / pixelate)",
}


class AutoMatteBackgroundSwap:
    """Put the subject (inside the matte) over a new background (an image or a clip; scaled to cover the frame, a
    shorter clip loops). Edge pixels are rebuilt from the subject's own colour (its interior spread outwards) rather
    than the old background showing through the soft edge, so no old-background fringe stays; `light_wrap` lets the
    new background's light spill a little onto the edges; `harmonize` moves the subject's overall colour cast and
    brightness part of the way towards the new background's (0 = none). Outputs the composite and a report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK",),
                "background": ("IMAGE", {"tooltip": "One image or a clip's frames."}),
            },
            "optional": {
                "light_wrap": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "harmonize": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    @staticmethod
    def _cover(bg, h, w):
        """Scale (N,H,W,C) to cover h x w, centre crop."""
        bh, bw = bg.shape[1], bg.shape[2]
        s = max(h / bh, w / bw)
        nh, nw = max(h, round(bh * s)), max(w, round(bw * s))
        t = F.interpolate(bg.permute(0, 3, 1, 2), size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        return t[:, :, y0:y0 + h, x0:x0 + w].permute(0, 2, 3, 1)

    def run(self, images, matte, background, light_wrap=0.35, harmonize=0.3):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        m = _fit_mask(matte, n, h, w)
        dev = _device()
        short = min(h, w)
        r_fg = max(3, round(0.01 * short))
        r_wrap = max(3, round(0.015 * short))
        nb = background.shape[0]
        # colour statistics for harmonize: subject vs new background (whole clip, OKLab means)
        sub_s, bg_s = [], []
        for i in range(0, n, CHUNK):
            j = min(n, i + CHUNK)
            mm = m[i:j].to(dev) > 0.5
            if bool(mm.any()):
                sub_s.append(srgb_to_ok(images[i:j, ..., :3].to(dev).float())[mm].mean(0).cpu())
        for i in range(0, min(nb, 64), 8):
            bg_s.append(srgb_to_ok(self._cover(background[i:i + 8, ..., :3].to(dev).float(), 256, 256)).reshape(-1, 3).mean(0).cpu())
        sub_mean = torch.stack(sub_s).mean(0) if sub_s else torch.zeros(3)
        bg_mean = torch.stack(bg_s).mean(0) if bg_s else torch.zeros(3)
        shift = torch.zeros(3)
        if harmonize > 0 and sub_s:
            shift[0] = harmonize * 0.5 * float(bg_mean[0] - sub_mean[0])          # half of the lightness gap
            shift[1:] = harmonize * (bg_mean[1:] - sub_mean[1:]) * 0.5            # half of the colour cast
        shift = shift.to(dev)
        out = torch.empty_like(images[..., :3])
        for i in range(0, n, CHUNK):
            j = min(n, i + CHUNK)
            rgb = images[i:j, ..., :3].to(dev).float()
            mm = m[i:j].to(dev)
            idx = torch.arange(i, j) % nb
            bg = self._cover(background[idx][..., :3].to(dev).float(), h, w)
            # subject colour at soft edges: interior colour spread outwards
            core = (mm > 0.9).float()
            den = _box(_box(core, r_fg), r_fg)
            fg_est = _box(_box(rgb * core[..., None], r_fg), r_fg) / den.clamp_min(1e-4)[..., None]
            k = (_smoothstep_t(mm, 0.5, 0.95))[..., None]
            fg = torch.where((den > 0.02)[..., None], k * rgb + (1 - k) * fg_est, rgb)
            if harmonize > 0:
                ok = srgb_to_ok(fg)
                fg = ok_to_srgb(ok + shift)
            a = mm[..., None]
            comp = fg * a + bg * (1 - a)
            if light_wrap > 0:
                bg_blur = _box(_box(bg, r_wrap), r_wrap)
                edge = (a * (1 - _box(_box(mm, r_wrap), r_wrap)[..., None])).clamp(0, 1) * 2.0
                comp = comp + light_wrap * edge.clamp(0, 1) * (bg_blur - comp) * a
            out[i:j] = comp.clamp(0, 1).to(out.dtype).cpu()
        rep = {"edit": "background_swap", "background_frames": nb, "light_wrap": light_wrap, "harmonize": harmonize,
               "subject_lightness": round(float(sub_mean[0]), 1), "background_lightness": round(float(bg_mean[0]), 1),
               "harmonize_shift": [round(float(v), 2) for v in shift.cpu()], "notes": []}
        if nb > 1 and nb < n:
            rep["notes"].append(f"The background clip has {nb} frames for {n}: it loops")
        return (out, json.dumps(rep))


def _smoothstep_t(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


NODE_CLASS_MAPPINGS["AutoMatteBackgroundSwap"] = AutoMatteBackgroundSwap
NODE_DISPLAY_NAME_MAPPINGS["AutoMatteBackgroundSwap"] = "AutoMatte: Background Swap"
