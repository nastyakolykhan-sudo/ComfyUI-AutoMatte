"""AutoMatte: glue nodes for a fully automatic subject matte.

A hard subject mask (e.g. SAM 3.1) says WHAT the subject is, a soft matte
(e.g. LTX-2.5 Alpha Gen) says HOW SOFT its edges are, and these nodes do the
rest without anyone choosing a model. Model-agnostic: no model limits in the
logic, defaults change nothing.

- AutoMatteFit      resize + pad frames to a model's input rules (size caps,
                    size multiple, k*n+offset frames) and remember how to undo it.
- AutoMatteCombine  gate the soft matte to the subject mask; if the soft matte
                    made a solid object semi-transparent (a product, metal,
                    a label), fill the interior from the mask and rescale the
                    edges. Writes a JSON report of what it did.
- AutoMatteRestore  crop the padding off and scale back to the source size and
                    frame count.
- AutoMatteToRGBA   source frames + matte -> RGBA frames (straight alpha).

No model weights, only torch, so nothing here carries a model licence.
"""

import json
import re
import math

import torch
import torch.nn.functional as F

try:  # ComfyUI runtime
    import comfy.model_management as mm

    def _device():
        return mm.get_torch_device()
except Exception:  # unit tests outside ComfyUI
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK = 8  # frames per GPU chunk


def _resize_frames(x_nhwc, h, w, mode):
    """Resize an (N,H,W,C) batch in chunks on the compute device."""
    if x_nhwc.shape[1] == h and x_nhwc.shape[2] == w:
        return x_nhwc
    dev, out = _device(), []
    for i in range(0, x_nhwc.shape[0], CHUNK):
        t = x_nhwc[i:i + CHUNK].to(dev).permute(0, 3, 1, 2).float()
        kw = {"antialias": True} if mode in ("bilinear", "bicubic") else {}
        t = F.interpolate(t, size=(h, w), mode=mode, align_corners=False if mode != "area" else None, **kw)
        out.append(t.permute(0, 2, 3, 1).cpu())
    return torch.cat(out, 0)


def _morph(mask_nhw, radius, op):
    """Square dilation ('dil') or erosion ('ero') of an (N,H,W) mask in [0,1]."""
    if radius <= 0:
        return mask_nhw
    k = 2 * radius + 1
    dev, out = _device(), []
    for i in range(0, mask_nhw.shape[0], CHUNK):
        t = mask_nhw[i:i + CHUNK].to(dev).float()[:, None]
        if op == "dil":
            t = F.max_pool2d(t, k, stride=1, padding=radius)
        else:
            t = -F.max_pool2d(-t, k, stride=1, padding=radius)
        out.append(t[:, 0].cpu())
    return torch.cat(out, 0)


def _local_coverage(ag, sam, deep, short):
    """How much of the subject the soft matte covers, per pixel: the soft matte's mean over the subject's interior in
    a neighbourhood (~3% of the shorter side) and +-4 frames, computed at low resolution. Thin parts with no
    interior fall back to the whole subject mask. Returns (N,H,W) in [0,1]."""
    n, H, W = ag.shape
    s = max(1, int(math.ceil(short / 192)))
    h, w = max(1, H // s), max(1, W // s)
    def small(x):
        return F.adaptive_avg_pool2d(x[:, None].float(), (h, w))[:, 0]
    a, m, d = small(ag), small(sam), small(deep.float())
    r = max(1, round(0.03 * short / s))
    k = 2 * r + 1
    def blur(x):  # box blur in space (twice ~ gaussian), then +-4 frames
        x = F.avg_pool2d(x[:, None], k, stride=1, padding=r, count_include_pad=False)
        x = F.avg_pool2d(x, k, stride=1, padding=r, count_include_pad=False)[:, 0]
        t = x.permute(1, 2, 0).reshape(1, h * w, n)
        t = F.avg_pool1d(t, 9, stride=1, padding=4, count_include_pad=False)
        return t.reshape(h, w, n).permute(2, 0, 1)
    num_d, den_d = blur(a * d), blur(d)
    num_m, den_m = blur(a * m), blur(m)
    cov = torch.where(den_d > 0.02, num_d / den_d.clamp_min(1e-4), num_m / den_m.clamp_min(1e-4))
    cov = torch.where(den_m > 1e-3, cov, torch.ones_like(cov))
    return F.interpolate(cov[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0].clamp(0, 1)


def _median01(values):
    """Median of values in [0,1] via a 1024-bin histogram (cheap on huge sets)."""
    if values.numel() == 0:
        return 1.0
    hist = torch.histc(values.float().clamp(0, 1), bins=1024, min=0.0, max=1.0)
    cdf = torch.cumsum(hist, 0)
    idx = int(torch.searchsorted(cdf, cdf[-1] * 0.5).item())
    return (idx + 0.5) / 1024.0


class AutoMatteFit:
    """Fit a frame batch to a model's input rules, reversibly (undo with Restore Size).

    - Scale down (never up) so the long side <= max_long_side and the short side <= max_short_side
      (whichever orientation the clip has). 0 = no limit.
    - Pad right/bottom to a multiple of size_multiple by repeating edge pixels. 1 = off.
    - Pad the frame count to frame_multiple * n + frame_offset by repeating the last frame. 1/0 = off.
    Pads, never stretches. Defaults change nothing; e.g. LTX-2.5 wants long 1920, short 1088,
    size multiple 32, frames 8n+1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_long_side": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 = no limit."}),
                "max_short_side": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 = no limit."}),
                "size_multiple": ("INT", {"default": 1, "min": 1, "max": 512, "tooltip": "Pad width and height up to a multiple of this. 1 = off."}),
                "frame_multiple": ("INT", {"default": 1, "min": 1, "max": 256, "tooltip": "Frame count becomes frame_multiple*n + frame_offset. 1 = off."}),
                "frame_offset": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "AUTOMATTE_FIT", "STRING")
    RETURN_NAMES = ("images", "width", "height", "length", "fit", "fit_info")
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    def run(self, images, max_long_side, max_short_side, size_multiple, frame_multiple, frame_offset):
        n, h, w, _ = images.shape
        m = max(1, size_multiple)
        long_cap = max_long_side - max_long_side % m if max_long_side else 0
        short_cap = max_short_side - max_short_side % m if max_short_side else 0
        max_w, max_h = (long_cap, short_cap) if w >= h else (short_cap, long_cap)
        s = 1.0
        if max_w:
            s = min(s, max_w / w)
        if max_h:
            s = min(s, max_h / h)
        cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
        if max_w:
            cw = min(cw, max_w)
        if max_h:
            ch = min(ch, max_h)
        W, H = int(math.ceil(cw / m) * m), int(math.ceil(ch / m) * m)
        x = _resize_frames(images, ch, cw, "bicubic").clamp(0, 1)
        if (W, H) != (cw, ch):
            x = F.pad(x.permute(0, 3, 1, 2), (0, W - cw, 0, H - ch), mode="replicate").permute(0, 2, 3, 1)
        k, off = max(1, frame_multiple), frame_offset % max(1, frame_multiple)
        L = n
        if k > 1 or off:
            L = max(n, off)
            while (L - off) % k:
                L += 1
        if L > n:
            x = torch.cat([x, x[-1:].repeat(L - n, 1, 1, 1)], 0)
        fit = {"orig_w": w, "orig_h": h, "orig_n": n, "content_w": cw, "content_h": ch, "W": W, "H": H, "L": L}
        return (x.contiguous(), W, H, L, fit, json.dumps(fit))


class AutoMatteCombine:
    """Final matte from a hard subject mask (e.g. SAM 3.1) and a soft matte (e.g. LTX-2.5 Alpha Gen).

    interior = median soft-matte value deep inside the subject mask (whole clip).
    - subject mask empty in every frame -> the soft matte as is (its own subject pick).
    - interior >= solid_threshold or keep_transparency
                                          -> soft-matte edges, gated to the subject.
    - otherwise (the soft matte made a solid object see-through)
                                          -> max(eroded subject mask, soft matte / interior),
                                             gated to the subject.
    - interior < min_interior (the soft matte missed the subject) -> the subject mask, lightly feathered.
    A named subject missing for up to hold_frames frames holds the nearest mask; longer absences
    (the subject is out of shot) come out empty. subject_text wired and empty = no subject named:
    the soft matte as is. Radii are 0 = automatic (a fraction of the shorter side).
    use_soft_matte off = the subject mask only, lightly feathered (same out-of-shot / dropout handling); the soft
    matte input is lazy, so whatever makes it is never run (a faster, lower-quality choice)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soft_matte": ("IMAGE", {"lazy": True, "tooltip": "Soft matte as grey frames (white = subject). Not computed when use_soft_matte is off."}),
                "subject_mask": ("MASK", {"tooltip": "Hard mask of the chosen subject, same frames."}),
                "solid_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "keep_transparency": ("BOOLEAN", {"default": False,
                                                   "tooltip": "On for subjects that really are see-through: never repair the interior."}),
                "gate_grow_px": ("INT", {"default": 0, "min": 0, "max": 256, "tooltip": "0 = auto (1.1% of shorter side)."}),
                "core_shrink_px": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "0 = auto (0.4% of shorter side)."}),
            },
            "optional": {
                "subject_text": ("STRING", {"forceInput": True, "tooltip": "What the user asked for. Wired and empty = no subject named: the soft matte is used as is."}),
                "hold_frames": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "Gaps up to this many frames hold the nearest subject mask; longer absences come out empty."}),
                "min_interior": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Below this, the soft matte missed the subject: the subject mask is used instead."}),
                "use_soft_matte": ("BOOLEAN", {"default": True, "tooltip": "Off = the subject mask only, lightly feathered; the soft matte is never computed (faster, harder edges)."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("matte", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    def check_lazy_status(self, soft_matte=None, use_soft_matte=True, **kwargs):
        return ["soft_matte"] if use_soft_matte and soft_matte is None else []

    def run(self, soft_matte, subject_mask, solid_threshold, keep_transparency, gate_grow_px, core_shrink_px,
            subject_text=None, hold_frames=4, min_interior=0.25, use_soft_matte=True):
        sam = subject_mask.float()
        if sam.dim() == 2:
            sam = sam[None]
        use_soft = bool(use_soft_matte) and soft_matte is not None
        if use_soft:
            ag = soft_matte[..., :3].float().mean(-1).clamp(0, 1)  # (N,H,W)
            n, H, W = ag.shape
            if sam.shape[1:] != (H, W):
                sam = F.interpolate(sam[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            if sam.shape[0] < n:  # pad frame count by repeating the last mask
                sam = torch.cat([sam, sam[-1:].repeat(n - sam.shape[0], 1, 1)], 0)
        else:
            ag = None
            n, H, W = sam.shape
        sam = (sam[:n] > 0.5).float()
        wired = subject_text is not None
        asked = bool(wired and subject_text.strip())

        short = min(H, W)
        r_gate = gate_grow_px or max(2, round(0.011 * short))
        r_core = core_shrink_px or max(1, round(0.004 * short))
        r_deep = max(r_gate, round(0.014 * short))

        present = sam.flatten(1).amax(1) > 0  # (N,)
        report = {"frames": n, "size": [W, H], "subject": (subject_text or "").strip(),
                  "subject_frames": int(present.sum()), "gate_grow_px": r_gate, "core_shrink_px": r_core,
                  "soft_matte_used": use_soft}

        if not use_soft and not bool(present.any()):
            reasons = (["The subject you named was not found; the matte is empty"] if asked
                       else ["Nothing was found to cut out; name the subject"])
            report.update(mode="empty", notes=[], needs_review=True, review_reasons=reasons)
            return (torch.zeros_like(sam), json.dumps(report))
        if use_soft and ((wired and not asked) or not bool(present.any())):
            reasons = (["The subject you named was not found; the cutout is the model's own choice of main subject"]
                       if asked else [])
            notes = [] if asked else ["No subject named: the cutout is the main subject"]
            report.update(mode="soft_matte_only", notes=notes, needs_review=bool(reasons), review_reasons=reasons)
            return (ag, json.dumps(report))

        # frames without the subject: bridge short gaps by holding the nearest mask, leave long absences empty
        idx = torch.nonzero(present).flatten().tolist()
        held, absent_ranges = 0, []
        i = 0
        while i < n:
            if bool(present[i]):
                i += 1
                continue
            j = i
            while j < n and not bool(present[j]):
                j += 1
            gap = j - i
            bounded = i > 0 or j < n
            if gap <= hold_frames and bounded:
                for f in range(i, j):
                    near = min(idx, key=lambda q: abs(q - f))
                    sam[f] = sam[near]
                held += gap
            else:
                absent_ranges.append([i, j - 1])
            i = j
        absent = torch.zeros(n, dtype=torch.bool)
        for a, b in absent_ranges:
            absent[a:b + 1] = True
        have = ~absent

        def range_notes(notes):
            if absent_ranges:
                notes.append("The subject is not in frames " + ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in absent_ranges)
                             + "; those frames are empty")
            if held:
                notes.append(f"The subject was briefly lost in {held} frame(s); the nearest mask was held")
            return notes

        if not use_soft:  # subject mask only, lightly feathered
            out = F.avg_pool2d(sam[:, None], 3, stride=1, padding=1)[:, 0]
            out = torch.where(absent[:, None, None], torch.zeros_like(out), out)
            notes = range_notes(["No subject named: the cutout is what the tracker picked"] if wired and not asked else [])
            report.update(mode="subject_mask", frames_without_subject=int(absent.sum()), frames_held=held,
                          absent_frames=absent_ranges, notes=notes, needs_review=False, review_reasons=[])
            return (out.clamp(0, 1).contiguous(), json.dumps(report))

        gate = _morph(sam, r_gate, "dil")
        deep = _morph(sam, r_deep, "ero") > 0.5
        # parts of the subject the soft matte missed (e.g. a second named object it didn't pick): tracker outline there
        miss_lo, miss_hi = 0.16 * min_interior, 0.48 * min_interior
        cov = _local_coverage(ag, sam, deep, short)
        missed = ((miss_hi - cov) / max(miss_hi - miss_lo, 1e-6)).clamp(0, 1) * have[:, None, None].float()
        dsel = deep & have[:, None, None]
        covered = dsel & (missed < 0.5)
        if bool(covered.any()):
            interior = _median01(ag[covered])
        else:
            interior = _median01(ag[dsel]) if bool(dsel.any()) else _median01(ag[(sam > 0.5) & have[:, None, None]])
        off_target = float((ag * (1 - gate))[have].sum() / ag[have].sum().clamp_min(1e-6))

        reasons, notes = [], []
        items = AutoMatteCountPromptItems.parse(subject_text or "")
        multi = len(items) > 1 or any(c > 1 for _, c in items)
        # clearly missed even when see-through is wanted: a see-through subject is faint, not absent
        if interior < (0.2 * min_interior if keep_transparency else min_interior):
            # the soft matte did not cover the subject: use the subject mask, lightly feathered
            out = F.avg_pool2d(sam[:, None], 3, stride=1, padding=1)[:, 0]
            mode = "subject_mask"
            reasons.append(f"The soft matte did not cover the subject ({interior:.2f}); the edges come from the tracker and are harder")
        elif keep_transparency or interior >= solid_threshold:
            out = ag * gate
            mode = "soft_edges"
        else:
            core = _morph(sam, r_core, "ero")
            out = torch.maximum(core, (ag / max(interior, 1e-3)).clamp(0, 1) * gate)
            mode = "interior_repaired"
            notes.append(f"The soft matte was partly see-through inside the subject ({interior:.2f}); the inside was filled solid")
        missed_share = float((missed * sam).sum() / sam[have].sum().clamp_min(1.0))
        if mode != "subject_mask" and missed_share > 0.005:
            feather = F.avg_pool2d(sam[:, None], 3, stride=1, padding=1)[:, 0]
            out = missed * torch.maximum(out, feather) + (1 - missed) * out
            if not multi and missed_share > 0.5:
                reasons.append(f"The soft matte covered only {1 - missed_share:.0%} of the subject; the rest has the "
                               "tracker's harder edges")
            elif missed_share > 0.02:
                notes.append(f"The soft matte did not cover part of the subject ({missed_share:.0%}, e.g. a second named "
                             "object); there the edges come from the tracker and are harder")
        out = torch.where(absent[:, None, None], torch.zeros_like(out), out)

        after = float(out[dsel].mean()) if bool(dsel.any()) else None
        range_notes(notes)
        if off_target > 0.25:
            notes.append("The soft matte also covered things outside the subject; they were removed")
        report.update(mode=mode, interior_soft_matte=round(interior, 3), missed_share=round(missed_share, 4),
                      interior_after=None if after is None else round(after, 3),
                      off_target_share=round(off_target, 4), frames_without_subject=int(absent.sum()),
                      frames_held=held, absent_frames=absent_ranges,
                      notes=notes, needs_review=bool(reasons), review_reasons=reasons)
        return (out.clamp(0, 1).contiguous(), json.dumps(report))


class AutoMatteRestore:
    """Undo AutoMatte Fit on a mask: drop padded frames and pixels, then scale
    back to the source size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "fit": ("AUTOMATTE_FIT",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    def run(self, mask, fit):
        m = mask[: fit["orig_n"], : fit["content_h"], : fit["content_w"]].float()
        m = _resize_frames(m[..., None], fit["orig_h"], fit["orig_w"], "bilinear")[..., 0]
        return (m.clamp(0, 1).contiguous(),)


class AutoMatteToRGBA:
    """Source frames + matte -> RGBA frames with straight (unpremultiplied) alpha."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "matte": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    def run(self, images, matte):
        n, h, w, _ = images.shape
        m = matte.float()
        if m.shape[1:] != (h, w):
            m = _resize_frames(m[..., None], h, w, "bilinear")[..., 0]
        if m.shape[0] < n:
            m = torch.cat([m, m[-1:].repeat(n - m.shape[0], 1, 1)], 0)
        return (torch.cat([images[..., :3], m[:n, ..., None].clamp(0, 1)], -1).contiguous(),)


class AutoMatteCountPromptItems:
    """Read a comma-separated prompt list with optional ':N' counts ("cake:2, apron, window panel:4") the way
    text-prompted detectors such as SAM 3 read it (parentheses dropped, an item without a count = default_count)
    and return the total, clamped to 1..max_total: e.g. for a tracker's max-objects setting. Also returns the clean
    item list and a JSON block for the report (wire it to Save Report's `extra`). `source` (optional) says where the
    text came from (e.g. typed / auto) and goes into the JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "max_total": ("INT", {"default": 64, "min": 1, "max": 4096, "tooltip": "Upper limit for the total."}),
                "default_count": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "Count for an item without ':N'."}),
            },
            "optional": {
                "source": ("STRING", {"forceInput": True, "tooltip": "Where the text came from; copied into the JSON."}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("total", "items", "json")
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    @staticmethod
    def parse(text, default_count=1):
        parts = [p.strip() for p in (text or "").replace("(", "").replace(")", "").split(",") if p.strip()]
        out = []
        for part in parts:
            m = re.match(r"^(.+?)\s*:\s*([\d.]+)\s*$", part)
            out.append((m.group(1).strip(), max(1, round(float(m.group(2))))) if m else (part, default_count))
        return out

    def run(self, text, max_total=64, default_count=1, source=None):
        items = self.parse(text, default_count)
        asked = sum(n for _, n in items) if items else default_count
        total = int(min(max(asked, 1), max_total))
        notes = []
        if len(items) > 1 or any(n > 1 for _, n in items):
            notes.append(f"Up to {total} objects are followed in total, shared by all subjects (not per subject)")
        if asked > max_total:
            notes.append(f"{asked} objects were asked for; capped at {max_total}")
        info = {"subject_items": [{"text": t, "count": n} for t, n in items], "max_objects": total}
        if source is not None:
            info["subject_source"] = source
        if notes:
            info["notes"] = notes
        return (total, ", ".join(t if n == default_count else f"{t}:{n}" for t, n in items), json.dumps(info))


class AutoMatteSaveReport:
    """Write the Combine report as a JSON file in the output folder and return it as a file
    output, so API runners (comfy deploy run, Brand Brain) download it with the images.
    `extra` (optional JSON object) is merged in: its notes / review_reasons are appended, other keys are added
    when the report doesn't have them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "automatte/report"})},
                "optional": {"extra": ("STRING", {"forceInput": True, "tooltip": "JSON object merged into the report."})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "AutoMatte"

    @staticmethod
    def merge(data, extra):
        try:
            more = json.loads(extra) if extra else {}
        except Exception:
            more = {"extra": extra}
        if not isinstance(more, dict):
            more = {"extra": more}
        for k, v in more.items():
            if k in ("notes", "review_reasons") and isinstance(v, list):
                data[k] = list(data.get(k) or []) + v
            elif k not in data:
                data[k] = v
        return data

    def run(self, report, filename_prefix, extra=None):
        import os
        import folder_paths

        try:
            data = json.loads(report)
        except Exception:
            data = {"report": report}
        data = self.merge(data, extra)
        out_dir = folder_paths.get_output_directory()
        folder, name = os.path.split(filename_prefix.strip().strip("/") or "report")
        full = os.path.join(out_dir, folder)
        os.makedirs(full, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(full, f"{name}_{i:05d}.json")):
            i += 1
        fname = f"{name}_{i:05d}.json"
        with open(os.path.join(full, fname), "w") as f:
            json.dump(data, f, indent=1)
        return {"ui": {"files": [{"filename": fname, "subfolder": folder, "type": "output"}],
                       "text": [json.dumps(data, indent=1)]}}


NODE_CLASS_MAPPINGS = {
    "AutoMatteFit": AutoMatteFit,
    "AutoMatteCombine": AutoMatteCombine,
    "AutoMatteRestore": AutoMatteRestore,
    "AutoMatteToRGBA": AutoMatteToRGBA,
    "AutoMatteSaveReport": AutoMatteSaveReport,
    "AutoMatteCountPromptItems": AutoMatteCountPromptItems,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteFit": "AutoMatte: Fit Frames",
    "AutoMatteCombine": "AutoMatte: Combine Subject Mask + Soft Matte",
    "AutoMatteRestore": "AutoMatte: Restore Size",
    "AutoMatteToRGBA": "AutoMatte: To RGBA",
    "AutoMatteSaveReport": "AutoMatte: Save Report",
    "AutoMatteCountPromptItems": "AutoMatte: Count Prompt Items",
}
