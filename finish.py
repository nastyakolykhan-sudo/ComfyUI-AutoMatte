"""AutoMatte finishing nodes on a layer (a matte): grade only the subject or only the background, and hide things
(clearance: faces without a release, third-party logos, number plates) by blur or pixelation that follows them.

- AutoMatteLayerGrade  exposure (stops, in linear light), contrast (around the layer's own mid-tone), saturation
                       and warmth (OKLab), applied to the subject or to the background through the soft matte.
- AutoMatteObscure     blur or pixelate inside a matte, sized from each frame's object (so a near face and a far
                       face get the same look), feathered; or a solid fill.
- AutoMatteBackgroundSwap  the subject over a new background: edge rebuild, light wrap, harmonize, and an automatic
                       contact shadow where the subject stands on something (no model: the matte's own silhouette).
- AutoMatteBackgroundFrame  one frame of the new background (for a yes/no vision question about it).

Only torch; no model weights. Pixels outside the edited layer are returned unchanged.
"""

import json
import math

import torch
import torch.nn.functional as F

from .nodes import _device, _resize_frames, CHUNK, GPU_CHUNK_PX, chunk_frames, make_room, progress
from .layers import srgb_to_ok, ok_to_srgb, _fit_mask, _box


def _lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(0, 1) + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _mask_chunk(matte, i, j, h, w):
    """Frames i..j-1 of a matte as (j-i, h, w) in [0, 1]: resized to the frame, padded with its last frame or cut
    (as _fit_mask, one chunk at a time, so no full-clip copy is made)."""
    idx = torch.arange(i, j).clamp(max=matte.shape[0] - 1)
    m = matte[idx].float()
    if m.shape[1:] != (h, w):
        m = _resize_frames(m[..., None], h, w, "bilinear")[..., 0]
    return m.clamp(0, 1)


# shared with the other layer nodes (nodes.py): GPU chunks sized by pixels, room made before the first chunk
_chunk_frames, _make_room, _progress = chunk_frames, make_room, progress


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
        # GPU chunks sized by pixels (1 frame at 4K) after asking ComfyUI to unload cached models, masks fitted per
        # chunk (no full-clip copies), one OKLab conversion per frame, statistics from a strided pixel sample, and
        # progress per chunk.
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        back = str(target).lower().startswith("back")
        rep = {"edit": "layer_grade", "target": target, "exposure_stops": exposure, "contrast": contrast,
               "saturation": saturation, "warmth": warmth}
        if not (exposure or contrast or saturation or warmth):
            rep["notes"] = ["All grade values are 0: the clip is returned unchanged"]
            return (images, json.dumps(rep))
        dev = _device()
        s = 4 if min(h, w) >= 512 else 1  # statistics stride
        k = _chunk_frames(h, w)
        _make_room(dev, k * h * w)
        chunks = [(i, min(n, i + k)) for i in range(0, n, k)]
        pbar = _progress(2 * len(chunks))

        def layer(i, j):
            m = _mask_chunk(matte, i, j, h, w).to(dev)
            return 1 - m if back else m

        # the layer's median lightness over the whole clip (contrast pivot), from a strided sample
        samples = []
        for i, j in chunks:
            sel = layer(i, j)[:, ::s, ::s] > 0.5
            if bool(sel.any()):
                L = srgb_to_ok(images[i:j, ::s, ::s, :3].to(dev).float())[..., 0][sel]
                samples.append(L[torch.linspace(0, L.numel() - 1, min(L.numel(), 50000), device=dev).long()].cpu())
            if pbar:
                pbar.update(1)
        pivot = float(torch.cat(samples).median()) if samples else 50.0
        gain = 2.0 ** exposure
        k_con = 1.0 + contrast
        k_sat = max(0.0, 1.0 + saturation)
        out = images.clone()
        lsum = [0.0, 0.0, 0]
        for i, j in chunks:
            mm = layer(i, j)[..., None]
            if bool((mm > 0).any()):
                rgb = images[i:j, ..., :3].to(dev).float()
                x = _srgb(_lin(rgb) * gain) if exposure else rgb
                ok = srgb_to_ok(x)
                L = pivot + (ok[..., 0] - pivot) * k_con
                a = ok[..., 1] * k_sat + 2.0 * warmth
                b = ok[..., 2] * k_sat + 10.0 * warmth
                del ok, x
                new = ok_to_srgb(torch.stack([L.clamp(0, 100), a, b], -1))
                del L, a, b
                res = torch.where(mm > 0, rgb + (new - rgb) * mm, rgb)
                del new
                out[i:j, ..., :3] = res.to(out.dtype).cpu()
                sel = mm[:, ::s, ::s, 0] > 0.5
                if bool(sel.any()):
                    lsum[0] += float(srgb_to_ok(rgb[:, ::s, ::s])[..., 0][sel].mean())
                    lsum[1] += float(srgb_to_ok(res[:, ::s, ::s])[..., 0][sel].mean())
                    lsum[2] += 1
                del rgb, res
            if pbar:
                pbar.update(1)
        rep["contrast_pivot_L"] = round(pivot, 1)
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
    brightness part of the way towards the new background's (0 = none).

    `contact_shadow` (0 = never) darkens the new background where the subject stands on it, so it doesn't float: the
    subject's own silhouette squashed onto the ground behind its lowest point (flatter for tall subjects), blurred
    and faded with distance, plus a tight dark core under the parts that touch. It is skipped by itself when no
    contact is in shot (the subject runs off the bottom of the frame), and, when the optional yes/no answers are
    linked, when the subject isn't standing or resting on anything (`subject_on_surface`) or the new background has
    no ground where it would stand (`background_has_ground`); those answers are only asked for when the matte shows
    a contact. A subject that leaves the ground (a jump) gets a shadow that stays down and fades.
    Outputs the composite and a report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK",),
            },
            "optional": {
                "background": ("IMAGE", {"lazy": True, "tooltip": "One image or a clip's frames. Not computed when a background video is given."}),
                "background_video": ("VIDEO", {"tooltip": "A background clip (wins over the image when it is a real file)."}),
                "light_wrap": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "harmonize": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "contact_shadow": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "How dark the shadow under the subject is where it touches the ground (0 = never)."}),
                "subject_on_surface": ("BOOLEAN", {"lazy": True, "forceInput": True,
                                                   "tooltip": "Is the subject standing or resting on something? Asked only when the matte shows a contact."}),
                "background_has_ground": ("BOOLEAN", {"lazy": True, "forceInput": True,
                                                      "tooltip": "Does the new background have ground for it to stand on? Asked only after a yes above."}),
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

    def _geometry(self, matte, n, h, w):
        """Contact geometry, cached per matte (check_lazy_status and run both need it)."""
        key = (id(matte), n, h, w)
        if getattr(self, "_geo_key", None) != key:
            self._geo = _contact_geometry(matte, n, h, w)
            self._geo_key = key
        return self._geo

    def check_lazy_status(self, **kw):
        # ComfyUI passes a linked-but-unevaluated lazy input as None and leaves an unlinked one out
        from .layers import _video_given
        need = []
        if not _video_given(kw.get("background_video")) and "background" in kw and kw["background"] is None:
            need.append("background")
        if float(kw.get("contact_shadow", 0.7) or 0) > 0 and kw.get("matte") is not None and kw.get("images") is not None:
            ims = kw["images"]
            if self._geometry(kw["matte"], ims.shape[0], ims.shape[1], ims.shape[2])["ok"]:
                if "subject_on_surface" in kw and kw["subject_on_surface"] is None:
                    need.append("subject_on_surface")
                elif kw.get("subject_on_surface") is not False and "background_has_ground" in kw and kw["background_has_ground"] is None:
                    need.append("background_has_ground")
        return need

    def _read_video(self, video, n, h, w):
        """Background clip frames scaled to cover h x w while decoding (at most n frames; light on memory)."""
        frames = []
        try:
            import av
            with av.open(video.get_stream_source(), mode="r") as c:
                for fr in c.decode(c.streams.video[0]):
                    t = torch.from_numpy(fr.to_ndarray(format="rgb24")).float()[None] / 255.0
                    frames.append(self._cover(t, h, w)[0])
                    if len(frames) >= n:
                        break
        except Exception:
            frames = []
        if frames:
            return torch.stack(frames, 0)
        return self._cover(video.get_components().images[:n, ..., :3].float(), h, w)

    def run(self, images, matte, background=None, background_video=None, light_wrap=0.35, harmonize=0.3,
            contact_shadow=0.7, subject_on_surface=None, background_has_ground=None):
        from .layers import _video_given
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if _video_given(background_video):
            background = self._read_video(background_video, n, h, w)
        if background is None:
            raise ValueError("Give a background image or a background video.")
        m = _fit_mask(matte, n, h, w)
        dev = _device()
        short = min(h, w)
        r_fg = max(3, round(0.01 * short))
        r_wrap = max(3, round(0.015 * short))
        nb = background.shape[0]
        # contact shadow: decided once for the clip
        geo = self._geometry(matte, n, h, w) if contact_shadow > 0 else None
        sh_rep = {"strength": contact_shadow}
        if contact_shadow <= 0:
            use_shadow, sh_rep["reason"] = False, "Contact Shadow is 0"
        elif not geo["ok"]:
            use_shadow, sh_rep["reason"] = False, geo["reason"]
        elif subject_on_surface is False:
            use_shadow, sh_rep["reason"] = False, "the subject isn't standing or resting on anything (vision check)"
        elif background_has_ground is False:
            use_shadow, sh_rep["reason"] = False, "the new background has no ground for the subject to stand on (vision check)"
        else:
            use_shadow, sh_rep["reason"] = True, "the subject stands on something"
        sh_rep["applied"] = use_shadow
        if geo is not None:
            sh_rep.update({k: geo[k] for k in ("frames_with_contact", "frames_with_subject", "squash", "airborne_frames")})
        if subject_on_surface is not None:
            sh_rep["subject_on_surface"] = bool(subject_on_surface)
        if background_has_ground is not None:
            sh_rep["background_has_ground"] = bool(background_has_ground)
        sh_dark = []
        # colour statistics for harmonize: subject vs new background (whole clip, OKLab means)
        sub_s, bg_s = [], []
        ck = _chunk_frames(h, w)
        _make_room(dev, ck * h * w)
        for i in range(0, n, ck):
            j = min(n, i + ck)
            mm = m[i:j].to(dev) > 0.5
            if bool(mm.any()):   # pixel-weighted: the same mean whatever the chunk size
                px = srgb_to_ok(images[i:j, ..., :3].to(dev).float())[mm]
                sub_s.append((px.sum(0).cpu(), px.shape[0]))
        for i in range(0, min(nb, 64), 8):
            bg_s.append(srgb_to_ok(self._cover(background[i:i + 8, ..., :3].to(dev).float(), 256, 256)).reshape(-1, 3).mean(0).cpu())
        sub_mean = (torch.stack([s for s, _ in sub_s]).sum(0) / sum(c for _, c in sub_s)) if sub_s else torch.zeros(3)
        bg_mean = torch.stack(bg_s).mean(0) if bg_s else torch.zeros(3)
        shift = torch.zeros(3)
        if harmonize > 0 and sub_s:
            shift[0] = harmonize * 0.5 * float(bg_mean[0] - sub_mean[0])          # half of the lightness gap
            shift[1:] = harmonize * (bg_mean[1:] - sub_mean[1:]) * 0.5            # half of the colour cast
        shift = shift.to(dev)
        out = torch.empty_like(images[..., :3])
        for i in range(0, n, ck):
            j = min(n, i + ck)
            rgb = images[i:j, ..., :3].to(dev).float()
            mm = m[i:j].to(dev)
            idx = torch.arange(i, j) % nb
            bg = self._cover(background[idx][..., :3].to(dev).float(), h, w)
            if use_shadow:
                sa = _contact_shadow_alpha(m[i:j].to(dev), geo, i, contact_shadow, h, w)
                if sa is not None:
                    bg = bg * (1 - sa[..., None])
                    sh_dark.append(float(sa.amax()))
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
        if use_shadow:
            sh_rep["max_darkening"] = round(max(sh_dark), 2) if sh_dark else 0.0
        rep = {"edit": "background_swap", "background_frames": nb, "light_wrap": light_wrap, "harmonize": harmonize,
               "subject_lightness": round(float(sub_mean[0]), 1), "background_lightness": round(float(bg_mean[0]), 1),
               "harmonize_shift": [round(float(v), 2) for v in shift.cpu()], "contact_shadow": sh_rep, "notes": []}
        if nb > 1 and nb < n:
            rep["notes"].append(f"The background clip has {nb} frames for {n}: it loops")
        if contact_shadow > 0 and not use_shadow:
            rep["notes"].append("No contact shadow: " + sh_rep["reason"])
        return (out, json.dumps(rep))


class AutoMatteBackgroundFrame:
    """One frame of the new background (the background clip's first frame, or the image), e.g. for a yes/no vision
    question about it. The image is not computed when a background clip is given."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "background": ("IMAGE", {"lazy": True}),
                "background_video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def check_lazy_status(self, **kw):
        from .layers import _video_given
        if _video_given(kw.get("background_video")):
            return []
        return ["background"] if "background" in kw and kw["background"] is None else []

    def run(self, background=None, background_video=None):
        from .layers import _video_given
        if _video_given(background_video):
            try:
                import av
                with av.open(background_video.get_stream_source(), mode="r") as c:
                    for fr in c.decode(c.streams.video[0]):
                        return (torch.from_numpy(fr.to_ndarray(format="rgb24")).float()[None] / 255.0,)
            except Exception:
                return (background_video.get_components().images[:1, ..., :3].float(),)
        if background is None:
            raise ValueError("Give a background image or a background video.")
        return (background[:1, ..., :3].float(),)


def _smoothstep_t(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------- contact shadow
_SH_SHORT = 540        # the shadow is built at this short side (it is soft), then scaled up


def _sh_size(h, w):
    f = min(1.0, _SH_SHORT / min(h, w))
    return max(8, round(h * f)), max(8, round(w * f))


def _contact_geometry(matte, n, h, w):
    """Where the subject touches the ground, per frame, from the matte at the shadow's working size (the clip is n
    frames of h x w; the matte is resized straight to the working size and padded or cut to n frames, as _fit_mask).

    The subject is split into groups of columns separated by empty gaps (two products side by side, or people at
    different depths, each get their own contact line). Per group: lowest row (the contact line), height, width,
    and whether it runs off the bottom of the frame (then its contact isn't in shot). A clip gets a shadow only when
    the contact is in shot in at least 30% of the frames that show the subject."""
    hs, ws = _sh_size(h, w)
    nm = matte.shape[0]
    frames, ratios, heights = [], [], []
    present = 0
    for i in range(0, n, 16):
        idx = torch.arange(i, min(n, i + 16)).clamp_max(nm - 1)
        md = F.interpolate(matte[idx][:, None].float(), size=(hs, ws), mode="area")[:, 0] > 0.5
        for b in md:
            area = int(b.sum())
            if area < max(4, 0.0005 * hs * ws):
                frames.append([])
                continue
            present += 1
            col = b.sum(0)
            occ = (col > 0).nonzero().flatten().tolist()
            gap = max(2, round(0.02 * ws))
            runs, x0, prev = [], occ[0], occ[0]
            for x in occ[1:]:
                if x - prev > gap:
                    runs.append((x0, prev + 1))
                    x0 = x
                prev = x
            runs.append((x0, prev + 1))
            groups = []
            for x0, x1 in runs:
                sub = b[:, x0:x1]
                ga = int(sub.sum())
                if ga < max(4, 0.08 * area):           # specks and stray bits
                    continue
                rows = sub.sum(1)
                minc = max(1, round(0.02 * (x1 - x0)))
                ry = (rows >= minc).nonzero().flatten()
                if ry.numel() == 0:
                    continue
                ytop, ybot = int(ry[0]), int(ry[-1])
                cut = float(sub[hs - 2:].float().sum(1).mean()) >= max(2, 0.05 * (x1 - x0))
                groups.append({"x0": x0, "x1": x1, "ybot": ybot, "ytop": ytop, "hg": ybot - ytop + 1, "wg": x1 - x0,
                               "cut": cut, "area": ga})
            frames.append(groups)
            live = [g for g in groups if not g["cut"]]
            if live:
                big = max(live, key=lambda g: g["area"])
                ratios.append(big["wg"] / max(1, big["hg"]))
                heights.append(big["hg"])
    contact = [any(not g["cut"] for g in gr) for gr in frames]
    nc = sum(contact)
    geo = {"frames": frames, "contact": contact, "hs": hs, "ws": ws, "frames_with_subject": present,
           "frames_with_contact": nc, "squash": None, "airborne_frames": 0, "ok": False, "reason": ""}
    if present == 0:
        geo["reason"] = "no subject in the matte"
        return geo
    if nc < max(1, 0.3 * present):
        geo["reason"] = "the subject runs off the bottom of the frame, so where it touches the ground isn't in shot"
        return geo
    r = sorted(ratios)[len(ratios) // 2]
    hmed = float(sorted(heights)[len(heights) // 2])
    s = min(0.55, max(0.08, 0.5 * r))           # shadow height = half the subject's width, at most 55% of its height
    # a jump: the lowest point AND the top rise above where they were both shortly before and shortly after (a
    # camera push or tilt moves them one way only; in a walk the lowest point swaps feet but the head stays level)
    low = [max((g["ybot"] for g in gr if not g["cut"]), default=None) for gr in frames]
    top = [min((g["ytop"] for g in gr if not g["cut"]), default=None) for gr in frames]

    def rise(v, t):
        before = [v[u] for u in range(max(0, t - 8), t) if v[u] is not None]
        after = [v[u] for u in range(t + 1, min(n, t + 9)) if v[u] is not None]
        return min(max(before), max(after)) - v[t] if before and after else 0.0

    lift = [0.0] * n
    for t in range(n):
        if low[t] is not None:
            lift[t] = max(0.0, min(rise(low, t), rise(top, t)) - 0.03 * hmed)
    # per-frame weight: contact in shot (eased over a few frames, no popping) x fades while airborne
    cw = torch.tensor([1.0 if c else 0.0 for c in contact])
    if n > 1:
        cw = F.avg_pool1d(cw[None, None], 9, stride=1, padding=4, count_include_pad=False)[0, 0]
    wts = [float(cw[t]) * math.exp(-lift[t] / (0.08 * hmed)) if contact[t] else 0.0 for t in range(n)]
    geo.update(ok=True, squash=round(s, 3), hmed=hmed, lift=lift, weight=wts,
               airborne_frames=sum(1 for t in range(n) if lift[t] > 0.02 * hmed))
    return geo


def _contact_shadow_alpha(m, geo, i0, strength, h, w):
    """Shadow opacity (B,h,w) for frames i0.. of the clip, or None when none of them gets a shadow.

    Main shadow: the silhouette squashed by `squash` about each group's contact line (pushed down while the
    subject is in the air), faded with distance from the contact, blurred (soft). Core: only the parts within ~6% of
    the subject's height of the contact line, squashed to half, lightly blurred (dark, tight)."""
    b = m.shape[0]
    hs, ws = geo["hs"], geo["ws"]
    wts = geo["weight"][i0:i0 + b]
    if max(wts) <= 1e-3:
        return None
    dev = m.device
    s = geo["squash"]
    md = F.interpolate(m[:, None].float(), size=(hs, ws), mode="area")             # (B,1,hs,ws)
    yb = torch.full((b, ws), -1e4, device=dev)       # contact row per column (-1e4: no shadow in this column)
    hg = torch.ones((b, ws), device=dev)
    drop = 0.004 * min(hs, ws)
    for k in range(b):
        for g in geo["frames"][i0 + k]:
            if g["cut"]:
                continue
            yb[k, g["x0"]:g["x1"]] = g["ybot"]
            hg[k, g["x0"]:g["x1"]] = g["hg"]
    yy = torch.arange(hs, device=dev, dtype=torch.float32)[None, :, None]           # (1,hs,1)
    xx = torch.arange(ws, device=dev, dtype=torch.float32)[None, None, :]
    # the shadow stays on the ground while the subject is in the air (anchor pushed down by the lift)
    lifts = torch.tensor([geo["lift"][i0 + k] for k in range(b)], device=dev)[:, None, None]
    ybc = yb[:, None, :]                                                             # (B,1,ws)
    hgc = hg[:, None, :]
    anchor = ybc + lifts + drop
    gx = ((xx + 0.5) / ws * 2 - 1).expand(b, hs, ws)

    def sample(src_y):
        gy = (src_y + 0.5) / hs * 2 - 1
        grid = torch.stack([gx, gy.expand(b, hs, ws)], -1)
        return F.grid_sample(md, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]

    valid = (ybc > -1e3).float()
    # main: squash by s about the contact line, fade over the shadow's length
    src = ybc - (anchor - yy) / s
    main = sample(src) * valid
    dist = (anchor - yy).clamp_min(0)
    fade = (1 - dist / (1.2 * s * hgc)).clamp(0, 1) ** 2
    main = main * fade
    # core: the bottom band only, squashed to half
    band = 0.06 * hgc
    src_c = ybc - (anchor - yy) / 0.5
    core = sample(src_c) * valid * (src_c >= ybc - band).float()
    hm = float(geo["hmed"])
    r1 = max(1, round(0.12 * s * hm))
    r2 = max(1, round(0.012 * hm))
    for _ in range(3):
        main = _box(main, r1)
    for _ in range(2):
        core = _box(core, r2)
    a = 1 - (1 - 0.75 * strength * main) * (1 - min(1.0, 1.15 * strength) * core)
    a = a * torch.tensor(wts, device=dev)[:, None, None]
    a = F.interpolate(a[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return a.clamp(0, 0.95)


NODE_CLASS_MAPPINGS["AutoMatteBackgroundSwap"] = AutoMatteBackgroundSwap
NODE_CLASS_MAPPINGS["AutoMatteBackgroundFrame"] = AutoMatteBackgroundFrame
NODE_DISPLAY_NAME_MAPPINGS["AutoMatteBackgroundSwap"] = "AutoMatte: Background Swap"
NODE_DISPLAY_NAME_MAPPINGS["AutoMatteBackgroundFrame"] = "AutoMatte: Background Frame"
