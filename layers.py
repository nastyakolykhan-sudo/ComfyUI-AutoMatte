"""AutoMatte layer edits: cheap, localized edits on a clip once a subject matte exists.

- AutoMatteGivenOrMade  use a matte video when one is given, otherwise the matte the graph makes
                        (lazy: the segmentation branch only runs when no matte is given).
- AutoMatteRecolour     recolour the subject to an exact colour (hex or name) inside the matte, or only
                        correct a subject that drifts from it (brand colour lock), with a CIEDE2000 report.

Colour maths in CIE Lab (D65, sRGB). Pixels outside the matte are returned unchanged, bit for bit.
No model weights, only torch (and PyAV, which ComfyUI already ships, to read a matte video).
"""

import json
import math
import os
import re

import torch
import torch.nn.functional as F

from .nodes import _device, _resize_frames, CHUNK

# ----------------------------------------------------------------------------------------------- colour names
_CSS = (
    "aliceblue f0f8ff antiquewhite faebd7 aqua 00ffff aquamarine 7fffd4 azure f0ffff beige f5f5dc bisque ffe4c4 "
    "black 000000 blanchedalmond ffebcd blue 0000ff blueviolet 8a2be2 brown a52a2a burlywood deb887 cadetblue 5f9ea0 "
    "chartreuse 7fff00 chocolate d2691e coral ff7f50 cornflowerblue 6495ed cornsilk fff8dc crimson dc143c cyan 00ffff "
    "darkblue 00008b darkcyan 008b8b darkgoldenrod b8860b darkgray a9a9a9 darkgreen 006400 darkgrey a9a9a9 "
    "darkkhaki bdb76b darkmagenta 8b008b darkolivegreen 556b2f darkorange ff8c00 darkorchid 9932cc darkred 8b0000 "
    "darksalmon e9967a darkseagreen 8fbc8f darkslateblue 483d8b darkslategray 2f4f4f darkslategrey 2f4f4f "
    "darkturquoise 00ced1 darkviolet 9400d3 deeppink ff1493 deepskyblue 00bfff dimgray 696969 dimgrey 696969 "
    "dodgerblue 1e90ff firebrick b22222 floralwhite fffaf0 forestgreen 228b22 fuchsia ff00ff gainsboro dcdcdc "
    "ghostwhite f8f8ff gold ffd700 goldenrod daa520 gray 808080 green 008000 greenyellow adff2f grey 808080 "
    "honeydew f0fff0 hotpink ff69b4 indianred cd5c5c indigo 4b0082 ivory fffff0 khaki f0e68c lavender e6e6fa "
    "lavenderblush fff0f5 lawngreen 7cfc00 lemonchiffon fffacd lightblue add8e6 lightcoral f08080 lightcyan e0ffff "
    "lightgoldenrodyellow fafad2 lightgray d3d3d3 lightgreen 90ee90 lightgrey d3d3d3 lightpink ffb6c1 "
    "lightsalmon ffa07a lightseagreen 20b2aa lightskyblue 87cefa lightslategray 778899 lightslategrey 778899 "
    "lightsteelblue b0c4de lightyellow ffffe0 lime 00ff00 limegreen 32cd32 linen faf0e6 magenta ff00ff maroon 800000 "
    "mediumaquamarine 66cdaa mediumblue 0000cd mediumorchid ba55d3 mediumpurple 9370db mediumseagreen 3cb371 "
    "mediumslateblue 7b68ee mediumspringgreen 00fa9a mediumturquoise 48d1cc mediumvioletred c71585 "
    "midnightblue 191970 mintcream f5fffa mistyrose ffe4e1 moccasin ffe4b5 navajowhite ffdead navy 000080 "
    "oldlace fdf5e6 olive 808000 olivedrab 6b8e23 orange ffa500 orangered ff4500 orchid da70d6 palegoldenrod eee8aa "
    "palegreen 98fb98 paleturquoise afeeee palevioletred db7093 papayawhip ffefd5 peachpuff ffdab9 peru cd853f "
    "pink ffc0cb plum dda0dd powderblue b0e0e6 purple 800080 rebeccapurple 663399 red ff0000 rosybrown bc8f8f "
    "royalblue 4169e1 saddlebrown 8b4513 salmon fa8072 sandybrown f4a460 seagreen 2e8b57 seashell fff5ee "
    "sienna a0522d silver c0c0c0 skyblue 87ceeb slateblue 6a5acd slategray 708090 slategrey 708090 snow fffafa "
    "springgreen 00ff7f steelblue 4682b4 tan d2b48c teal 008080 thistle d8bfd8 tomato ff6347 turquoise 40e0d0 "
    "violet ee82ee wheat f5deb3 white ffffff whitesmoke f5f5f5 yellow ffff00 yellowgreen 9acd32 "
    # common design names (approximate swatches, not standards)
    "cream fffdd0 rust b7410e burgundy 800020 mustard ffdb58 emerald 50c878 camel c19a6b charcoal 36454f "
    "mint 98ff98 terracotta e2725b cobalt 0047ab sky 87ceeb sage b2ac88 lilac c8a2c8 mauve e0b0ff ochre cc7722"
).split()
COLOUR_NAMES = dict(zip(_CSS[0::2], _CSS[1::2]))


def parse_colour(text):
    """'#2E7D32', '2e7d32', '#abc', 'rgb(46, 125, 50)', 'forest green', 'Navy' -> (r, g, b) ints and '#RRGGBB'."""
    raw = (text or "").strip()
    t = raw.lower()
    m = re.fullmatch(r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", t)
    if m:
        rgb = tuple(min(255, int(v)) for v in m.groups())
    else:
        h = t.lstrip("#")
        if re.fullmatch(r"[0-9a-f]{6}", h):
            rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        elif re.fullmatch(r"[0-9a-f]{3}", h) and t.startswith("#"):
            rgb = tuple(int(c * 2, 16) for c in h)
        else:
            key = re.sub(r"[\s\-_']", "", t)
            if key not in COLOUR_NAMES:
                raise ValueError(f"Colour '{raw}' isn't a hex code (#2E7D32), rgb(46, 125, 50) or a known colour name "
                                 f"(CSS names such as navy or forest green).")
            hx = COLOUR_NAMES[key]
            rgb = tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))
    return rgb, "#%02X%02X%02X" % rgb


# ------------------------------------------------------------------------------------------------- Lab maths
_WHITE = (0.95047, 1.0, 1.08883)
_RGB2XYZ = ((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750), (0.0193339, 0.1191920, 0.9503041))
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def srgb_to_lab(rgb):
    """(..., 3) sRGB in [0, 1] -> (..., 3) Lab."""
    c = rgb.clamp(0, 1)
    c = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = torch.tensor(_RGB2XYZ, dtype=c.dtype, device=c.device)
    xyz = c @ m.T / torch.tensor(_WHITE, dtype=c.dtype, device=c.device)
    f = torch.where(xyz > _EPS, xyz.clamp_min(1e-12) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab):
    """(..., 3) Lab -> (..., 3) sRGB in [0, 1] (out-of-gamut values clipped)."""
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    f3 = f ** 3
    xyz = torch.where(f3 > _EPS, f3, (116 * f - 16) / _KAPPA) * torch.tensor(_WHITE, dtype=lab.dtype, device=lab.device)
    m = torch.linalg.inv(torch.tensor(_RGB2XYZ, dtype=torch.float64)).to(lab.dtype).to(lab.device)
    c = xyz @ m.T
    c = torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055)
    return c.clamp(0, 1)


def rgb255_to_lab(rgb):
    return srgb_to_lab(torch.tensor([v / 255.0 for v in rgb], dtype=torch.float64)).tolist()


def lab_to_hex(lab):
    rgb = lab_to_srgb(torch.tensor(lab, dtype=torch.float64)).tolist()
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb)


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference between two Lab triples (floats)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    cb = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(cb ** 7 / (cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0
    dl, dc = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        dh = dh - 360 if dh > 180 else dh + 360 if dh < -180 else dh
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    lbp, cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    rc = 2 * math.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / math.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -math.sin(math.radians(2 * dtheta)) * rc
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dH / sh) ** 2 + rt * (dc / sc) * (dH / sh))


def _smoothstep(x, e0, e1):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _quantile(x, q):
    """torch.quantile on at most 1M evenly spaced values (it refuses very large inputs)."""
    x = x.float().flatten()
    if x.numel() > 1_000_000:
        x = x[torch.linspace(0, x.numel() - 1, 1_000_000).long()]
    return float(torch.quantile(x, q))


def _hue_diff(h, h0):
    """Signed angle h - h0 in radians, wrapped to (-pi, pi]."""
    return torch.remainder(h - h0 + math.pi, 2 * math.pi) - math.pi


def _fit_mask(m, n, h, w):
    """(Nm,Hm,Wm) mask -> (n,h,w): resized, frames padded with the last one or cut."""
    m = m.float()
    if m.shape[1:] != (h, w):
        m = _resize_frames(m[..., None], h, w, "bilinear")[..., 0]
    if m.shape[0] < n:
        m = torch.cat([m, m[-1:].repeat(n - m.shape[0], 1, 1)], 0)
    return m[:n].clamp(0, 1).contiguous()


# --------------------------------------------------------------------------------------- Given or Made matte
def _video_given(video):
    """True when a VIDEO input really points at a file (an empty WorkflowIO Video Input resolves to the input folder)."""
    if video is None:
        return False
    try:
        src = video.get_stream_source()
    except Exception:
        return True
    if isinstance(src, (str, bytes, os.PathLike)):
        return os.path.isfile(src)
    return src is not None


def _read_matte_video(video):
    """Matte video -> (N,H,W) float in [0,1], read frame by frame as grey (light on memory)."""
    frames = []
    try:
        import av
        src = video.get_stream_source()
        with av.open(src, mode="r") as c:
            stream = c.streams.video[0]
            for fr in c.decode(stream):
                frames.append(torch.from_numpy(fr.to_ndarray(format="rgb24")).float().mean(-1).to(torch.uint8))
    except Exception:
        frames = []
    if frames:
        m = torch.stack(frames, 0).float() / 255.0
    else:  # fall back to ComfyUI's own decoder
        m = video.get_components().images[..., :3].float().mean(-1)
    return ((m - 0.02) / 0.96).clamp(0, 1)  # compression noise near black/white -> exact 0/1


class AutoMatteGivenOrMade:
    """Pick the subject matte for an edit: the matte video when one is given (e.g. saved from an earlier
    segmentation run of the same clip), otherwise the matte the graph makes. `matte`, `report` and `extra` are lazy:
    whatever makes them only runs when no matte video is given. A given matte is checked against the clip (same
    aspect, about the same frame count), scaled to the frame size and padded/cut to the frame count.
    Outputs the matte, a JSON report (the segmentation report + extra when made) and `made`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames: sets the size and frame count."}),
                "matte": ("MASK", {"lazy": True, "tooltip": "Matte made by the graph; only computed when no matte video is given."}),
            },
            "optional": {
                "matte_video": ("VIDEO", {"tooltip": "A saved matte of the same clip (white = subject). Empty = make one."}),
                "report": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "Segmentation report (JSON); only computed when the matte is made."}),
                "extra": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "More JSON merged into the report when the matte is made."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "BOOLEAN")
    RETURN_NAMES = ("matte", "report", "made")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def check_lazy_status(self, images, matte=None, matte_video=None, **kw):
        if _video_given(matte_video):
            return []
        need = ["matte"] if matte is None else []
        return need + [k for k in ("report", "extra") if k in kw and kw[k] is None]

    def run(self, images, matte=None, matte_video=None, report=None, extra=None):
        from .nodes import AutoMatteSaveReport
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if _video_given(matte_video):
            m = _read_matte_video(matte_video)
            nm, hm, wm = m.shape
            if abs(math.log((wm / hm) / (w / h))) > 0.02:
                raise ValueError(f"The matte is {wm}x{hm} but the video is {w}x{h}: use a matte made from this clip.")
            if abs(nm - n) > max(2, round(0.02 * n)):
                raise ValueError(f"The matte has {nm} frames but the video has {n}: use a matte made from this clip.")
            rep = {"matte_source": "given", "matte_frames": nm, "matte_size": [wm, hm]}
            if nm != n:
                rep["notes"] = [f"The matte had {nm} frames for {n} video frames; the last frame was "
                                f"{'repeated' if nm < n else 'dropped'}"]
            return (_fit_mask(m, n, h, w), json.dumps(rep), False)
        if matte is None:
            raise ValueError("No matte video was given and no matte was made.")
        try:
            rep = json.loads(report) if report else {}
        except Exception:
            rep = {"segmentation_report": report}
        rep = AutoMatteSaveReport.merge(rep if isinstance(rep, dict) else {"segmentation_report": rep}, extra)
        rep["matte_source"] = "made"
        return (_fit_mask(matte, n, h, w), json.dumps(rep), True)


# --------------------------------------------------------------------------------------------------- OKLab
# Edits run in OKLab (hue-linear: a blue stays blue when it gets lighter or darker), scaled to CIE Lab-like units
# (L x100, a/b x400) so chroma thresholds read like Lab's. CIEDE2000 is still measured in CIE Lab.
_OK_M1 = ((0.4122214708, 0.5363325363, 0.0514459929), (0.2119034982, 0.6806995451, 0.1073969566),
          (0.0883024619, 0.2817188376, 0.6299787005))
_OK_M2 = ((0.2104542553, 0.7936177850, -0.0040720468), (1.9779984951, -2.4285922050, 0.4505937099),
          (0.0259040371, 0.7827717662, -0.8086757660))
_OK_M2_INV = ((1.0, 0.3963377774, 0.2158037573), (1.0, -0.1055613458, -0.0638541728),
              (1.0, -0.0894841775, -1.2914855480))
_OK_M1_INV = ((4.0767416621, -3.3077115913, 0.2309699292), (-1.2684380046, 2.6097574011, -0.3413193965),
              (-0.0041960863, -0.7034186147, 1.7076147010))
_OK_SCALE = (100.0, 400.0, 400.0)


def _t(m, x):
    return torch.tensor(m, dtype=x.dtype, device=x.device)


def srgb_to_ok(rgb):
    """(..., 3) sRGB in [0, 1] -> scaled OKLab (L 0..100, a/b roughly -130..130)."""
    c = rgb.clamp(0, 1)
    c = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = c @ _t(_OK_M1, c).T
    lms = torch.sign(lms) * lms.abs().clamp_min(1e-12) ** (1 / 3)
    return (lms @ _t(_OK_M2, c).T) * _t(_OK_SCALE, c)


def ok_to_srgb(ok):
    """Scaled OKLab -> sRGB in [0, 1] (out-of-gamut values clipped)."""
    x = ok / _t(_OK_SCALE, ok)
    lms = x @ _t(_OK_M2_INV, ok).T
    lms = lms * lms * lms
    c = lms @ _t(_OK_M1_INV, ok).T
    c = torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055)
    return c.clamp(0, 1)


def _fade(L):
    """Chroma allowed at lightness L: none at pure black/white, full in between (whites stay white)."""
    return ((100 - L) / 10).clamp(0, 1) * (L / 6).clamp(0, 1)


def _box_nchw(t, r):
    """Separable box blur (rows then columns) of (N,C,H,W), edges normalised: same result as a 2D box, O(r) not O(r^2)."""
    k = 2 * r + 1
    t = F.avg_pool2d(t, (1, k), stride=1, padding=(0, r), count_include_pad=False)
    return F.avg_pool2d(t, (k, 1), stride=1, padding=(r, 0), count_include_pad=False)


def _box(x, r):
    """Box blur of (N,H,W) or (N,H,W,C) with radius r (edges normalised)."""
    if x.dim() == 4:
        return _box_nchw(x.permute(0, 3, 1, 2), r).permute(0, 2, 3, 1)
    return _box_nchw(x[:, None], r)[:, 0]


def _guided(guide, p, r, eps=1e-3):
    """Edge-aware smoothing of p (N,H,W) along the guide's edges (He et al. guided filter)."""
    mi, mp = _box(guide, r), _box(p, r)
    a = (_box(guide * p, r) - mi * mp) / (_box(guide * guide, r) - mi * mi + eps)
    b = mp - a * mi
    return (_box(a, r) * guide + _box(b, r)).clamp(0, 1)


# ------------------------------------------------------------------------------------------------- Recolour
class AutoMatteRecolour:
    """Recolour the subject inside a matte to an exact colour.

    Recolour: the subject's main colour becomes the target (its mean lightness, hue and chroma, calibrated so the
      measured result lands on the colour); the light and dark variation is kept (compressed only if it would clip),
      so shading, reflections and print survive. On a coloured subject, parts that are already neutral (a white
      sole, a white label, specular highlights, black print) stay, a colour gradient keeps its fade, and parts in a
      clearly different hue (a gold cap on a blue bottle, a logo) are kept. A neutral or barely coloured subject
      (white, grey, black, denim) takes the colour everywhere except clearly coloured details.
    Correct: for a subject that should be this colour but drifts (e.g. generated footage): every frame's main
      colour is turned onto the target hue and chroma; lightness is untouched (the scene's light stays).
    Edits run in OKLab; statistics over the whole clip (Recolour) or per frame smoothed over 5 frames (Correct), so
    nothing flickers. Soft edges change only by the subject's own colour change (the background showing through an
    edge stays as it was), and outside the matte the frames are returned bit for bit. The report measures the
    subject's main colour before and after against the target in CIEDE2000."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK", {"tooltip": "White = the subject to recolour (soft edges are blended)."}),
                "colour": ("STRING", {"default": "#2E7D32", "tooltip": "Hex (#2E7D32), rgb(46, 125, 50) or a colour name (navy, forest green)."}),
                "mode": (["Recolour", "Correct"], {"default": "Recolour", "tooltip": "Recolour = change the subject to the colour; Correct = pull a subject that drifts back onto it."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    SAMPLES_PER_FRAME = 40000

    # -- statistics ---------------------------------------------------------------------------------------
    def _samples(self, images, matte, dev):
        """sRGB samples of the subject core (matte > 0.5): (K,3), (K,) frame index, per-frame core pixel counts."""
        px_all, idx, counts = [], [], []
        for i in range(0, images.shape[0], CHUNK):
            rgb = images[i:i + CHUNK, ..., :3].to(dev).float()
            m = matte[i:i + CHUNK].to(dev)
            for j in range(rgb.shape[0]):
                core = m[j] > 0.5
                k = int(core.sum().item())
                counts.append(k)
                if k == 0:
                    continue
                px = rgb[j][core]
                if k > self.SAMPLES_PER_FRAME:
                    px = px[torch.linspace(0, k - 1, self.SAMPLES_PER_FRAME, device=dev).long()]
                px_all.append(px.cpu())
                idx.append(torch.full((px.shape[0],), i + j, dtype=torch.long))
        if not px_all:
            return None, None, counts
        return torch.cat(px_all), torch.cat(idx), counts

    @staticmethod
    def _dominant(ok):
        """Main colour from OKLab samples -> dict(kind, h, c_ref | c_lim, share)."""
        a, b = ok[:, 1], ok[:, 2]
        C = torch.hypot(a, b)
        chromatic = C >= 6
        share = float(chromatic.float().mean())
        info = {"share": share}
        if share >= 0.2 and int(chromatic.sum()) >= 200:
            ab = torch.stack([a[chromatic], b[chromatic]], 1)
            bins, lo, hi = 110, -130.0, 130.0
            q = ((ab - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
            hist = torch.zeros(bins * bins)
            hist.index_add_(0, q[:, 0] * bins + q[:, 1], torch.ones(q.shape[0]))
            hist = hist.view(1, 1, bins, bins)
            for _ in range(2):
                hist = F.avg_pool2d(hist, 3, stride=1, padding=1, count_include_pad=False)
            p = int(hist.view(-1).argmax())
            pa, pb = lo + (p // bins + 0.5) * (hi - lo) / bins, lo + (p % bins + 0.5) * (hi - lo) / bins
            near = ((ab[:, 0] - pa) ** 2 + (ab[:, 1] - pb) ** 2) < 15 ** 2
            da, db = (float(ab[near, 0].mean()), float(ab[near, 1].mean())) if int(near.sum()) > 20 else (pa, pb)
            h0 = math.atan2(db, da)
            same = _hue_diff(torch.atan2(ab[:, 1], ab[:, 0]), h0).abs() < math.radians(25)
            c_ref = _quantile(C[chromatic][same], 0.75) if int(same.sum()) > 20 else math.hypot(da, db)
            if c_ref >= 18:
                info.update(kind="chromatic", h=h0, c_ref=c_ref)
                return info
            info.update(kind="neutral", c_lim=max(22.0, 1.8 * c_ref))
            return info
        info.update(kind="neutral", c_lim=22.0)
        return info

    @staticmethod
    def _weights(ok, d):
        """Selection weight w (pixel belongs to the main colour) and colour amount r."""
        C = torch.hypot(ok[..., 1], ok[..., 2])
        if d["kind"] == "chromatic":
            dh = _hue_diff(torch.atan2(ok[..., 2], ok[..., 1]), d["h"]).abs()
            return 1 - _smoothstep(dh, math.radians(30), math.radians(55)), (C / d["c_ref"]).clamp(0, 1.25)
        return 1 - _smoothstep(C, 0.6 * d["c_lim"], d["c_lim"]), torch.ones_like(C)

    @staticmethod
    def _recolour(ok, w, r, d, P):
        L, a, b = ok[..., 0], ok[..., 1], ok[..., 2]
        l_full = P["tl"] + P["l_off"] + P["s"] * (L - P["mean_l"])
        if d["kind"] == "chromatic":
            amt = r.clamp(max=1.0)
            l_new = L + (l_full - L) * amt
            dh = _hue_diff(torch.atan2(b, a), d["h"])
            h_new = P["th"] + P["h_off"] + 0.5 * dh * (dh.abs() < math.radians(30)).to(ok.dtype)
            c_new = P["tc"] * P["gain"] * r
        else:
            l_new = l_full
            h_new = torch.full_like(L, P["th"] + P["h_off"])
            c_new = torch.full_like(L, P["tc"] * P["gain"])
        l_new = l_new.clamp(0, 100)
        c_new = c_new * _fade(l_new)
        return torch.stack([l_new, c_new * torch.cos(h_new), c_new * torch.sin(h_new)], -1)

    @staticmethod
    def _correct(ok, d, fa, fb, P):
        """fa, fb: per-frame main colour (broadcastable to ok[..., 0])."""
        L, a, b = ok[..., 0], ok[..., 1], ok[..., 2]
        if d["kind"] == "chromatic":
            fh, fc = torch.atan2(fb, fa), torch.hypot(fa, fb).clamp_min(1.0)
            h_new = torch.atan2(b, a) + _hue_diff(torch.full_like(fh, P["th"]), fh)
            c_new = torch.hypot(a, b) * (P["tc"] / fc).clamp(0.5, 2.0)
            return torch.stack([L, c_new * torch.cos(h_new), c_new * torch.sin(h_new)], -1)
        return torch.stack([L, a + (P["ta"] - fa), b + (P["tb"] - fb)], -1)

    # -- the edit ------------------------------------------------------------------------------------------
    def run(self, images, matte, colour, mode="Recolour"):
        rgb_t, hex_t = parse_colour(colour)
        lab_t = rgb255_to_lab(rgb_t)
        ok_t = srgb_to_ok(torch.tensor([v / 255.0 for v in rgb_t], dtype=torch.float64)).tolist()
        mode = "Correct" if str(mode).strip().lower() in ("correct", "lock", "colour lock", "color lock") else "Recolour"
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        matte = _fit_mask(matte, n, h, w)
        dev = _device()
        report = {"edit": "recolour", "mode": mode, "colour_input": colour, "colour": hex_t,
                  "colour_lab": [round(v, 2) for v in lab_t], "notes": [], "review_reasons": []}
        P = {"tl": ok_t[0], "ta": ok_t[1], "tb": ok_t[2], "tc": math.hypot(ok_t[1], ok_t[2]),
             "th": math.atan2(ok_t[2], ok_t[1]), "l_off": 0.0, "gain": 1.0, "h_off": 0.0, "s": 1.0, "mean_l": 50.0}

        rgb_s, idx_s, counts = self._samples(images, matte, dev)
        report["frames_without_subject"] = int(sum(1 for k in counts if k == 0))
        if rgb_s is None or rgb_s.shape[0] < 50:
            report["review_reasons"].append("No subject in the matte: nothing was recoloured")
            report["needs_review"] = True
            return (images, json.dumps(report))
        ok_s = srgb_to_ok(rgb_s.float())
        d = self._dominant(ok_s)
        w_s, r_s = self._weights(ok_s, d)
        main = (w_s >= 0.5) & ((r_s >= 0.6) if d["kind"] == "chromatic" else torch.ones_like(w_s, dtype=torch.bool))
        if int(main.sum()) < 50:
            main = torch.ones_like(w_s, dtype=torch.bool)
        lab_main = srgb_to_lab(rgb_s[main].double())
        before = lab_main.mean(0).tolist()
        c_s = torch.hypot(ok_s[:, 1], ok_s[:, 2])
        kept_share = float(((w_s < 0.5) & (c_s >= 6)).float().mean())   # parts in another colour, kept
        neutral_kept = float(((c_s < 6) & ~main).float().mean()) if d["kind"] == "chromatic" else 0.0
        report.update(subject_colour_kind=d["kind"], subject_chromatic_share=round(d["share"], 3),
                      subject_colour_before=lab_to_hex(before),
                      delta_e00_before=round(ciede2000(before, lab_t), 2), kept_share=round(kept_share, 3),
                      neutral_kept_share=round(neutral_kept, 3))

        L_main = ok_s[main, 0]
        P["mean_l"] = float(L_main.mean())
        if mode == "Recolour":
            p2, p98 = _quantile(L_main, 0.02), _quantile(L_main, 0.98)
            s = min((97 - P["tl"]) / max(p98 - P["mean_l"], 1.0), (P["tl"] - 3) / max(P["mean_l"] - p2, 1.0), 1.0)
            P["s"] = max(s, 0.35)
            report["shading_kept"] = round(P["s"], 3)
            if P["s"] < 0.6:
                report["notes"].append("The subject's light and dark range was compressed to fit the colour")
            # calibrate on the samples so the measured main colour lands on the target (gamut clipping included)
            sub = ok_s[main]
            if sub.shape[0] > 200000:
                sub = sub[torch.linspace(0, sub.shape[0] - 1, 200000).long()]
            ws, rs = self._weights(sub, d)
            for _ in range(4):
                back = srgb_to_ok(ok_to_srgb(self._recolour(sub, ws, rs, d, P)))
                mL, ma, mb = back[:, 0].mean().item(), back[:, 1].mean().item(), back[:, 2].mean().item()
                P["l_off"] = max(-20.0, min(20.0, P["l_off"] + P["tl"] - mL))
                if P["tc"] > 2:
                    P["gain"] = max(0.5, min(1.8, P["gain"] * max(0.7, min(1.4, P["tc"] / max(math.hypot(ma, mb), 1e-3)))))
                    P["h_off"] += float(_hue_diff(torch.tensor(P["th"]), torch.tensor(math.atan2(mb, ma))))
        else:
            fa, fb, fl = (torch.full((n,), float("nan")) for _ in range(3))
            fi = idx_s[main]
            okm = ok_s[main]
            for f in torch.unique(fi).tolist():
                sel = fi == f
                if int(sel.sum()) >= 20:
                    fl[f], fa[f], fb[f] = okm[sel, 0].mean(), okm[sel, 1].mean(), okm[sel, 2].mean()
            valid = ~torch.isnan(fa)
            k5 = torch.ones(1, 1, 5)

            def smooth(x):
                num = F.conv1d(torch.where(valid, x, torch.zeros_like(x)).view(1, 1, -1), k5, padding=2).view(-1)
                den = F.conv1d(valid.float().view(1, 1, -1), k5, padding=2).view(-1)
                return torch.where(den > 0, num / den.clamp_min(1e-6), torch.full_like(x, float("nan")))
            sa, sb = smooth(fa), smooth(fb)
            sa = torch.where(torch.isnan(sa), torch.full_like(sa, P["ta"]), sa)
            sb = torch.where(torch.isnan(sb), torch.full_like(sb, P["tb"]), sb)
            de_f = []
            for f in torch.unique(fi).tolist():
                sel = fi == f
                if int(sel.sum()) >= 20:
                    lab_f = srgb_to_lab(rgb_s[main][sel].double()).mean(0).tolist()
                    de_f.append(ciede2000(lab_f, [lab_f[0], lab_t[1], lab_t[2]]))
            if de_f:
                report["delta_e00_hue_chroma_before"] = {"mean": round(sum(de_f) / len(de_f), 2), "max": round(max(de_f), 2)}
            if report["delta_e00_before"] > 25:
                report["review_reasons"].append("The subject is far from this colour: Correct only fixes drift; "
                                                "use Recolour to change the colour")

        # -- apply in chunks -------------------------------------------------------------------------------
        short = min(h, w)
        r_guide, r_fg = max(2, round(0.004 * short)), max(3, round(0.01 * short))
        out = images.clone()
        after = []
        for i in range(0, n, CHUNK):
            m = matte[i:i + CHUNK].to(dev).float()
            if not bool((m > 0).any()):
                continue
            rgb = images[i:i + CHUNK, ..., :3].to(dev).float()
            ok = srgb_to_ok(rgb)
            wsel, r = self._weights(ok, d)
            wsel = _guided(ok[..., 0] / 100.0, wsel, r_guide)
            # the subject's own colour near each edge pixel (interior colours spread outwards)
            core = (m > 0.9).float()
            den = _box(_box(core, r_fg), r_fg)
            fg = _box(_box(rgb * core[..., None], r_fg), r_fg) / den.clamp_min(1e-4)[..., None]
            fg = torch.where((den > 0.02)[..., None], fg, rgb)
            ok_fg = srgb_to_ok(fg)
            w_fg, r_fg_amt = self._weights(ok_fg, d)
            if mode == "Recolour":
                new = ok_to_srgb(self._recolour(ok, wsel, r, d, P))
                new_fg = ok_to_srgb(self._recolour(ok_fg, w_fg, r_fg_amt, d, P))
            else:
                fr = torch.arange(i, min(i + CHUNK, n))
                fa_c, fb_c = sa[fr].to(dev).view(-1, 1, 1), sb[fr].to(dev).view(-1, 1, 1)
                new = ok_to_srgb(self._correct(ok, d, fa_c, fb_c, P))
                new_fg = ok_to_srgb(self._correct(ok_fg, d, fa_c, fb_c, P))
            k = torch.where(den > 0.02, _smoothstep(m, 0.5, 0.95), torch.ones_like(m))[..., None]
            change = k * wsel[..., None] * (new - rgb) + (1 - k) * w_fg[..., None] * (new_fg - fg)
            res = (rgb + m[..., None] * change).clamp(0, 1)
            res = torch.where((m > 0)[..., None], res, rgb)  # exact outside the matte
            out[i:i + CHUNK, ..., :3] = res.to(out.dtype).cpu()
            sel = (m > 0.5) & (wsel >= 0.5)
            if d["kind"] == "chromatic":
                sel = sel & (r >= 0.6)
            if bool(sel.any()):
                px = res[sel]
                if px.shape[0] > 20000:
                    px = px[torch.linspace(0, px.shape[0] - 1, 20000, device=dev).long()]
                after.append(px.cpu())

        if after:
            lab_after = srgb_to_lab(torch.cat(after).double()).mean(0).tolist()
            report["subject_colour_after"] = lab_to_hex(lab_after)
            if mode == "Recolour":
                de = ciede2000(lab_after, lab_t)
                report["delta_e00_after"] = round(de, 2)
                if de > 3:
                    report["review_reasons"].append(f"The recoloured subject is {de:.1f} dE00 from the colour "
                                                    "(a very light, dark or saturated colour can't be reached on this subject)")
            else:
                report["delta_e00_hue_chroma_after"] = round(ciede2000(lab_after, [lab_after[0], lab_t[1], lab_t[2]]), 2)
        if kept_share > 0.1:
            report["notes"].append(f"{kept_share:.0%} of the subject is in another colour and was kept (a logo, a cap, "
                                   "skin); name the part to recolour (e.g. 'bottle body') to be sure")
        if neutral_kept > 0.1:
            report["notes"].append(f"White, grey and black parts ({neutral_kept:.0%} of the subject) kept their colour")
        if report["frames_without_subject"]:
            report["notes"].append(f"{report['frames_without_subject']} frames have no subject and are unchanged")
        report["needs_review"] = bool(report["review_reasons"])
        return (out, json.dumps(report))


# --------------------------------------------------------------------------------------------- Protect people
_PEOPLE_WORDS = {"person", "persons", "people", "man", "men", "woman", "women", "girl", "girls", "boy", "boys", "child",
                 "children", "kid", "kids", "baby", "model", "dancer", "face", "faces", "skin", "hair", "hand", "hands",
                 "arm", "arms", "leg", "legs", "lips", "lip", "nail", "nails", "eye", "eyes", "eyebrows", "body",
                 "neck", "head", "beard", "makeup", "make-up", "lipstick", "nail polish", "teeth", "tattoo", "feet", "foot"}


def names_people(text):
    words = set(re.findall(r"[a-z\-]+", (text or "").lower()))
    return bool(words & _PEOPLE_WORDS) or "nail polish" in (text or "").lower()


class AutoMatteProtect:
    """Take parts that must not change out of a matte, e.g. people's faces, necks, hands and hair when a garment or a
    held product is edited (a soft matte of a coat reaches into the neck and the hand). `protect` is lazy: whatever
    makes it (e.g. SAM on "face, neck, hand, hair") only runs when the subject text doesn't itself name people or a
    body part (then nothing is protected: "hair", "lips", "woman" are the subject). The protect mask is grown a
    little and feathered, then removed: matte * (1 - protect). `report` (JSON, optional) gets the protected share."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matte": ("MASK",),
                "protect": ("MASK", {"lazy": True, "tooltip": "What must stay as it is (white). Only computed when the subject doesn't name people."}),
            },
            "optional": {
                "subject_text": ("STRING", {"forceInput": True, "tooltip": "What is edited; naming people or a body part turns protection off."}),
                "report": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("matte", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Layers"

    def check_lazy_status(self, matte, protect=None, subject_text=None, report=None):
        if names_people(subject_text):
            return []
        return ["protect"] if protect is None else []

    def run(self, matte, protect=None, subject_text=None, report=None):
        try:
            rep = json.loads(report) if report else {}
        except Exception:
            rep = {"report": report}
        n, h, w = matte.shape
        if names_people(subject_text) or protect is None:
            rep["people_protected"] = False
            return (matte, json.dumps(rep))
        p = _fit_mask(protect, n, h, w)
        short = min(h, w)
        r = max(1, round(0.004 * short))
        k = 2 * r + 1
        p = F.max_pool2d(p[:, None], (1, k), stride=1, padding=(0, r))
        p = F.max_pool2d(p, (k, 1), stride=1, padding=(r, 0))[:, 0]
        p = _box(p, r).clamp(0, 1)
        before = float((matte > 0.5).float().sum()) or 1.0
        out = (matte.float() * (1 - p)).clamp(0, 1)
        share = 1 - float((out > 0.5).float().sum()) / before
        rep["people_protected"] = True
        rep["protected_share_of_matte"] = round(share, 4)
        if share > 0.002:
            notes = rep.setdefault("notes", [])
            notes.append(f"Faces, necks, hands and hair inside the matte ({share:.1%} of it) were kept as they are")
        return (out.contiguous(), json.dumps(rep))


NODE_CLASS_MAPPINGS = {
    "AutoMatteGivenOrMade": AutoMatteGivenOrMade,
    "AutoMatteRecolour": AutoMatteRecolour,
    "AutoMatteProtect": AutoMatteProtect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteGivenOrMade": "AutoMatte: Given or Made Matte",
    "AutoMatteRecolour": "AutoMatte: Recolour",
    "AutoMatteProtect": "AutoMatte: Protect People",
}
