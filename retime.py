"""AutoMatte retime: change the speed of a clip, or of one layer of it, without touching what doesn't change.

- AutoMatteTimeMap           which source moment each output frame shows: speed, speed ramps around a hero frame, a
                             target duration, frame-rate conform, and the layer modes (Subject Only, Background Only,
                             Hero Freeze). JSON, so the report and an agent can read it.
- AutoMattePickInterpModel   Auto / Fast / Large Motion between two core frame-interpolation models (core "Load Frame
                             Interpolation Model": RIFE, FILM). Lazy: only the chosen loader runs. Auto measures how
                             far things move between frames.
- AutoMatteGivenOrMadePlate  the clean plate (background without the subject) for the layer modes: a given video (Video
                             Object Removal's output), else a median plate when the camera is still and the subject
                             uncovers every background pixel, else the graph's own removal branch (lazy).
- AutoMatteRetime            renders the time map. Whole-number source times are the source frames, bit for bit;
                             in-betweens come from the core model at any timestep (RIFE carries the matte along with
                             the same flow); speed-ups get a synthetic 180-degree shutter. Layer modes recomposite
                             out = I(ts) + (1 - a(ts)) * (B(tb) - B(ts)) over a clean background B (the original's
                             pixels except under the subject and its shadow/light), which gives back the original
                             frame exactly wherever both layers show the same moment; a camera lock (similarity fit on
                             the background) keeps the subject in the scene or on the frame. Seamless loops (cut,
                             morph bridge or ping-pong). JSON report with review reasons.

The interpolation models come from ComfyUI core's loader (ComfyUI >= 0.41: RIFE 4.25/4.26, FILM); this module has no
weights, only torch.
"""

import json
import math
import re

import torch
import torch.nn.functional as F

from .nodes import _device, CHUNK
from .layers import _fit_mask, _box, srgb_to_ok

MODES = ["Whole Clip", "Subject Only", "Background Only", "Hero Freeze"]
RAMPS = ["Constant", "Slow at Hero Frame", "Ease Into", "Ease Out Of"]
RATES = ["Same as Source", "24", "25", "30", "60"]
PICKS = ["Auto", "Fast", "Large Motion"]
LOCKS = ["Auto", "Scene", "Frame"]
EPS_T = 1e-4          # a source time this close to a whole frame is that frame
SHUTTER = 0.5         # 180-degree shutter: open for half of the time an output frame covers
LARGE_MOTION = 0.04   # Auto picks Large Motion above this (90th-percentile shift / short side, per source frame)
MAX_TAPS = 9


def _pick(value, options, default):
    """Case- and dash-tolerant enum match (WorkflowIO Enum inputs arrive as plain strings)."""
    v = re.sub(r"[\s\-_]+", " ", str(value or "")).strip().lower()
    for o in options:
        if o.lower() == v:
            return o
    return default


def _rate(value, fps):
    v = _pick(value, RATES, "Same as Source")
    return float(fps) if v == "Same as Source" else float(v)


# ============================================================================================== time map (maths)
def _ease(x):
    return 0.5 - 0.5 * torch.cos(math.pi * x.clamp(0, 1))


def _bump(x, c, w):
    d = ((x - c).abs() / max(w, 1e-6)).clamp(0, 1)
    return 0.5 * (1 + torch.cos(math.pi * d))


def parse_speed_curve(text, fps):
    """'0s:1, 1.2s:0.25, 2.4s:1' (times in the clip; '36f' = frame 36, 1-based) -> sorted [(seconds, speed)] or None."""
    keys = []
    for part in re.split(r"[,;\n]+", (text or "").strip()):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(s|f)?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*x?", part, re.I)
        if not m:
            raise ValueError(f"Speed Curve: '{part}' isn't a time:speed pair such as 1.2s:0.25 or 36f:0.25.")
        t = float(m.group(1))
        if (m.group(2) or "").lower() == "f":
            t = max(0.0, t - 1) / fps
        keys.append((t, min(16.0, max(0.02, float(m.group(3))))))
    keys.sort()
    return keys or None


def _curve_speed(keys, t):
    ks = torch.tensor([k[0] for k in keys], dtype=t.dtype)
    vs = torch.tensor([k[1] for k in keys], dtype=t.dtype)
    if len(keys) == 1:
        return torch.full_like(t, float(vs[0]))
    idx = torch.searchsorted(ks, t).clamp(1, len(keys) - 1)
    t0, t1, v0, v1 = ks[idx - 1], ks[idx], vs[idx - 1], vs[idx]
    v = v0 + (v1 - v0) * _ease((t - t0) / (t1 - t0).clamp_min(1e-9))
    v = torch.where(t <= ks[0], vs[0], v)
    return torch.where(t >= ks[-1], vs[-1], v)


def speed_profile(t, ramp, speed, hero_t, width, keys=None):
    """Speed (source seconds per output second) at source time t (seconds)."""
    if keys:
        return _curve_speed(keys, t)
    if ramp == "Slow at Hero Frame":
        return 1 - (1 - speed) * _bump(t, hero_t, width)
    if ramp == "Ease Into":
        return 1 - (1 - speed) * _ease((t - (hero_t - width)) / width)
    if ramp == "Ease Out Of":
        return speed + (1 - speed) * _ease((t - hero_t) / width)
    return torch.full_like(t, float(speed))


def _interp1(x, xp, fp):
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)


def _snap(frames, n):
    r = frames.round()
    return torch.where((frames - r).abs() < EPS_T, r, frames).clamp(0, n - 1)


def whole_clip_times(n, fps, fps_out, speed=1.0, ramp="Constant", target=0.0, hero=None, keys=None):
    """Source frame (float) for every output frame of a whole-clip retime, plus what was solved.

    Constant speed: the result lasts L / speed (L = first to last frame). Target Duration > 0 wins: with Constant the
    speed is L / target; with a ramp the slow speed is solved so the clip lasts exactly the target (or the whole ramp
    is scaled when the target can't be reached with slow motion); a Speed Curve is scaled as a whole. The first and
    last source frames are kept exactly."""
    notes = []
    if n < 2:
        return [0.0], {"speed": 1.0, "out_seconds": 0.0, "notes": ["One frame: nothing to retime"]}
    L = (n - 1) / fps
    hero_t = (hero if hero is not None else (n - 1) / 2) / fps
    width = min(1.5, max(0.3, 0.25 * L))
    grid = torch.linspace(0, L, max(4001, 8 * n + 1), dtype=torch.float64)

    def tau_of(spd):
        v = speed_profile(grid, ramp, spd, hero_t, width, keys).clamp(0.02, 16.0)
        dt = (grid[1:] - grid[:-1]) / (0.5 * (v[1:] + v[:-1]))
        return torch.cat([torch.zeros(1, dtype=grid.dtype), dt.cumsum(0)])

    solved = float(speed)
    if target > 0 and not keys:
        if ramp == "Constant":
            solved = L / target
        elif float(tau_of(1.0)[-1]) >= target:
            notes.append("Target Duration is shorter than the clip with a slow-motion ramp: the whole ramp was sped up "
                         "to fit")
        else:
            lo, hi = 0.02, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if float(tau_of(mid)[-1]) > target:
                    lo = mid
                else:
                    hi = mid
            solved = 0.5 * (lo + hi)
    tau = tau_of(solved)
    T = float(tau[-1])
    if target > 0:
        tau = tau * (target / T)
        T = float(target)
    n_out = max(2, int(round(T * fps_out)) + 1)
    t = _interp1(torch.linspace(0, T, n_out, dtype=torch.float64), tau, grid)
    frames = _snap(t * fps, n)
    return frames.tolist(), {"speed": round(solved, 4), "out_seconds": round(T, 3), "ramp_width_s": round(width, 3),
                             "notes": notes}


def layer_times(n, fps, fps_out, mode, speed, hero, loop=False):
    """Subject and background source frames for the layer modes, in sync at the hero frame (the anchor: there the
    output is the original frame). Output length = the longest span in which no layer runs out of footage.
    Hero Freeze: both layers play up to the hero frame, then the subject holds while the background plays at `speed`;
    with a loop only the held part is made (a cinemagraph). Returns (subject, background, anchor output index)."""
    r = fps / fps_out
    k, last = float(hero), n - 1
    if mode == "Hero Freeze":
        s_bg = speed if speed > 0 else 1.0
        pre = [] if loop else [k - r * j for j in range(int(math.floor(k / r + 1e-9)), 0, -1)]
        post = [k + s_bg * r * j for j in range(int(math.floor((last - k) / (s_bg * r) + 1e-9)) + 1)]
        sub, bg, anchor = pre + [k] * len(post), pre + post, len(pre)
    else:
        s_sub, s_bg = (speed, 1.0) if mode == "Subject Only" else (1.0, speed)

        def reach(s, room):
            return math.inf if s <= 0 else room / (s * r)

        a = int(math.floor(min(reach(s_sub, k), reach(s_bg, k)) + 1e-9))
        b = int(math.floor(min(reach(s_sub, last - k), reach(s_bg, last - k)) + 1e-9))
        sub = [k + s_sub * r * m for m in range(-a, b + 1)]
        bg = [k + s_bg * r * m for m in range(-a, b + 1)]
        anchor = a
    snap = lambda v: _snap(torch.tensor(v, dtype=torch.float64), n).tolist()  # noqa: E731
    return snap(sub), snap(bg), anchor


def _steps(times):
    """Source frames each output frame covers (local speed in frames per output frame)."""
    t = torch.tensor(times, dtype=torch.float64)
    if t.numel() < 2:
        return [0.0] * t.numel()
    d = (t[1:] - t[:-1]).abs()
    s = torch.cat([d[:1], 0.5 * (d[1:] + d[:-1]), d[-1:]])
    return s.tolist()


def _taps(step):
    """Shutter samples for one output frame: 1 (a plain frame) unless the shutter spans more than one source frame."""
    span = SHUTTER * step
    return 1 if span <= 1.0 + 1e-6 else min(MAX_TAPS, 2 * int(math.ceil(span)) + 1)


def _offsets(k):
    return [0.0] if k == 1 else torch.linspace(-SHUTTER / 2, SHUTTER / 2, k).tolist()


# ================================================================================================ image helpers
def _thumbs(images, short=96):
    """(N,H,W,C) -> (N,h,w) grey thumbnails on the CPU (area resize, chunked on the compute device)."""
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    s = short / min(h, w)
    th, tw = max(8, round(h * s)), max(8, round(w * s))
    dev, out = _device(), []
    for i in range(0, n, CHUNK * 4):
        x = images[i:i + CHUNK * 4]
        x = x[..., :3].mean(-1) if x.dim() == 4 else x
        out.append(F.interpolate(x.to(dev).float()[:, None], size=(th, tw), mode="area")[:, 0].cpu())
    return torch.cat(out, 0)


def _dil(m, r):
    """Square dilation of (N,H,W) (separable max pool)."""
    if r <= 0:
        return m
    k = 2 * r + 1
    t = F.max_pool2d(m[:, None].float(), (1, k), stride=1, padding=(0, r))
    return F.max_pool2d(t, (k, 1), stride=1, padding=(r, 0))[:, 0]


def _hann(n, dev):
    return torch.hann_window(n, periodic=False, dtype=torch.float32, device=dev) if n > 1 else torch.ones(1, device=dev)


def _phase_corr(a, b):
    """Sub-pixel shift (dy, dx) such that b is a moved by (-dy, -dx) (same convention as the numpy prototype)."""
    A, B = torch.fft.fft2(a.float()), torch.fft.fft2(b.float())
    R = A * torch.conj(B)
    r = torch.fft.ifft2(R / (R.abs() + 1e-9)).real
    H, W = r.shape
    flat = int(torch.argmax(r))
    y, x = divmod(flat, W)

    def sub(v, i, n):
        l, c, rr = float(v[(i - 1) % n]), float(v[i]), float(v[(i + 1) % n])
        den = 2 * (l - 2 * c + rr)
        return i + ((l - rr) / den if abs(den) > 1e-12 else 0.0)

    yy, xx = sub(r[:, x], y, H), sub(r[y, :], x, W)
    yy = yy - H if yy > H / 2 else yy
    xx = xx - W if xx > W / 2 else xx
    return yy, xx


def _sample(img, ys, xs, pad="border"):
    """Bilinear samples of (H,W) or (H,W,C) img at pixel coordinates ys, xs (same shape)."""
    H, W = img.shape[0], img.shape[1]
    grid = torch.stack([2 * xs / max(W - 1, 1) - 1, 2 * ys / max(H - 1, 1) - 1], -1)[None].float()
    x = img[None, None] if img.dim() == 2 else img.permute(2, 0, 1)[None]
    o = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode=pad, align_corners=True)[0]
    return o[0] if img.dim() == 2 else o.permute(1, 2, 0)


def warp(img, T, pad="border"):
    """Move a frame or matte by the 3x3 transform T (pixel (y, x) of the source -> pixel of the result)."""
    H, W = img.shape[0], img.shape[1]
    Ti = torch.linalg.inv(torch.as_tensor(T, dtype=torch.float64)).float().to(img.device)
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device, dtype=torch.float32),
                            torch.arange(W, device=img.device, dtype=torch.float32), indexing="ij")
    sy = Ti[0, 0] * ys + Ti[0, 1] * xs + Ti[0, 2]
    sx = Ti[1, 0] * ys + Ti[1, 1] * xs + Ti[1, 2]
    return _sample(img, sy, sx, pad).to(img.dtype)


def _rot(scale, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64) * scale


def similarity_fm(g0, g1, n_ang=360, n_rad=512):
    """Scale, rotation and shift from grey frame g0 to g1, whole frame (Fourier-Mellin, about the centre). Fallback for
    `similarity` when too few tiles carry texture. Returns (scale, angle, dy, dx): a pixel p of g0 lands at
    A (p - c) + c + d in g1, A = scale * R(angle)."""
    dev = g0.device
    H, W = g0.shape
    win = _hann(H, dev)[:, None] * _hann(W, dev)[None]
    yy = torch.arange(H, device=dev, dtype=torch.float32)[:, None] - H // 2
    xx = torch.arange(W, device=dev, dtype=torch.float32)[None] - W // 2
    hp = 1 - torch.exp(-(xx ** 2 / (W / 8) ** 2 + yy ** 2 / (H / 8) ** 2))

    def mag(g):
        Fm = torch.fft.fftshift(torch.fft.fft2((g - g.mean()) * win).abs())
        return torch.log1p(Fm) * hp

    M0, M1 = mag(g0.float()), mag(g1.float())
    rmax = min(H, W) / 2
    base = rmax ** (1 / n_rad)
    ang = torch.linspace(0, math.pi, n_ang + 1, device=dev)[:-1]
    rad = base ** torch.arange(n_rad, device=dev, dtype=torch.float32)
    cy, cx = H // 2, W // 2
    ys = cy + rad[None] * torch.sin(ang)[:, None]
    xs = cx + rad[None] * torch.cos(ang)[:, None]
    L0, L1 = _sample(M0, ys, xs), _sample(M1, ys, xs)
    da, dr = _phase_corr(L1, L0)
    scale = base ** (-dr)
    angle = -da * math.pi / n_ang
    if abs(angle) > math.radians(10):      # camera roll is rare: a big angle is noise
        angle = 0.0
    Mi = _rot(scale, angle)
    yg, xg = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    sy = Mi[0, 0] * (yg - cy) + Mi[0, 1] * (xg - cx) + cy
    sx = Mi[1, 0] * (yg - cy) + Mi[1, 1] * (xg - cx) + cx
    g1u = _sample(g1.float(), sy.float().to(dev), sx.float().to(dev))
    dyu, dxu = _phase_corr(g1u * win, g0.float() * win)
    d = Mi @ torch.tensor([dyu, dxu], dtype=torch.float64)
    return float(scale), float(angle), float(d[0]), float(d[1])


def _tile_fit(g0, g1, keep=None, grid=4):
    """Similarity from per-tile shifts (phase correlation on overlapping tiles), fitted robustly: tiles on a moving
    object and flat tiles drop out. With a `keep` mask (1 = background to use) only tiles entirely inside it count
    (third-size tiles on a 5x5 grid; half-size tiles, mostly inside it, when too few are). Returns (scale, angle, dy,
    dx, tiles used) or None when fewer than 3 tiles are usable."""
    dev = g0.device
    H, W = g0.shape
    cy, cx = H // 2, W // 2

    def collect(th, tw, n_grid, ok):
        win = _hann(th, dev)[:, None] * _hann(tw, dev)[None]
        rows, rhs = [], []
        for y0 in torch.linspace(0, H - th, n_grid).round().long().unique().tolist():
            for x0 in torch.linspace(0, W - tw, n_grid).round().long().unique().tolist():
                if keep is not None and not ok(keep[y0:y0 + th, x0:x0 + tw]):
                    continue
                a = g0[y0:y0 + th, x0:x0 + tw].float()
                b = g1[y0:y0 + th, x0:x0 + tw].float()
                if float(a.std()) < 0.01 or float(b.std()) < 0.01:
                    continue
                dy, dx = _phase_corr((b - b.mean()) * win, (a - a.mean()) * win)
                py, px = y0 + th / 2 - cy, x0 + tw / 2 - cx
                rows += [[py, -px, 1.0, 0.0], [px, py, 0.0, 1.0]]
                rhs += [dy, dx]
        return rows, rhs

    rows, rhs = [], []
    if keep is not None:
        rows, rhs = collect(max(16, H // 3), max(16, W // 3), 5, lambda k: float(k.min()) > 0.5)
    if len(rows) < 8:
        rows, rhs = collect(max(16, H // 2), max(16, W // 2), grid, lambda k: float(k.mean()) > 0.6)
    if len(rows) < 6:
        return None
    A = torch.tensor(rows, dtype=torch.float64)
    y = torch.tensor(rhs, dtype=torch.float64)
    w = torch.ones(len(rhs), dtype=torch.float64)
    for _ in range(4):                                  # iteratively reweighted least squares
        sw = w.sqrt()[:, None]
        u = torch.linalg.lstsq(A * sw, y[:, None] * sw).solution[:, 0]
        r = (A @ u - y).view(-1, 2).norm(dim=1)
        sig = max(float(r.median()) * 1.4826, 0.15)
        wt = 1.0 / (1.0 + (r / (2 * sig)) ** 2)
        wt = torch.where(r > 4 * sig, torch.zeros_like(wt), wt)
        w = wt.repeat_interleave(2)
    a1, b = float(u[0]) + 1.0, float(u[1])
    if abs(math.atan2(b, a1)) < math.radians(0.5):      # small roll is noise: refit scale + shift only
        A3 = A[:, [0, 2, 3]]
        sw = w.sqrt()[:, None]
        u3 = torch.linalg.lstsq(A3 * sw, y[:, None] * sw).solution[:, 0]
        return 1.0 + float(u3[0]), 0.0, float(u3[1]), float(u3[2]), int((w[::2] > 0).sum())
    return math.hypot(a1, b), math.atan2(b, a1), float(u[2]), float(u[3]), int((w[::2] > 0).sum())


def similarity(g0, g1, keep=None):
    """Scale, rotation and shift from grey frame g0 to g1: per-tile shifts fitted robustly (a moving object or the
    masked subject doesn't steer it), refined once after undoing the first estimate; Fourier-Mellin when the frame
    has too little texture. Returns (scale, angle, dy, dx) about the centre: pixel p of g0 lands at A (p - c) + c + d
    in g1, A = scale * R(angle)."""
    H, W = g0.shape
    c = (H // 2, W // 2)
    first = _tile_fit(g0, g1, keep)
    if first is None:
        return similarity_fm(g0, g1)
    T = _mat(first[0], first[1], first[2], first[3], c)
    # undo the estimate on g1 (g1u(p) = g1(T p)) and fit what is left
    yg, xg = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    sy = T[0, 0] * yg + T[0, 1] * xg + T[0, 2]
    sx = T[1, 0] * yg + T[1, 1] * xg + T[1, 2]
    g1u = _sample(g1.float(), sy.float().to(g1.device), sx.float().to(g1.device))
    rest = _tile_fit(g0, g1u, keep)
    if rest is not None:
        T = T @ _mat(rest[0], rest[1], rest[2], rest[3], c)
    A = T[:2, :2]
    scale = float(torch.sqrt(torch.det(A).abs()))
    angle = math.atan2(float(A[1, 0]), float(A[0, 0]))
    if abs(angle) > math.radians(10):
        angle = 0.0
    d = T[:2, 2] - torch.tensor(c, dtype=torch.float64) + A @ torch.tensor(c, dtype=torch.float64)
    return scale, angle, float(d[0]), float(d[1])


def _mat(scale, angle, dy, dx, centre):
    A = _rot(scale, angle)
    c = torch.tensor(centre, dtype=torch.float64)
    T = torch.eye(3, dtype=torch.float64)
    T[:2, :2] = A
    T[:2, 2] = c + torch.tensor([dy, dx], dtype=torch.float64) - A @ c
    return T


# ============================================================================================ interpolation core
class _Interp:
    """The core frame-interpolation model (the INTERP_MODEL of 'Load Frame Interpolation Model') at any timesteps.

    RIFE (IFNet): one pass per frame pair for all timesteps; with a matte, the IFNet block loop is run here so the
    matte is warped with the same flow and blend mask as the picture (mirrors comfy_extras/frame_interpolation_models/
    ifnet.py, ComfyUI 0.41-0.45). FILM: forward_multi_timestep; a matte goes through the model as a grey picture.
    Anything else with forward(img0, img1, timestep) works too (unit-test stand-ins)."""

    def __init__(self, interp_model):
        self.patcher = interp_model
        self.net = getattr(interp_model, "model", interp_model)
        self.device = getattr(interp_model, "load_device", None) or _device()
        dt = None
        try:
            dt = interp_model.model_dtype()
        except Exception:
            try:
                dt = next(self.net.parameters()).dtype
            except Exception:
                dt = torch.float32
        self.dtype = dt
        self.align = int(getattr(self.net, "pad_align", 1) or 1)
        self.kind = type(self.net).__name__
        self.co_warp = all(hasattr(self.net, a) for a in ("encode", "blocks", "scale_list", "warp", "_build_warp_grids"))
        self.film = hasattr(self.net, "forward_multi_timestep")
        self.batch = 8
        self._fc = {}
        self._loaded = None
        self.pairs = 0
        self.made = 0

    def load(self, h, w):
        if self._loaded == (h, w):
            return
        try:
            import comfy.model_management as mm
            mem = self.net.memory_used_forward((1, h, w, 3), self.dtype) if hasattr(self.net, "memory_used_forward") else 0
            mm.load_models_gpu([self.patcher], memory_required=mem)
        except Exception:
            if hasattr(self.net, "to"):
                self.net.to(self.device)
        self._loaded = (h, w)

    def _prep(self, x):
        """(H,W,C) -> (1,C,Hp,Wp) on the model's device and dtype, padded to the model's size multiple."""
        t = x.to(self.device, self.dtype).permute(2, 0, 1)[None] if x.dim() == 3 else x.to(self.device, self.dtype)[None, None]
        ph, pw = (-t.shape[2]) % self.align, (-t.shape[3]) % self.align
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        return t

    def _feats(self, key, img):
        if not hasattr(self.net, "extract_features"):
            return None
        if key is not None and key in self._fc:
            return self._fc[key]
        f = self.net.extract_features(img)
        if key is not None:
            if len(self._fc) >= 6:
                self._fc.pop(next(iter(self._fc)))
            self._fc[key] = f
        return f

    def _ifnet(self, x0, x1, ts, c0, c1, a0, a1):
        net = self.net
        B, (Hp, Wp) = len(ts), x0.shape[2:]
        t = torch.tensor(ts, device=x0.device, dtype=x0.dtype).view(B, 1, 1, 1).expand(B, 1, Hp, Wp)
        img0, img1 = x0.expand(B, -1, -1, -1), x1.expand(B, -1, -1, -1)
        cache = {"img0": c0, "img1": c1} if c0 is not None else None
        if a0 is None:
            return net(img0, img1, timestep=t, cache=cache), None
        net._build_warp_grids(Hp, Wp, x0.device)
        f0 = c0.expand(B, -1, -1, -1) if c0 is not None else net.encode(img0)
        f1 = c1.expand(B, -1, -1, -1) if c1 is not None else net.encode(img1)
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, block in enumerate(net.blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, t), 1), None, scale=net.scale_list[i])
            else:
                fd, mask, feat = block(torch.cat((w0, w1, net.warp(f0, flow[:, :2]), net.warp(f1, flow[:, 2:4]), t, mask,
                                                  feat), 1), flow, scale=net.scale_list[i])
                flow = flow + fd
            w0, w1 = net.warp(img0, flow[:, :2]), net.warp(img1, flow[:, 2:4])
        s = torch.sigmoid(mask)
        out = torch.lerp(w1, w0, s)
        aw = torch.lerp(net.warp(a0.expand(B, -1, -1, -1), flow[:, :2]), net.warp(a1.expand(B, -1, -1, -1), flow[:, 2:4]), s)
        return out, aw[:, 0]

    def _film(self, x0, x1, ts, c0, c1, a0, a1):
        cache = {"img0": c0, "img1": c1} if c0 is not None else None
        out = self.net.forward_multi_timestep(x0, x1, list(ts), cache=cache)
        aw = None
        if a0 is not None:
            aw = self.net.forward_multi_timestep(a0.expand(1, 3, -1, -1), a1.expand(1, 3, -1, -1), list(ts)).mean(1)
        return out, aw

    def _generic(self, x0, x1, ts, a0, a1):
        B, (Hp, Wp) = len(ts), x0.shape[2:]
        t = torch.tensor(ts, device=x0.device, dtype=x0.dtype).view(B, 1, 1, 1).expand(B, 1, Hp, Wp)
        out = self.net(x0.expand(B, -1, -1, -1), x1.expand(B, -1, -1, -1), timestep=t)
        aw = None
        if a0 is not None:
            aw = self.net(a0.expand(B, 3, -1, -1), a1.expand(B, 3, -1, -1), timestep=t).mean(1)
        return out, aw

    @torch.no_grad()
    def between(self, f0, f1, ts, a0=None, a1=None, keys=(None, None)):
        """In-betweens of frames f0, f1 ((H,W,3)) at timesteps ts (0 < t < 1). Mattes a0, a1 ((H,W)) are carried along
        when given. Returns (len(ts),H,W,3) [, (len(ts),H,W)] float32 on the CPU."""
        H, W = f0.shape[0], f0.shape[1]
        x0, x1 = self._prep(f0[..., :3]), self._prep(f1[..., :3])
        p0 = self._prep(a0) if a0 is not None else None
        p1 = self._prep(a1) if a1 is not None else None
        c0, c1 = self._feats(keys[0], x0), self._feats(keys[1], x1)
        outs, alphas, i = [], [], 0
        ts = list(ts)
        while i < len(ts):
            part = ts[i:i + self.batch]
            try:
                if self.co_warp:
                    o, a = self._ifnet(x0, x1, part, c0, c1, p0, p1)
                elif self.film:
                    o, a = self._film(x0, x1, part, c0, c1, p0, p1)
                else:
                    o, a = self._generic(x0, x1, part, p0, p1)
            except RuntimeError as e:   # out of memory: fewer timesteps per pass
                if "out of memory" not in str(e).lower() or self.batch <= 1:
                    raise
                self.batch = max(1, self.batch // 2)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            outs.append(o[:, :3, :H, :W].float().clamp(0, 1).permute(0, 2, 3, 1).cpu())
            if a is not None:
                alphas.append(a[:, :H, :W].float().clamp(0, 1).cpu())
            i += len(part)
        self.pairs += 1
        self.made += len(ts)
        return torch.cat(outs, 0), (torch.cat(alphas, 0) if alphas else None)


class _Frames:
    """A clip (and its matte) at any source time: whole times are the frames, untouched; others are in-betweens from
    the model (a linear blend when no model is connected)."""

    def __init__(self, frames, alpha=None, interp=None, tag="I"):
        self.f, self.a, self.ip, self.tag = frames, alpha, interp, tag
        self.n = frames.shape[0]
        self.blended = 0

    def render(self, times):
        n, H, W = self.n, self.f.shape[1], self.f.shape[2]
        out = torch.empty((len(times), H, W, 3), dtype=torch.float32)
        oa = torch.empty((len(times), H, W), dtype=torch.float32) if self.a is not None else None
        pending = {}
        for p, t in enumerate(times):
            t = min(max(float(t), 0.0), n - 1.0)
            i = int(math.floor(t))
            fr = t - i
            if fr < EPS_T or fr > 1 - EPS_T or i >= n - 1:
                j = i + 1 if (fr > 1 - EPS_T and i < n - 1) else i
                out[p] = self.f[j, ..., :3]
                if oa is not None:
                    oa[p] = self.a[j]
            else:
                pending.setdefault(i, {}).setdefault(round(fr, 6), []).append(p)
        for i in sorted(pending):
            fracs = sorted(pending[i])
            if self.ip is None:
                f0, f1 = self.f[i, ..., :3].float(), self.f[i + 1, ..., :3].float()
                res = torch.stack([f0 + (f1 - f0) * w for w in fracs])
                ra = torch.stack([self.a[i] + (self.a[i + 1] - self.a[i]) * w for w in fracs]) if oa is not None else None
                self.blended += len(fracs)
            else:
                res, ra = self.ip.between(self.f[i], self.f[i + 1], fracs,
                                          self.a[i] if oa is not None else None,
                                          self.a[i + 1] if oa is not None else None,
                                          keys=((self.tag, i), (self.tag, i + 1)))
            for q, fr in enumerate(fracs):
                for p in pending[i][fr]:
                    out[p] = res[q]
                    if oa is not None:
                        oa[p] = ra[q] if ra is not None else (self.a[i] + (self.a[i + 1] - self.a[i]) * fr)
        return out, oa

    def morph(self, a, b, w):
        """A frame between two (possibly far apart) frames a and b of this clip, at weight w."""
        if self.ip is None:
            o = self.f[a, ..., :3] * (1 - w) + self.f[b, ..., :3] * w
            oa = self.a[a] * (1 - w) + self.a[b] * w if self.a is not None else None
            self.blended += 1
            return o, oa
        res, ra = self.ip.between(self.f[a], self.f[b], [w], self.a[a] if self.a is not None else None,
                                  self.a[b] if self.a is not None else None, keys=((self.tag, a), (self.tag, b)))
        oa = (ra[0] if ra is not None else self.a[a] * (1 - w) + self.a[b] * w) if self.a is not None else None
        return res[0], oa


# ====================================================================================================== camera
class _Camera:
    """Similarity camera path of a clip (fit on the clean background, so the subject doesn't steer it): scale, roll and
    shift chained every few frames, interpolated in between. T(ts, tb) moves a pixel of frame ts to where the same
    scene point is in frame tb."""

    def __init__(self, frames, step=4, short=256, support=None):
        n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
        g = _thumbs(frames, short)
        sup = None
        if support is not None:
            sup = F.interpolate(support.float()[:, None], size=g.shape[1:], mode="area")[:, 0]
        self.g, self.sup = g, sup
        self.n, self.h, self.w = n, h, w
        lh, lw = g.shape[1], g.shape[2]
        self.f = (h / lh, w / lw)
        self.c = ((lh // 2) * self.f[0], (lw // 2) * self.f[1])
        dev = _device()
        keys = list(range(0, n, step))
        if keys[-1] != n - 1:
            keys.append(n - 1)
        # cumulative centre-form parameters per key frame: log scale, angle, dy, dx (full-res pixels)
        P = [(0.0, 0.0, 0.0, 0.0)]
        self.steps = []
        for k0, k1 in zip(keys[:-1], keys[1:]):
            keep = None
            if sup is not None:
                keep = 1 - _box(_dil(torch.maximum(sup[k0], sup[k1])[None], 3), 2)[0].clamp(0, 1)
                keep = keep.to(dev) if float(keep.mean()) > 0.2 else None
            s, a, dy, dx = similarity(g[k0].to(dev), g[k1].to(dev), keep)
            dy, dx = dy * self.f[0], dx * self.f[1]
            self.steps.append((k0, k1, s, a, dy, dx))
            ls, an, py, px = P[-1]
            Ar = _rot(s, a)
            d = Ar @ torch.tensor([py, px], dtype=torch.float64) + torch.tensor([dy, dx], dtype=torch.float64)
            P.append((ls + math.log(s), an + a, float(d[0]), float(d[1])))
        kt = torch.tensor(keys, dtype=torch.float64)
        Pt = torch.tensor(P, dtype=torch.float64)
        frames_t = torch.arange(n, dtype=torch.float64)
        self.par = torch.stack([_interp1(frames_t, kt, Pt[:, j]) for j in range(4)], 1) if n > 1 else Pt
        steps = torch.tensor([[math.log(s[2]), s[3], s[4], s[5]] for s in self.steps], dtype=torch.float64) \
            if self.steps else torch.zeros(1, 4, dtype=torch.float64)
        per = (torch.tensor([s[1] - s[0] for s in self.steps], dtype=torch.float64).clamp_min(1)
               if self.steps else torch.ones(1, dtype=torch.float64))
        short_px = min(h, w)
        self.shift_per_frame = float((steps[:, 2:].norm(dim=1) / per).median()) / short_px
        self.scale_per_frame = float((steps[:, 0].abs() / per).median())
        tot = self.par[-1]
        drift = float(self.par[:, 2:].norm(dim=1).max()) / short_px
        zoom = float(self.par[:, 0].abs().max())
        roll = float(self.par[:, 1].abs().max())
        self.static = drift < 0.01 and zoom < 0.005 and roll < math.radians(0.3)
        self.total = {"scale": round(math.exp(float(tot[0])), 4), "angle_deg": round(math.degrees(float(tot[1])), 2),
                      "shift_px": [round(float(tot[2]), 1), round(float(tot[3]), 1)]}

    def _C(self, t):
        t = min(max(float(t), 0.0), self.n - 1.0)
        i = int(math.floor(t))
        j = min(i + 1, self.n - 1)
        p = self.par[i] + (self.par[j] - self.par[i]) * (t - i)
        return _mat(math.exp(float(p[0])), float(p[1]), float(p[2]), float(p[3]), self.c)

    def T(self, ts, tb):
        M = self._C(tb) @ torch.linalg.inv(self._C(ts))
        return None if float((M - torch.eye(3, dtype=torch.float64)).abs().max()) < 1e-6 else M

    def set_subject(self, matte, short=128):
        """The subject's track in the original clip: centroid and size (sqrt of the matte area), smoothed over +-3
        frames. For a subject that stays put in the scene this is the camera at the subject's own depth (parallax
        included), so it is what the Scene lock follows."""
        m = _thumbs(matte, short)
        fy, fx = self.h / m.shape[1], self.w / m.shape[2]
        n = m.shape[0]
        raw = []
        for i in range(n):
            ys, xs = torch.nonzero(m[i] > 0.5, as_tuple=True)
            raw.append(None if ys.numel() < 6 else
                       (float(ys.float().mean()) * fy, float(xs.float().mean()) * fx, math.sqrt(ys.numel() * fy * fx)))
        tr = []
        for i in range(n):
            near = [raw[j] for j in range(max(0, i - 3), min(n, i + 4)) if raw[j] is not None]
            tr.append(None if raw[i] is None or not near else
                      tuple(float(torch.tensor([v[q] for v in near]).median()) for q in range(3)))
        self.track = tr

    def _track_at(self, t):
        t = min(max(float(t), 0.0), self.n - 1.0)
        i = int(math.floor(t))
        j = min(i + 1, self.n - 1)
        a, b = self.track[i], self.track[j]
        if a is None or b is None:
            return None
        f = t - i
        return tuple(a[q] + (b[q] - a[q]) * f for q in range(3))

    def T_scene(self, ts, tb):
        """Move the subject's layer from ts into tb's camera: its own track when it is known at both moments (exact at
        its depth), else the background's similarity path."""
        if getattr(self, "track", None) is not None:
            a, b = self._track_at(ts), self._track_at(tb)
            if a is not None and b is not None:
                s = b[2] / max(a[2], 1e-6)
                M = torch.eye(3, dtype=torch.float64)
                M[0, 0] = M[1, 1] = s
                M[0, 2], M[1, 2] = b[0] - s * a[0], b[1] - s * a[1]
                return None if float((M - torch.eye(3, dtype=torch.float64)).abs().max()) < 1e-6 else M
        return self.T(ts, tb)

    def lock_rule(self, matte, follow=0.45):
        """'static', 'Frame' (the camera follows the subject: it keeps it on the frame) or 'Scene' (the subject is
        carried by the scene: it stays in place there). Between key frames, compare how much the subject moves on screen
        (position and size) with how much the scene would carry it if it stood still in it (the background's
        similarity at the subject). Frame when the subject is clearly steadier on screen (under `follow` times that),
        else Scene. Borderline shots: Lock To overrides."""
        self.set_subject(matte)
        if self.static:
            return "static", {}
        screen, carried = [], []
        for k0, k1, s, a, dy, dx in self.steps:
            t0, t1 = self.track[k0], self.track[k1]
            if t0 is None or t1 is None:
                continue
            c0 = torch.tensor(t0[:2], dtype=torch.float64)
            c1 = torch.tensor(t1[:2], dtype=torch.float64)
            radius = 0.25 * (t0[2] + t1[2])
            grow = math.log(t1[2] / max(t0[2], 1e-6))
            pred = _mat(s, a, dy, dx, self.c)[:2] @ torch.cat([c0, torch.ones(1, dtype=torch.float64)])
            screen.append(float((c1 - c0).norm()) + radius * abs(grow))
            carried.append(float((pred - c0).norm()) + radius * abs(math.log(s)))
        if not screen:
            return "Frame", {}
        sm, cm = float(torch.tensor(screen).median()), float(torch.tensor(carried).median())
        ratio = sm / max(cm, 1e-6)
        return ("Frame" if ratio < follow else "Scene"), {"subject_screen_px_per_step": round(sm, 2),
                                                          "scene_carry_px_per_step": round(cm, 2),
                                                          "screen_vs_carry": round(ratio, 2)}


# ===================================================================================================== loops
def loop_plan(frames, fps, window=2):
    """Video-texture loop points on a clip: frames i < j (j - i >= max(half the clip, 1 s)) whose neighbourhoods
    match best (picture and motion), and the style that hides the seam: 'cut' (seam no bigger than a normal frame
    step), 'morph' (a short flow-aware bridge) or 'pingpong' (forward then back, eased at the turns)."""
    n = frames.shape[0]
    if n < 2 * window + 4:
        return {"style": "pingpong", "start": 0, "end": n, "seam_ratio": None}
    g = _thumbs(frames, 36).reshape(n, -1)
    D = torch.cdist(g[None], g[None])[0] / math.sqrt(g.shape[1])
    typical = float(torch.diagonal(D, 1).median().clamp_min(1e-6))
    m = n - 2 * window
    S = sum(D[window + k:window + k + m, window + k:window + k + m] for k in range(-window, window + 1)) / (2 * window + 1)
    min_len = min(m - 1, max(n // 2, int(round(fps))))
    ii = torch.arange(m)[:, None]
    jj = torch.arange(m)[None]
    S = torch.where(jj >= ii + min_len, S, torch.full_like(S, float("inf")))
    best = int(torch.argmin(S))
    i, j = divmod(best, m)
    i, j = i + window, j + window
    score = float(S.flatten()[best])
    if not math.isfinite(score):
        return {"style": "pingpong", "start": 0, "end": n, "seam_ratio": None}
    ratio = score / typical
    far = D[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=max(1, min_len))]
    unrelated = float(far.median()) if far.numel() else score
    rel = score / max(unrelated, 1e-6)          # vs two unrelated frames of the clip: 1 = no better than random
    if ratio <= 1.5 and rel <= 0.5:
        style = "cut"
    elif ratio <= 6 and rel <= 0.8:
        style = "morph"
    else:
        style = "pingpong"
    return {"style": style, "start": i, "end": j, "seam_ratio": round(ratio, 2), "seam_vs_unrelated": round(rel, 3),
            "seam": round(score, 4), "typical_change": round(typical, 4)}


def loop_frames(R, fps, plan=None):
    """Build one seamless loop of the clip R (a _Frames). Returns frames, matte (or None), plan and the nominal source
    position of each output frame."""
    n = R.n
    plan = plan or loop_plan(R.f, fps)
    style, i, j = plan["style"], plan["start"], plan["end"]
    if style == "cut":
        idx = list(range(i, j))
        o, a = R.render([float(x) for x in idx])
        return o, a, plan, [float(x) for x in idx]
    if style == "morph":
        K = int(min(max(4, round(2 * plan["seam_ratio"])), max(4, round(fps / 2)), i, j - i - 1))
        if K < 2:
            style = plan["style"] = "pingpong"
        else:
            plan["bridge_frames"] = K
            idx = list(range(i, j))
            o, a = R.render([float(x) for x in idx])
            for q in range(K):
                w = (q + 1) / (K + 1)
                fa, fb = j - K + q, i - K + q
                mo, ma = R.morph(fa, fb, w)
                o[len(idx) - K + q] = mo
                if a is not None and ma is not None:
                    a[len(idx) - K + q] = ma
            return o, a, plan, [float(x) for x in idx]
    # ping-pong over the whole clip, eased so the motion slows to a stop at both turns
    span = n - 1
    fwd = []
    for q in range(span):
        u = q / span
        fwd.append(span * (u - math.sin(2 * math.pi * u) / (2 * math.pi)))
    times = fwd + [span - t for t in fwd]
    o, a = R.render(times)
    plan.update(style="pingpong", start=0, end=n)
    return o, a, plan, times


# =================================================================================================== metrics
_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _sharpness(images, matte, idx, short=256):
    """Mean |Laplacian| of the picture inside the matte core, and the core's area share, for frames idx."""
    dev = _device()
    sub = images[idx]
    g = _thumbs(sub, short).to(dev)
    m = F.interpolate(matte[idx].to(dev).float()[:, None], size=g.shape[1:], mode="area")[:, 0]
    lap = F.conv2d(g[:, None], _LAP.to(dev)[None, None], padding=1)[:, 0].abs()
    core = (m > 0.5).float()
    sharp = (lap * core).sum((1, 2)) / core.sum((1, 2)).clamp_min(1)
    return sharp.cpu(), core.mean((1, 2)).cpu()


def hero_freeze_frame(images, matte, given=None):
    """The frame to hold: the sharpest frame with the largest subject in the middle 60% of the clip; a given frame is
    refined to the sharpest within +-3 frames (a held frame keeps its motion blur)."""
    n = images.shape[0]
    if given is None:
        lo, hi = int(0.2 * (n - 1)), int(0.8 * (n - 1))
    else:
        lo, hi = max(0, given - 3), min(n - 1, given + 3)
    idx = list(range(lo, hi + 1)) or [n // 2]
    sharp, area = [], []
    for c in range(0, len(idx), CHUNK * 2):
        s, a = _sharpness(images, matte, idx[c:c + CHUNK * 2])
        sharp.append(s)
        area.append(a)
    sharp, area = torch.cat(sharp), torch.cat(area)
    if float(area.max()) <= 0:
        return (given if given is not None else (n - 1) // 2), False
    score = sharp / sharp.max().clamp_min(1e-9)
    if given is None:
        score = score * (area / area.max()).sqrt()
    score = torch.where(area > 0, score, torch.zeros_like(score))
    return idx[int(torch.argmax(score))], True


def peak_motion_frame(images):
    g = _thumbs(images)
    if g.shape[0] < 2:
        return 0
    d = (g[1:] - g[:-1]).abs().mean((1, 2))
    d = F.avg_pool1d(d[None, None], 5, 1, 2, count_include_pad=False)[0, 0]
    return int(torch.argmax(d))


def measure_motion(images, pairs=24, tiles=3, short=128):
    """How far things move between neighbouring frames: 90th percentile of per-tile shifts (phase correlation) over up
    to `pairs` frame pairs, as a share of the short side."""
    n = images.shape[0]
    if n < 2:
        return 0.0
    g = _thumbs(images, short)
    dev = _device()
    H, W = g.shape[1], g.shape[2]
    th, tw = H // tiles, W // tiles
    win = _hann(th, dev)[:, None] * _hann(tw, dev)[None]
    shifts = []
    for i in torch.linspace(0, n - 2, min(pairs, n - 1)).round().long().unique().tolist():
        a, b = g[i].to(dev), g[i + 1].to(dev)
        for ty in range(tiles):
            for tx in range(tiles):
                pa = a[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                pb = b[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
                if float(pa.std()) < 0.01:
                    continue
                dy, dx = _phase_corr((pa - pa.mean()) * win, (pb - pb.mean()) * win)
                shifts.append(math.hypot(dy, dx))
    if not shifts:
        return 0.0
    return float(torch.tensor(shifts).quantile(0.9)) / min(H, W)


def _edge_agreement(image, matte, short=256):
    """How well the matte's edge follows edges in the picture (about 1 or more when it does; low when the matte runs
    through flat background: an overreaching matte)."""
    dev = _device()
    g = _thumbs(image[None], short)[0].to(dev)
    m = F.interpolate(matte[None, None].to(dev).float(), size=g.shape, mode="area")[0, 0] > 0.5
    if int(m.sum()) < 16:
        return None
    gy = (g[1:, :] - g[:-1, :]).abs()
    gx = (g[:, 1:] - g[:, :-1]).abs()
    gm = torch.zeros_like(g)
    gm[:-1, :] += gy
    gm[:, :-1] += gx
    mf = m.float()[None]
    band = (_dil(mf, 1)[0] > 0.5) & ~(_dil(1 - mf, 1)[0] > 0.5)
    ring = (_dil(mf, 5)[0] > 0.5) & ~(_dil(1 - mf, 5)[0] > 0.5)
    if int(band.sum()) < 8 or int(ring.sum()) < 8:
        return None
    return float(gm[band].mean() / gm[ring].mean().clamp_min(1e-6))


# ======================================================================================================= nodes
def _load_tm(text):
    try:
        tm = json.loads(text) if isinstance(text, str) else dict(text or {})
    except Exception:
        raise ValueError("time_map must be the JSON from AutoMatte Time Map")
    if tm.get("version") != 1:
        raise ValueError("time_map must be the JSON from AutoMatte Time Map (version 1)")
    return tm


class AutoMatteTimeMap:
    """Which source moment each output frame shows (source frames as decimals; whole numbers = the frames as shot).

    Whole Clip: Speed (1 = as shot), a Speed Ramp around the hero frame or a Speed Curve, or a Target Duration (wins
    over Speed). Layer modes: the subject (Subject Only) or the background (Background Only) plays at Speed while the
    other layer plays as shot; Hero Freeze holds the subject from the hero frame on while the background plays at
    Speed. The hero frame is picked automatically when 0: the moment of peak motion for ramps, the sharpest frame with
    the largest subject for Hero Freeze (needs the matte: lazy, only asked for in Hero Freeze), the middle otherwise.
    Frame Rate other than the source's conforms (high-frame-rate footage gives real slow motion)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames."}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "The clip's frame rate."}),
                "mode": (MODES, {"default": "Whole Clip"}),
                "speed": ("FLOAT", {"default": 0.5, "min": 0.02, "max": 16.0, "step": 0.01,
                                    "tooltip": "1 = as shot; below 1 slow motion; above 1 faster. Hero Freeze: the background's speed."}),
                "speed_ramp": (RAMPS, {"default": "Constant", "tooltip": "Whole Clip: how the speed changes around the hero frame."}),
                "target_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                                              "tooltip": "Whole Clip: length of the result in seconds (wins over Speed); 0 = off."}),
                "seamless_loop": ("BOOLEAN", {"default": False}),
                "frame_rate": (RATES, {"default": "Same as Source"}),
            },
            "optional": {
                "speed_curve": ("STRING", {"default": "", "tooltip": "Whole Clip: time:speed keys (1.2s:0.25, or 36f:0.25), used instead of Speed and Speed Ramp."}),
                "hero_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "1-based frame of the hero moment; 0 = pick automatically."}),
                "matte": ("MASK", {"lazy": True, "tooltip": "Subject matte; only computed in Hero Freeze (to pick the frame to hold)."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("time_map", "fps_out", "frames_out")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Retime"

    def check_lazy_status(self, **kw):
        mode = _pick(kw.get("mode"), MODES, "Whole Clip")
        return ["matte"] if mode == "Hero Freeze" and "matte" in kw and kw["matte"] is None else []

    def run(self, images, fps, mode="Whole Clip", speed=0.5, speed_ramp="Constant", target_duration=0.0,
            seamless_loop=False, frame_rate="Same as Source", speed_curve="", hero_frame=0, matte=None):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        fps = float(fps)
        if fps <= 0:
            raise ValueError("The clip's frame rate is 0.")
        mode = _pick(mode, MODES, "Whole Clip")
        ramp = _pick(speed_ramp, RAMPS, "Constant")
        fps_out = _rate(frame_rate, fps)
        speed = float(speed)
        keys = parse_speed_curve(speed_curve, fps)
        notes, reasons = [], []
        given = int(hero_frame) - 1 if int(hero_frame or 0) > 0 else None
        if given is not None and given > n - 1:
            notes.append(f"Hero frame {hero_frame} is past the last frame ({n}): the last frame is used")
            given = n - 1
        hero_source = "given" if given is not None else "middle"
        hero = given if given is not None else (n - 1) // 2
        tm = {"version": 1, "mode": mode, "n_src": n, "fps_src": fps, "fps_out": fps_out, "loop": bool(seamless_loop)}
        if mode == "Whole Clip":
            if given is None and (ramp != "Constant") and not keys:
                hero, hero_source = peak_motion_frame(images), "auto: peak motion"
            sub, info = whole_clip_times(n, fps, fps_out, speed, ramp, float(target_duration or 0), hero, keys)
            bg, anchor = sub, None
            notes += info.pop("notes", [])
            tm.update(info)
            tm["ramp"] = "Speed Curve" if keys else ramp
            if keys:
                tm["speed_curve"] = [[round(k[0], 3), k[1]] for k in keys]
        else:
            if ramp != "Constant" or keys or target_duration:
                notes.append("Speed Ramp, Speed Curve and Target Duration apply to Whole Clip only: ignored")
            if mode == "Hero Freeze":
                if matte is not None:
                    m = _fit_mask(matte, n, h, w)
                    hero, found = hero_freeze_frame(images, m, given)
                    hero_source = ("given, sharpest within 3 frames" if given is not None else "auto: sharpest frame, largest subject")
                    if not found:
                        reasons.append("The matte is empty: no subject to hold")
                if hero >= n - 1:
                    reasons.append("The hero frame is the last frame: nothing plays after the freeze")
            sub, bg, anchor = layer_times(n, fps, fps_out, mode, speed, hero, bool(seamless_loop))
            tm["speed"] = speed
        tm.update(hero_frame=int(hero) + 1, hero_source=hero_source, anchor=anchor, subject=[round(t, 4) for t in sub],
                  background=[round(t, 4) for t in bg], n_out=len(sub))
        steps = [max(a, b) for a, b in zip(_steps(sub), _steps(bg))]
        taps = [_taps(s) for s in steps]
        frac = any(abs(t - round(t)) > EPS_T for t in sub + bg)
        tm["shutter_taps_max"] = max(taps) if taps else 1
        tm["needs_interp"] = bool(frac or max(taps or [1]) > 1 or seamless_loop)
        tm["out_seconds"] = round((len(sub) - 1) / fps_out, 3)
        tm["source_range_s"] = {"subject": [round(min(sub) / fps, 3), round(max(sub) / fps, 3)],
                                "background": [round(min(bg) / fps, 3), round(max(bg) / fps, 3)]}
        slowest = min([s for s in _steps(sub) if s > 0] or [1.0]) * fps_out / fps   # slow-motion factor
        if slowest < 0.25 - 1e-9 and fps < 48:
            reasons.append(f"Slow motion below 0.25x from a {fps:g} fps clip: fast or thin motion can smear or ghost "
                           "(Large Motion model, or footage shot at a higher frame rate, helps)")
        if fps_out > fps * 1.01:
            notes.append(f"Frame Rate {fps_out:g} is above the clip's {fps:g}: the extra frames are in-betweens")
        if mode == "Subject Only" and speed < 1:
            notes.append(f"The subject shows {tm['source_range_s']['subject'][0]}-{tm['source_range_s']['subject'][1]} s "
                         "of its action (it plays slower than the background)")
        tm["notes"], tm["review_reasons"] = notes, reasons
        return (json.dumps(tm), float(fps_out), len(sub))


class AutoMattePickInterpModel:
    """Pick the frame-interpolation model: Fast (RIFE), Large Motion (FILM) or Auto, which measures how far things move
    between neighbouring frames and takes Large Motion above 4% of the short side. Both model inputs are lazy: only the
    chosen loader runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip (Auto measures its motion)."}),
                "choice": (PICKS, {"default": "Auto"}),
            },
            "optional": {
                "fast": ("INTERP_MODEL", {"lazy": True, "tooltip": "RIFE (Load Frame Interpolation Model)."}),
                "large": ("INTERP_MODEL", {"lazy": True, "tooltip": "FILM (Load Frame Interpolation Model)."}),
            },
        }

    RETURN_TYPES = ("INTERP_MODEL", "STRING")
    RETURN_NAMES = ("interp_model", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Retime"

    def _decide(self, images, choice):
        choice = _pick(choice, PICKS, "Auto")
        key = (id(images), tuple(images.shape))
        if getattr(self, "_cache", (None,))[0] == key:
            return self._cache[1], self._cache[2], choice
        motion = measure_motion(images) if choice == "Auto" else None
        pick = choice if choice != "Auto" else ("Large Motion" if motion > LARGE_MOTION else "Fast")
        self._cache = (key, pick, motion)
        return pick, motion, choice

    def check_lazy_status(self, **kw):
        pick, _, _ = self._decide(kw["images"], kw.get("choice", "Auto"))
        want = "large" if pick == "Large Motion" else "fast"
        return [want] if want in kw and kw[want] is None else []

    def run(self, images, choice="Auto", fast=None, large=None):
        pick, motion, asked = self._decide(images, choice)
        model = large if pick == "Large Motion" else fast
        notes = []
        if model is None:
            model = fast if large is None else large
            notes.append(f"{pick} model not connected: the other one was used")
        if model is None:
            raise ValueError("Connect a Load Frame Interpolation Model (RIFE and/or FILM).")
        rep = {"interp_choice": asked, "interp_model": pick, "notes": notes}
        if motion is not None:
            rep["motion_p90_share"] = round(motion, 4)
        return (model, json.dumps(rep))


def _read_plate(video, n, h, w):
    """Clean-plate clip frames, resized to h x w while decoding (at most n + 8 frames), and the clip's own size."""
    frames, src = [], None
    try:
        import av
        with av.open(video.get_stream_source(), mode="r") as c:
            for fr in c.decode(c.streams.video[0]):
                t = torch.from_numpy(fr.to_ndarray(format="rgb24")).float()[None] / 255.0
                src = src or (t.shape[1], t.shape[2])
                if t.shape[1:3] != (h, w):
                    t = F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False,
                                      antialias=True).permute(0, 2, 3, 1)
                frames.append(t[0])
                if len(frames) >= n + 8:
                    break
    except Exception:
        frames = []
    if frames:
        return torch.stack(frames, 0), src
    x = video.get_components().images[..., :3].float()
    src = (x.shape[1], x.shape[2])
    return F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1), src


def _fit_frames(x, n, h, w):
    x = x[..., :3].float()
    if x.shape[1:3] != (h, w):
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False,
                          antialias=True).permute(0, 2, 3, 1)
    if x.shape[0] < n:
        x = torch.cat([x, x[-1:].repeat(n - x.shape[0], 1, 1, 1)], 0)
    return x[:n].contiguous()


def median_plate_check(images, matte):
    """Can a median plate stand in for a removal? Needs a still camera and every background pixel uncovered by the
    subject in at least 15% of the frames (and 3 frames). Light check at thumbnail size."""
    n = images.shape[0]
    if n < 5:
        return False, {"median_plate": "too few frames"}
    g = _thumbs(images, 128)
    m = F.interpolate(matte.float()[:, None], size=g.shape[1:], mode="area")[:, 0]
    free = m < 0.02
    cover = free.float().mean(0)
    ok_cover = bool(((cover >= 0.15) & (free.sum(0) >= 3)).float().mean() >= 0.999)
    dev = _device()
    shifts, diffs = [], []
    gap = max(1, n // 6)
    for i in range(0, n - gap, gap):
        j = i + gap
        keep = (1 - _dil(((~free[i]) | (~free[j])).float()[None], 3)[0]).to(dev)
        s, a, dy, dx = similarity(g[i].to(dev), g[j].to(dev), keep if float(keep.mean()) > 0.2 else None)
        shifts.append(max(abs(math.log(s)) * min(g.shape[1:]), math.hypot(dy, dx)))   # thumbnail pixels
        both = free[i] & free[j]
        if int(both.sum()) > 64:
            diffs.append(float((g[i] - g[j]).abs()[both].mean()))
    still = bool(shifts) and max(shifts) < 0.5 and (not diffs or float(torch.tensor(diffs).median()) < 0.03)
    return (ok_cover and still), {"median_plate_cover_ok": ok_cover, "median_plate_still_camera": still}


def median_plate(images, matte):
    """Per pixel, the median over the frames where the subject doesn't cover it (row bands, light on memory)."""
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    dev = _device()
    out = torch.empty((h, w, 3), dtype=torch.float32)
    band = max(8, int(2 ** 24 // max(1, n * w * 3)))
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        x = images[:, r0:r1, :, :3].to(dev).float().clone()
        free = (matte[:, r0:r1].to(dev) < 0.02)[..., None].expand_as(x)
        x[~free] = float("nan")
        med = torch.nanmedian(x, dim=0).values
        fallback = images[:, r0:r1, :, :3].to(dev).float().median(dim=0).values
        out[r0:r1] = torch.where(torch.isnan(med), fallback, med).cpu()
    return out[None].repeat(n, 1, 1, 1)


class AutoMatteGivenOrMadePlate:
    """The clean plate (the clip without the subject) for the layer modes. A given video wins (Video Object Removal's
    output, checked against the clip: same aspect, about the same frame count). Otherwise a median plate when the
    camera is still and the subject uncovers every background pixel in at least 15% of the frames (no model, exact
    background). Otherwise the graph's removal branch (`made`, lazy: only computed then)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matte": ("MASK",),
            },
            "optional": {
                "plate_video": ("VIDEO", {"tooltip": "A clean plate of this clip (Video Object Removal's video)."}),
                "made": ("IMAGE", {"lazy": True, "tooltip": "A removal branch's frames; only computed when no plate is given and a median plate can't stand in."}),
                "matte_report": ("STRING", {"forceInput": True, "tooltip": "The matte's report (JSON), merged in."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("plate", "report", "made")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Retime"

    def _median_ok(self, images, matte):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        key = (id(images), id(matte), n)
        if getattr(self, "_cache", (None,))[0] != key:
            self._cache = (key,) + median_plate_check(images, _fit_mask(matte, n, h, w))
        return self._cache[1], self._cache[2]

    def check_lazy_status(self, **kw):
        from .layers import _video_given
        if _video_given(kw.get("plate_video")):
            return []
        ok, _ = self._median_ok(kw["images"], kw["matte"])
        return [] if ok else (["made"] if "made" in kw and kw["made"] is None else [])

    def run(self, images, matte, plate_video=None, made=None, matte_report=None):
        from .layers import _video_given
        from .nodes import AutoMatteSaveReport
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        rep = {"notes": [], "review_reasons": []}
        if _video_given(plate_video):
            p, (ph, pw) = _read_plate(plate_video, n, h, w)
            np_ = p.shape[0]
            if abs(np_ - n) > max(2, round(0.02 * n)):
                raise ValueError(f"The Clean Plate has {np_} frames but the video has {n}: use a plate made from this clip.")
            plate, made_flag = _fit_frames(p, n, h, w), False
            rep.update(plate_source="given", plate_frames=np_)
            if abs(math.log((pw / ph) / (w / h))) > 0.02:
                raise ValueError(f"The Clean Plate is {pw}x{ph} but the video is {w}x{h}: use a plate made from this clip.")
            if np_ != n:
                rep["notes"].append(f"The Clean Plate had {np_} frames for {n}: the last frame was "
                                    f"{'repeated' if np_ < n else 'dropped'}")
        else:
            ok, info = self._median_ok(images, matte)
            rep.update(info)
            if ok:
                plate, made_flag = median_plate(images[..., :3], _fit_mask(matte, n, h, w)), True
                rep["plate_source"] = "median (still camera; the subject uncovers the whole background)"
            elif made is not None:
                plate, made_flag = _fit_frames(made, n, h, w), True
                rep["plate_source"] = "made (removal branch)"
            else:
                raise ValueError("No Clean Plate was given, and a median plate can't stand in (the camera moves or the "
                                 "subject never uncovers part of the background): give a Clean Plate from Video Object "
                                 "Removal.")
        out = AutoMatteSaveReport.merge(rep, matte_report) if matte_report else rep
        return (plate, json.dumps(out), made_flag)


class AutoMatteRetime:
    """Render a time map (AutoMatte Time Map). Whole-number source times are the source frames, bit for bit;
    in-betweens come from the frame-interpolation model at any timestep (lazy: only loaded when needed); speed-ups get
    a 180-degree shutter (several in-betweens averaged). Layer modes put the subject's moment over the background's:
        B(t)  = S P(t) + (1 - S) I(t)      clean background: the plate only under the subject and its shadow/light
        out   = I(ts) + (1 - a(ts)) (B(tb) - B(ts))
    exact where both layers show the same moment; the soft edge loses its old background; shadows and light found
    with AutoMatte Effect Zone/Mask move with the subject. A camera lock moves the subject's layer with the camera
    (Scene) or keeps it on the frame (Frame); Auto decides from how the subject moves against the background. A seamless
    loop (cut, morph bridge or ping-pong) on request. Matte, plate and model are lazy: Whole Clip needs neither matte
    nor plate. Outputs the frames, the retimed matte (layer modes), audio when the timing is unchanged, and a report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames."}),
                "time_map": ("STRING", {"forceInput": True, "tooltip": "JSON from AutoMatte Time Map."}),
            },
            "optional": {
                "interp_model": ("INTERP_MODEL", {"lazy": True, "tooltip": "Only loaded when in-betweens are needed."}),
                "matte": ("MASK", {"lazy": True, "tooltip": "Subject matte (layer modes only)."}),
                "plate": ("IMAGE", {"lazy": True, "tooltip": "Clean plate (layer modes only)."}),
                "layer_report": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "Plate/matte report (layer modes only)."}),
                "lock": (LOCKS, {"default": "Auto"}),
                "audio": ("AUDIO", {"tooltip": "Source audio: passed through only when the timing is unchanged."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "matte", "has_matte", "audio", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Retime"

    def check_lazy_status(self, **kw):
        tm = _load_tm(kw.get("time_map"))
        need = ["interp_model"] if tm.get("needs_interp") else []
        if tm.get("mode") != "Whole Clip":
            need += ["matte", "plate", "layer_report"]
        return [k for k in need if k in kw and kw[k] is None]

    # ------------------------------------------------------------------------------------------- whole clip
    @staticmethod
    def _render_single(R, times, max_samples=24):
        q_all = len(times)
        steps = _steps(times)
        out = torch.empty((q_all, R.f.shape[1], R.f.shape[2], 3), dtype=torch.float32)
        q, taps_used = 0, 1
        while q < q_all:
            plan, samples = [], []
            while q < q_all and (not samples or len(samples) + _taps(steps[q]) <= max_samples):
                k = _taps(steps[q])
                taps_used = max(taps_used, k)
                ts = [min(max(times[q] + o * steps[q], 0.0), R.n - 1.0) for o in _offsets(k)]
                plan.append((q, len(samples), k))
                samples += ts
                q += 1
            fr, _ = R.render(samples)
            for qq, s0, k in plan:
                out[qq] = fr[s0] if k == 1 else fr[s0:s0 + k].mean(0)
        return out, taps_used

    # ----------------------------------------------------------------------------------------- layer modes
    @staticmethod
    def _support(images, m, plate):
        """S: the subject's support (alpha > 0.02 grown a little) plus its shadow/light (AutoMatte Effect Zone + Mask
        on the original vs the plate), feathered by 1 px."""
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        r = max(1, round(0.003 * min(h, w)))
        dev = _device()
        core = torch.empty((n, h, w), dtype=torch.float32)
        for i in range(0, n, CHUNK):
            c = _dil((m[i:i + CHUNK].to(dev) > 0.02).float(), r)
            core[i:i + CHUNK] = _box(c, 1).clamp(0, 1).cpu() if min(h, w) > 4 else c.cpu()
        info = {}
        try:
            from .effects import AutoMatteEffectZone, AutoMatteEffectMask
            zone = AutoMatteEffectZone().run(m, images)[0]
            paste, erep = AutoMatteEffectMask().run(images, plate, m, zone, object_grow=0.003)
            S = torch.maximum(core, paste.float())
            eff = (paste > 0.5) & (m < 0.02)
            info["effects_share"] = round(float(eff.float().mean()), 5)
            try:
                info["effects_report"] = json.loads(erep)
            except Exception:
                pass
        except Exception as e:  # older pack, or no effects found
            S = core
            info["effects"] = f"not separated ({type(e).__name__})"
        return S, info

    def _layers(self, images, m, P, tm, ip, lock, rep):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        dev = _device()
        S, info = self._support(images, m, P)
        rep.update(info)
        B = torch.empty_like(images[..., :3], dtype=torch.float32)
        for i in range(0, n, CHUNK):
            s = S[i:i + CHUNK].to(dev)[..., None]
            B[i:i + CHUNK] = (s * P[i:i + CHUNK].to(dev) + (1 - s) * images[i:i + CHUNK, ..., :3].to(dev).float()).cpu()
        # camera and lock
        cam = _Camera(B, support=S)
        cam.set_subject(m)
        lock = _pick(lock, LOCKS, "Auto")
        decision, linfo = cam.lock_rule(m) if lock == "Auto" else (lock, {})
        use_lock = decision == "Scene" and not cam.static
        rep["camera"] = {"static": cam.static, "total_over_clip": cam.total,
                         "shift_share_per_frame": round(cam.shift_per_frame, 5),
                         "scale_change_per_frame": round(cam.scale_per_frame, 5), "lock": decision, **linfo}
        RI = _Frames(images[..., :3], m, ip, "I")
        RB = _Frames(B, None, ip, "B")
        TS, TB = tm["subject"], tm["background"]
        if tm["mode"] == "Hero Freeze" and tm.get("loop"):
            # cinemagraph: one seamless loop of the background behind the held subject
            Bseq, _ = RB.render(TB)
            loop_src = _Frames(Bseq, None, ip, "Bseq")
            Bl, _, plan, nominal = loop_frames(loop_src, tm["fps_out"])
            rep["loop"] = plan
            TB = [TB[min(len(TB) - 1, int(round(x)))] for x in nominal]
            TS = [TS[0]] * len(TB)
            bg_frames = Bl
        else:
            bg_frames = None
        Q = len(TS)
        st_s, st_b = _steps(TS), _steps(TB)
        out = torch.empty((Q, h, w, 3), dtype=torch.float32)
        mo = torch.empty((Q, h, w), dtype=torch.float32)
        fill, busy, light, taps_used, q = [], [], [], 1, 0
        probe = set(torch.linspace(0, Q - 1, min(Q, 24)).round().long().tolist())
        while q < Q:
            plan, flat_s, flat_b = [], [], []
            while q < Q and (not flat_s or len(flat_s) < 16):
                k = 1 if bg_frames is not None else _taps(max(st_s[q], st_b[q]))
                taps_used = max(taps_used, k)
                offs = _offsets(k)
                plan.append((q, len(flat_s), k))
                flat_s += [min(max(TS[q] + o * st_s[q], 0.0), n - 1.0) for o in offs]
                flat_b += [min(max(TB[q] + o * st_b[q], 0.0), n - 1.0) for o in offs]
                q += 1
            Is, As = RI.render(flat_s)
            Bs, _ = RB.render(flat_s)
            if bg_frames is not None:
                Bb = torch.stack([bg_frames[qq] for qq, _, _ in plan])
            else:
                Bb, _ = RB.render(flat_b)
            for qq, s0, k in plan:
                T = cam.T_scene(TS[qq], TB[qq]) if use_lock else None
                acc = acc_a = None
                for j in range(s0, s0 + k):
                    i_s, a_s, b_s, b_b = Is[j].to(dev), As[j].to(dev), Bs[j].to(dev), Bb[j].to(dev)
                    if T is not None:
                        i_s, b_s, a_s = warp(i_s, T), warp(b_s, T), warp(a_s, T, pad="zeros")
                    comp = (i_s + (1 - a_s[..., None]) * (b_b - b_s)).clamp(0, 1)
                    acc = comp if acc is None else acc + comp
                    acc_a = a_s if acc_a is None else acc_a + a_s
                out[qq] = (acc / k).cpu() if k > 1 else acc.cpu()
                mo[qq] = (acc_a / k).cpu() if k > 1 else acc_a.cpu()
                if qq in probe:
                    tb_i = int(round(min(max(TB[qq], 0), n - 1)))
                    a_c = mo[qq].to(dev)
                    s_b = S[tb_i].to(dev)
                    fill.append(float(((1 - a_c) * s_b).mean()))
                    subj = a_c > 0.5
                    if int(subj.sum()) > 16:
                        b_now = Bb[s0].to(dev)
                        b_then = Bs[s0].to(dev) if T is None else warp(Bs[s0].to(dev), T)
                        busy.append(float(((b_now - b_then).abs().mean(-1)[subj] > 0.12).float().mean()))
                    outside = (S[tb_i].to(dev) < 0.5) & (S[int(round(min(max(TS[qq], 0), n - 1)))].to(dev) < 0.5)
                    if int(outside.sum()) > 64:
                        Lb = srgb_to_ok(Bb[s0].to(dev))[..., 0][outside].mean()
                        Ls = srgb_to_ok(Bs[s0].to(dev))[..., 0][outside].mean()
                        light.append(abs(float(Lb - Ls)))
        self._blended = RI.blended + RB.blended
        rep["shutter_taps_max"] = taps_used
        rep["plate_fill_share_max"] = round(max(fill), 4) if fill else 0.0
        rep["busy_behind_subject_max"] = round(max(busy), 4) if busy else 0.0
        rep["lighting_change_max"] = round(max(light), 2) if light else 0.0
        # self-check: at the anchor both layers show the hero frame: the output must be that frame
        a = tm.get("anchor")
        if a is not None and not tm.get("loop") and 0 <= a < Q and abs(TS[a] - TB[a]) < EPS_T \
                and abs(TS[a] - round(TS[a])) < EPS_T and _taps(max(st_s[a], st_b[a])) == 1:
            err = float((out[a] - images[int(round(TS[a])), ..., :3].float()).abs().max())
            rep["self_check_max_error"] = round(err * 255, 4)
            if err > 1 / 255:
                rep["review_reasons"].append(f"Internal check: the anchor frame differs from the source by "
                                             f"{err * 255:.2f}/255 (it should be exact)")
        # contact and matte-edge checks
        k_hero = int(tm["hero_frame"]) - 1
        agree = _edge_agreement(images[k_hero, ..., :3].float(), m[k_hero])
        if agree is not None:
            rep["matte_edge_agreement"] = round(agree, 3)
            if agree < 0.7:
                rep["review_reasons"].append(
                    f"At frame {k_hero + 1} the matte's edge doesn't follow the picture closely: old background inside "
                    "the matte may travel with the subject (a halo). A tighter matte or another hero frame helps")
        rep["contact_score"] = round(self._contact(B, m), 2)
        if rep["contact_score"] > 1.5:
            rep["review_reasons"].append("The subject touches something that moves (a hand, a leash, a moving "
                                         "surface): name both in the Subject so they keep the same timing")
        if rep["plate_fill_share_max"] > 0.15:
            rep["review_reasons"].append(f"Up to {rep['plate_fill_share_max'] * 100:.0f}% of a frame is rebuilt "
                                         "background (where the subject was): check the plate there")
        if rep["busy_behind_subject_max"] > 0.25:
            rep["review_reasons"].append("Something moves where the subject is drawn: check that nothing passes in "
                                         "front of it (it would be hidden behind the subject)")
        if rep["lighting_change_max"] > 4:
            rep["review_reasons"].append("The light on the background changes between the two moments shown: the "
                                         "subject may not match it")
        if not cam.static and decision == "Frame" and lock == "Auto":
            rep["notes"].append("The camera follows the subject: the subject stays where it is on the frame")
        if not cam.static and decision == "Scene":
            rep["notes"].append("The subject is moved with the camera so it stays in place in the scene")
        return out, mo

    @staticmethod
    def _contact(B, m, samples=12, r=4):
        """Background change right around the subject vs elsewhere (consecutive frames, thumbnails). High = the subject
        touches something moving."""
        n = B.shape[0]
        if n < 2:
            return 0.0
        idx = torch.linspace(0, n - 2, min(samples, n - 1)).round().long().unique().tolist()
        g = _thumbs(torch.stack([B[i] for i in idx] + [B[i + 1] for i in idx]), 128)
        mm = F.interpolate(torch.stack([m[i] for i in idx]).float()[:, None], size=g.shape[1:], mode="area")[:, 0] > 0.5
        k = len(idx)
        vals = []
        for q in range(k):
            a = mm[q].float()[None]
            ring = (_dil(a, r)[0] > 0.5) & ~mm[q]
            far = ~(_dil(a, 4 * r)[0] > 0.5)
            d = (g[k + q] - g[q]).abs()
            if int(ring.sum()) > 8 and int(far.sum()) > 8:
                vals.append((float(d[ring].mean()) + 1e-3) / (float(d[far].mean()) + 1e-3))
        return float(torch.tensor(vals).median()) if vals else 0.0

    # ------------------------------------------------------------------------------------------------ audio
    @staticmethod
    def _audio(audio, tm, rep):
        if audio is None:
            return None
        n_out, fps, fps_out = tm["n_out"], tm["fps_src"], tm["fps_out"]
        bg = tm["background"]
        same = (abs(fps - fps_out) < 1e-6 and not tm.get("loop") and len(bg) > 0
                and all(abs(bg[i] - (bg[0] + i)) < EPS_T for i in range(len(bg))))
        if tm["mode"] == "Whole Clip":
            same = same and all(abs(a - b) < EPS_T for a, b in zip(tm["subject"], bg))
        if not same:
            rep["notes"].append("Audio left out: the timing changed")
            return None
        try:
            wf, sr = audio["waveform"], int(audio["sample_rate"])
            s0 = int(round(bg[0] / fps * sr))
            s1 = s0 + int(round(n_out / fps * sr))
            return {"waveform": wf[..., s0:s1], "sample_rate": sr}
        except Exception:
            return audio

    # -------------------------------------------------------------------------------------------------- run
    def run(self, images, time_map, interp_model=None, matte=None, plate=None, layer_report=None, lock="Auto",
            audio=None):
        tm = _load_tm(time_map)
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if tm["n_src"] != n:
            raise ValueError(f"The time map is for {tm['n_src']} frames but the clip has {n}.")
        ip = _Interp(interp_model) if interp_model is not None else None
        if ip is not None:
            ip.load(h, w)
        rep = {"edit": "retime", "mode": tm["mode"], "frames_in": n, "fps_in": tm["fps_src"],
               "frames_out": tm["n_out"], "fps_out": tm["fps_out"], "seconds_out": tm["out_seconds"],
               "hero_frame": tm["hero_frame"], "hero_source": tm["hero_source"],
               "source_range_s": tm["source_range_s"], "notes": list(tm.get("notes", [])),
               "review_reasons": list(tm.get("review_reasons", []))}
        for k in ("speed", "ramp", "speed_curve", "out_seconds", "ramp_width_s"):
            if k in tm:
                rep[k] = tm[k]
        if layer_report:
            try:
                lr = json.loads(layer_report)
                for k, v in lr.items():
                    if k in ("notes", "review_reasons") and isinstance(v, list):
                        rep[k] += v
                    elif k not in rep:
                        rep[k] = v
            except Exception:
                pass
        has_matte = tm["mode"] != "Whole Clip"
        if not has_matte:
            R = _Frames(images[..., :3], None, ip, "I")
            out, taps = self._render_single(R, tm["subject"])
            rep["shutter_taps_max"] = taps
            mo = None
            blended = R.blended
        else:
            if matte is None or plate is None:
                raise ValueError(f"{tm['mode']} needs the subject's matte and a clean plate.")
            m = _fit_mask(matte, n, h, w)
            P = _fit_frames(plate, n, h, w)
            out, mo = self._layers(images, m, P, tm, ip, lock, rep)
            blended = getattr(self, "_blended", 0)
        if tm.get("loop") and not (tm["mode"] == "Hero Freeze"):
            Lp = _Frames(out, mo, ip, "L")
            out, mo, plan, _ = loop_frames(Lp, tm["fps_out"])
            rep["loop"] = plan
            blended += Lp.blended
        if "loop" in rep:
            rep["frames_out"] = out.shape[0]
            rep["seconds_out"] = round(out.shape[0] / tm["fps_out"], 3)
            if rep["loop"].get("seam_ratio") is not None and rep["loop"]["style"] == "pingpong":
                rep["notes"].append("No matching loop points: the loop plays forward then back")
        if ip is not None:
            rep["interp"] = {"model": ip.kind, "matte_same_flow": bool(ip.co_warp), "frame_pairs": ip.pairs,
                             "in_betweens": ip.made}
        if blended:
            rep["notes"].append(f"{blended} in-betweens were linear blends (no interpolation model connected)")
        audio_out = self._audio(audio, tm, rep)
        if mo is not None:
            matte_img = mo[..., None].repeat(1, 1, 1, 3)
        else:
            matte_img = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
        return (out, matte_img, has_matte, audio_out, json.dumps(rep))


NODE_CLASS_MAPPINGS = {
    "AutoMatteTimeMap": AutoMatteTimeMap,
    "AutoMattePickInterpModel": AutoMattePickInterpModel,
    "AutoMatteGivenOrMadePlate": AutoMatteGivenOrMadePlate,
    "AutoMatteRetime": AutoMatteRetime,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteTimeMap": "AutoMatte: Time Map (retime)",
    "AutoMattePickInterpModel": "AutoMatte: Pick Interpolation Model",
    "AutoMatteGivenOrMadePlate": "AutoMatte: Given or Made Clean Plate",
    "AutoMatteRetime": "AutoMatte: Retime",
}
