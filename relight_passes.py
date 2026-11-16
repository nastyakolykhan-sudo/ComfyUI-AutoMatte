"""AutoMatte relight from passes: the same take lit by a chosen sun, worked out from the clip's own geometry (a
SwitchLight-style relight), so the light direction is exact instead of guessed by a video model.

- AutoMatteRelightFromPasses  source frames + normal, base colour and depth passes (AutoMatte Marigold Passes and
                              Depth Passes) + the Relight Plan (Look, Sun Direction) -> relit frames.

How it works, per frame, at a processing size (the light is smooth, so it's computed small and scaled up):
1. New light: a sun from the plan's direction (rotation 0 = from the camera side, 90 = from the right, 180 = from
   behind; elevation above the horizon) with the Look's colour and strength, a sky/ground hemisphere fill, Lambert
   shading on the normals (softened for soft looks) and cast shadows marched through the depth pass toward the sun
   (screen-space heightfield shadows: only things nearer the camera can shadow; the penumbra follows the look).
2. Light as shot: source / base colour, smoothed along the depth pass's edges (so it follows geometry, not texture,
   and doesn't bleed across a silhouette), with the clip's own light colour taken out, so the old tint doesn't stay.
3. Output = source pixels x (new light / light as shot): every texture, face and print stays the source's; only the
   light changes. Exposure is set once per clip (no pumping), the light is steadied over time along the optical flow
   (the passes are estimated per frame; the light should only change where something moves), and sky (the far end of
   the depth pass, if bright) takes the Look's sky tint instead of being shaded.

Torch, plus OpenCV's optical flow when present (a pack requirement); no model weights. Pass-agnostic: any normal (camera space, (n+1)/2), sRGB base colour and near = 1 depth
works. Night looks are not handled here (Relight's Day-to-Night route does them).
"""

import json
import math

import torch
import torch.nn.functional as F

try:                       # OpenCV (a pack requirement) gives optical flow; without it the light is steadied in place
    import cv2
except Exception:          # noqa: BLE001
    cv2 = None

from .nodes import _device, _resize_frames, CHUNK
from .layers import _box
from .relight import _lin, _srgb, _luma, _gauss, _key, structure_lock

# Look -> light. sun / sky / ground colours are linear RGB multipliers; wrap softens the terminator (0 = hard);
# hardness sets the shadow penumbra (high = crisp); shadow = how dark full shadow gets (1 = only fill remains);
# exposure = the clip's median brightness relative to the source; sky_tint multiplies the sky.
LIGHT_LOOKS = {
    "golden hour": dict(sun=(1.00, 0.66, 0.36), sun_k=2.3, sky=(0.33, 0.40, 0.62), sky_k=0.45, ground=(0.45, 0.30, 0.18),
                        ground_k=0.22, wrap=0.05, hardness=14.0, shadow=1.0, exposure=0.95, sky_tint=(1.18, 0.92, 0.70)),
    "warm afternoon": dict(sun=(1.00, 0.84, 0.62), sun_k=2.1, sky=(0.42, 0.50, 0.70), sky_k=0.50, ground=(0.40, 0.33, 0.25),
                           ground_k=0.22, wrap=0.12, hardness=10.0, shadow=0.95, exposure=1.0, sky_tint=(1.08, 1.0, 0.90)),
    "midday sun": dict(sun=(1.00, 0.97, 0.92), sun_k=2.5, sky=(0.48, 0.60, 0.86), sky_k=0.55, ground=(0.38, 0.36, 0.33),
                       ground_k=0.20, wrap=0.0, hardness=24.0, shadow=1.0, exposure=1.05, sky_tint=(0.95, 1.0, 1.10)),
    "low sun": dict(sun=(1.00, 0.86, 0.70), sun_k=2.4, sky=(0.40, 0.48, 0.70), sky_k=0.45, ground=(0.40, 0.32, 0.24),
                    ground_k=0.22, wrap=0.02, hardness=20.0, shadow=1.0, exposure=0.95, sky_tint=(1.06, 0.98, 0.92)),
    "soft daylight": dict(sun=(1.00, 0.97, 0.94), sun_k=1.1, sky=(0.60, 0.67, 0.82), sky_k=0.95, ground=(0.40, 0.38, 0.35),
                          ground_k=0.30, wrap=0.45, hardness=4.0, shadow=0.6, exposure=1.0, sky_tint=(1.0, 1.0, 1.04)),
    "overcast": dict(sun=(0.90, 0.92, 0.96), sun_k=0.35, sky=(0.72, 0.76, 0.83), sky_k=1.15, ground=(0.38, 0.38, 0.38),
                     ground_k=0.35, wrap=0.8, hardness=2.5, shadow=0.35, exposure=0.9, sky_tint=(0.92, 0.95, 1.0)),
    "backlit": dict(sun=(1.00, 0.80, 0.55), sun_k=2.6, sky=(0.32, 0.38, 0.56), sky_k=0.50, ground=(0.42, 0.32, 0.22),
                    ground_k=0.25, wrap=0.0, hardness=16.0, shadow=1.0, exposure=0.85, sky_tint=(1.15, 0.95, 0.78)),
}
DEFAULT_LOOK = "golden hour"
STEADY_SIGMA = 0.08  # source log-luminance change that counts as "the picture changed here" (grain is ~0.02)
SKY_DEPTH = 0.02     # the far end of the depth pass (clip-normalised, near = 1) that can be sky


SIDE_PULL = 20.0   # a pure side sun (rotation +-90) is pulled this far toward the camera, as a 3/4 key


def key_rotation(rotation):
    """The rotation the light is built at: a pure side sun leaves every camera-facing surface (faces, walls) exactly on
    the terminator, so the frame goes flat; pulling it 20 degrees toward the camera keeps it reading as side light
    while those surfaces get modelling. Front, 45-degree and back directions are kept as given."""
    r = ((float(rotation) + 180.0) % 360.0) - 180.0
    if 60.0 <= abs(r) <= 120.0:
        return math.copysign(max(0.0, abs(r) - SIDE_PULL * (1 - abs(abs(r) - 90.0) / 30.0)), r)
    return r


def light_vector(rotation, elevation):
    """Unit vector toward the sun in camera space (x right, y up, z toward the camera), Sphere-Light-Render's
    convention: rotation 0 = from the camera side, 90 = from the right, 180 = from behind, -90/270 = from the left."""
    az, el = math.radians(float(rotation)), math.radians(float(elevation))
    return (math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))


def cast_shadow(height, L, hardness, steps=48, reach=0.22):
    """Screen-space heightfield shadows. height: (N,H,W) in pixels toward the camera (near = high). L: unit vector
    toward the light. March toward the light in image space; only samples nearer the camera than the receiver can
    occlude (so a receding floor or a flat wall never shadows itself), and they must stand above the ray, which climbs
    by L_z per pixel of L_xy when the light is on the camera side (flat for side and back light). Occluders fade out
    toward the end of the reach (no hard cut), and the map is blurred by a penumbra that widens as the look gets
    softer (hardness: high = crisp). Returns (N,H,W) in [0, 1], 1 = lit."""
    n, h, w = height.shape
    lx, ly, lz = L
    lxy = math.hypot(lx, ly)
    if lxy < 1e-3:                                   # light straight from the camera or straight behind: no screen shadow
        return torch.ones_like(height)
    step = max(1.0, reach * w / steps)
    dx, dy = lx / lxy * step, -ly / lxy * step       # image y runs down
    dz = max(0.0, lz / lxy * step)                   # ray height gained per step (light on the camera side only)
    dev = height.device
    ys = torch.arange(h, device=dev, dtype=height.dtype).view(1, h, 1).expand(n, h, w)
    xs = torch.arange(w, device=dev, dtype=height.dtype).view(1, 1, w).expand(n, h, w)
    flat = height.reshape(n, -1)
    lit = torch.ones_like(height)
    bias = 0.006 * w
    for t in range(1, steps + 1):
        x, y = xs + dx * t, ys + dy * t
        inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        idx = (y.clamp(0, h - 1).round().long() * w + x.clamp(0, w - 1).round().long()).reshape(n, -1)
        sample = torch.gather(flat, 1, idx).reshape(n, h, w)
        blocked = inside & (sample > height + bias) & (sample > height + dz * t + bias)
        fade = min(1.0, max(0.0, (t / steps - 0.6) / 0.4))   # occluders near the end of the reach count less
        lit = torch.where(blocked, torch.minimum(lit, torch.full_like(lit, fade)), lit)
    return _gauss(lit, max(0.5, 0.15 * w / hardness)).clamp(0, 1)


def _guided_log_ratio(guide, num, den, r, eps=1e-3):
    """Light as shot, smoothed along the guide's edges: exp(guided(log(num / den))), so it follows geometry (the depth
    pass) instead of texture and doesn't bleed across a silhouette. (N,H,W) in, (N,H,W) out, unclamped."""
    p = torch.log(num.clamp_min(1e-4) / den.clamp_min(0.02))
    mi, mp = _box(guide, r), _box(p, r)
    a = (_box(guide * p, r) - mi * mp) / (_box(guide * guide, r) - mi * mi + eps)
    b = mp - a * mi
    return torch.exp(_box(a, r) * guide + _box(b, r))


def new_light(normal, depth, L, look, proc_w):
    """(N,H,W,3) linear light from the look's sun + hemisphere fill, with cast shadows; and the sun-only term."""
    N = normal * 2 - 1
    N = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    Lt = torch.tensor(L, dtype=N.dtype, device=N.device)
    ndl = (N * Lt).sum(-1)
    wrap = look["wrap"]
    ndl = ((ndl + wrap) / (1 + wrap)).clamp(0, 1)
    height = depth * (0.30 * proc_w)                 # relative depth -> pixels toward the camera
    sh = cast_shadow(height, L, look["hardness"])
    sh = 1 - look["shadow"] * (1 - sh)
    up = ((N[..., 1] + 1) / 2)[..., None]            # hemisphere fill: sky from above, bounce from below
    t = lambda c: torch.tensor(c, dtype=N.dtype, device=N.device)  # noqa: E731
    fill = up * t(look["sky"]) * look["sky_k"] + (1 - up) * t(look["ground"]) * look["ground_k"]
    sun = (ndl * sh)[..., None] * t(look["sun"]) * look["sun_k"]
    return fill + sun, sh


def _flow(a_u8, b_u8):
    """Dense flow (H,W,2) such that a(x) ~ b(x + flow(x)) (Farneback), or zeros without OpenCV."""
    if cv2 is None:
        return torch.zeros(a_u8.shape[0], a_u8.shape[1], 2)
    import numpy as np
    f = cv2.calcOpticalFlowFarneback(np.ascontiguousarray(a_u8), np.ascontiguousarray(b_u8), None,
                                     0.5, 4, 21, 3, 7, 1.5, 0)
    return torch.from_numpy(f)


def _warp(x, flow):
    """Sample x (H,W,C) at x + flow (H,W,2, pixels); also returns where that lands inside the frame (H,W)."""
    h, w = x.shape[0], x.shape[1]
    ys = torch.arange(h, device=x.device, dtype=x.dtype).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=x.device, dtype=x.dtype).view(1, w).expand(h, w)
    px, py = xs + flow[..., 0], ys + flow[..., 1]
    inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    grid = torch.stack([(px + 0.5) / w * 2 - 1, (py + 0.5) / h * 2 - 1], -1)[None]
    out = F.grid_sample(x.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0), inside


def steady_over_time(n, frame_log_gain, ysrc, strength_frames, sigma, dev):
    """Steady the light over time, following the motion: a forward and a backward recursive filter; each frame blends
    in its neighbour's (already steadied) light warped along the optical flow, by alpha = s / (s + 1) times a
    confidence that the warp is right (the source's own log luminance matches after the warp; 0 where the flow
    leaves the frame or meets something new). The two directions are averaged, so nothing lags.
    frame_log_gain(t) -> (H,W,3) on dev; ysrc (N,H,W) log luminance; returns (N,H,W,3) half on the CPU."""
    alpha = strength_frames / (strength_frames + 1.0)
    h, w = ysrc.shape[1], ysrc.shape[2]
    fh, fw = max(16, h // 2), max(16, w // 2)

    def gray(t):                                   # 8-bit sRGB luma at half the processing size, for the flow
        y = _srgb(torch.exp(ysrc[t].float()))[None, None]
        y = F.interpolate(y, size=(fh, fw), mode="area")[0, 0]
        return (y * 255 + 0.5).clamp(0, 255).byte().numpy()

    def flow_to(a, b, ga, gb):                     # flow from frame a to frame b at the processing size, on dev
        f = _flow(ga, gb).permute(2, 0, 1)[None].float()
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        return (f * torch.tensor([w / fw, h / fh])).to(dev)

    def run(order):
        res = [None] * n
        prev_t, prev, g_prev = None, None, None
        for t in order:
            cur = frame_log_gain(t)
            g_t = gray(t)
            if prev is None:
                out = cur
            else:
                fl = flow_to(t, prev_t, g_t, g_prev)
                y_t = ysrc[t].float().to(dev)
                warped, inside = _warp(torch.cat([prev, ysrc[prev_t].float().to(dev)[..., None]], -1), fl)
                conf = torch.exp(-((y_t - warped[..., 3]) ** 2) / (2 * sigma ** 2)) * inside.float()
                a = (alpha * conf)[..., None]
                out = (1 - a) * cur + a * warped[..., :3]
            res[t] = out.half().cpu()
            prev_t, prev, g_prev = t, out, g_t
        return res

    fwd = run(range(n))
    bwd = run(range(n - 1, -1, -1))
    return torch.stack([(fwd[t].float() + bwd[t].float()) / 2 for t in range(n)]).half()


class AutoMatteRelightFromPasses:
    """Relight a clip from its passes: the plan's Look and sun direction become a physical light on the clip's normals
    and depth (cast shadows included); the source pixels are kept and only their light changes (source x new light /
    light as shot). Exposure is set once per clip; the light is smoothed over neighbouring frames; sky takes the Look's
    tint. Outputs the relit frames, the new light (for review) and a JSON report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The source frames."}),
                "normal": ("IMAGE", {"tooltip": "Normal pass (camera space, (n+1)/2): AutoMatte Marigold Passes."}),
                "base_color": ("IMAGE", {"tooltip": "Base colour (sRGB albedo): AutoMatte Marigold Passes."}),
                "depth": ("IMAGE", {"tooltip": "Depth, near = white, normalised over the clip: AutoMatte Depth Passes."}),
                "plan": ("RELIGHT_PLAN", {"tooltip": "AutoMatte Relight Plan: Look, sun rotation and elevation."}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                                       "tooltip": "How much of the light change is applied (1 = all)."}),
                "processing_long_side": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32,
                                                 "tooltip": "Size the light is worked out at (it is smooth)."}),
                "temporal_smoothing": ("INT", {"default": 12, "min": 0, "max": 24,
                                               "tooltip": "About how many frames the light is steadied over, following "
                                                          "the motion (0 = off)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "light", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Relight"

    def run(self, images, normal, base_color, depth, plan, strength=1.0, processing_long_side=1024,
            temporal_smoothing=12):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        if plan.get("route") == "night":
            raise ValueError("Relight from Passes makes daylight looks; night looks run on the Day-to-Night route.")
        lk = _key(plan.get("look", "")) or DEFAULT_LOOK
        look = LIGHT_LOOKS.get(lk, LIGHT_LOOKS[DEFAULT_LOOK])
        rot_used = key_rotation(plan.get("rotation", -90.0))
        L = light_vector(rot_used, plan.get("elevation", 20.0))
        s = processing_long_side / max(h, w)
        ph, pw = max(16, round(h * s)), max(16, round(w * s))
        dev = _device()

        def small(x, c=3):
            x = x[..., :c].float()
            return _resize_frames(x, ph, pw, "bilinear") if x.shape[1:3] != (ph, pw) else x

        # clip constants: the source's light colour (divided out, so the old tint doesn't stay) and median brightness
        idx = torch.linspace(0, n - 1, min(n, 16)).round().long().unique()
        s_lin, a_lin = _lin(small(images[idx]).to(dev)), _lin(small(base_color[idx]).to(dev))
        d_s = small(depth[idx], 1)[..., 0].to(dev)
        ok = (d_s > SKY_DEPTH) & (_luma(a_lin) > 0.03) & (_luma(s_lin) > 0.01)
        if int(ok.sum()) < 64:
            ok = (_luma(a_lin) > 0.03) & (_luma(s_lin) > 0.01)
        old_tint = torch.stack([(s_lin[..., c][ok] / a_lin[..., c][ok].clamp_min(1e-3)).median() for c in range(3)])
        old_tint = old_tint / (old_tint * torch.tensor([0.2126, 0.7152, 0.0722], device=dev)).sum().clamp_min(1e-4)
        y_med = math.log(float(_luma(s_lin).median().clamp_min(1e-4)))
        sky_gain = torch.tensor(look["sky_tint"], device=dev) / old_tint

        def sky_weight(dep, ylog):
            """Sky = the far end of the depth pass AND brighter than the clip's median (a dim far wall at the end of the
            depth range stays land); both ramps are soft, so a value near a limit can't flip from frame to frame."""
            far = (1 - dep / SKY_DEPTH).clamp(0, 1)
            bright = ((ylog - y_med - 0.1) / 0.3).clamp(0, 1)
            return _gauss(far * bright, 1.0)

        # pass 1, per frame at the processing size (half precision): the land gain = new light / light as shot (smoothed
        # along depth edges), the sky weight, the source's log luminance (temporal weights), new light and shadow
        r_g = max(3, round(0.02 * pw))
        gains, ysrcs, shades = [], [], []
        for i in range(0, n, CHUNK):
            sl = slice(i, i + CHUNK)
            src, alb = _lin(small(images[sl]).to(dev)), _lin(small(base_color[sl]).to(dev))
            nrm, dep = small(normal[sl]).to(dev), small(depth[sl], 1)[..., 0].to(dev)
            light, sh = new_light(nrm, dep, L, look, pw)
            shot = _guided_log_ratio(dep, _luma(src), _luma(alb), r_g).clamp(0.05, 8)
            ylog = torch.log(_gauss(_luma(src), 1.0).clamp_min(1e-4))
            sky = sky_weight(dep, ylog)
            gains.append((light / (shot[..., None] * old_tint)).half().cpu())
            ysrcs.append(ylog.half().cpu())
            shades.append(torch.stack([_luma(light), sh, sky], -1).half().cpu())
        gain, ysrc, shade = torch.cat(gains, 0), torch.cat(ysrcs, 0), torch.cat(shades, 0)
        del gains, ysrcs, shades

        # exposure, once per clip: the median relit brightness (sky left out) lands on the source's x the look's
        # exposure; the sky only takes the look's tint, so a darker scene never blows it out
        src_s = _lin(small(images[idx]).to(dev))
        rel_s = src_s * gain[idx].float().to(dev)
        land = shade[idx, ..., 2].float().to(dev) < 0.5
        if int(land.sum()) < 64:
            land = torch.ones_like(land)
        k_exp = float(look["exposure"] * _luma(src_s)[land].median() / _luma(rel_s)[land].median().clamp_min(1e-5))
        l_top = float(torch.quantile(shade[idx, ..., 0].float().flatten()[::7], 0.995).clamp_min(1e-4))

        def log_gain(jj):
            g = gain[jj].float().to(dev)
            sky = shade[jj, ..., 2].float().to(dev)[..., None]
            return torch.log((g * k_exp).clamp(0.08, 8.0) * (1 - sky) + sky_gain * sky)

        # pass 2: steady the light over time along the motion (the passes are estimated per frame and flicker; the
        # light should only change where something moves), set strength, apply at full size
        sm = int(temporal_smoothing) if n > 1 else 0
        if sm > 0:
            lgain = steady_over_time(n, lambda t: log_gain(torch.tensor([t]))[0], ysrc, sm, STEADY_SIGMA, dev)
        out, lock, light_out = [], [], []
        for i in range(0, n, CHUNK):
            j = torch.arange(i, min(i + CHUNK, n))
            lg = lgain[j].float().to(dev) if sm > 0 else log_gain(j)
            g = torch.exp(lg * float(strength))
            src = images[j, ..., :3].to(dev).float()
            g_full = F.interpolate(g.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                                   align_corners=False).permute(0, 2, 3, 1)
            o = _srgb(_lin(src) * g_full)
            lock.append(structure_lock(src, o).cpu())
            out.append(o.cpu())
            lv = _srgb((shade[j, ..., 0].float().to(dev) / l_top).clamp(0, 1))
            light_out.append(F.interpolate(lv[:, None], size=(h, w), mode="bilinear",
                                           align_corners=False)[:, 0, ..., None].expand(-1, -1, -1, 3).cpu())
        res = torch.cat(out, 0).clamp(0, 1).contiguous()
        light_img = torch.cat(light_out, 0).clamp(0, 1).contiguous()
        shadow = shade[..., 1].float()
        lk_t = torch.cat(lock, 0)
        rep = {"edit": "relight_from_passes", "look": plan.get("look"), "sun_direction": plan.get("sun_direction"),
               "rotation": plan.get("rotation"), "light_rotation": round(rot_used, 1), "elevation": plan.get("elevation"),
               "light_vector": [round(v, 3) for v in L], "processing_size": [pw, ph],
               "old_light_tint": [round(float(v), 3) for v in old_tint], "exposure_gain": round(k_exp, 3),
               "shadow_share": round(float((shadow < 0.5).float().mean()), 3),
               "sky_share": round(float((shade[..., 2] > 0.5).float().mean()), 3),
               "structure_lock_mean": round(float(lk_t.mean()), 3), "structure_lock_min": round(float(lk_t.min()), 3),
               "strength": strength, "temporal_smoothing": temporal_smoothing,
               "motion_following": cv2 is not None}
        return (res, light_img, json.dumps(rep))


NODE_CLASS_MAPPINGS = {"AutoMatteRelightFromPasses": AutoMatteRelightFromPasses}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoMatteRelightFromPasses": "AutoMatte: Relight from Passes (sun from geometry)"}
