"""AutoMatte relight: the same take in other light (golden hour, a new sun direction, overcast, night) with
Lightricks' LTX IC-LoRAs, while the subject's fine detail and label text stay as shot.

- AutoMatteRelightPlan     turns a Look (+ Sun Direction) and a VLM's scene notes into everything technical:
                           which LoRA route (sun relight or day-to-night), the prompt, the light-ball angles, the
                           generation canvas, the temporal window, the conditioning fps, the subject to protect and
                           warnings (interior + sun look, portrait, high fps).
- AutoMatteRelightGuide    the IC-LoRA reference: frames scaled to the canvas, padded to /32 (mirror), and for the sun
                           route a light-direction ball drawn in the top-right corner of every frame (in a mirrored
                           strip added on the right, or over the picture), frames padded to 8n+1.
- AutoMatteRelightRestore  back to the source: crop the strip/padding, scale up, put the source's fine detail back
                           (frequency split in linear light, stronger inside the subject matte so label strokes and
                           faces come from the source), optionally keep the subject exactly as shot, and a per-frame
                           structure-lock score (do the edges still match the source?) in the report.

The ball renders in assets/light_balls/ were baked once from Sphere-Light-Render-ComfyUI @ cc89da6 (MIT, three.js r128,
same scene and light maths); that node renders in the browser only, so an API run could not change the direction.
See assets/light_balls/NOTICE.md. No model weights, only torch (PIL, which ComfyUI ships, to read the PNGs).
"""

import json
import math
import os
import re

import torch
import torch.nn.functional as F

from .nodes import _device, _resize_frames, CHUNK
from .layers import _video_given, _fit_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
BALL_DIR = os.path.join(_HERE, "assets", "light_balls")

# ------------------------------------------------------------------------------------------------ the looks
# Sun looks use the trained phrases of LTX-2.3 Relight (model card); elevations follow its altitude bands
# (< 33 low, 33-57 mid, >= 58 high). auto: how Sun Direction = Auto is chosen.
LOOKS = {
    "golden hour": {"night": False, "phrase": None, "elevation": 15, "auto": "mirror"},
    "warm afternoon": {"night": False, "phrase": "soft warm afternoon light", "elevation": 35, "auto": "mirror"},
    "midday sun": {"night": False, "phrase": "hard high-angle sunlight", "elevation": 70, "auto": "mirror"},
    "low sun": {"night": False, "phrase": "hard low-angle sunlight", "elevation": 20, "auto": "mirror"},
    "soft daylight": {"night": False, "phrase": "soft diffused daylight", "elevation": 45, "auto": "front left"},
    "overcast": {"night": False, "phrase": "dim overcast light", "elevation": 60, "auto": "front left"},
    "backlit": {"night": False, "phrase": "strong backlight with rim light", "elevation": 25, "auto": "behind"},
    "night": {"night": True, "style": "moon"},
    "night with city lights": {"night": True, "style": "city"},
}
LOOK_OPTIONS = "Golden Hour, Warm Afternoon, Midday Sun, Low Sun, Soft Daylight, Overcast, Backlit, Night, Night with City Lights"
DIRECTION_OPTIONS = "Auto, Left, Front Left, Front, Front Right, Right, Back Left, Behind, Back Right"
# rotation as in Sphere-Light-Render: 0 = light from the camera side, 90 = from the right, 180 = from behind
DIRECTIONS = {"front": 0, "front right": 45, "right": 90, "back right": 135, "behind": 180, "back left": -135,
              "left": -90, "front left": -45}
DIR_PHRASE = {0: "the front", 45: "the front-right", 90: "the right", 135: "the back-right", 180: "behind",
              -135: "the back-left", -90: "the left", -45: "the front-left"}
RELIGHT_TRIGGER = "relight the video to match the light-direction ball."
NIGHT_TAIL = "Only the lighting changes from day to night; identical composition, framing, camera movement and motion."
NIGHT_STYLE = {
    ("moon", "exterior"): "Photorealistic moonlight with cool blue ambient light and deep natural shadows.",
    ("moon", "interior"): "Dim cool moonlight through the windows, low-key light and deep natural shadows.",
    ("city", "exterior"): "Warm street lamps and lit windows cast amber pools of light against a cool night sky, "
                          "with deep shadows.",
    ("city", "interior"): "Warm lamps light the room with soft amber light while the windows show a dark night outside.",
}
# The question the VLM (Qwen3-VL 4B, core TextGenerate) answers on the clip's middle frame; Plan parses the answer.
VLM_PROMPT = (
    "This is a frame from a video clip. Answer with one line of JSON and nothing else, with these keys: "
    "\"setting\": \"exterior\" or \"interior\"; "
    "\"light\": where the sun or main light comes from as seen from the camera, one of \"left\", \"right\", "
    "\"front\", \"behind\", \"above\", \"overcast\", \"none\"; "
    "\"scene\": one short sentence on what is in the shot, without light, time of day or weather; "
    "\"subject\": the one object, person or animal at the centre of attention, as a short generic noun phrase an "
    "object detector can find (for example serum bottle, woman in beige coat, dalmatian, red car; not a brand name "
    "or the text on it), or \"none\" for a landscape or a shot without one main subject."
)

TRAINED_SUN = (1280, 704)        # LTX-2.3 Relight: 1280x704, 121 frames
NIGHT_CAP = (1920, 1088)         # Lightricks' 2.5 ladder stage 1 (FullHD, one spatial tile)
BALL_FRAC, MARGIN_FRAC = 143 / 704, 22 / 704   # ball patch and margin as a share of the canvas's short side
_DAY_WORDS = r"\b(sunny|sunlit|sun-drenched|sunlight|sunshine|daylight|daytime|bright|midday|morning|afternoon|" \
             r"blue sky|clear sky|golden hour|sunset|sunrise)\b"


def _ceil(v, m):
    return int(math.ceil(v / m) * m)


def _key(text):
    return re.sub(r"[\s\-_]+", " ", (text or "").strip().lower())


def parse_scene(text):
    """VLM answer -> {setting, light, scene, subject, parsed}. Tolerates code fences, prose around the JSON and
    single keys; anything missing falls back to 'unknown' / ''."""
    out = {"setting": "unknown", "light": "unknown", "scene": "", "subject": "", "parsed": False}
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = None
    if not isinstance(data, dict):  # key by key, e.g. a truncated answer
        data = {}
        for k in ("setting", "light", "scene", "subject"):
            mk = re.search(r'"%s"\s*:\s*"([^"]*)' % k, t)
            if mk:
                data[k] = mk.group(1)
    if data:
        out["parsed"] = True
    s = _key(data.get("setting", ""))
    if "inter" in s or "indoor" in s or "inside" in s:
        out["setting"] = "interior"
    elif "exter" in s or "outdoor" in s or "outside" in s:
        out["setting"] = "exterior"
    li = _key(data.get("light", ""))
    for word in ("overcast", "behind", "above", "front", "left", "right", "none"):
        if word in li:
            out["light"] = word
            break
    if "back" in li and out["light"] in ("unknown", "left", "right"):
        out["light"] = "behind"
    scene = re.sub(_DAY_WORDS, "", str(data.get("scene", "")), flags=re.I)
    scene = re.sub(r"\s+", " ", scene).strip(" .,;")
    out["scene"] = scene[:200]
    subj = str(data.get("subject", "")).strip().strip(".\"'")
    out["subject"] = "" if _key(subj) in ("", "none", "no", "n/a", "nothing", "unknown") else subj[:60]
    return out


def auto_rotation(rule, light):
    """Sun Direction = Auto: a side unlike the light as shot ('mirror'), or the look's fixed default."""
    if rule == "behind":
        return 180
    if rule == "front left":
        return -45
    return {"left": 90, "right": -90, "front": 90}.get(light, -45)


def golden_phrase(rot):
    if rot == 0:
        return "warm golden low front sun"
    if abs(rot) >= 135:
        return "soft hazy backlight"
    return "warm golden low side sun"


def relight_canvas(w, h):
    """Picture size for the sun relight: the trained window (1280x704 or 704x1280; area for near-square clips),
    never larger than the source; padded up to /32."""
    long_, short = max(w, h), min(w, h)
    if long_ / short < 1.3:
        s = min(1.0, math.sqrt(TRAINED_SUN[0] * TRAINED_SUN[1] / (w * h)))
    else:
        s = min(1.0, TRAINED_SUN[0] / long_, TRAINED_SUN[1] / short)
    return max(32, round(w * s)), max(32, round(h * s))


def night_canvas(w, h):
    """Picture size for day-to-night: Lightricks' FullHD stage-1 canvas (long 1920, short 1088), never larger than
    the source."""
    long_cap, short_cap = NIGHT_CAP
    s = min(1.0, (long_cap if w >= h else short_cap) / w, (short_cap if w >= h else long_cap) / h)
    return max(32, round(w * s)), max(32, round(h * s))


def window_frames(n_padded, trained):
    """Temporal window for Tiled Fusion streaming, as Lightricks' tile_frames_for: the whole clip up to 121 frames
    (no chunking), else the LoRA's trained length (97 or 121). Always 8n+1."""
    return n_padded if n_padded <= 121 else trained


def make_plan(look, sun_direction, width, height, frames, fps, scene_text="", subject_typed="", matte_given=False,
              prompt_suffix="", ball_placement="strip", ball_intensity=0.0):
    lk = _key(look)
    if lk not in LOOKS:
        raise ValueError(f"Look '{look}' isn't one of: {LOOK_OPTIONS}.")
    L = LOOKS[lk]
    scene = parse_scene(scene_text)
    notes, warnings = [], []
    if scene_text and not scene["parsed"]:
        notes.append("The scene check's answer could not be read; defaults were used (Auto direction = front left)")
    n8 = frames if (frames - 1) % 8 == 0 else frames + (8 - (frames - 1) % 8)
    plan = {"edit": "relight", "look": look.strip(), "route": "night" if L["night"] else "sun",
            "source_size": [width, height], "frames": frames, "frames_padded": n8, "fps": round(float(fps), 3),
            "cond_fps": round(min(float(fps), 30.0), 3), "setting": scene["setting"], "light_as_shot": scene["light"]}
    if L["night"]:
        cw, ch = night_canvas(width, height)
        style = NIGHT_STYLE[(L["style"], "interior" if scene["setting"] == "interior" else "exterior")]
        head = "A realistic nighttime scene." if scene["setting"] == "interior" else "A realistic nighttime outdoor scene."
        parts = [head]
        if scene["scene"]:
            parts.append(scene["scene"] + ".")
        parts.append(style)
        if (prompt_suffix or "").strip():
            parts.append(prompt_suffix.strip().rstrip(".") + ".")
        parts.append(NIGHT_TAIL)
        plan.update(prompt=" ".join(parts), rotation=0.0, elevation=0.0, intensity=0.0, ball=None,
                    sun_direction="(not used at night)", lora="ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors",
                    tile_frames=window_frames(n8, 97))
        strip = 0
        if (sun_direction or "").strip() and _key(sun_direction) != "auto":
            notes.append("Sun Direction is not used by night looks")
    else:
        cw, ch = relight_canvas(width, height)
        sd = _key(sun_direction) or "auto"
        if sd == "auto":
            rot = auto_rotation(L["auto"], scene["light"])
            how = "auto"
        elif sd in DIRECTIONS:
            rot, how = DIRECTIONS[sd], "chosen"
        else:
            raise ValueError(f"Sun Direction '{sun_direction}' isn't one of: {DIRECTION_OPTIONS}.")
        phrase = golden_phrase(rot) if L["phrase"] is None else L["phrase"]
        dphrase = DIR_PHRASE[rot]
        prompt = f"{RELIGHT_TRIGGER} {phrase} from {dphrase}"
        intensity = float(ball_intensity) if ball_intensity and ball_intensity > 0 else 1.5
        W, H = _ceil(cw, 32), _ceil(ch, 32)
        short = min(W, H)
        patch, margin = round(BALL_FRAC * short), round(MARGIN_FRAC * short)
        placement = "overlay" if _key(ball_placement) == "overlay" else "strip"
        strip = _ceil(patch + 2 * margin, 32) if placement == "strip" else 0
        name = next((k for k, v in DIRECTIONS.items() if v == rot), "")
        plan.update(prompt=prompt, rotation=float(rot), elevation=float(L["elevation"]), intensity=intensity,
                    sun_direction=f"{name.title()} ({how})", lora="ltx-2.3-22b-ic-lora-relight-1.0.safetensors",
                    tile_frames=window_frames(n8, 121),
                    ball={"patch": patch, "margin": margin, "placement": placement})
        if scene["setting"] == "interior":
            warnings.append("The shot looks like an interior: the sun relight model was trained on exteriors only, "
                            "so the change may be weak (Night looks handle interiors)")
        if height > width * 1.05:
            notes.append("Portrait clip: the sun relight model was trained on landscape clips")
    W, H = _ceil(cw, 32), _ceil(ch, 32)
    plan.update(content_w=cw, content_h=ch, canvas_w=W, canvas_h=H, strip_w=strip, gen_w=W + strip, gen_h=H)
    if fps > 30.5:
        notes.append(f"{fps:g} fps clip: both LoRAs were trained at 24 fps; conditioning fps capped at 30")
    if frames > 361:
        notes.append(f"Long clip ({frames} frames): run time grows with every extra window")
    # the subject to keep: a given matte, else the typed Subject, else the VLM's main subject
    typed = (subject_typed or "").strip()
    if matte_given:
        src, subj = "given", typed
    elif typed:
        src, subj = "typed", typed
    elif scene["subject"]:
        src, subj = "auto", scene["subject"]
    else:
        src, subj = "none", ""
    plan.update(subject_source=src, subject=subj, protect=src != "none", scene=scene["scene"])
    if notes:
        plan["notes"] = notes
    if warnings:
        plan["review_reasons"] = warnings
    return plan


class AutoMatteRelightPlan:
    """Look (+ Sun Direction) and the scene check -> route, prompt, light-ball angles, canvas, temporal window,
    conditioning fps, the subject to keep, and a JSON plan for the report. Nothing technical is asked."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames (size and frame count)."}),
                "fps": ("FLOAT", {"forceInput": True}),
                "look": ("STRING", {"forceInput": True, "tooltip": LOOK_OPTIONS}),
                "sun_direction": ("STRING", {"forceInput": True, "tooltip": DIRECTION_OPTIONS}),
            },
            "optional": {
                "scene": ("STRING", {"forceInput": True, "tooltip": "The VLM's JSON answer to VLM_PROMPT."}),
                "subject": ("STRING", {"forceInput": True, "tooltip": "Typed subject; empty = the VLM's main subject."}),
                "matte_video": ("VIDEO", {"tooltip": "A given matte: the subject is protected without searching."}),
                "prompt_suffix": ("STRING", {"forceInput": True, "tooltip": "Night looks: extra words on the lights."}),
                "ball_placement": ("STRING", {"default": "strip",
                                   "tooltip": "strip = ball in a mirrored strip right of the picture (cropped later); "
                                              "overlay = over the picture's corner, as in training."}),
                "ball_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1,
                                   "tooltip": "0 = automatic (1.5, the node's default). Baked: 1.5 and 0.8."}),
            },
        }

    RETURN_TYPES = ("RELIGHT_PLAN", "BOOLEAN", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("plan", "night", "prompt", "tile_frames", "gen_width", "gen_height", "cond_fps", "subject",
                    "plan_json")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Relight"

    def run(self, images, fps, look, sun_direction, scene="", subject="", matte_video=None, prompt_suffix="",
            ball_placement="strip", ball_intensity=0.0):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        p = make_plan(look, sun_direction, w, h, n, float(fps), scene or "", subject or "", _video_given(matte_video),
                      prompt_suffix or "", ball_placement, ball_intensity)
        return (p, p["route"] == "night", p["prompt"], p["tile_frames"], p["gen_w"], p["gen_h"], p["cond_fps"],
                p["subject"], json.dumps(p))


# ------------------------------------------------------------------------------------------------ the guide
_BALLS = {}
BALL_ROTATIONS = (0, 45, 90, 135, 180, 225, 270, 315)
BALL_ELEVATIONS = (15, 20, 25, 35, 45, 60, 70)
BALL_INTENSITIES = (1.5, 0.8)


def ball_file(rotation, elevation, intensity):
    """Nearest baked render: rotation snapped to 45 degrees, elevation and intensity to the baked values."""
    r = int(round((float(rotation) % 360) / 45.0)) % 8 * 45
    e = min(BALL_ELEVATIONS, key=lambda v: abs(v - float(elevation)))
    i = min(BALL_INTENSITIES, key=lambda v: abs(v - float(intensity)))
    return os.path.join(BALL_DIR, f"ball_r{r:03d}_e{e:02d}_i{int(round(i * 10)):02d}.png"), r, e, i


def load_ball(rotation, elevation, intensity, size):
    """(size, size, 3) float ball patch, downscaled from the 512x512 render with antialiasing."""
    path, r, e, i = ball_file(rotation, elevation, intensity)
    if path not in _BALLS:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Light-ball render missing: {path} (copy relight assets/light_balls/ next to "
                                    f"this module).")
        from PIL import Image
        import numpy as np
        _BALLS[path] = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
    t = _BALLS[path].permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return t[0].permute(1, 2, 0).clamp(0, 1)


def _pad_right_bottom(x, pr, pb):
    """(N,H,W,C) padded on the right/bottom by mirroring (replicate when the pad is too large to mirror)."""
    if pr <= 0 and pb <= 0:
        return x
    t = x.permute(0, 3, 1, 2)
    mode = "reflect" if pr < t.shape[3] and pb < t.shape[2] else "replicate"
    return F.pad(t, (0, max(0, pr), 0, max(0, pb)), mode=mode).permute(0, 2, 3, 1)


class AutoMatteRelightGuide:
    """The IC-LoRA reference for the plan: frames scaled to the canvas and mirror-padded to /32; for the sun route the
    light-direction ball drawn top-right on every frame (in a mirrored strip on the right, or over the picture), after
    every resize; frames padded to 8n+1 with the last frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "plan": ("RELIGHT_PLAN",)}}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("guide", "width", "height", "length")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Relight"

    def run(self, images, plan):
        n = images.shape[0]
        cw, ch, W, H, strip = plan["content_w"], plan["content_h"], plan["canvas_w"], plan["canvas_h"], plan["strip_w"]
        x = _resize_frames(images[..., :3], ch, cw, "bicubic").clamp(0, 1)
        x = _pad_right_bottom(x, W - cw, H - ch)
        if strip:
            x = _pad_right_bottom(x, strip, 0)
        ball = plan.get("ball")
        if ball:
            p, m = ball["patch"], ball["margin"]
            patch = load_ball(plan["rotation"], plan["elevation"], plan["intensity"], p)
            gw = x.shape[2]
            x = x.clone()
            x[:, m:m + p, gw - m - p:gw - m, :] = patch.to(x.dtype)
        L = plan["frames_padded"]
        if L > n:
            x = torch.cat([x, x[-1:].repeat(L - n, 1, 1, 1)], 0)
        return (x.contiguous(), int(x.shape[2]), int(x.shape[1]), int(x.shape[0]))


# ---------------------------------------------------------------------------------------------- the restore
def _lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(0, 1) + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _luma(lin):
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def _gauss(x, sigma):
    """Separable Gaussian blur of (N,H,W), mirrored edges."""
    if sigma <= 0.05:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    r = min(r, x.shape[1] - 1, x.shape[2] - 1)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    t = F.pad(x[:, None], (r, r, 0, 0), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, 1, -1))
    t = F.pad(t, (0, 0, r, r), mode="reflect")
    return F.conv2d(t, k.view(1, 1, -1, 1))[:, 0]


def _detail_ratio(y, sigma, lo, hi, night):
    """Source detail above `sigma` as a multiplier: y / blur(y), clamped; at night ratios above 1 (sun glints,
    day highlights) are compressed (1 + 0.6 ln d: a 9x glint comes back ~2.3x, a 1.5x texture 1.24x) so the day's
    highlights don't come back, while dark print (ratios below 1) stays as it is."""
    d = y / _gauss(y, sigma).clamp_min(1e-4)
    d = d.clamp(lo, hi)
    if night:
        d = torch.where(d > 1, 1 + 0.6 * torch.log(d.clamp_min(1)), d)
    return d


def _edges(p):
    """Sobel magnitude of (N,H,W)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=p.dtype, device=p.device).view(1, 1, 3, 3)
    t = F.pad(p[:, None], (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(t, kx), F.conv2d(t, kx.transpose(2, 3))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)[:, 0]


def structure_lock(src_srgb, out_srgb, short=256):
    """Per-frame correlation of edge maps (source vs output) at a small size: 1 = the same structure; a drop flags
    a frame that drifted (light changes lower it evenly, drift lowers it locally in time)."""
    n, h, w = src_srgb.shape[:3]
    s = min(1.0, short / min(h, w))
    hh, ww = max(8, round(h * s)), max(8, round(w * s))
    def small(x):
        y = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
        return F.interpolate(y[:, None], size=(hh, ww), mode="area")[:, 0]
    a, b = _edges(small(src_srgb)), _edges(small(out_srgb))
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)


SUBJECT_LIGHT_OPTIONS = "Match the New Light, Keep as Shot"


class AutoMatteRelightRestore:
    """Back to the source size and frame count with the source's fine detail: the relit clip gives the light (its low
    frequencies), the source gives everything finer (y / blur(y), in linear light, luminance only, so the new light's
    colour stays). Over the whole frame only what the canvas could not hold comes back; inside the subject matte
    a wider band does, so label strokes and faces are the source's. 'Keep as Shot' puts the original subject back
    unchanged. `matte` is optional and lazy: only computed when it is linked and the plan protects a subject. Writes
    a JSON report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The source frames (full size)."}),
                "relit": ("IMAGE", {"tooltip": "The decoded generation (canvas size, with strip/padding)."}),
                "plan": ("RELIGHT_PLAN",),
            },
            "optional": {
                "matte": ("MASK", {"lazy": True, "tooltip": "Subject matte (e.g. AutoMatte Given or Made); only "
                                   "computed when the plan protects a subject. Unlinked = no subject."}),
                "subject_light": ("STRING", {"forceInput": True, "tooltip": SUBJECT_LIGHT_OPTIONS}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Relight"

    def check_lazy_status(self, images, relit, plan, **kw):
        # an optional lazy input is only lazy when linked: request it only when present (linked) and not computed
        return ["matte"] if plan.get("protect") and "matte" in kw and kw["matte"] is None else []

    def run(self, images, relit, plan, matte=None, subject_light="Match the New Light"):
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        cw, ch = plan["content_w"], plan["content_h"]
        if relit.shape[0] < n:
            raise ValueError(f"The generation has {relit.shape[0]} frames for a {n}-frame clip.")
        rel = relit[:n, :ch, :cw, :3]
        keep = _key(subject_light) == "keep as shot"
        night = plan["route"] == "night"
        ratio = min(h, w) / max(1, min(ch, cw))
        sig_g = 0.5 * ratio if ratio > 1.05 else 0.0
        sig_s = max(2.0, 0.0035 * min(h, w), 2.0 * sig_g)
        m = None
        if plan.get("protect") and matte is not None:
            m = _fit_mask(matte, n, h, w)
        dev, out, lock = _device(), [], []
        for i in range(0, n, CHUNK):
            s = images[i:i + CHUNK, ..., :3].to(dev).float()
            r = _resize_frames(rel[i:i + CHUNK], h, w, "bicubic").to(dev).float().clamp(0, 1)
            s_lin, r_lin = _lin(s), _lin(r)
            ys, yr = _luma(s_lin), _luma(r_lin)
            yo = yr if sig_g == 0 else _gauss(yr, sig_g) * _detail_ratio(ys, sig_g, 0.25, 4.0, night)
            if m is not None:
                mc = m[i:i + CHUNK].to(dev).float()
                inner = -F.max_pool2d(-mc[:, None], 3, stride=1, padding=1)[:, 0]          # 1 px in from the edge
                inner = F.avg_pool2d(inner[:, None], 3, stride=1, padding=1, count_include_pad=False)[:, 0]
                ys_sub = _gauss(yr, sig_s) * _detail_ratio(ys, sig_s, 0.02, 10.0, night)  # black print on white: ~0.05
                yo = yo + (ys_sub - yo) * inner
            gain = (yo / yr.clamp_min(1e-4)).clamp(0, 8)[..., None]
            o_lin = torch.where(yr[..., None] > 1e-4, r_lin * gain, yo[..., None].expand_as(r_lin))
            o = _srgb(o_lin)
            if keep and m is not None:
                mk = F.avg_pool2d(mc[:, None], 3, stride=1, padding=1, count_include_pad=False)[:, 0][..., None]
                o = o + (s - o) * mk
            lock.append(structure_lock(s, o).cpu())
            out.append(o.cpu())
        res = torch.cat(out, 0).clamp(0, 1).contiguous()
        lk = torch.cat(lock, 0)
        mean = float(lk.mean())
        flagged = [int(j) for j in torch.nonzero(lk < mean - 0.15).flatten().tolist()]
        rep = {k: v for k, v in plan.items() if k not in ("ball",)}
        rep.update(subject_light="Keep as Shot" if keep else "Match the New Light",
                   matte_used=m is not None, detail_sigma_frame=round(sig_g, 2), detail_sigma_subject=round(sig_s, 2),
                   structure_lock_mean=round(mean, 3), structure_lock_min=round(float(lk.min()), 3),
                   structure_lock_flagged_frames=flagged[:50])
        if plan.get("ball"):
            rep["ball"] = dict(plan["ball"], file=os.path.basename(ball_file(plan["rotation"], plan["elevation"],
                                                                               plan["intensity"])[0]))
        notes = list(rep.get("notes") or [])
        if keep and m is None:
            notes.append("Keep as Shot needs a subject; none was found, so the whole frame was relit")
        if notes:
            rep["notes"] = notes
        reasons = list(rep.get("review_reasons") or [])
        if flagged:
            reasons.append(f"{len(flagged)} frame(s) drift from the source's structure (structure lock well below "
                           f"the clip's mean): check them")
        rep["review_reasons"] = reasons
        rep["needs_review"] = bool(reasons)
        return (res, json.dumps(rep))


NODE_CLASS_MAPPINGS = {
    "AutoMatteRelightPlan": AutoMatteRelightPlan,
    "AutoMatteRelightGuide": AutoMatteRelightGuide,
    "AutoMatteRelightRestore": AutoMatteRelightRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteRelightPlan": "AutoMatte: Relight Plan",
    "AutoMatteRelightGuide": "AutoMatte: Relight Guide (light ball)",
    "AutoMatteRelightRestore": "AutoMatte: Relight Restore (detail + subject)",
}
