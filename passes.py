"""VFX passes: one source from an image OR a video, material passes (normal, base colour, roughness, metallic,
specular) from Marigold, and clip-wide depth renders (greyscale and heat map).

Model-agnostic where it can be: the Marigold node takes any diffusers Marigold normals / intrinsics folder by name and
reads what each model predicts from its own config (target names), so a newer Marigold release drops in by name.
"""
import json
import logging
import os
import tempfile

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .layers import _video_given

log = logging.getLogger("AutoMatte")


# --------------------------------------------------------------------------------------------- Image or Video
class AutoMatteImageOrVideo:
    """One source for a workflow that takes an image OR a video (both optional, as an app form can't switch inputs).
    The video is checked first and the image is only computed when no video file was given, so an empty image input
    never runs. Outputs the frames (one for an image), the frame rate, the audio (video only) and which one it was."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                        "tooltip": "Frame rate reported for a still image."}),
            },
            "optional": {
                "video": ("VIDEO", {"lazy": True, "tooltip": "Used when it points at a file."}),
                "image": ("IMAGE", {"lazy": True, "tooltip": "Used only when no video is given."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO", "BOOLEAN", "BOOLEAN", "INT")
    RETURN_NAMES = ("images", "fps", "audio", "is_video", "is_image", "frame_count")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Passes"

    def check_lazy_status(self, image_fps, **kw):
        # Only linked optional inputs arrive in kw (None until computed). The video goes first (an IO Video Input
        # only resolves a path); the image is asked for only when no video file came.
        if "video" in kw and kw["video"] is None:
            return ["video"]
        if _video_given(kw.get("video")):
            return []
        return ["image"] if "image" in kw and kw["image"] is None else []

    def run(self, image_fps, video=None, image=None):
        if _video_given(video):
            c = video.get_components()
            frames = c.images
            if frames is None or frames.shape[0] == 0:
                raise ValueError("The video has no frames.")
            fps = float(c.frame_rate) if c.frame_rate else float(image_fps)
            return (frames.float(), fps, c.audio, True, False, int(frames.shape[0]))
        if image is None:
            raise ValueError("Add an image or a video.")
        img = image[:1].float()
        return (img, float(image_fps), None, False, True, 1)


# --------------------------------------------------------------------------------------------- Marigold
_PIPES = {}


SHARED_PREFIX = "marigold-sd2-shared"  # files identical across Marigold models (SD2 text encoder, VAE, tokenizer, scheduler)


def _marigold_dir(name):
    """A diffusers folder for a Marigold model. Looks for models/diffusers/<name>/ first; then for the flat files a
    build stages (builds take no sub-folders): `<name>__unet__config.json`, … plus `marigold-sd2-shared__…` for the
    parts every Marigold shares, linked into a folder in the temp dir; else downloads the fp16 files from HF."""
    roots = list(folder_paths.get_folder_paths("diffusers")) if "diffusers" in folder_paths.folder_names_and_paths else []
    roots.append(os.path.join(folder_paths.models_dir, "diffusers"))
    short = name.split("/")[-1]
    for r in roots:
        p = os.path.join(r, short)
        if os.path.isfile(os.path.join(p, "model_index.json")):
            return p
    for r in roots:
        if not os.path.isdir(r):
            continue
        files = os.listdir(r)
        if f"{short}__model_index.json" not in files:
            continue
        target = os.path.join(tempfile.gettempdir(), "automatte_marigold", short)
        for prefix in (SHARED_PREFIX, short):  # the model's own files win
            for f in files:
                if not f.startswith(prefix + "__"):
                    continue
                rel = f[len(prefix) + 2:].replace("__", os.sep)
                dst = os.path.join(target, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.join(r, f), dst)
        return target
    target = os.path.join(roots[-1], short)
    repo = name if "/" in name else f"prs-eth/{name}"
    log.warning("AutoMatte: %s not staged, downloading %s to %s", short, repo, target)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=target,
                      allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])
    return target


def _load_pipe(name):
    if name in _PIPES:
        return _PIPES[name]
    import diffusers
    path = _marigold_dir(name)
    with open(os.path.join(path, "model_index.json")) as f:
        cls_name = json.load(f).get("_class_name", "")
    cls = getattr(diffusers, cls_name, None)
    if cls is None or "Marigold" not in cls_name:
        raise ValueError(f"{name}: not a diffusers Marigold pipeline ({cls_name!r}); diffusers {diffusers.__version__}")
    fp16 = any(f.endswith("fp16.safetensors") for _, _, fs in os.walk(path) for f in fs)
    pipe = cls.from_pretrained(path, variant="fp16" if fp16 else None, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    _PIPES.clear()  # keep one pipeline resident at a time
    _PIPES[name] = pipe
    return pipe


def _run_marigold(pipe, frames, steps, resolution, seed, blend, pbar):
    """Per-frame Marigold with one shared starting noise and the previous frame's latent blended in, so a clip stays
    steady (a still image is just the first frame). Returns predictions (N, T, 3, H, W) in [0, 1] (normals in [-1, 1])
    at the frames' size."""
    dev = mm.get_torch_device()
    pipe.to(dev)
    n, h, w, _ = frames.shape
    ph, pw = (h * resolution // max(h, w), w * resolution // max(h, w)) if resolution > 0 else (h, w)
    lh, lw = (ph + 7) // 8, (pw + 7) // 8  # the pipeline's own latent size
    ch = pipe.unet.config.out_channels
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    common = torch.randn((1, ch, lh, lw), generator=g).to(dev, torch.float16)
    prev, out = None, []
    try:
        for i in range(n):
            lat = common if prev is None else (1 - blend) * common + blend * prev
            img = frames[i:i + 1].permute(0, 3, 1, 2).to(dev, torch.float16)
            r = pipe(img, num_inference_steps=steps, ensemble_size=1, processing_resolution=resolution,
                     match_input_resolution=True, latents=lat, output_latent=True, output_type="pt")
            prev = r.latent
            p = r.prediction.float()  # (T, 3, H, W) for one image
            if p.dim() == 3:
                p = p[None]
            out.append(p.cpu())
            pbar.update(1)
    finally:
        pipe.to(mm.unet_offload_device())
        mm.soft_empty_cache()
    return torch.stack(out, 0)


def _srgb_to_lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


class AutoMatteMarigoldPasses:
    """Material passes from Marigold (diffusers): Normal (camera space, flat = lavender blue), Base Color (sRGB albedo,
    no light or shading), Roughness and Metallic (greyscale, linear), and Specular, the standard PBR reflectance at
    normal incidence: mix(0.04, base colour, metallic) worked out in linear light. Each model runs per frame with
    one shared starting noise and the previous frame's latent blended in, so a clip stays steady."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "normals_model": ("STRING", {"default": "prs-eth/marigold-normals-v1-1",
                                             "tooltip": "diffusers Marigold normals folder in models/diffusers (or HF repo id)."}),
                "material_model": ("STRING", {"default": "prs-eth/marigold-iid-appearance-v1-1",
                                              "tooltip": "diffusers Marigold intrinsics (appearance) folder: albedo + roughness/metallicity."}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "processing_resolution": ("INT", {"default": 768, "min": 0, "max": 2048, "step": 8,
                                                  "tooltip": "Long side the models see (0 = native). 768 is what they were trained for."}),
                "temporal_blend": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.9, "step": 0.01,
                                             "tooltip": "How much of the previous frame's latent starts the next frame (video only)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal", "base_color", "roughness", "metallic", "specular")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Passes"

    def run(self, images, normals_model, material_model, seed, steps, processing_resolution, temporal_blend):
        n = images.shape[0]
        pbar = comfy.utils.ProgressBar(2 * n)

        pipe = _load_pipe(normals_model)
        nrm = _run_marigold(pipe, images, steps, processing_resolution, seed, temporal_blend, pbar)[:, 0]
        nrm = F.normalize(nrm, dim=1)  # unit normals in [-1, 1]
        normal = ((nrm + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1)

        pipe = _load_pipe(material_model)
        names = list((pipe.target_properties or {}).get("target_names", ["albedo", "material"]))
        pred = _run_marigold(pipe, images, steps, processing_resolution, seed, temporal_blend, pbar).clamp(0, 1)
        props = pipe.target_properties or {}

        def target(name):
            return pred[:, names.index(name)] if name in names else None

        albedo = target("albedo")
        if albedo is None:
            raise ValueError(f"{material_model} predicts {names}, not an albedo")
        if (props.get("albedo", {}) or {}).get("prediction_space") == "linear":
            albedo = _lin_to_srgb(albedo)
        rough = metal = None
        for tname in names:
            subs = (props.get(tname, {}) or {}).get("sub_target_names")
            if subs:
                t = target(tname)
                for ci, sub in enumerate(subs):
                    if sub == "roughness":
                        rough = t[:, ci]
                    elif sub in ("metallicity", "metallic", "metalness"):
                        metal = t[:, ci]
        if rough is None and "roughness" in names:
            rough = target("roughness").mean(1)
        if metal is None:
            for k in ("metallicity", "metallic", "metalness"):
                if k in names:
                    metal = target(k).mean(1)
        if rough is None or metal is None:
            raise ValueError(f"{material_model} predicts {names}: no roughness/metallic")

        base = albedo.permute(0, 2, 3, 1)
        m = metal[:, :, :, None]
        spec_lin = 0.04 * (1 - m) + _srgb_to_lin(base) * m
        spec = _lin_to_srgb(spec_lin)
        grey = lambda x: x[:, :, :, None].repeat(1, 1, 1, 3).contiguous()  # noqa: E731
        size = images.shape[1:3]

        def fit(x):  # Marigold matches the input size already; guard against an off-by-one from rounding
            if tuple(x.shape[1:3]) == tuple(size):
                return x.contiguous()
            return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="bilinear",
                                 align_corners=False).permute(0, 2, 3, 1).contiguous()

        return (fit(normal), fit(base), fit(grey(rough)), fit(grey(metal)), fit(spec))


# --------------------------------------------------------------------------------------------- Depth
def _turbo(x):
    """Google's Turbo colormap (polynomial fit), x in [0, 1] -> RGB."""
    x = x.clamp(0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return torch.stack([r, g, b], -1).clamp(0, 1)


class AutoMatteDepthPasses:
    """Depth renders normalised over the whole clip (not per frame, so nothing pumps): greyscale with near = white,
    and a heat map (Turbo: near = red, far = dark blue). Takes raw depth (far = larger) or disparity (near = larger)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE", {"tooltip": "Raw depth or disparity per frame (first channel is read)."}),
                "input_is": (["depth (far = larger)", "disparity (near = larger)"],),
                "low_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "high_percentile": ("FLOAT", {"default": 99.0, "min": 80.0, "max": 100.0, "step": 0.1}),
                "align_frames": ("BOOLEAN", {"default": True,
                                             "tooltip": "Per-frame depth models guess each frame's scale on its own: fit "
                                                        "each frame's scale and offset to the frame before (video)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("depth", "heat_map")
    FUNCTION = "run"
    CATEGORY = "AutoMatte/Passes"

    def run(self, depth, input_is, low_percentile, high_percentile, align_frames=True):
        d = depth[..., 0].float()
        if input_is.startswith("depth"):
            d = 1.0 / d.clamp_min(1e-6)  # to disparity: spreads near detail like a depth pass reads
        if align_frames and d.shape[0] > 1:
            d = d.clone()
            step = max(1, int((d.shape[1] * d.shape[2] / 65536) ** 0.5))
            for i in range(1, d.shape[0]):
                x, y = d[i, ::step, ::step].flatten(), d[i - 1, ::step, ::step].flatten()
                xm, ym = x.mean(), y.mean()
                var = ((x - xm) ** 2).mean()
                s_ = (((x - xm) * (y - ym)).mean() / var) if var > 1e-12 else torch.tensor(1.0)
                s_ = s_.clamp(0.5, 2.0)  # a cut or a big move: don't let one fit swing the scale
                d[i] = s_ * d[i] + (ym - s_ * xm)
        flat = d.flatten()
        if flat.numel() > 4_000_000:
            flat = flat[torch.randperm(flat.numel())[:4_000_000]]
        lo = torch.quantile(flat, low_percentile / 100.0)
        hi = torch.quantile(flat, high_percentile / 100.0)
        x = ((d - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)  # near = 1
        grey = x[..., None].repeat(1, 1, 1, 3).contiguous()
        return (grey, _turbo(x).contiguous())


NODE_CLASS_MAPPINGS = {
    "AutoMatteImageOrVideo": AutoMatteImageOrVideo,
    "AutoMatteMarigoldPasses": AutoMatteMarigoldPasses,
    "AutoMatteDepthPasses": AutoMatteDepthPasses,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMatteImageOrVideo": "Image or Video (AutoMatte)",
    "AutoMatteMarigoldPasses": "Material Passes: Marigold (AutoMatte)",
    "AutoMatteDepthPasses": "Depth Passes: grey + heat map (AutoMatte)",
}
