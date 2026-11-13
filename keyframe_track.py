"""AutoMatte: SAM3 Track From Keyframe.

ComfyUI's core Run SAM3 Video Track always starts from the first frame it gets: its
initial_mask belongs to frame 0. This node starts from any frame instead. It runs the
same core SAM3 tracker twice from the keyframe, forward to the last frame and backward
over the reversed frames to the first, and joins the two into one SAM3_TRACK_DATA for
the whole clip, so core SAM3 Track to Mask and SAM3 Track Preview read it as usual.

Typical chain: a video's keyframe (for example a trim node's keyframe output) → points
→ SAM3 Detect on that frame → its mask here as initial_mask, with the keyframe's index
in the frames; the frames are the whole clip.

Only core ComfyUI SAM3 / SAM 3.1 (comfy.ldm.sam3) is used; imported at run time.
"""

TRACK_DATA_TYPE = "SAM3_TRACK_DATA"  # core SAM3 Video Track's output type
DIRECTIONS = ["both", "forward", "backward"]


def _as_packed(result):
    """(packed [M, O, h, w/8] uint8 or None, scores) from a forward_video result."""
    if isinstance(result, dict):
        return result.get("packed_masks"), list(result.get("scores") or [])
    # SAM 3 (non-multiplex) returns mask logits [M, O, h, w].
    from comfy.ldm.sam3.tracker import pack_masks
    return pack_masks(result), []


def join_tracks(n_frames, keyframe, forward, backward):
    """One packed track for frames 0..n_frames-1 from a forward run (frames keyframe..end,
    keyframe first) and a backward run (frames keyframe..0, keyframe first); either may be
    None (not run) or have no objects. Frames no run covers are empty."""
    import torch

    runs = [p for p in (forward, backward) if p is not None]
    if not runs:
        return None
    n_obj = max(p.shape[1] for p in runs)
    sample = runs[0]
    empty = torch.zeros((n_obj, *sample.shape[2:]), dtype=sample.dtype, device=sample.device)

    def frame(p, i):
        m = p[i]
        if m.shape[0] < n_obj:
            m = torch.cat([m, empty[:n_obj - m.shape[0]]], dim=0)
        return m

    frames = [empty] * n_frames
    if backward is not None:
        for i in range(backward.shape[0]):  # backward[i] is frame keyframe - i
            frames[keyframe - i] = frame(backward, i)
    if forward is not None:  # forward wins on the keyframe itself
        for i in range(forward.shape[0]):
            frames[keyframe + i] = frame(forward, i)
    return torch.stack(frames, dim=0)


class AutoMatteKeyframeTrack:
    CATEGORY = "AutoMatte"
    FUNCTION = "track"
    RETURN_TYPES = (TRACK_DATA_TYPE,)
    RETURN_NAMES = ("track_data",)
    OUTPUT_TOOLTIPS = ("The track over the whole clip, for SAM3 Track to Mask or SAM3 Track Preview.",)
    DESCRIPTION = (
        "Tracks objects through a clip from any frame: the mask on the keyframe (for example from SAM3 Detect "
        "with points on that frame) is followed forward to the end and backward to the start with core SAM3's "
        "tracker, and the two runs are joined. Core Run SAM3 Video Track only starts from the first frame."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The clip's frames."}),
                "model": ("MODEL", {"tooltip": "The SAM3 / SAM 3.1 model, as for Run SAM3 Video Track."}),
                "initial_mask": ("MASK", {"tooltip": "The object mask(s) on the keyframe, one per object "
                                                     "(for example SAM3 Detect's mask on that frame)."}),
                "keyframe_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                           "tooltip": "Which frame initial_mask belongs to, counted from 0 in "
                                                      "images (for example a trim node's keyframe_index)."}),
                "direction": (DIRECTIONS, {"default": "both",
                                           "tooltip": "'both' = the whole clip; 'forward' = from the keyframe "
                                                      "to the end; 'backward' = from the start to the keyframe. "
                                                      "Frames not tracked come out empty."}),
            },
        }

    def track(self, images, model, initial_mask, keyframe_index, direction):
        import torch
        import torch.nn.functional as F
        import comfy.model_management as mm
        import comfy.utils

        n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        if n == 0:
            raise ValueError("AutoMatte: SAM3 Track From Keyframe: no frames.")
        k = min(max(0, int(keyframe_index)), n - 1)

        masks = initial_mask if initial_mask.dim() == 3 else initial_mask.reshape(-1, *initial_mask.shape[-2:])
        if masks.shape[-2:] != (h, w):
            masks = F.interpolate(masks[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]

        mm.load_model_gpu(model)
        device = mm.get_torch_device()
        dtype = model.model.get_dtype()
        sam = model.model.diffusion_model
        init = masks.unsqueeze(1).to(device=device, dtype=dtype)
        frames = images[..., :3].movedim(-1, 1)  # [N, 3, H, W], CPU, as core passes them

        run_fwd = direction in ("both", "forward")
        run_bwd = direction in ("both", "backward") and k > 0
        pbar = comfy.utils.ProgressBar((n - k if run_fwd else 0) + (k + 1 if run_bwd else 0))

        def run(clip):
            result = sam.forward_video(images=clip, initial_masks=init, pbar=pbar, text_prompts=None,
                                       new_det_thresh=0.5, max_objects=0, detect_interval=1,
                                       target_device=device, target_dtype=dtype)
            return _as_packed(result)

        forward = backward = None
        scores = []
        if run_fwd:
            forward, scores = run(frames[k:])
        if run_bwd:
            backward, back_scores = run(torch.flip(frames[:k + 1], dims=[0]))
            scores = scores or back_scores

        packed = join_tracks(n, k, forward, backward)
        return ({"packed_masks": packed, "n_frames": n, "scores": scores, "orig_size": (h, w)},)


NODE_CLASS_MAPPINGS = {"AutoMatteKeyframeTrack": AutoMatteKeyframeTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoMatteKeyframeTrack": "AutoMatte: SAM3 Track From Keyframe"}
