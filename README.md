# ComfyUI-AutoMatte

Glue nodes for a fully automatic subject matte. A **hard subject mask** (for example from SAM 3.1) says *what*
the subject is, a **soft matte** (for example from LTX-2.5 Alpha Gen) says *how soft* its edges are, and these nodes
do what a person would otherwise do by hand: fit the clip to the model's input rules, decide between the two mattes,
and undo the fit. Model-agnostic: no model's limits in the logic, and every default changes nothing.

| Node | What it does |
|---|---|
| AutoMatte: Fit Frames | Scales down (never up) so the long side ≤ `max_long_side` and the short side ≤ `max_short_side`, whichever way the clip is oriented; pads right/bottom to a multiple of `size_multiple` by repeating edge pixels; pads the frame count to `frame_multiple·n + frame_offset` by repeating the last frame. Outputs the frames, width, height, length and a `fit` record. Defaults (0, 0, 1, 1, 0) = no change. |
| AutoMatte: Combine Subject Mask + Soft Matte | Gates the soft matte to the subject mask. Measures the soft matte's opacity deep inside the subject over the whole clip: solid (≥ `solid_threshold`, 0.95) → keeps its soft edges; see-through on a solid object → fills the interior from the mask and rescales the edges. Soft matte missing the subject (interior below `min_interior`, 0.25) → the subject mask, lightly feathered. A subject missing for up to `hold_frames` (4) frames holds the nearest mask; longer absences (out of shot) come out empty. Optional `subject_text`: wired and empty = no subject named, the soft matte is used as is; typed but never found → needs review. |
| AutoMatte: Restore Size | Drops the padded frames and pixels and scales back to the source size. |
| AutoMatte: To RGBA | Source frames + matte → RGBA frames, straight alpha. |
| AutoMatte: Save Report | Writes Combine's JSON report as a file output, so API runners download it with the images. Optional `extra` (JSON object) is merged in: its `notes` / `review_reasons` are appended, other keys added when missing. |
| AutoMatte: Count Prompt Items | Reads a comma-separated prompt list with optional `:N` counts (`cake:2, apron, window panel:4`) the way text-prompted detectors such as SAM 3 do (parentheses dropped, no count = `default_count`, 1) and returns the total, clamped to 1..`max_total` (64), e.g. for a tracker's max-objects setting; plus the clean list and a JSON block for Save Report's `extra` (items, total, optional `source`, a note that the total is shared by all items). |

`keep_transparency` (Combine) skips the interior repair, for subjects that really are see-through (glass, water, smoke).

`use_soft_matte` (Combine, default on): off = the subject mask only, lightly feathered, with the same out-of-shot and
dropout handling. `soft_matte` is a lazy input, so with this off whatever produces the soft matte is never run: a
faster, harder-edged choice (wire a boolean to it to offer both in one workflow). Nothing found → an empty matte,
needs review.

Report fields: `mode` (`soft_edges`, `interior_repaired`, `subject_mask`, `soft_matte_only`, `empty`), `soft_matte_used`, `interior_soft_matte`,
`interior_after`, `off_target_share`, `frames_without_subject`, `absent_frames` (ranges left empty), `frames_held`,
`notes`, `needs_review`, `review_reasons`.

Example settings for LTX-2.5 (Alpha Gen IC-LoRA): Fit with long 1920, short 1088, size multiple 32, frames 8n+1
(`frame_multiple` 8, `frame_offset` 1).

Only torch, no model weights, so nothing in this pack carries a model licence. Check the licences of the models you
pair it with.
