# ComfyUI-AutoMatte

Glue nodes for a fully automatic subject matte. A **hard subject mask** (for example from SAM 3.1) says *what*
the subject is, a **soft matte** (for example from LTX-2.5 Alpha Gen) says *how soft* its edges are, and these nodes
do what a person would otherwise do by hand: fit the clip to the model's input rules, decide between the two mattes,
and undo the fit. Model-agnostic: no model's limits in the logic, and every default changes nothing.

| Node | What it does |
|---|---|
| AutoMatte: Fit Frames | Scales down (never up) so the long side ≤ `max_long_side` and the short side ≤ `max_short_side`, whichever way the clip is oriented; pads right/bottom to a multiple of `size_multiple` by repeating edge pixels; pads the frame count to `frame_multiple·n + frame_offset` by repeating the last frame. Outputs the frames, width, height, length and a `fit` record. Defaults (0, 0, 1, 1, 0) = no change. |
| AutoMatte: Combine Subject Mask + Soft Matte | Gates the soft matte to the subject mask. Measures the soft matte's opacity deep inside the subject over the whole clip: solid (≥ `solid_threshold`, 0.95) → keeps its soft edges; see-through on a solid object → fills the interior from the mask and rescales the edges. Soft matte missing the subject (interior below `min_interior`, 0.25) → the subject mask, lightly feathered. Parts of the subject the soft matte leaves out (local coverage near 0, e.g. a second named object it didn't pick) use the subject mask there, so nothing named vanishes; the whole-subject decision is measured on the covered part only. A subject missing for up to `hold_frames` (4) frames holds the nearest mask; longer absences (out of shot) come out empty. Optional `subject_text`: wired and empty = no subject named, the soft matte is used as is; typed but never found → needs review. |
| AutoMatte: Restore Size | Drops the padded frames and pixels and scales back to the source size. |
| AutoMatte: To RGBA | Source frames + matte → RGBA frames, straight alpha. |
| AutoMatte: Save Report | Writes Combine's JSON report as a file output, so API runners download it with the images. Optional `extra` (JSON object) is merged in: its `notes` / `review_reasons` are appended, other keys added when missing. |
| AutoMatte: Count Prompt Items | Reads a comma-separated prompt list with optional `:N` counts (`cake:2, apron, window panel:4`) the way text-prompted detectors such as SAM 3 do (parentheses dropped, no count = `default_count`, 1) and returns the total, clamped to 1..`max_total` (64), e.g. for a tracker's max-objects setting; plus the clean list and a JSON block for Save Report's `extra` (items, total, optional `source`, a note that the total is shared by all items). |

`keep_transparency` (Combine) skips the interior repair, for subjects that really are see-through (glass, water, smoke).

`use_soft_matte` (Combine, default on): off = the subject mask only, lightly feathered, with the same out-of-shot and
dropout handling. `soft_matte` is a lazy input, so with this off whatever produces the soft matte is never run: a
faster, harder-edged choice (wire a boolean to it to offer both in one workflow). Nothing found → an empty matte,
needs review.

Report fields: `mode` (`soft_edges`, `interior_repaired`, `subject_mask`, `soft_matte_only`, `empty`), `soft_matte_used`, `interior_soft_matte`, `missed_share`,
`interior_after`, `off_target_share`, `frames_without_subject`, `absent_frames` (ranges left empty), `frames_held`,
`notes`, `needs_review`, `review_reasons`.

Example settings for LTX-2.5 (Alpha Gen IC-LoRA): Fit with long 1920, short 1088, size multiple 32, frames 8n+1
(`frame_multiple` 8, `frame_offset` 1).

## Layer edits (`layers.py`)
Cheap, localized edits on a clip once a subject matte exists (Miracle-style layer editing). Pixels outside the
matte come back bit for bit.

| Node | What it does |
|---|---|
| AutoMatte: Given or Made Matte | Picks the matte for an edit: the `matte_video` when one is given (a saved matte of the same clip, e.g. from an earlier segmentation run), otherwise the graph's own `matte`. `matte`, `report` and `extra` are lazy, so the segmentation branch only runs when no matte video is given. An empty WorkflowIO Video Input counts as "not given". A given matte must match the clip (aspect within 2 %, frame count within 2 % or 2 frames); it is scaled to the frame size and padded/cut to the frame count. Outputs the matte, a JSON report (`matte_source` given/made, plus the segmentation report when made) and `made` (e.g. for an output's `include`). |
| AutoMatte: Recolour | **Recolour**: the subject's main colour becomes an exact colour (hex `#2E7D32`, `rgb(46, 125, 50)`, a CSS name such as `navy` or `forest green`, or a few design names: cream, rust, burgundy, mustard, emerald, camel, charcoal, mint, terracotta, cobalt, sky, sage, lilac, mauve, ochre). Lightness, hue and chroma are calibrated on the clip so the measured main colour lands on the target; the light/dark variation is kept (compressed only if it would clip), so shading, reflections and print survive. On a coloured subject, neutral parts (white sole, white label, highlights, black print) stay, gradients keep their fade, and parts in a clearly different hue (a gold cap, a logo) are kept; a neutral or barely coloured subject (white, grey, black, denim) takes the colour everywhere except clearly coloured details. **Correct** (brand colour lock): for a subject that should be this colour but drifts, each frame's main colour is turned onto the target hue and chroma, lightness untouched. Edits in OKLab (hue-linear), statistics over the whole clip (Recolour) or per frame smoothed over 5 frames (Correct): no flicker. Soft edges change only by the subject's own colour change, so the background seen through an edge stays. JSON report: target, subject colour before/after, CIEDE2000 before/after (Correct: hue/chroma only, per-frame mean/max before), share of the subject kept in another colour, shading kept, review reasons. |
| AutoMatte: Protect People | Takes faces, necks, hands and hair (any mask wired to the lazy `protect`, e.g. SAM 3 on `face, neck, hand, hair`) out of a matte before an edit: a garment's or a held product's soft matte reaches into the neck and the hand. Skipped (and `protect` never computed) when `subject_text` names people or a body part (`hair`, `lips`, `woman`). Grown a little and feathered; adds `people_protected` / `protected_share_of_matte` to the report. |
| AutoMatte: Effect Zone | Where an object's shadow, caustic or reflection can fall: the object plus an ellipse around where it meets the surface (its mask's bottom), `wide_factor` (1.4) object widths across and `depth_factor` (0.6) heights deep, never above its lower part; per frame, smoothed over ±4 frames, at the size and frame count of `images` (e.g. frames padded for a model). Use it as the area a generative fill rebuilds. |
| AutoMatte: Effect Mask | After the fill: the object (grown by `object_grow`, 3 % of the shorter side) plus the parts of the zone where the fill differs from the original by a smooth change of light or colour connected to the object (its shadow, caustic, glow), measured low-pass in OKLab against the fill's own noise in the outer zone, followed at most `reach` (1.2) object widths, held ±2 frames, feathered. Paste the fill back only there: everything else stays original, even inside the zone. Report: frames with effects, effect area vs object, thresholds. |

Name the part to recolour in the segmentation subject (`bottle body`, `denim jacket`): a matte of a whole person
recolours skin and hair too.

Only torch, no model weights, so nothing in this pack carries a model licence. Check the licences of the models you
pair it with.
