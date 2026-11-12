# ComfyUI-AutoMatte

Glue nodes for a fully automatic subject matte: **SAM 3.1** decides *what* the subject is,
**LTX-2.5 Alpha Gen** decides *how soft* its edges are, and these nodes do everything a person
would otherwise do by hand (resizing for LTX, choosing between the two mattes, undoing the resize).

| Node | What it does |
|---|---|
| AutoMatte: Fit Clip for LTX | Scales down to fit 1920×1088 (never up), pads to sides /32 and 8n+1 frames by repeating edges / the last frame. Outputs the fitted frames, width, height, length and a `fit` record. |
| AutoMatte: Combine SAM + Alpha Gen | Gates the Alpha Gen matte to the SAM subject. Measures Alpha Gen's opacity deep inside the subject over the whole clip: solid (≥ 0.95) → keeps Alpha Gen's edges; see-through on a solid object → fills the interior from SAM and rescales the edges. SAM found nothing → Alpha Gen as is. Writes a JSON report (`mode`, `interior_alpha_gen`, `off_target_share`, `needs_review`, reasons). |
| AutoMatte: Restore Size | Drops the padded frames/pixels and scales back to the source size. |
| AutoMatte: To RGBA | Source frames + matte → RGBA frames, straight alpha. |

`keep_transparency` (Combine) skips the interior repair, for subjects that really are see-through (glass, water, smoke).

Tested behaviour (2026-11-11, two 65-frame clips): a person with locs kept Alpha Gen's edges (interior 1.00);
a gold-pump serum bottle that Alpha Gen made 80% opaque was repaired to 1.00 and a passing hummingbird was gated out.

Only torch, no model weights: nothing in this pack carries a model licence. The models it is used with:
SAM 3.1 (SAM License, commercial use allowed with restrictions), LTX-2.5 Alpha Gen (LTX-2 Community License:
free under $10M annual group revenue, paid licence above).
