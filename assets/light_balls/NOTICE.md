# Light-ball renders (relight reference)

112 PNG renders (512×512) of the light-direction sphere that the LTX-2.3 Relight (Sun Direction) IC-LoRA reads from
the top-right corner of its reference video.

- Name: `ball_r<rotation>_e<elevation>_i<intensity×10>.png`.
  - rotation 0/45/…/315: 0 = light from the camera side, 90 = from the right, 180 = from behind, 270 = from the left.
  - elevation 15/20/25/35/45/60/70°.
  - intensity 1.5 (the node's default) and 0.8.
- How they were made: baked 2026-11-13 in a browser (WebGL) with the exact scene and light maths of
  **Sphere-Light-Render-ComfyUI** @ cc89da6 (`js/preview.js` buildScene/renderLight, `js/light.js` lightPosition) and
  its vendored three.js r128. Script: `../../tools/bake_light_balls/` (`bake.html` + `server.py`).
- Why baked: that node renders in the browser only (the server just decodes a `render_b64` baked into the saved
  workflow), so an API or Brand Brain run could not change the light direction per run.
- Licence: Sphere-Light-Render-ComfyUI is MIT, © 2026 eric-venti-seeds and © 2026 Christopher Connock
  (`LICENSE-Sphere-Light-Render.txt`). three.js is MIT, © 2010–2021 three.js authors.
- Check: the plane grey runs from 156 (15° sun) to 243 (70°). The Relight model card measures ≈ RGB(173,173,173) on its
  low-sun example, which falls in that range. Compare one render with the `render_b64` in the gated official
  `LTX-2.3_Relight_ICLoRA_SingleStage_Distilled.json` before production.
