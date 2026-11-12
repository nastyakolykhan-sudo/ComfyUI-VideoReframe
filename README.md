# ComfyUI-VideoReframe

Nodes for reframing video with an outpainting model (MiniMax H3 Max Reframe, Luma Ray 3.2 Reframe, LTX 2.3, …) while keeping
the original clip's own pixels and sound. Built for the Silverside **Video Reframer** workflow (v02). ComfyUI 0.45+;
no extra dependencies (PyAV, numpy, scipy and Pillow ship with ComfyUI).

| Node | What it does |
|---|---|
| **Reframe Paste Back** (`VideoReframePasteBack`) | Pastes the original back over the model's master. Measures where the model put it (normalised cross-correlation, scale 0.97–1.04 and shift ±40 px at half resolution, refined at full resolution, median of 3 frames), samples both clips to N = min(frames) with the workflow's map `round(i*(T-1)/(N-1))`, blends through a seam mask (Gaussian sigma = `seam` × the original's short side, eroded 3 sigma first, seam sides only), streams frame by frame, writes H.264 (crf 12, yuv420p) with the source's audio. `output_size` **source** (default) upscales the model's frames so the original comes back at its own resolution, pasted 1:1 (long side capped by `max_side`, default 4096); **master** keeps the model's size. `ratio` (w/h, 0 = off): the most extreme ratio the master was made for; when the scaled master falls a little short of it (models round their canvases: MiniMax 768P 9:16 is 768x1344, 21:9 about 2.29:1), only the generated margins are stretched to reach it (up to 6%), so an exact-ratio crop never cuts the original. Outputs the video and a JSON placement report. |
| **Crop Video To Ratio** (`VideoCropToRatio`) | Largest centred rectangle of exactly `ratio_w:ratio_h`, even sizes. Lazy (`as_cropped`) for files, keeps audio, no preview output. List-friendly: one call per ratio. |
| **Retime Video** (`VideoRetime`) | Same frames at fps × speed; audio tempo-changed (atempo, pitch kept) or dropped. Speed 1 passes the video through. For Luma's 10 s limit: ×2 before, ×0.5 after. |
| **Video Info (Reframe)** (`VideoReframeInfo`) | Width, height, frames, fps, duration without decoding. |
| **LTX Outpaint Canvas** (`VideoReframeLTXCanvas`) | Guide frames for LTX 2.3 IC-LoRA outpainting (open weights): the clip centred on a black canvas of `ratio` (w/h), about `pixels` in area (never above the clip's own scale), sides multiples of 32 with the expanded side rounded up so an exact-ratio crop never cuts the clip; sampled to 8n+1 frames with the paste-back's map, so the master lines up frame for frame; power curve `gamma` (2, as in the Comfy LTX 2.3 outpainting template). Streams the clip; clips over `max_seconds` (default 10) stop with a message. Outputs the frames, canvas size, frame count and the fps that keeps the clip's duration. |
| **Image Gamma (Reframe)** (`VideoReframeImageGamma`) | `v ** gamma` on images (clamped to 0–1); 0.5 on the decoded LTX frames undoes the guide curve. |

Why measure placement: on 25 masters (2026-11-11) Luma kept the original's scale but shifted it 0, 8 or 16 px; MiniMax
kept it centred but enlarged it 1.000–1.025×, different per clip. A paste at the centred fit ghosts or misregisters.

Offline use (no ComfyUI): `python reframe_core.py measure SRC MASTER`, `paste SRC MASTER OUT`, `retime SRC OUT SPEED`.
