# ComfyUI-VideoReframe

Nodes for reframing video with an outpainting model (MiniMax H3 Max Reframe, Luma Ray 3.2 Reframe, …) while keeping
the original clip's own pixels and sound. Built for the Silverside **Video Reframer** workflow (v02). ComfyUI 0.45+;
no extra dependencies (PyAV, numpy, scipy and Pillow ship with ComfyUI).

| Node | What it does |
|---|---|
| **Reframe Paste Back** (`VideoReframePasteBack`) | Pastes the original back over the model's master. Measures where the model put it (normalised cross-correlation, scale 0.97–1.04 and shift ±40 px at half resolution, refined at full resolution, median of 3 frames), samples both clips to N = min(frames) with the workflow's map `round(i*(T-1)/(N-1))`, blends through a seam mask (Gaussian sigma = `seam` × the original's short side, eroded 3 sigma first, seam sides only), streams frame by frame, writes H.264 (crf 12, yuv420p) with the source's audio. `output_size` **source** (default) upscales the model's frames so the original comes back at its own resolution, pasted 1:1 (long side capped by `max_side`, default 4096); **master** keeps the model's size. Outputs the video and a JSON placement report. |
| **Crop Video To Ratio** (`VideoCropToRatio`) | Largest centred rectangle of exactly `ratio_w:ratio_h`, even sizes. Lazy (`as_cropped`) for files, keeps audio, no preview output. List-friendly: one call per ratio. |
| **Retime Video** (`VideoRetime`) | Same frames at fps × speed; audio tempo-changed (atempo, pitch kept) or dropped. Speed 1 passes the video through. For Luma's 10 s limit: ×2 before, ×0.5 after. |
| **Video Info (Reframe)** (`VideoReframeInfo`) | Width, height, frames, fps, duration without decoding. |

Why measure placement: on 25 masters (2026-11-11) Luma kept the original's scale but shifted it 0, 8 or 16 px; MiniMax
kept it centred but enlarged it 1.000–1.025×, different per clip. A paste at the centred fit ghosts or misregisters.

Offline use (no ComfyUI): `python reframe_core.py measure SRC MASTER`, `paste SRC MASTER OUT`, `retime SRC OUT SPEED`.
