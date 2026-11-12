"""ComfyUI nodes for video reframing (classic node API, like ComfyUI-VideoTrimmer). The work is in reframe_core."""
import json, os, uuid

import folder_paths

from . import reframe_core as rc

try:
    from comfy_api.latest import InputImpl
    VideoFromFile = InputImpl.VideoFromFile
except ImportError:  # older ComfyUI
    from comfy_api.input_impl import VideoFromFile

CATEGORY = "video/reframe"


def _temp(prefix, ext=".mp4"):
    d = folder_paths.get_temp_directory()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{prefix}_{uuid.uuid4().hex[:12]}{ext}")


def _as_file(video):
    """A plain file path for `video`: the file itself when it is an untrimmed, uncropped file on disk, else a temp
    copy written by the video's own save_to (trims, crops, in-memory files and tensor videos)."""
    if isinstance(video, VideoFromFile) and getattr(video, "_VideoFromFile__crop", None) is None:
        start, duration = video.get_active_trim_window()
        src = video.get_stream_source()
        if isinstance(src, str) and not start and not duration:
            return src
    path = _temp("reframe_in")
    video.save_to(path)
    return path


class VideoReframePasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("VIDEO", {"tooltip": "The original clip the master was made from."}),
                "master": ("VIDEO", {"tooltip": "The reframing model's output (MiniMax, Luma, ...): the original "
                                                "somewhere inside a bigger canvas."}),
                "seam": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.2, "step": 0.005,
                                   "tooltip": "Blend width: Gaussian sigma as a share of the original's short side. "
                                              "The original's area is eroded 3 sigma first, so the blend reaches "
                                              "zero before the original ends (no hard line)."}),
            },
            "optional": {
                "placement": (["measure", "centred fit"], {"default": "measure",
                              "tooltip": "measure: find where the model put the original (scale and shift searched "
                                         "on 3 frames). centred fit: assume it is fitted and centred."}),
                "output_size": (["source", "master"], {"default": "source",
                                "tooltip": "source: the original comes back at its own resolution, pasted 1:1, and "
                                           "only the model's generated edges are upscaled to match. master: keep the "
                                           "model's output size."}),
                "max_side": ("INT", {"default": 4096, "min": 0, "max": 16384, "step": 2,
                             "tooltip": "With output_size source: cap on the output's long side (0 = no cap). "
                                        "4096 keeps H.264 files playable everywhere."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "placement")
    OUTPUT_TOOLTIPS = ("The master with the original's own pixels pasted back where the model put them, H.264, "
                       "with the source's audio.",
                       "JSON: the measured rectangle, scale, shift and match errors.")
    FUNCTION = "paste"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Paste the original clip back over a reframing model's output. Measures where the model placed "
                   "it (models shift or scale it by a few pixels), samples both clips to the same frame count, "
                   "blends the seam on the sides that meet generated pixels, and keeps the source's sound.")

    def paste(self, source, master, seam, placement="measure", output_size="source", max_side=4096):
        src, mst = _as_file(source), _as_file(master)
        rect = None
        if placement == "centred fit":
            sp, mp = rc.probe(src), rc.probe(mst)
            fit = min(mp["width"] / sp["width"], mp["height"] / sp["height"])
            w, h = sp["width"] * fit, sp["height"] * fit
            rect = [(mp["width"] - w) / 2, (mp["height"] - h) / 2, w, h]
        out = _temp("reframe_paste")
        report = rc.paste_back(src, mst, out, rect=rect, seam=seam, output_size=output_size, max_side=max_side)
        return (VideoFromFile(out), json.dumps(report))


class VideoCropToRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "ratio_w": ("INT", {"default": 9, "min": 1, "max": 100000}),
                "ratio_h": ("INT", {"default": 16, "min": 1, "max": 100000}),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT")
    RETURN_NAMES = ("video", "width", "height")
    FUNCTION = "crop"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The largest centred rectangle of exactly ratio_w:ratio_h inside the video, even sizes. Lazy for "
                   "files (nothing is decoded until the video is saved) and keeps the audio. Takes lists, one crop "
                   "per ratio.")

    def crop(self, video, ratio_w, ratio_h):
        W, H = video.get_dimensions()
        x, y, w, h = rc.crop_to_ratio(W, H, ratio_w, ratio_h)
        if (w, h) == (W, H):
            return (video, w, h)
        return (video.as_cropped(x, y, w, h), w, h)


class VideoRetime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "speed": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 20.0, "step": 0.01,
                                    "tooltip": "2 = twice as fast (half as long, double the frame rate); 0.5 slows "
                                               "it back. Every frame is kept. 1 returns the video untouched."}),
                "audio": (["tempo", "drop"], {"default": "tempo",
                          "tooltip": "tempo: speed the sound up or down by the same factor (pitch kept). drop: no "
                                     "audio."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "retime"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Change a video's speed by changing its frame rate: the same frames, faster or slower. Used to "
                   "send a clip of 10 s or more to Luma Ray 3.2 (which refuses those) and slow the result back.")

    def retime(self, video, speed, audio):
        if abs(float(speed) - 1.0) < 1e-6:
            return (video,)
        out = _temp("reframe_retime")
        rc.retime(_as_file(video), out, float(speed), audio)
        return (VideoFromFile(out),)


class VideoReframeLTXCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The clip to outpaint."}),
                "ratio": ("FLOAT", {"default": 0.5625, "min": 0.1, "max": 10.0, "step": 0.0001,
                                    "tooltip": "The canvas ratio, width / height (9:16 = 0.5625). The clip is padded "
                                               "to it, nothing is cut."}),
                "pixels": ("INT", {"default": 921600, "min": 65536, "max": 8294400, "step": 1024,
                                   "tooltip": "Canvas area in pixels (921600 = 1280x720). Never above the clip's own "
                                              "scale; each side a multiple of 32."}),
                "max_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 600.0, "step": 0.5,
                                          "tooltip": "Longer clips stop the run with a clear message (0 = no limit)."}),
                "gamma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 4.0, "step": 0.05,
                                    "tooltip": "Power curve on the guide frames, as in the LTX 2.3 outpainting "
                                               "template; Image Gamma 0.5 on the decoded frames undoes it."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("guide", "width", "height", "frames", "fps", "fps_int")
    OUTPUT_TOOLTIPS = ("The clip fitted inside a black canvas of the ratio, 8n+1 frames.", "Canvas width.",
                       "Canvas height.", "Frame count (8n+1).",
                       "Frame rate that keeps the clip's duration with these frames.", "The same, rounded.")
    FUNCTION = "canvas"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Guide frames for LTX 2.3 IC-LoRA outpainting: the clip centred on a black canvas of the given ratio "
                   "and area, sampled to 8n+1 frames with the same map Reframe Paste Back uses, so the master lines up "
                   "with the source frame for frame. Streams the clip; only canvas-size frames are held.")

    def canvas(self, video, ratio, pixels, max_seconds, gamma):
        import torch
        frames, info = rc.ltx_canvas_frames(_as_file(video), float(ratio), int(pixels), float(max_seconds),
                                            float(gamma))
        return (torch.from_numpy(frames), info["width"], info["height"], info["frames"], info["fps"],
                max(1, int(round(info["fps"]))))


class VideoReframeImageGamma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 4.0, "step": 0.05,
                                    "tooltip": "Each value v becomes v ** gamma (below 1 brightens, above darkens)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "A power curve on images (v ** gamma, values clamped to 0-1 first). 0.5 undoes the LTX guide curve."

    def apply(self, image, gamma):
        return (image.clamp(0.0, 1.0).pow(float(gamma)),)


class VideoReframeInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "frames", "fps", "duration")
    FUNCTION = "info"
    CATEGORY = CATEGORY
    DESCRIPTION = "Width, height, frame count, frame rate and duration of a video, without decoding its frames."

    def info(self, video):
        src = video.get_stream_source() if isinstance(video, VideoFromFile) else None
        if isinstance(src, str) and getattr(video, "_VideoFromFile__crop", None) is None \
                and video.get_active_trim_window() == (0.0, 0.0):
            p = rc.probe(src)
            return (p["width"], p["height"], p["frames"], float(p["fps"]), float(p["duration"]))
        W, H = video.get_dimensions()
        return (W, H, int(video.get_frame_count()), float(video.get_frame_rate()), float(video.get_duration()))


NODE_CLASS_MAPPINGS = {
    "VideoReframePasteBack": VideoReframePasteBack,
    "VideoCropToRatio": VideoCropToRatio,
    "VideoRetime": VideoRetime,
    "VideoReframeInfo": VideoReframeInfo,
    "VideoReframeLTXCanvas": VideoReframeLTXCanvas,
    "VideoReframeImageGamma": VideoReframeImageGamma,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoReframeLTXCanvas": "LTX Outpaint Canvas",
    "VideoReframeImageGamma": "Image Gamma (Reframe)",
    "VideoReframePasteBack": "Reframe Paste Back",
    "VideoCropToRatio": "Crop Video To Ratio",
    "VideoRetime": "Retime Video",
    "VideoReframeInfo": "Video Info (Reframe)",
}
