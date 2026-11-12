"""Core of ComfyUI-VideoReframe: pure functions over video files (PyAV + numpy + scipy + Pillow), no ComfyUI imports,
so everything here can be tested outside ComfyUI.

- probe(path)                      size, fps, frame count, duration, audio, without decoding
- sample_indices(total, n)         the workflow's even sampling map
- measure_placement(src, master)   where a reframing model put the original inside its master
- seam_mask(...)                   1 on the original, smooth to 0 on the seam sides only
- paste_back(src, master, out)     the original pasted back at the measured place, H.264 + the source's audio
- crop_to_ratio(w, h, rw, rh)      the largest centred rectangle of an exact ratio, even sizes
- retime(src, out, speed)          same frames at fps * speed (audio tempo-changed or dropped)
- ltx_canvas_frames(src, ratio, …) the clip on a black canvas of `ratio`, 8n+1 frames, for LTX 2.3 outpainting
"""
import json, math, os, time
from fractions import Fraction

import av
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------------------------------- probing


def probe(path):
    """Size, frame rate, frame count (demuxed packets, pts >= 0), duration and audio codec, without decoding."""
    with av.open(path) as c:
        if not c.streams.video:
            raise ValueError(f"No video stream in {os.path.basename(str(path))}")
        vs = c.streams.video[0]
        fps = Fraction(vs.average_rate or vs.guessed_rate or 24).limit_denominator(100_000)
        n = 0
        for p in c.demux(vs):
            if p.pts is not None and p.size and p.pts >= 0 and not getattr(p, "is_discard", False):
                n += 1
        dur = float(c.duration / av.time_base) if c.duration else n / float(fps)
        audio = c.streams.audio[0].codec_context.name if c.streams.audio else None
        return {"width": vs.codec_context.width, "height": vs.codec_context.height, "fps": fps, "frames": n,
                "duration": dur, "audio": audio}


def sample_indices(total, n):
    """The workflow's even sampling: n of total frames, index round(i * (total - 1) / (n - 1))."""
    if n <= 1:
        return [total // 2]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def iter_frames(path, wanted=None, fmt="rgb24"):
    """Yield (index, ndarray) for decoded frames in order; only indices in `wanted` (a set) when given."""
    last = max(wanted) if wanted else None
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for i, f in enumerate(c.decode(vs)):
            if wanted is None or i in wanted:
                yield i, f.to_ndarray(format=fmt)
            if last is not None and i >= last:
                break


# ---------------------------------------------------------------------------------------------------- placement


def _resize(a, w, h, resample=Image.BILINEAR):
    return np.asarray(Image.fromarray(a).resize((max(1, int(round(w))), max(1, int(round(h)))), resample),
                      np.float32)


def _ncc_map(region, tmpl):
    """Normalised cross-correlation of tmpl at every valid position inside region (2-D float32 arrays)."""
    th, tw = tmpl.shape
    n = th * tw
    t = tmpl - tmpl.mean()
    tn = math.sqrt(float((t * t).sum())) + 1e-6
    num = fftconvolve(region, t[::-1, ::-1], mode="valid")
    # local sums of region and region^2 over th x tw windows (integral images)
    def box(a):
        c = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0))), 0), 1)
        return c[th:, tw:] - c[:-th, tw:] - c[th:, :-tw] + c[:-th, :-tw]
    s1 = box(region.astype(np.float64)); s2 = box(region.astype(np.float64) ** 2)
    var = np.maximum(s2 - s1 * s1 / n, 1e-6)
    return num / (np.sqrt(var) * tn)


def _search(M, S, fw, fh, scales, cx, cy, radius, margin=0.12):
    """Best (ncc, scale, dx, dy): S resized to (fw, fh) * scale, centred at (cx + dx, cy + dy) in M."""
    MH, MW = M.shape
    best = (-2.0, 1.0, 0, 0)
    for s in scales:
        w, h = fw * s, fh * s
        T = _resize(S, w, h)
        th, tw = T.shape
        m = int(margin * min(th, tw))
        T = T[m:th - m, m:tw - m]
        # where the inner template's top-left sits when centred, then the search window around it
        x0 = int(round(cx - tw / 2)) + m; y0 = int(round(cy - th / 2)) + m
        rx0, ry0 = max(0, x0 - radius), max(0, y0 - radius)
        rx1, ry1 = min(MW, x0 + T.shape[1] + radius), min(MH, y0 + T.shape[0] + radius)
        R = M[ry0:ry1, rx0:rx1]
        if R.shape[0] < T.shape[0] or R.shape[1] < T.shape[1]:
            continue
        ncc = _ncc_map(R, T)
        iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
        if ncc[iy, ix] > best[0]:
            best = (float(ncc[iy, ix]), float(s), (rx0 + ix) - x0, (ry0 + iy) - y0)
    return best


def measure_placement(src, master, frames=3, sp=None, mp=None):
    """Where the model put the original inside its master: contain-fit as the start, scale 0.97-1.04 and shift
    +-40 px searched at half resolution, refined at full resolution, median over `frames` frames.
    Returns {"rect": [x, y, w, h], "scale", "dx", "dy", "fit_rect", "ncc", "err_fit", "err_measured", "per_frame",
    "seconds"} in master pixels (the rect may run past the master's edges)."""
    t0 = time.time()
    sp = sp or probe(src); mp = mp or probe(master)
    SW, SH, MW, MH = sp["width"], sp["height"], mp["width"], mp["height"]
    n = min(sp["frames"], mp["frames"])
    smap, mmap = sample_indices(sp["frames"], n), sample_indices(mp["frames"], n)
    picks = sorted({int(round(t * (n - 1))) for t in np.linspace(0.2, 0.8, frames)})
    sw = {smap[i] for i in picks}; mw = {mmap[i] for i in picks}
    sf = dict(iter_frames(src, sw, "gray")); mf = dict(iter_frames(master, mw, "gray"))
    fit = min(MW / SW, MH / SH); fw, fh = SW * fit, SH * fit
    per = []
    for i in picks:
        S = sf[smap[i]]; M = mf[mmap[i]].astype(np.float32)
        M2 = _resize(mf[mmap[i]], MW / 2, MH / 2, Image.BOX)
        S2 = np.asarray(Image.fromarray(S).resize((max(1, SW // 2), max(1, SH // 2)), Image.BOX))
        c = _search(M2, S2, fw / 2, fh / 2, np.arange(0.97, 1.0401, 0.005), MW / 4, MH / 4, 20)
        _, s0, dx0, dy0 = c
        f = _search(M, S, fw, fh, np.arange(s0 - 0.004, s0 + 0.00401, 0.001), MW / 2 + 2 * dx0, MH / 2 + 2 * dy0, 3)
        ncc, s, dx, dy = f
        dx += 2 * dx0; dy += 2 * dy0
        per.append({"frame": i, "ncc": round(ncc, 4), "scale": round(s, 4), "dx": int(dx), "dy": int(dy)})
    s = float(np.median([p["scale"] for p in per])); dx = float(np.median([p["dx"] for p in per]))
    dy = float(np.median([p["dy"] for p in per]))
    w, h = fw * s, fh * s
    rect = [MW / 2 - w / 2 + dx, MH / 2 - h / 2 + dy, w, h]
    fit_rect = [(MW - fw) / 2, (MH - fh) / 2, fw, fh]
    i = picks[len(picks) // 2]
    out = {"rect": [round(v, 1) for v in rect], "scale": round(s, 4), "dx": dx, "dy": dy,
           "fit_rect": [round(v, 1) for v in fit_rect], "ncc": float(np.median([p["ncc"] for p in per])),
           "err_fit": _inner_err(mf[mmap[i]], sf[smap[i]], fit_rect),
           "err_measured": _inner_err(mf[mmap[i]], sf[smap[i]], rect), "per_frame": per,
           "seconds": round(time.time() - t0, 2)}
    return out


def _inner_err(M, S, rect, margin=0.12):
    """Mean |master - original| (0-1) over the inner part of rect."""
    x, y, w, h = rect
    T = _resize(S, w, h) / 255.0
    X, Y = int(round(x)), int(round(y))
    m = int(margin * min(T.shape))
    xa, ya = max(X + m, 0), max(Y + m, 0)
    xb, yb = min(X + T.shape[1] - m, M.shape[1]), min(Y + T.shape[0] - m, M.shape[0])
    if xb - xa < 10 or yb - ya < 10:
        return None
    return round(float(np.abs(M[ya:yb, xa:xb] / 255.0 - T[ya - Y:yb - Y, xa - X:xb - X]).mean()), 4)


# ---------------------------------------------------------------------------------------------------- paste-back


def seam_mask(MW, MH, X, Y, W, H, seam=0.02, pad=64):
    """Mask (MH x MW x 1, float32) in master space: 1 on the original, smooth to 0 at its seam sides. Sigma = seam *
    the original's short side (at least 3 px); the area is eroded 3 sigma (taxicab, like a cross-kernel erosion)
    first so the mask is ~0 where the original ends. Sides at or past the master's edge stay 1 up to the edge."""
    sg = max(3, round(min(W, H) * seam)); er = 3 * sg
    CH, CW = MH + 2 * pad, MW + 2 * pad
    area = np.zeros((CH, CW), bool)
    x0, y0, x1, y1 = X, Y, X + W, Y + H
    left = 0 if x0 <= 2 else x0 + pad
    top = 0 if y0 <= 2 else y0 + pad
    right = CW if x1 >= MW - 2 else x1 + pad
    bottom = CH if y1 >= MH - 2 else y1 + pad
    area[max(0, top):max(0, bottom), max(0, left):max(0, right)] = True
    dist = ndimage.distance_transform_cdt(area, metric="taxicab")
    m = (dist > er).astype(np.float32)
    m = ndimage.gaussian_filter(m, sg)[pad:pad + MH, pad:pad + MW]
    keep = np.zeros_like(m)
    keep[max(0, y0):min(MH, y1), max(0, x0):min(MW, x1)] = 1
    return (m * keep)[..., None], sg


def _open_writer(path, fps, width, height, crf=12, preset="fast"):
    out = av.open(path, mode="w")
    vs = out.add_stream("libx264", rate=fps, options={"crf": str(crf), "preset": preset})
    vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
    vs.codec_context.time_base = Fraction(1) / fps
    return out, vs


def _put(out, vs, arr, pts, src_frame_props=None):
    f = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
    f.pts = pts
    f.time_base = vs.codec_context.time_base
    for p in vs.encode(f):
        out.mux(p)


def _flush(out, vs):
    for p in vs.encode():
        out.mux(p)


def _add_audio(src, out, tempo=1.0):
    """Add the source's first audio stream to `out` BEFORE any packet is muxed (a stream added after the header is
    written crashes libav). Returns a writer to call after the video is done, or None when there's no audio.
    Packets are copied when AAC and no tempo change; otherwise decoded, (tempo-changed via atempo) and encoded AAC."""
    with av.open(src) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        copy = tempo == 1.0 and a.codec_context.name == "aac"
        rate = a.codec_context.sample_rate or 48000
        layout = a.codec_context.layout.name if a.codec_context.layout else "stereo"
        oa = out.add_stream_from_template(a) if copy else out.add_stream("aac", rate=rate, layout=layout)

    def write():
        with av.open(src) as c:
            a = c.streams.audio[0]
            if copy:
                for p in c.demux(a):
                    if p.dts is None:
                        continue
                    p.stream = oa
                    out.mux(p)
                return
            graph = None
            if tempo != 1.0:
                graph = av.filter.Graph()
                node = graph.add_abuffer(template=a)
                chain, t = [], tempo
                while t > 2.0:
                    chain.append(2.0); t /= 2.0
                while t < 0.5:
                    chain.append(0.5); t /= 0.5
                chain.append(t)
                for k in chain:
                    nxt = graph.add("atempo", f"{k:.6f}"); node.link_to(nxt); node = nxt
                sink = graph.add("abuffersink"); node.link_to(sink)
                graph.configure()
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)

            def enc(fr):
                for rf in resampler.resample(fr):
                    for p in oa.encode(rf):
                        out.mux(p)

            def drain():
                while True:
                    try:
                        enc(graph.pull())
                    except (av.BlockingIOError, av.EOFError):
                        break

            for frame in c.decode(a):
                if graph is None:
                    enc(frame)
                else:
                    graph.push(frame); drain()
            if graph is not None:
                graph.push(None); drain()
            enc(None)  # flush the resampler
            for p in oa.encode(None):
                out.mux(p)

    return write


def output_scale(rect_w, src_w, master_w, master_h, mode="source", max_side=4096):
    """Factor applied to the master before the paste. 'source': the original comes back at its own resolution
    (factor = source width / the original's width in the master), capped so the long side stays <= max_side
    (0 = no cap). 'master': 1 (output at the model's size)."""
    if mode != "source":
        return 1.0
    f = src_w / rect_w
    if max_side and max(master_w, master_h) * f > max_side:
        f = max_side / max(master_w, master_h)
    return f


def stretch_plan(MW, MH, X, Y, W, H, ratio, max_stretch=0.06):
    """When a master is a little short of `ratio` (w/h) on the side it was made for, how to widen (or heighten) it by
    stretching only the generated margins, so an exact-ratio crop never has to cut the original. Models round their
    canvases (MiniMax 768P: 9:16 is 768x1344, 0.571), which leaves the master 1-2% short of the ratio it was asked
    for. Returns None (nothing to do, or the shortfall is over max_stretch of the margins) or
    {"axis": "x"|"y", "size": new width|height, "a": first margin, "b": last margin, "a2", "b2": their new sizes,
     "shift": the original's move along the axis}."""
    if not ratio or W <= 0 or H <= 0:
        return None
    src_r = W / H
    if ratio > src_r * 1.001 and MW < MH * ratio - 0.5:          # wide: needs more width
        axis, size, lo, span = "x", 2 * math.ceil(MH * ratio / 2), X, (MW, W)
    elif ratio < src_r / 1.001 and MH < MW / ratio - 0.5:        # tall: needs more height
        axis, size, lo, span = "y", 2 * math.ceil(MW / ratio / 2), Y, (MH, H)
    else:
        return None
    full, orig = span
    a = max(0, lo); b = max(0, full - (lo + orig))              # the generated margins before and after the original
    extra = size - full
    if a + b <= 0 or extra / float(a + b) > max_stretch:
        return None
    a2 = a + int(round(extra * a / float(a + b))); b2 = b + (extra - (a2 - a))
    return {"axis": axis, "size": size, "a": a, "b": b, "a2": a2, "b2": b2, "shift": a2 - a}


def _apply_stretch(o, sp):
    """Stretch frame o's two generated margins along sp['axis'] (the middle, the original's span, is kept as is)."""
    if sp["axis"] == "x":
        Hh, Ww = o.shape[:2]; a, b = sp["a"], sp["b"]
        parts = [_resize_f(o[:, :a], sp["a2"], Hh) if a else o[:, :0], o[:, a:Ww - b],
                 _resize_f(o[:, Ww - b:], sp["b2"], Hh) if b else o[:, :0]]
        return np.concatenate(parts, axis=1)
    Hh, Ww = o.shape[:2]; a, b = sp["a"], sp["b"]
    parts = [_resize_f(o[:a], Ww, sp["a2"]) if a else o[:0], o[a:Hh - b], _resize_f(o[Hh - b:], Ww, sp["b2"]) if b else o[:0]]
    return np.concatenate(parts, axis=0)


def _resize_f(a, w, h):
    """Resize a float32 HxWx3 array (0-255) with Lanczos."""
    return np.asarray(Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8)).resize((int(w), int(h)), Image.LANCZOS),
                      np.float32)


def paste_back(src, master, out_path, rect=None, seam=0.02, crf=12, preset="fast", placement_frames=3,
               output_size="source", max_side=4096, ratio=0.0, max_stretch=0.06):
    """Paste the original back over the model's master where the model put it, streaming frame by frame.
    Both clips are sampled to N = min(frames) (the workflow's map); fps = master fps * N / master frames, so the
    duration stays the master's. output_size 'source' (default) upscales the model's frames so the original comes
    back at its own resolution, pasted 1:1 (only the generated edges are upscaled); 'master' keeps the model's size.
    ratio (w/h, 0 = off): the most extreme ratio this master was made for; when the scaled master falls a little
    short of it (the model's canvas is rounded), the generated margins are stretched to reach it (up to max_stretch
    of their size), so cropping that ratio never cuts the original.
    Writes H.264 (yuv420p, even size) + the source's audio to out_path. Returns the placement report (dict)."""
    t0 = time.time()
    sp, mp = probe(src), probe(master)
    pl = measure_placement(src, master, placement_frames, sp, mp) if rect is None else {"rect": list(rect)}
    x, y, w, h = pl["rect"]
    MW0, MH0 = mp["width"], mp["height"]

    def layout(f):
        MW, MH = int(round(MW0 * f)), int(round(MH0 * f))
        if output_size == "source" and abs(f - sp["width"] / w) < 1e-9:
            W, H = sp["width"], sp["height"]  # the original exactly at its own size: no resampling
        else:
            W, H = int(round(w * f)), int(round(h * f))
        X, Y = int(round(x * f)), int(round(y * f))
        return MW, MH, W, H, X, Y, stretch_plan(MW, MH, X, Y, W, H, ratio, max_stretch)

    f = output_scale(w, sp["width"], MW0, MH0, output_size, 0)
    MW, MH, W, H, X, Y, st = layout(f)
    TW, TH = (st["size"], MH) if st and st["axis"] == "x" else ((MW, st["size"]) if st else (MW, MH))
    if output_size == "source" and max_side and max(TW, TH) > max_side:   # the cap applies to the final size
        f *= max_side / float(max(TW, TH))
        MW, MH, W, H, X, Y, st = layout(f)
        TW, TH = (st["size"], MH) if st and st["axis"] == "x" else ((MW, st["size"]) if st else (MW, MH))
    if st:
        X, Y = (X + st["shift"], Y) if st["axis"] == "x" else (X, Y + st["shift"])
    n = min(sp["frames"], mp["frames"])
    smap, mmap = sample_indices(sp["frames"], n), sample_indices(mp["frames"], n)
    fps = mp["fps"] if n == mp["frames"] else Fraction(mp["fps"] * n / mp["frames"]).limit_denominator(100_000)
    mask, sg = seam_mask(TW, TH, X, Y, W, H, seam)
    xa, ya, xb, yb = max(0, X), max(0, Y), min(TW, X + W), min(TH, Y + H)
    k = mask[ya:yb, xa:xb]
    OW, OH = TW - TW % 2, TH - TH % 2
    out, vs = _open_writer(out_path, fps, OW, OH, crf, preset)
    write_audio = _add_audio(src, out)
    S = iter_frames(src, set(smap)); M = iter_frames(master, set(mmap))
    count = 0
    try:
        for (_, s_f), (_, m_f) in zip(S, M):
            z = s_f.astype(np.float32) if (W, H) == (s_f.shape[1], s_f.shape[0]) else _resize(s_f, W, H, Image.LANCZOS)
            o = m_f.astype(np.float32) if (MW, MH) == (MW0, MH0) else _resize(m_f, MW, MH, Image.LANCZOS)
            if st:
                o = _apply_stretch(o, st)
            o[ya:yb, xa:xb] = z[ya - Y:yb - Y, xa - X:xb - X] * k + o[ya:yb, xa:xb] * (1 - k)
            _put(out, vs, np.clip(o + 0.5, 0, 255).astype(np.uint8)[:OH, :OW], count)
            count += 1
        _flush(out, vs)
        audio = write_audio is not None
        if audio:
            write_audio()
    finally:
        out.close()
    pl.update({"frames": count, "fps": str(fps), "size": [OW, OH], "master_size": [MW0, MH0],
               "output_size": output_size, "upscale": round(f, 4), "original_pasted_at": [X, Y, W, H],
               "margin_stretch": ({"axis": st["axis"], "from": [st["a"], st["b"]], "to": [st["a2"], st["b2"]]} if st else None),
               "seam_sigma_px": sg, "audio": audio, "total_seconds": round(time.time() - t0, 2)})
    return pl


# ---------------------------------------------------------------------------------------------------- LTX canvas


def ltx_frame_count(total):
    """LTX generates 8n+1 frames: the largest such count <= total (at least 9)."""
    if total < 9:
        raise ValueError(f"LTX needs at least 9 frames; this clip has {total}.")
    return 8 * ((total - 1) // 8) + 1


def ltx_canvas_size(W, H, ratio, pixels, multiple=32, min_side=256):
    """The LTX canvas for a W x H clip outpainted to `ratio` (w/h): the clip padded to that ratio (nothing cut), scaled
    to about `pixels` in area (never above the clip's own scale), each side a multiple of `multiple`. The expanded
    side is rounded up, so the canvas is never short of `ratio` there and the exact-ratio crop never cuts the clip.
    Returns (cw, ch, x, y, fw, fh): the canvas size and the fitted clip's rectangle in it (even sizes, centred)."""
    CW, CH = max(W, H * ratio), max(H, W / ratio)
    s = min(1.0, math.sqrt(pixels / (CW * CH)))
    if H * ratio >= W:  # wider (or the same): the clip fills the height, the width grows
        ch = max(min_side, int(multiple * round(H * s / multiple)))
        cw = max(min_side, int(multiple * math.ceil(ch * ratio / multiple - 1e-9)))
    else:  # taller: the clip fills the width, the height grows
        cw = max(min_side, int(multiple * round(W * s / multiple)))
        ch = max(min_side, int(multiple * math.ceil(cw / ratio / multiple - 1e-9)))
    f = min(cw / W, ch / H)
    fw, fh = min(cw, 2 * round(W * f / 2)), min(ch, 2 * round(H * f / 2))
    return cw, ch, (cw - fw) // 2, (ch - fh) // 2, int(fw), int(fh)


def ltx_canvas_frames(src, ratio, pixels, max_seconds=10.0, gamma=2.0):
    """The clip as LTX outpainting guide frames: sampled to 8n+1 frames with the workflow's map (so Reframe Paste Back
    lines the master up with the source frame for frame), fitted inside the black canvas from ltx_canvas_size, then
    raised to `gamma` (the LTX 2.3 outpainting template's guide curve; 0.5 on the output undoes it). Streams the source,
    so only canvas-size frames are held. Returns (frames float32 [n, ch, cw, 3] in 0-1, info dict)."""
    sp = probe(src)
    if max_seconds and sp["duration"] > max_seconds + 0.5 / float(sp["fps"]):
        raise ValueError(f"LTX 2.3 takes clips up to {max_seconds:g} s; this one is {sp['duration']:.2f} s. Trim it or "
                         f"pick MiniMax H3 Max (up to 300 s).")
    total = sp["frames"]
    n = ltx_frame_count(total)
    idx = sample_indices(total, n)
    cw, ch, x, y, fw, fh = ltx_canvas_size(sp["width"], sp["height"], ratio, pixels)
    out = np.zeros((n, ch, cw, 3), np.float32)
    pos = {}
    for k, i in enumerate(idx):
        pos.setdefault(i, []).append(k)
    for i, f in iter_frames(src, set(idx)):
        a = _resize(f, fw, fh, Image.LANCZOS) / 255.0
        np.clip(a, 0.0, 1.0, out=a)
        if gamma != 1.0:
            a **= gamma
        for k in pos.get(i, ()):
            out[k, y:y + fh, x:x + fw] = a
    fps = Fraction(sp["fps"] * n / total).limit_denominator(100_000) if n != total else sp["fps"]
    return out, {"frames": n, "source_frames": total, "fps": float(fps), "width": cw, "height": ch,
                 "fit_rect": [x, y, fw, fh], "source_size": [sp["width"], sp["height"]]}


# ---------------------------------------------------------------------------------------------------- crop / retime


def crop_to_ratio(width, height, rw, rh):
    """(x, y, w, h): the largest centred rect of exactly rw:rh inside width x height, even sizes."""
    cw = 2 * round(min(width - width % 2, height * rw / rh) / 2)
    ch = 2 * round(min(height - height % 2, width * rh / rw) / 2)
    return (width - cw) // 2, (height - ch) // 2, int(cw), int(ch)


def retime(src, out_path, speed, audio="tempo", crf=12, preset="fast"):
    """Same frames at fps * speed (speed 2 = twice as fast, half as long). Audio tempo-changed by `speed` or
    dropped. Returns a small report."""
    sp = probe(src)
    fps = Fraction(sp["fps"] * Fraction(speed).limit_denominator(1000)).limit_denominator(100_000)
    out, vs = _open_writer(out_path, fps, sp["width"] - sp["width"] % 2, sp["height"] - sp["height"] % 2, crf, preset)
    write_audio = _add_audio(src, out, tempo=float(speed)) if audio == "tempo" else None
    n = 0
    try:
        for _, f in iter_frames(src):
            _put(out, vs, f[:vs.height, :vs.width], n)
            n += 1
        _flush(out, vs)
        wrote_audio = write_audio is not None
        if wrote_audio:
            write_audio()
    finally:
        out.close()
    return {"frames": n, "fps": str(fps), "speed": speed, "audio": wrote_audio}


if __name__ == "__main__":  # offline use: python reframe_core.py paste SRC MASTER OUT | measure SRC MASTER
    import sys
    cmd = sys.argv[1]
    if cmd == "measure":
        print(json.dumps(measure_placement(sys.argv[2], sys.argv[3]), indent=1))
    elif cmd == "paste":  # paste SRC MASTER OUT [source|master]
        mode = sys.argv[5] if len(sys.argv) > 5 else "source"
        print(json.dumps(paste_back(sys.argv[2], sys.argv[3], sys.argv[4], output_size=mode), indent=1))
    elif cmd == "retime":
        print(json.dumps(retime(sys.argv[2], sys.argv[3], float(sys.argv[4])), indent=1))
