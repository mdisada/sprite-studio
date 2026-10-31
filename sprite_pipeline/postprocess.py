"""Frame post-processing: chroma key -> crop -> pixelize -> sprite sheet.

Turns the H3 video frames (PNGs on a flat green background) into transparent,
palette-locked pixel-art sprites plus a sprite sheet and animated previews.
A single color palette is built across ALL frames of an animation so the
sprite does not flicker between palettes frame to frame.
"""
import glob
import json
import math
import os
import shutil

import numpy as np
from PIL import Image, ImageFilter


# ----------------------------------------------------------------- chroma key
def chroma_key_green(img, dominance=40, despill=True, lo=36):
    """Return RGBA with the backdrop keyed to transparency.

    The backdrop color is sampled per frame (median of a 4px border), so it
    survives the video model drifting the green toward cyan. Then:

    - hard bg/fg split at color distance `lo`: H.264 ringing (light, washed
      green just outside the outline) falls below it. A soft distance ramp
      used to keep those pixels half-opaque and despill them grey - the grey
      halo visible on dark backgrounds;
    - enclosed islands only count as backdrop when they LOOK like backdrop
      (gaps between arm and body) - teal lenses, green eyes stay solid;
    - flat green contact shadows touching the backdrop are dropped;
    - the 1px boundary gets alpha from projecting the pixel onto the line
      backdrop -> local foreground color, and takes that foreground color,
      so no backdrop is ever carried into the sprite;
    - only the outer 3px shell is despilled (yuv420 chroma bleed), so
      genuinely green costume colors survive inside.

    Falls back to the plain distance key when scipy is unavailable.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return _distance_key(img, dominance, despill)
    c = np.asarray(img.convert("RGB")).astype(np.float32)
    border = np.concatenate([c[:4].reshape(-1, 3), c[-4:].reshape(-1, 3),
                             c[:, :4].reshape(-1, 3), c[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((c - bg) ** 2).sum(axis=-1))
    mask = dist > lo

    # contact shadows: flat, wide, darker green-dominant floor patches
    lum = c.mean(axis=-1)
    gdom = c[..., 1] - np.maximum(c[..., 0], c[..., 2])
    shadow = mask & (gdom > 18) & (lum < bg.mean() - 8) & (lum > 25)
    lab, _ = ndimage.label(shadow)
    touch = ndimage.binary_dilation(~mask)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = lab[sl] == i
        if h <= 0.06 * c.shape[0] and w >= 2.5 * h and (touch[sl] & m).any():
            mask[sl][m] = False

    # enclosed "holes" that aren't backdrop-colored are mis-keyed foreground
    lab, n = ndimage.label(~mask)
    edge_ids = set(np.unique(np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if i in edge_ids or sl is None:
            continue
        m = lab[sl] == i
        if (dist[sl][m] < 30).mean() < 0.6:
            mask[sl][m] = True

    # 1px boundary: projection unmix against the local foreground color
    core = ndimage.binary_erosion(mask)
    band = mask & ~core
    w = core.astype(np.float32)
    den = ndimage.uniform_filter(w, 7)
    fg = np.stack([ndimage.uniform_filter(c[..., i] * w, 7) for i in range(3)],
                  axis=-1) / np.maximum(den, 1e-6)[..., None]
    known = den > 1e-6
    d = fg - bg
    proj = np.clip(((c - bg) * d).sum(-1) / np.maximum((d * d).sum(-1), 1.0), 0, 1)
    proj[proj < 0.2] = 0.0
    alpha = core.astype(np.float32)
    alpha = np.where(band & known, proj, alpha)
    alpha = np.where(band & ~known, 1.0, alpha)     # 1px strands stay solid
    out = np.where((band & known)[..., None], fg, c)
    if despill and bg[1] > max(bg[0], bg[2]) + dominance // 2:
        shell = mask & ~ndimage.binary_erosion(mask, iterations=3)
        cap = np.maximum(out[..., 0], out[..., 2])
        out[..., 1] = np.where(shell, np.minimum(out[..., 1], cap), out[..., 1])
    rgba = np.dstack([out, alpha * 255.0])
    return Image.fromarray(np.round(rgba).clip(0, 255).astype(np.uint8), "RGBA")


def _distance_key(img, dominance=40, despill=True, lo=32, hi=88):
    """Soft distance-ramp key (no-scipy fallback). Leaves a light rim of
    despilled codec ringing around dark outlines - prefer chroma_key_green."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=-1))
    alpha = np.clip((dist - lo) * (255.0 / max(hi - lo, 1)), 0, 255).astype(np.uint8)
    out = arr.clip(0, 255).astype(np.uint8)
    if despill and bg[1] > max(bg[0], bg[2]) + dominance // 2:
        g_kept = out[..., 1]
        cap = np.maximum(out[..., 0], out[..., 2])
        spill = (alpha > 0) & (g_kept > cap)
        out[..., 1][spill] = cap[spill]
    rgba = np.dstack([out, alpha])
    return Image.fromarray(rgba, "RGBA")


def key_glow(img, floor=10, gain=2.2):
    """Background-subtraction key for luminous VFX (glows, lightning, fire).

    Subtracts the per-frame backdrop (border median) and derives alpha from
    the luminous excess above it - the way compositors isolate an additive
    effect from a plate. Soft halos keep soft alpha; the backdrop nets zero.
    Green spill is neutralized by clamping green to max(r, b).
    """
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.clip(arr - bg, 0, 255)
    alpha = np.clip((diff.max(axis=-1) - floor) * gain, 0, 255).astype(np.uint8)
    out = arr.clip(0, 255).astype(np.uint8)
    g, cap = out[..., 1], np.maximum(out[..., 0], out[..., 2])
    spill = g > cap
    out[..., 1][spill] = cap[spill]
    return Image.fromarray(np.dstack([out, alpha]), "RGBA")


def key_frame(img, key_mode="auto", dominance=40):
    if key_mode == "glow":
        return key_glow(img)
    return chroma_key_green(img, dominance=dominance)


def union_bbox(frames, alpha_thresh=8, pad=4):
    """Smallest box containing the character across every frame (+padding)."""
    boxes = []
    for f in frames:
        a = np.asarray(f)[..., 3]
        ys, xs = np.where(a > alpha_thresh)
        if len(xs):
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        return (0, 0, frames[0].width, frames[0].height)
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(frames[0].width, max(b[2] for b in boxes) + pad)
    y1 = min(frames[0].height, max(b[3] for b in boxes) + pad)
    return (int(x0), int(y0), int(x1), int(y1))


# ------------------------------------------------------------------ pixelize
def build_shared_palette(frames, colors=48, method=None, kmeans=0):
    """Palette from all frames' opaque pixels, as a palette image.

    Measured on real sprites: MEDIANCUT + k-means refinement is the faithful
    choice (median cut splits by pixel counts, k-means then pulls each slot
    onto the actual tones - skin and gold stay warm). MAXCOVERAGE spreads
    slots to cover the RGB space evenly and tints mid-tones (greenish skin);
    avoid it for character sprites."""
    pixels = []
    for f in frames:
        arr = np.asarray(f)
        opaque = arr[arr[..., 3] > 128][:, :3]
        if len(opaque):
            step = max(1, len(opaque) // 20000)
            pixels.append(opaque[::step])
    allpix = np.concatenate(pixels) if pixels else np.zeros((1, 3), np.uint8)
    strip = Image.fromarray(allpix.reshape(1, -1, 3), "RGB")
    try:
        return strip.quantize(colors=colors, kmeans=kmeans,
                              method=method if method is not None else Image.MEDIANCUT)
    except Exception:                      # MAXCOVERAGE can reject some inputs
        return strip.quantize(colors=colors, method=Image.MEDIANCUT)


def snap_box_to_grid(box, factor, size):
    """Align a crop box to the global factor x factor sample grid.

    The pixel grid must have the same phase in every animation of a
    character, or the same art downsamples into different pixels per clip
    (and per frame if the box ever changes). Snapping the origin down and
    the far edge up to multiples of `factor` keeps every cell on one global
    grid anchored at the source's (0,0)."""
    x0, y0, x1, y1 = box
    w, h = size
    x0 -= x0 % factor
    y0 -= y0 % factor
    x1 = min(w, x1 + (-x1) % factor)
    y1 = min(h, y1 + (-y1) % factor)
    # a partial trailing cell (source edge not divisible by factor) is
    # dropped by the // in the downscaler; that is fine for sprites
    return (int(x0), int(y0), int(x1), int(y1))


def estimate_pixel_grid(img, lo=3, hi=24):
    """Estimate the art's own pixel-cluster size in source pixels.

    AI-generated 'pixel art' is painted at high resolution with an implied
    pixel grid; edge energy repeats at the cluster period, so the strongest
    autocorrelation lag of row/column gradient energy is the cluster size.
    Downscaling by (approximately) this factor is what makes the output's
    pixel squares match the art instead of fighting it."""
    a = np.asarray(img.convert("L"), np.float32)
    score = {}
    for axis in (0, 1):
        g = np.abs(np.diff(a, axis=axis)).sum(axis=1 - axis)
        g = g - g.mean()
        denom = float((g * g).mean()) or 1.0
        for p in range(lo, hi + 1):
            v = float((g[:-p] * g[p:]).mean()) / denom
            score[p] = score.get(p, 0.0) + v
    return max(score, key=score.get)


def kcentroid_downscale(frame, factor, iters=6):
    """K-centroid downscale (the Retro Diffusion / Astropulse method).

    Each output pixel's source cell is clustered into k=2 colors and the
    dominant cluster's centroid wins. Unlike NEAREST (one arbitrary sample
    per cell, noisy) or averaging (mushes outlines into mid-tones), this
    preserves hard edges AND denoises - the two things pixel art needs.
    Alpha is a majority vote per cell, weighted so color comes only from
    opaque source pixels. Vectorized over all cells of a frame at once."""
    if factor <= 1:
        return frame
    arr = np.asarray(frame).astype(np.float32)
    h2, w2 = arr.shape[0] // factor, arr.shape[1] // factor
    arr = arr[:h2 * factor, :w2 * factor]
    cells = (arr.reshape(h2, factor, w2, factor, 4)
                .transpose(0, 2, 1, 3, 4)
                .reshape(h2 * w2, factor * factor, 4))
    rgb, a = cells[..., :3], cells[..., 3]
    opaque = a >= 128
    out_a = np.where(opaque.mean(axis=1) >= 0.5, 255, 0).astype(np.uint8)
    w = opaque.astype(np.float32)
    w[w.sum(axis=1) == 0] = 1.0            # fully transparent cell: use all
    n = len(cells)
    idx = np.arange(n)
    luma = rgb @ np.array([.299, .587, .114], np.float32)
    masked = np.where(w > 0, luma, np.nan)
    cent = np.stack([rgb[idx, np.nanargmin(masked, axis=1)],
                     rgb[idx, np.nanargmax(masked, axis=1)]], axis=1)  # (n,2,3)
    for _ in range(iters):
        d = ((rgb[:, :, None, :] - cent[:, None, :, :]) ** 2).sum(-1)
        assign = d.argmin(-1)
        for j in (0, 1):
            m = ((assign == j) & (w > 0)).astype(np.float32)
            cnt = m.sum(1)
            s = (rgb * m[..., None]).sum(1)
            upd = cnt > 0
            cent[upd, j] = s[upd] / cnt[upd, None]
    d = ((rgb[:, :, None, :] - cent[:, None, :, :]) ** 2).sum(-1)
    assign = d.argmin(-1)
    counts = np.stack([((assign == j) & (w > 0)).sum(1) for j in (0, 1)], 1)
    out_rgb = cent[idx, counts.argmax(1)].clip(0, 255).astype(np.uint8)
    out = np.concatenate([out_rgb, out_a[:, None]], axis=1)
    return Image.fromarray(out.reshape(h2, w2, 4), "RGBA")


def expand_outlines(frame, strength=0.8):
    """Fatten thin dark lines before downscaling (PixelOE's key idea, lite).

    An integer 4-5x divide erases 1-2px dark outlines. Where a dark line
    runs nearby (local median much brighter than the local minimum), blend
    each pixel toward its darkest neighbor - lines thicken enough to win
    the dominant-color vote in kcentroid_downscale, flat areas untouched."""
    rgb = frame.convert("RGB")
    lum = rgb.convert("L")
    med = np.asarray(lum.filter(ImageFilter.MedianFilter(5)), np.float32)
    lmin = np.asarray(lum.filter(ImageFilter.MinFilter(3)), np.float32)
    w = np.clip((med - lmin) / 64.0, 0, 1) * strength
    eroded = np.asarray(rgb.filter(ImageFilter.MinFilter(3)), np.float32)
    orig = np.asarray(rgb, np.float32)
    out = (eroded * w[..., None] + orig * (1 - w[..., None])).astype(np.uint8)
    return Image.merge("RGBA", (*Image.fromarray(out).split(),
                                frame.split()[3]))


def _detect_period(small, min_period=12):
    """Cycle length of a looping clip by self-similarity: the lag P at which
    frame t looks like frame t+P for every t. (Comparing against frame 0
    alone fails for treadmill walks, which never return to the standing
    reference stance.) Returns the first strong local minimum, or None."""
    n = len(small)
    if n < 2 * min_period + 2:
        return None
    lags = list(range(min_period, n // 2 + 1))
    vals = np.array([np.mean([float(np.abs(small[t] - small[t + lag]).mean())
                              for t in range(0, n - lag, 2)]) for lag in lags])
    gmin, gmax = float(vals.min()), float(vals.max())
    if gmax <= gmin + 1e-6:
        return None
    thresh = gmin + 0.15 * (gmax - gmin)
    for i in range(1, len(vals) - 1):
        if vals[i] <= thresh and vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
            return lags[i]
    return lags[int(vals.argmin())]


def _full_gait_period(small, period, tol=1.2):
    """Left and right strides look alike, so self-similarity finds ONE
    stride; a walk loop needs both (or the same leg leads every cycle).
    Use the doubled period when frames still match at that lag."""
    n = len(small)
    if not period or 2 * period > n - 2:
        return period

    def score(lag):
        return float(np.mean([np.abs(small[t] - small[t + lag]).mean()
                              for t in range(0, n - lag, 2)]))
    return 2 * period if score(2 * period) <= score(period) * tol else period


def _select_pixel_frames(frames, k, loop, min_period=12, quiet=None,
                         gait=False):
    """Collapse a 24fps clip to k classic keyframes.

    Real pixel animation is 4-12 frames per action at ~8-12fps (NES 2-4,
    SNES/GBA 4-8, Metal Slug 8-12) - fewer, well-chosen frames both read
    as authentic and shrink the flicker surface ~8x.

    Loops: the cycle period comes from self-similarity (_detect_period);
    the cycle START is the frame whose loop seam (frame s vs s+P) is
    tightest, which skips the clip's ease-in from the reference stance.
    k frames are sampled across ONE cycle. With `quiet` (0-1), only the
    calmest `quiet` fraction of the cycle is used and the caller ping-pongs
    it - a subtle idle from an over-animated clip.
    One-shots sample evenly first-to-last. Returns (indices, period).
    """
    n = len(frames)
    if n <= k:
        return list(range(n)), None
    small = [np.asarray(f.convert("L").resize((32, 32)), dtype=np.float32)
             for f in frames]
    period = _detect_period(small, min_period) if loop else None
    if period and gait:
        period = _full_gait_period(small, period)
    if period:
        # best-closing cycle start: minimal seam error, earliest on ties
        seams = [(float(np.abs(small[s] - small[s + period]).mean()), s)
                 for s in range(0, n - period)]
        seam_min = min(v for v, _ in seams)
        start = min(s for v, s in seams if v <= seam_min * 1.25 + 1e-6)
        if quiet:
            # calmest window: least motion energy across the window
            win = max(3, int(round(period * min(1.0, max(0.1, quiet)))))
            energy = [float(np.mean([np.abs(small[t + 1] - small[t]).mean()
                                     for t in range(s, s + win - 1)]))
                      for s in range(start, min(n - win, start + period) + 1)]
            start = start + int(np.argmin(energy))
            idx = sorted({min(n - 1, int(round(v)))
                          for v in np.linspace(start, start + win - 1, k)})
        else:   # one full cycle, endpoint excluded (frame start+P ~ start)
            idx = sorted({min(n - 1, start + int(round(i * period / float(k))))
                          for i in range(k)})
    else:       # one-shot (or no cycle found): even spread, keep the ends
        idx = sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})
    return idx, period


def stabilize_pixels(down_frames, quant_frames, tol=14):
    """Temporal palette-hold: a pixel keeps its previous frame's color
    unless the UNDERLYING art actually changed there.

    Palette quantization is a hard threshold, so a cell sitting near a
    palette boundary flips color every frame from invisible video noise -
    the classic AI-pixel-art shimmer. Comparing the pre-quantization cell
    colors between frames separates noise (< tol per channel: hold last
    frame's palette color) from real motion (pass through). Holds chain
    frame to frame, so static regions lock completely still."""
    downs = [np.asarray(f).astype(np.int16) for f in down_frames]
    outs = [np.array(f) for f in quant_frames]
    for t in range(1, len(outs)):
        src_d = np.abs(downs[t][..., :3] - downs[t - 1][..., :3]).max(axis=-1)
        hold = (src_d < tol) & (downs[t][..., 3] == downs[t - 1][..., 3])
        outs[t][hold] = outs[t - 1][hold]
    return [Image.fromarray(o, "RGBA") for o in outs]


def pixelize(frame, target_h, palette_img, alpha_thresh=96, soft_alpha=False,
             int_factor=None):
    """Downscale to sprite resolution, snap colors to the shared palette,
    and quantize alpha: binary for solid sprites (pixel art wants hard
    edges), 4 retro levels for glow VFX so halos survive.

    int_factor: downscale by an exact integer divisor of the SOURCE pixel
    grid (e.g. 640->128 is 5x). Keeping the sample grid identical across
    frames is what stops edge pixels shimmering between palette colors.
    """
    if int_factor and not soft_alpha:
        target_w = max(1, frame.width // int_factor)
        target_h = max(1, frame.height // int_factor)
        small = frame.resize((target_w, target_h), Image.NEAREST)
    else:
        scale = target_h / float(frame.height)
        target_w = max(1, int(round(frame.width * scale)))
        resample = Image.LANCZOS if soft_alpha else Image.NEAREST
        small = frame.resize((target_w, target_h), resample)
    rgb = small.convert("RGB").quantize(palette=palette_img, dither=Image.NONE).convert("RGB")
    alpha = np.asarray(small)[..., 3]
    if soft_alpha:
        q = (np.round(alpha / 85.0) * 85).clip(0, 255).astype(np.uint8)
    else:
        q = np.where(alpha >= alpha_thresh, 255, 0).astype(np.uint8)
    out = np.dstack([np.asarray(rgb), q])
    return Image.fromarray(out, "RGBA")


def find_action_range(sprites, rel_thresh=0.35, pad_before=2, pad_after=3):
    """Locate the FIRST burst of motion in a clip.

    Video models fill the mandatory clip length by repeating a one-shot
    action (jump twice, strike twice...); a game needs exactly one. Motion
    energy = fraction of pixels changing per frame; the first contiguous
    above-threshold burst (1-frame gaps tolerated) is the action. Returns
    [start, end) sprite indices; loops can ignore it.
    """
    if len(sprites) < 4:
        return [0, len(sprites)]
    # Pose departure from the start stance, not motion energy: our clips are
    # bookended at the neutral stance, so the character returns near frame 0
    # between repetitions (ending the burst) but is far from it at a jump's
    # airborne apex (so a mid-action lull can't end it).
    arrs = [np.asarray(s).astype(np.int16) for s in sprites]
    base = arrs[0]
    dep = np.array([float(np.abs(a - base).sum(axis=-1).mean()) for a in arrs])
    dep[0] = 0.0
    dep = np.convolve(dep, np.ones(3) / 3.0, mode="same")
    if dep.max() <= 1e-6:
        return [0, len(sprites)]
    active = dep > dep.max() * rel_thresh
    idx = np.where(active)[0]
    if not len(idx):
        return [0, len(sprites)]
    start = int(idx[0])
    end = start
    while end + 1 < len(active) and active[end + 1:end + 3].any():
        end += 1
    return [max(0, start - pad_before), min(len(sprites), end + 1 + pad_after)]


def despeckle(sprites, min_frac=0.004):
    """Drop tiny disconnected fragments left by keying.

    A character sprite is one connected body; stray 1-3px islands are key
    noise (a speck of costume colour that survived, a highlight cut off by
    the alpha threshold) and read as dirt around the sprite. Anything under
    `min_frac` of the biggest component's area goes.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return sprites
    out = []
    for s in sprites:
        arr = np.array(s)
        solid = arr[..., 3] > 128
        lab, n = ndimage.label(solid)
        if n <= 1:
            out.append(s)
            continue
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        keep = sizes >= max(4.0, sizes.max() * min_frac)
        drop = np.isin(lab, [i + 1 for i, k in enumerate(keep) if not k])
        arr[drop] = 0
        out.append(Image.fromarray(arr, "RGBA"))
    return out


def cel_clean(frame, factor):
    """Flatten video mottling by snapping to the clip's OWN cluster grid.

    H3 paints fake-pixel clusters with soft, noisy edges; sampling each
    cluster with k-centroid and scaling straight back up re-crisps the
    clusters and flattens the blotchy shading inside flat areas, without
    forcing a palette or a smaller sprite.
    """
    if factor <= 1:
        return frame
    small = kcentroid_downscale(frame, factor)
    return small.resize((small.width * factor, small.height * factor),
                        Image.NEAREST)


def edge_bleed(sprites, passes=4):
    """Push opaque edge colors outward under transparent pixels. Engines
    sampling RGBA with bilinear filtering blend the under-color into sprite
    edges - leftover green screen there becomes a green fringe."""
    out = []
    for s in sprites:
        arr = np.array(s)
        rgb = arr[..., :3].astype(np.float32)
        alive = arr[..., 3] > 0
        for _ in range(passes):
            if alive.all():
                break
            acc = np.zeros_like(rgb)
            cnt = np.zeros(alive.shape, np.float32)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sh_a = np.roll(alive, (dy, dx), (0, 1))
                sh_rgb = np.roll(rgb, (dy, dx), (0, 1))
                if dy == 1: sh_a[0] = False
                if dy == -1: sh_a[-1] = False
                if dx == 1: sh_a[:, 0] = False
                if dx == -1: sh_a[:, -1] = False
                acc += sh_rgb * sh_a[..., None]
                cnt += sh_a
            grow = (~alive) & (cnt > 0)
            rgb[grow] = acc[grow] / cnt[grow, None]
            alive = alive | grow
        arr[..., :3] = rgb.clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr, "RGBA"))
    return out


def _auto_range(sprites, ends_at_end=False):
    """Detected action window. Clips built with an explicit END pose are
    bookended: they deliberately finish at the target pose, so the action
    must run to the last frame no matter where motion energy peaked."""
    r = find_action_range(sprites)
    if ends_at_end:
        return [r[0], len(sprites)]
    return r


def deflicker(sprites):
    """Remove single-frame pixel pops: if a pixel is identical in the previous
    and next frame but differs in this one, it is video noise, not motion -
    snap it back. Real movement persists for 2+ frames and is untouched."""
    arrs = [np.array(s) for s in sprites]
    for t in range(1, len(arrs) - 1):
        prev, cur, nxt = arrs[t - 1], arrs[t], arrs[t + 1]
        pop = (prev == nxt).all(axis=-1) & (cur != prev).any(axis=-1)
        cur[pop] = prev[pop]
    return [Image.fromarray(a, "RGBA") for a in arrs]


# --------------------------------------------------------------- deliverables
def make_sheet(frames, columns=None):
    n = len(frames)
    columns = columns or min(n, 8)
    rows = int(math.ceil(n / float(columns)))
    fw, fh = frames[0].size
    sheet = Image.new("RGBA", (fw * columns, fh * rows), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % columns) * fw, (i // columns) * fh))
    return sheet, columns, rows


def save_gif(frames, path, fps, scale=None, bg=(34, 34, 46)):
    """Upscaled (nearest) GIF preview on a dark backdrop."""
    if scale is None:
        scale = max(1, 384 // frames[0].height)
    big = []
    for f in frames:
        up = f.resize((f.width * scale, f.height * scale), Image.NEAREST)
        canvas = Image.new("RGB", up.size, bg)
        canvas.paste(up, (0, 0), up)
        big.append(canvas)
    big[0].save(path, save_all=True, append_images=big[1:],
                duration=int(round(1000.0 / fps)), loop=0)


def save_webp(frames, path, fps, scale=None):
    """Full-color animated WebP preview with alpha - GIF's 256-color palette
    ruins smooth painted gradients; WebP keeps them intact and small."""
    if scale is None:
        scale = max(1, 384 // frames[0].height)
    big = [f if scale == 1 else
           f.resize((f.width * scale, f.height * scale), Image.NEAREST)
           for f in frames]
    big[0].save(path, save_all=True, append_images=big[1:],
                duration=int(round(1000.0 / fps)), loop=0,
                quality=88, method=4)


def save_apng(frames, path, fps, scale=None):
    """Transparent APNG preview (nearest upscale)."""
    if scale is None:
        scale = max(1, 384 // frames[0].height)
    big = [f.resize((f.width * scale, f.height * scale), Image.NEAREST) for f in frames]
    big[0].save(path, save_all=True, append_images=big[1:],
                duration=int(round(1000.0 / fps)), loop=0, default_image=False)


def smooth_resize(frame, target_h, sharpen=True):
    """Painted-style path: high-quality downscale, continuous alpha, no
    palette. A light unsharp mask restores the painted linework that video
    generation + downscaling softens.

    BOX (area) filter, not Lanczos: Lanczos' negative lobes ring at the
    sprite's dark outline and paint a light rim outside it. The sharpen
    likewise only touches the opaque interior - sharpening the rim brightens
    it against the transparent side (measured: doubled the halo)."""
    scale = target_h / float(frame.height)
    target_w = max(1, int(round(frame.width * scale)))
    out = frame.resize((target_w, target_h), Image.BOX)
    if sharpen:
        rgb = np.asarray(edge_bleed([out], passes=6)[0].convert("RGB").filter(
            ImageFilter.UnsharpMask(radius=1.6, percent=80, threshold=2)))
        arr = np.array(out)
        inner = arr[..., 3] == 255
        try:
            from scipy import ndimage
            inner = ndimage.binary_erosion(inner, iterations=2)
        except ImportError:
            pass
        arr[..., :3] = np.where(inner[..., None], rgb, arr[..., :3])
        out = Image.fromarray(arr, "RGBA")
    return out


def add_outline(sprite, mode="black", thresh=128):
    """Hard silhouette plus a 1px outline just outside it.

    The soft rim, where any leftover keying fringe lives, is replaced rather
    than shown, so the edge is clean on any background. Generated characters
    already carry a dark outline, so this thickens it to ~2px - a bolder
    "sticker" look, opt-in per sprite. mode: "black", or "selout" (darkened
    neighbouring color, the classic pixel-art selective outline)."""
    try:
        from scipy import ndimage
    except ImportError:
        return sprite
    a = np.array(sprite.convert("RGBA"))
    solid = a[..., 3] >= thresh
    ring = ndimage.binary_dilation(solid) & ~solid
    out = a.copy()
    out[..., 3] = np.where(solid | ring, 255, 0).astype(np.uint8)
    if mode == "selout":
        w = solid.astype(np.float32)
        rgb = a[..., :3].astype(np.float32)
        den = np.maximum(ndimage.uniform_filter(w, 3), 1e-6)[..., None]
        near = np.stack([ndimage.uniform_filter(rgb[..., i] * w, 3)
                         for i in range(3)], axis=-1) / den
        out[ring, :3] = np.clip(near * 0.3, 0, 255)[ring].astype(np.uint8)
    else:
        out[ring, :3] = (16, 12, 20)
    return Image.fromarray(out, "RGBA")


def process_animation(frame_paths, out_dir, sprite_h=96, colors=48,
                      take_every=2, src_fps=24.0, dominance=40, pad=4,
                      trim_start=0, trim_end=0, key_mode="auto",
                      style="pixel", range_override=None, ends_at_end=False,
                      loop=False, pixel_frames=None, pixel_fps=10.0,
                      loop_mode="cycle", quiet=0.35, drop=None, gait=False,
                      clean=False, outline=None):
    """Full post pipeline for one animation. Returns a metadata dict.

    take_every: subsample the 24fps video (2 -> 12fps sprite animation).
    trim_start/trim_end: drop leading/trailing frames (H3 clips sometimes
    ease in/out of the reference pose).
    pixel_frames: collapse the clip to this many classic keyframes played
    at pixel_fps (loops sample one detected cycle); 0/None = no collapse.
    loop_mode: "cycle" plays one full cycle; "pingpong" uses only the
    calmest `quiet` fraction of the cycle and plays it forward-then-back -
    a subtle idle from an over-animated clip.
    drop: sprite indices (after collapse) to cut from the final sequence.
    outline: None, "black" or "selout" - see add_outline.
    """
    frame_paths = sorted(frame_paths)
    if trim_end:
        frame_paths = frame_paths[trim_start:-trim_end]
    elif trim_start:
        frame_paths = frame_paths[trim_start:]
    collapse = key_mode != "glow" and bool(pixel_frames)
    # collapse picks its own keyframes from the full 24fps set
    frame_paths = frame_paths[::1 if collapse else take_every]
    if not frame_paths:
        raise ValueError("no frames to process")

    keyed = [key_frame(Image.open(p), key_mode=key_mode, dominance=dominance)
             for p in frame_paths]
    # glow sprites: crop to the bright core, not the faint halo field, or the
    # box balloons to the whole frame and kills sprite resolution
    box = union_bbox(keyed, pad=(8 if key_mode == "glow" else pad),
                     alpha_thresh=(48 if key_mode == "glow" else 8))
    factor = max(1, int(round(keyed[0].height / float(sprite_h))))
    art_grid = None
    loop_period = None
    if style != "painted" and key_mode != "glow":
        box = snap_box_to_grid(box, factor, keyed[0].size)
    cropped = [f.crop(box) for f in keyed]
    if collapse:
        # classic hand-animation timing: a few held keyframes instead of
        # smooth 24fps-derived motion - works for pixel AND painted sprites
        pingpong = loop and loop_mode == "pingpong"
        idx, loop_period = _select_pixel_frames(
            cropped, int(pixel_frames), loop,
            quiet=(quiet if pingpong else None), gait=gait)
        if pingpong and len(idx) > 2:
            idx = idx + idx[-2:0:-1]        # forward, then back: seamless
        cropped = [cropped[i] for i in idx]
        print("[post] collapsed %d source frames -> %d keyframes%s%s @ %g fps"
              % (len(frame_paths), len(cropped),
                 (" (cycle of %d)" % loop_period) if loop_period else "",
                 " ping-pong" if pingpong else "", pixel_fps))
    # NOTE: cutting is applied AFTER the sprites are built (further down), not
    # here - the editor needs every collapsed frame on disk so cut frames stay
    # visible, keep stable indices, and can be restored.
    all_count = len(cropped)
    if style == "painted":
        target_h = max(sprite_h, 384)  # 640->384 keeps painted linework sharp
        if clean:
            # snap to the video's own cluster grid first: flattens the
            # blotchy shading H3 leaves inside flat areas (a white coat is
            # the worst case) instead of letting unsharp amplify it
            g = estimate_pixel_grid(cropped[0])
            f = max(2, min(6, g))
            cropped = [cel_clean(c, f) for c in cropped]
        sprites = [smooth_resize(f, min(target_h, f.height), sharpen=not clean)
                   for f in cropped]
        sprites = edge_bleed(sprites)
    elif key_mode == "glow":
        palette = build_shared_palette(cropped, colors=colors)
        sprites = [pixelize(f, sprite_h, palette, soft_alpha=True)
                   for f in cropped]
        sprites = deflicker(sprites)
    else:
        # pixel-perfect path: k-centroid cells on a phase-locked grid,
        # palette built AFTER downscaling (so it reflects the actual output
        # colors, not the high-res gradients), then temporal palette-hold
        art_grid = estimate_pixel_grid(cropped[0])
        if art_grid and abs(art_grid - factor) > max(2, factor * 0.5):
            print("[post] note: art's own pixel clusters ~%dpx vs downscale "
                  "factor %d - if the output looks off-grid, retune sprite_h"
                  % (art_grid, factor))
        cropped = [expand_outlines(f) for f in cropped]
        downs = [kcentroid_downscale(f, factor) for f in cropped]
        palette = build_shared_palette(downs, colors=colors,
                                       method=Image.MEDIANCUT, kmeans=colors)
        quant = []
        for f in downs:
            rgb = f.convert("RGB").quantize(palette=palette,
                                            dither=Image.NONE).convert("RGB")
            quant.append(Image.merge("RGBA", (*rgb.split(), f.split()[3])))
        sprites = stabilize_pixels(downs, quant)
        sprites = deflicker(sprites)

    # keying leaves stray single pixels around any character sprite
    if key_mode != "glow":
        sprites = despeckle(sprites)
        if outline:
            sprites = [add_outline(s, outline) for s in sprites]

    os.makedirs(out_dir, exist_ok=True)
    # Every collapsed frame is written to all_frames/ using its ORIGINAL
    # index. The editor's strip reads these, so cut frames stay visible and
    # restorable and their numbering never shifts as cuts are made.
    adir = os.path.join(out_dir, "all_frames")
    if os.path.isdir(adir):
        shutil.rmtree(adir, ignore_errors=True)
    os.makedirs(adir, exist_ok=True)
    for i, s in enumerate(sprites):
        s.save(os.path.join(adir, "sprite_%03d.png" % i))

    # now apply the cuts, and shift the playback window with them so it keeps
    # covering the same poses instead of sliding onto different ones
    cut = sorted({int(x) for x in drop}) if drop else []
    if cut:
        keep = [i for i in range(len(sprites)) if i not in set(cut)]
        if len(keep) >= 2:
            if range_override:
                s0, e0 = int(range_override[0]), int(range_override[1])
                range_override = [sum(1 for i in keep if i < s0),
                                  sum(1 for i in keep if i < e0)]
            sprites = [sprites[i] for i in keep]
            print("[post] cut %d frame(s) %s -> %d frames%s"
                  % (len(cut), cut, len(sprites),
                     (", window -> %s" % range_override) if range_override else ""))
        else:
            cut = []

    fdir = os.path.join(out_dir, "frames")
    if os.path.isdir(fdir):
        for old in glob.glob(os.path.join(fdir, "sprite_*.png")):
            os.remove(old)
    os.makedirs(fdir, exist_ok=True)
    for i, s in enumerate(sprites):
        s.save(os.path.join(fdir, "sprite_%03d.png" % i))

    sheet, cols, rows = make_sheet(sprites)
    sheet.save(os.path.join(out_dir, "sheet.png"))
    fps = pixel_fps if collapse else src_fps / take_every
    save_gif(sprites, os.path.join(out_dir, "preview.gif"), fps)
    save_apng(sprites, os.path.join(out_dir, "preview_transparent.png"), fps)
    if style == "painted":
        save_webp(sprites, os.path.join(out_dir, "preview.webp"), fps)

    # TexturePacker-style atlas: imports directly into Godot / Unity /
    # Phaser. Pivot at bottom-center = the feet anchor the harness uses.
    name = os.path.basename(os.path.normpath(out_dir))
    fw, fh = sprites[0].size
    frames_dict, order = {}, []
    for i in range(len(sprites)):
        key = "%s_%d" % (name, i)
        frames_dict[key] = {
            "frame": {"x": (i % cols) * fw, "y": (i // cols) * fh,
                      "w": fw, "h": fh},
            "rotated": False, "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": fw, "h": fh},
            "sourceSize": {"w": fw, "h": fh},
            "pivot": {"x": 0.5, "y": 1.0},
        }
        order.append(key)
    atlas = {"frames": frames_dict, "animations": {name: order},
             "meta": {"app": "spirtes sprite_pipeline", "image": "sheet.png",
                      "format": "RGBA8888", "scale": "1",
                      "size": {"w": sheet.width, "h": sheet.height},
                      "framerate": fps}}
    with open(os.path.join(out_dir, "atlas.json"), "w") as fh_:
        json.dump(atlas, fh_, indent=1)

    # ground line = lowest opaque row of frame 0, the reference stance every
    # clip starts from. The union crop box's bottom can be a cape hem or a
    # swinging shield tip in some later frame, which would float the whole
    # animation above the ground in the harness; frame 0's feet never lie.
    a0 = np.asarray(keyed[0])[..., 3]
    opaque_rows = np.where((a0 > 8).any(axis=1))[0]
    ground_y = (int(opaque_rows.max()) + 1 if len(opaque_rows)
                else int(box[3]))
    # How tall the character is in the clip's FIRST frame, in source pixels.
    # Every clip starts from the reference stance, so this is comparable
    # across animations and lets the harness normalize a clip that the video
    # model happened to render larger or smaller than the rest. Measured
    # before any keyframe collapse, so a crouch or a collapse can't skew it.
    stand_h = (int(opaque_rows.max() - opaque_rows.min()) + 1
               if len(opaque_rows) else None)
    meta = {"frames": len(sprites), "fps": fps,
            "ground_y": ground_y, "stand_h": stand_h,
            "frame_size": list(sprites[0].size),
            "sheet_grid": [cols, rows], "colors": colors, "key_mode": key_mode,
            "style": style,
            "action_range": (list(range_override)[:2] if range_override
                             else _auto_range(sprites, ends_at_end)),
            "range_manual": bool(range_override),
            "source_frames": len(frame_paths), "crop_box": list(box),
            "source_size": [keyed[0].width, keyed[0].height],
            "pixel_factor": (factor if style != "painted"
                             and key_mode != "glow" else None),
            "art_grid": art_grid, "loop_period": loop_period,
            "collapsed": bool(collapse), "loop": bool(loop),
            "loop_mode": loop_mode if (collapse and loop) else None,
            "quiet": quiet if (collapse and loop and loop_mode == "pingpong") else None,
            "dropped": cut, "all_frames": all_count}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def pixel_conform_reference(path, out_path, sprite_h=128, colors=48):
    """Snap a reference image onto a true pixel grid before animation.

    The video model copies the reference's style, so pixel-perfection is
    enforced HERE (and re-imposed in post), not demanded of the animation:
    a hi-res image that only implies pixel art has soft, inconsistent
    cluster sizes; k-centroid down to the project's real sprite density,
    palette-lock, then NEAREST back up hands the animator genuine pixel art
    with exactly correct, consistent pixel-square sizes."""
    src = Image.open(path).convert("RGB")
    factor = max(1, int(round(src.height / float(sprite_h))))
    if factor <= 1:
        Image.open(path).save(out_path)
        return out_path
    src = src.crop((0, 0, (src.width // factor) * factor,
                    (src.height // factor) * factor))
    # key the backdrop out FIRST: the padded ref is mostly green, and a
    # palette fit over the whole image hands most of a small color budget
    # to greens - the character comes out washed gray (the bug behind
    # "loss of color" in rendered clips: H3 copies the washed ref)
    arr = np.asarray(src)
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    keyed = chroma_key_green(src)
    down = kcentroid_downscale(keyed, factor)
    # NO palette quantization here: the conform's job is grid + style for
    # the video model; palette discipline happens in post. Any small-palette
    # lock at this stage tints the ref (measured: MAXCOVERAGE turned skin
    # greenish) and the video faithfully copies the tint.
    out = np.asarray(down)[..., :3].copy()
    alpha = np.asarray(down)[..., 3]
    # backdrop returns as FLAT green - an even cleaner chroma stage for H3
    out[alpha == 0] = bg
    up = Image.fromarray(out, "RGB").resize(
        (down.width * factor, down.height * factor), Image.NEAREST)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    up.save(out_path)
    return out_path


def pad_reference(path, out_path, margin=0.18):
    """Shrink the reference inside its own canvas, adding backdrop margin.

    H3 copies the reference's framing, so a reference whose character fills
    the frame produces clips where capes, effects and airborne poses get cut
    off at the edges. Scaling the whole image down onto a same-size backdrop
    canvas buys that headroom without changing composition or identity.
    """
    src = Image.open(path).convert("RGB")
    w, h = src.size
    arr = np.asarray(src)
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    f = max(0.3, 1.0 - 2 * margin)
    small = src.resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    # centered horizontally; biased low so feet keep a small ground margin
    canvas.paste(small, ((w - small.width) // 2,
                         int((h - small.height) * 0.62)))
    # flatten the backdrop: the shrunk image's own green is never perfectly
    # uniform, and H3 copies that faint rectangle into every frame; snapping
    # near-backdrop pixels to the exact border color hands it one flat field
    arr = np.asarray(canvas).astype(np.int32)      # int16 squares overflow
    near = np.sqrt(((arr - np.array(bg, np.int32)) ** 2).sum(axis=-1)) < 40
    arr[near] = np.array(bg, np.int32)
    canvas = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)
    return out_path


def process_ref_image(path, out_path, sprite_h=192, colors=48, dominance=40,
                      style="pixel", outline=None):
    """Key (+ pixelize, for pixel style) a single still reference."""
    keyed = chroma_key_green(Image.open(path), dominance=dominance)
    box = union_bbox([keyed])
    if style == "painted":
        sprite = smooth_resize(keyed.crop(box),
                               min(max(sprite_h, 384), box[3] - box[1]))
    else:
        factor = max(1, int(round(keyed.height / float(sprite_h))))
        box = snap_box_to_grid(box, factor, keyed.size)
        down = kcentroid_downscale(keyed.crop(box), factor)
        palette = build_shared_palette([down], colors=colors,
                                       method=Image.MEDIANCUT, kmeans=colors)
        rgb = down.convert("RGB").quantize(palette=palette,
                                           dither=Image.NONE).convert("RGB")
        sprite = Image.merge("RGBA", (*rgb.split(), down.split()[3]))
    if outline:
        sprite = add_outline(sprite, outline)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sprite.save(out_path)
    return sprite
