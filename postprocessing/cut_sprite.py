import math
from pathlib import Path
from collections import Counter, deque

import numpy as np
from PIL import Image


# ---------- helpers ----------


def detect_background_color(arr: np.ndarray):
    """Pick the most likely background color using corners + sparse grid sample."""
    H, W, _ = arr.shape
    samples = []

    # corners & mid-edges
    coords = [
        (0, 0),
        (W - 1, 0),
        (0, H - 1),
        (W - 1, H - 1),
        (W // 2, 0),
        (0, H // 2),
        (W - 1, H // 2),
        (W // 2, H - 1),
    ]
    for x, y in coords:
        samples.append(tuple(int(v) for v in arr[y, x]))

    # sparse grid (20x20 approx)
    step_y = max(1, H // 20)
    step_x = max(1, W // 20)
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            samples.append(tuple(int(v) for v in arr[y, x]))

    return Counter(samples).most_common(1)[0][0]  # (r,g,b)


def color_distance_map(arr: np.ndarray, bg: tuple):
    """Euclidean distance in RGB to background color."""
    bg_vec = np.array(bg, dtype=np.float32)[None, None, :]
    arrf = arr.astype(np.float32)
    diff = arrf - bg_vec
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist


def crop_with_margin(box, W, H, margin):
    xmin, ymin, xmax, ymax = box
    return (
        max(0, xmin - margin),
        max(0, ymin - margin),
        min(W - 1, xmax + margin),
        min(H - 1, ymax + margin),
    )


def largest_component_mask(mask: np.ndarray):
    """
    Return (comp_mask, bbox, count) for the largest 4-neighborhood connected component in `mask`.
    comp_mask has the same shape as mask (bool).
    bbox = (xmin,ymin,xmax,ymax)
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords = None
    best_count = 0
    best_bbox = None

    for y in range(H):
        if not mask[y].any():
            continue
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                q = deque([(x, y)])
                visited[y, x] = True
                xs, ys = [], []
                count = 0
                while q:
                    cx, cy = q.popleft()
                    xs.append(cx)
                    ys.append(cy)
                    count += 1
                    # neighbors (4-neighborhood)
                    if cx + 1 < W and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                        visited[cy, cx + 1] = True
                        q.append((cx + 1, cy))
                    if cx - 1 >= 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                        visited[cy, cx - 1] = True
                        q.append((cx - 1, cy))
                    if cy + 1 < H and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                        visited[cy + 1, cx] = True
                        q.append((cx, cy + 1))
                    if cy - 1 >= 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                        visited[cy - 1, cx] = True
                        q.append((cx, cy - 1))
                if count > best_count:
                    best_count = count
                    best_coords = (xs, ys)
                    best_bbox = (min(xs), min(ys), max(xs), max(ys))

    if best_coords is None:
        return None, None, 0

    xs, ys = best_coords
    comp_mask = np.zeros_like(mask, dtype=bool)
    comp_mask[np.array(ys), np.array(xs)] = True
    return comp_mask, best_bbox, best_count


def make_alpha_from_distance(dist: np.ndarray, tol: float, feather: float):
    """
    Turn color distance into alpha. Pixels near the bg color get alpha 0,
    pixels clearly different get alpha 255. `feather` softens the edge in distance units.
    """
    if feather <= 0:
        return (dist > tol).astype(np.uint8) * 255
    # ramp from (tol - feather) -> 0 up to tol -> 255
    a = (dist - (tol - feather)) / max(1e-6, feather)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def erode_mask(mask: np.ndarray, iterations: int = 1):
    """
    Simple binary erosion (shrink foreground by N pixels).
    Uses 4-neighborhood (cross kernel).
    """
    result = mask.copy()
    for _ in range(iterations):
        H, W = result.shape
        eroded = np.zeros_like(result, dtype=bool)
        for y in range(H):
            for x in range(W):
                if result[y, x]:
                    # Check 4-neighbors
                    has_all_neighbors = True
                    if x == 0 or not result[y, x - 1]:
                        has_all_neighbors = False
                    if x == W - 1 or not result[y, x + 1]:
                        has_all_neighbors = False
                    if y == 0 or not result[y - 1, x]:
                        has_all_neighbors = False
                    if y == H - 1 or not result[y + 1, x]:
                        has_all_neighbors = False
                    if has_all_neighbors:
                        eroded[y, x] = True
        result = eroded
    return result


# ---------- main extractor ----------


def extract_single_sprite_transparent(
    png_path,
    out_path=None,
    *,
    bg_color=None,  # specific background color (r,g,b) or None to auto-detect
    tol=None,  # color distance tolerance vs. background (auto if None)
    min_size=10,  # min bbox width/height in pixels
    min_area=50,  # min bbox area in pixels
    margin=0,  # extra pixels around bbox in the output crop
    feather=6.0,  # softness of alpha edge (in color-distance units)
    expand_mask=0,  # erode the mask by N pixels to remove edge artifacts
):
    """
    Extract the most prominent sprite and save it as a transparent PNG.

    Args:
        png_path (str|Path): input image path.
        out_path (str|Path|None): where to save. If None, not saved (just returned).
        bg_color (tuple|None): Known background color (r,g,b) like (255,0,255) for magenta.
                              If None, auto-detect from corners/edges.
        tol (float|None): background-vs-foreground color distance threshold.
                          Default: ~5% of max RGB distance, but at least 25.
                          For known bg colors like magenta, consider 40-60 for pixel art.
        min_size (int): reject tiny components by width/height.
        min_area (int): reject tiny components by area.
        margin (int): expand the crop box outward by this many pixels (clamped to image).
        feather (float): softness of the alpha edge in color-distance units.
                        Set to 0 for hard edges (good for pixel art).
        expand_mask (int): shrink the foreground mask inward by this many pixels
                          to remove edge artifacts. Useful for pixel art with compression.

    Returns:
        PIL.Image.Image (RGBA), info dict with keys: bbox, bg, saved_to
    """
    png_path = Path(png_path)
    im = Image.open(png_path).convert("RGB")
    W, H = im.size
    rgb = np.array(im)

    # 1) Background color + distance map
    if bg_color is not None:
        bg = tuple(bg_color)
    else:
        bg = detect_background_color(rgb)
    dist = color_distance_map(rgb, bg)

    if tol is None:
        # For known bg colors (especially magenta), use higher tolerance
        if bg_color is not None:
            tol = 50.0  # more aggressive for known backgrounds
        else:
            tol = float(
                max(25, 0.05 * math.sqrt(255**2 * 3))
            )  # ≈5% of max RGB distance

    # 2) Foreground mask
    fg_mask = dist > tol

    # 3) Largest connected component (the sprite)
    comp_mask_full, bbox, count = largest_component_mask(fg_mask)
    if comp_mask_full is None:
        raise ValueError(
            "No sprite-like component found. Try lowering `tol` or thresholds."
        )

    # 4) Crop with margin
    xmin, ymin, xmax, ymax = bbox
    xmin2, ymin2, xmax2, ymax2 = crop_with_margin(
        (xmin, ymin, xmax, ymax), W, H, margin
    )

    # slice everything to the crop
    rgb_crop = rgb[ymin2 : ymax2 + 1, xmin2 : xmax2 + 1]
    dist_crop = dist[ymin2 : ymax2 + 1, xmin2 : xmax2 + 1]
    mask_crop = fg_mask[ymin2 : ymax2 + 1, xmin2 : xmax2 + 1]

    # 5) Within the crop, re-isolate the largest component to be safe
    comp_mask_crop, _, _ = largest_component_mask(mask_crop)
    if comp_mask_crop is None:
        # fallback: use the original fg mask within crop
        comp_mask_crop = mask_crop

    # 5b) Optionally erode the mask to remove edge artifacts
    if expand_mask > 0:
        comp_mask_crop = erode_mask(comp_mask_crop, iterations=expand_mask)

    # 6) Build alpha: feathered by distance, but only keep the chosen component
    alpha_soft = make_alpha_from_distance(dist_crop, tol=tol, feather=feather)
    alpha = alpha_soft * comp_mask_crop.astype(np.uint8)

    # 7) Compose RGBA with transparent background
    rgba = np.dstack([rgb_crop, alpha]).astype(np.uint8)
    out_img = Image.fromarray(rgba, mode="RGBA")

    saved_to = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path)
        saved_to = out_path

    return out_img, {
        "bbox": (xmin2, ymin2, xmax2, ymax2),
        "bg": bg,
        "saved_to": saved_to,
    }


# ---------- quick demo ----------
if __name__ == "__main__":
    # Example usage:
    #   python postprocessing/cut_sprite.py
    # Adjust paths and params below or wrap in argparse if you like.
    sprite, info = extract_single_sprite_transparent(
        "output_test/image-2.png",  # input
        out_path="output_test/clean.png",  # transparent PNG output
        # bg_color=(255, 0, 255),  # magenta background
        tol=50,  # higher tolerance to catch edge artifacts
        margin=5,  # small breathing room
        feather=0,  # hard edges for pixel art (no feathering)
        expand_mask=3,  # shrink mask by 1px to remove edge artifacts
    )
    print("Saved to:", info["saved_to"])
    print("Background color:", info["bg"])
    print("Cropped bbox:", info["bbox"])
