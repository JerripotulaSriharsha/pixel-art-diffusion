import math
from pathlib import Path
from collections import Counter, deque

from PIL import Image
import numpy as np

"""
Script to cut sprites from spritesheets. Detects background color and also rescales the sprites to 512x512 and 1024x1024.
"""


def detect_background_color(arr):
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


def color_distance_map(arr, bg):
    """Euclidean distance in RGB to background color."""
    bg_vec = np.array(bg, dtype=np.float32)[None, None, :]
    arrf = arr.astype(np.float32)
    diff = arrf - bg_vec
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist


def replace_background_color(
    img: Image.Image, original_bg: tuple, new_bg: tuple, tolerance: float
):
    """Replace background color in PIL Image with new color based on tolerance."""
    arr = np.array(img)

    # Calculate distance from original background color
    dist = color_distance_map(arr, original_bg)

    # Create mask for pixels that should be replaced (close to original background)
    bg_mask = dist <= tolerance

    # Replace background pixels with new color
    result_arr = arr.copy()
    result_arr[bg_mask] = new_bg

    return Image.fromarray(result_arr)


def connected_components(mask):
    """4-neighborhood connected components, returns list of (xmin,ymin,xmax,ymax,pixel_count)."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
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
                    # neighbors
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
                comps.append((min(xs), min(ys), max(xs), max(ys), count))
    return comps


def crop_with_margin(box, W, H, margin):
    xmin, ymin, xmax, ymax = box
    return (
        max(0, xmin - margin),
        max(0, ymin - margin),
        min(W - 1, xmax + margin),
        min(H - 1, ymax + margin),
    )


def save_with_padding_constant_scale(
    img: Image.Image, path: Path, size: int, bg_color: tuple, pad_px_final: int = 0
):
    """
    Save to size×size without changing the sprite's scale compared to a no-padding export.
    The sprite is scaled based on the *original crop* area, then centered on a canvas
    filled with `bg_color`. `pad_px_final` (in output pixels) reserves a uniform border.
    """
    # original crop size (no extra padding here)
    W, H = img.size

    # area available for the sprite after reserving final border
    avail = max(1, size - 2 * pad_px_final)

    # scale sprite by the "fit longest side" rule
    k = avail / max(W, H)
    new_w = max(1, int(round(W * k)))
    new_h = max(1, int(round(H * k)))

    # resize sprite with NEAREST (pixel art friendly)
    sprite = img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

    # make final canvas and paste centered
    canvas = Image.new("RGB", (size, size), color=bg_color)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas.paste(sprite, (ox, oy))

    canvas.save(path)


def process_image(path: Path, out_base: Path, args):
    out_raw = out_base / "raw"
    out_512 = out_base / "dataset-512"
    out_1024 = out_base / "dataset-1024"
    for d in (out_raw, out_512, out_1024):
        d.mkdir(parents=True, exist_ok=True)

    im = Image.open(path).convert("RGB")
    W, H = im.size
    arr = np.array(im)

    # Background & mask
    bg = detect_background_color(arr)
    dist = color_distance_map(arr, bg)
    tol = float(
        args.tol if args.tol is not None else max(25, 0.05 * math.sqrt(255**2 * 3))
    )
    mask = dist > tol

    # Optional pre-slice grid to avoid merging neighbors (useful if sprites are very close)
    boxes = []
    if args.grid and args.grid > 0:
        cell = int(args.grid)
        for y0 in range(0, H, cell):
            for x0 in range(0, W, cell):
                sub = mask[y0 : min(y0 + cell, H), x0 : min(x0 + cell, W)]
                comps = connected_components(sub)
                for xmin, ymin, xmax, ymax, cnt in comps:
                    boxes.append((x0 + xmin, y0 + ymin, x0 + xmax, y0 + ymax, cnt))
    else:
        boxes = connected_components(mask)

    # Filter components
    filtered = []
    for xmin, ymin, xmax, ymax, cnt in boxes:
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        if w < args.min_size or h < args.min_size:
            continue
        if area < args.min_area:
            continue
        # drop very wide + short bands in bottom credits
        if (ymin > int((100 - args.skip_bottom_pct) / 100.0 * H)) and (w > 0.5 * W):
            continue
        # also drop near-full-width short strips
        if (w > 0.75 * W) and (h < 0.12 * H):
            continue
        filtered.append((xmin, ymin, xmax, ymax, cnt))

    # Sort by reading order
    filtered.sort(key=lambda b: (b[1], b[0]))

    saved = 0
    for i, (xmin, ymin, xmax, ymax, _) in enumerate(filtered):
        bx = crop_with_margin((xmin, ymin, xmax, ymax), W, H, args.margin)
        crop = im.crop((bx[0], bx[1], bx[2] + 1, bx[3] + 1))

        # sanity check: ensure enough foreground inside crop
        sub = dist[bx[1] : bx[3] + 1, bx[0] : bx[2] + 1] > tol
        if sub.mean() < 0.10:  # <10% non-background → likely noise
            continue

        # Replace background color if specified
        if args.replace_bg:
            crop = replace_background_color(crop, bg, args.new_bg_color, tol)

        fn = f"{path.stem}_sprite_{i:04d}.png"

        # save raw crop (no resizing)
        crop.save(out_raw / fn)

        # choose canvas bg color: use new_bg if we replaced, else original bg
        canvas_bg = args.new_bg_color if args.replace_bg else bg

        # Save constant-scale padded versions
        save_with_padding_constant_scale(
            crop,
            out_512 / fn,
            512,
            canvas_bg,
            getattr(args, "pad_px_512", 8),
        )
        save_with_padding_constant_scale(
            crop,
            out_1024 / fn,
            1024,
            canvas_bg,
            getattr(args, "pad_px_1024", 16),
        )

        # caption stubs
        (out_512 / fn.replace(".png", ".txt")).write_text(
            "pixel art monster, retro SNES RPG style\n"
        )
        (out_1024 / fn.replace(".png", ".txt")).write_text(
            "pixel art monster, retro SNES RPG style\n"
        )
        saved += 1

    return saved, len(filtered)


def main():
    # Hardcoded arguments - modify these values as needed
    class Args:
        def __init__(self):
            self.input = "data/raw_spritesheets/"  # Input file or folder path
            self.out = "data/test"  # Output directory
            self.tol = 10.0  # Color distance tolerance vs. background
            self.min_size = 10  # Minimum bbox width/height
            self.min_area = 50  # Minimum bbox area
            self.margin = 0  # Margin (pixels) added around each bbox before cropping
            self.skip_bottom_pct = 8.0  # Ignore very wide comps in bottom X percent
            self.grid = 0  # Optional grid cell size to pre-slice (0 = off)
            self.exts = ".png,.jpg,.jpeg"  # Comma-separated list of image extensions

            # Background replacement settings
            self.replace_bg = True
            self.new_bg_color = (255, 0, 255)  # New background color (R, G, B)

            # Final export padding (uniform border on the *output canvas*, in output pixels)
            self.pad_px_512 = 8
            self.pad_px_1024 = 16

    args = Args()

    inp = Path(args.input)
    out_base = Path(args.out)

    # check if the output directory exists
    if out_base.exists():
        print(f"Error: {out_base} already exists")
        raise ValueError(f"Output directory {out_base} already exists")

    if inp.is_dir():
        img_exts = tuple(e.strip().lower() for e in args.exts.split(","))
        paths = [p for p in inp.rglob("*") if p.suffix.lower() in img_exts]
        total_saved = 0
        for i, path in enumerate(sorted(paths)):
            saved, kept = process_image(path, out_base, args)
            print(
                f"[{i+1}/{len(paths)}] {path.name}: saved {saved} crops (kept {kept} boxes)"
            )
            total_saved += saved
        print(f"Done. Total crops saved: {total_saved}")
    else:
        saved, kept = process_image(inp, out_base, args)
        print(f"{inp.name}: saved {saved} crops (kept {kept} boxes)")
        print(f"Output: {out_base}")


if __name__ == "__main__":
    main()
