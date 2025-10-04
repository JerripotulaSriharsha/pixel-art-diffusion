from PIL import Image
import numpy as np


def _smoothstep(x, a, b):
    t = np.clip((x - a) / max(1e-6, (b - a)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _count_neighbors(mask):
    m = mask.astype(np.uint8)
    p = np.pad(m, ((1, 1), (1, 1)), constant_values=0)
    return (
        p[:-2, 1:-1]
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[:-2, :-2]
        + p[:-2, 2:]
        + p[2:, :-2]
        + p[2:, 2:]
    )


def _despill_vec(rgb01, alpha01, bg_dir=(1.0, 0.0, 1.0), strength=1.2):
    # remove color in the direction of the background (magenta) near transparency
    B = np.array(bg_dir, dtype=np.float32).reshape(1, 1, 3)
    B /= max(1e-6, np.linalg.norm(B))
    proj = (np.sum(rgb01 * B, axis=-1, keepdims=True)) * B
    k = strength * (1.0 - alpha01[..., None])
    return np.clip(rgb01 - k * proj, 0.0, 1.0)


def remove_magenta_bleed(
    img: Image.Image,
    bg_color=(255, 0, 255),
    key_lo=35,
    key_hi=165,  # widen to grab more magenta
    hard_threshold=0.40,  # used only for morphology, alpha stays soft
    open_iters=1,
    close_iters=1,
    despill_strength=1.2,
    green_spill_strength=0.5,
) -> Image.Image:
    im = img.convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    a_in = arr[..., 3] / 255.0

    # 1) soft alpha from distance to magenta (no hard thresholding)
    diff = (
        rgb.astype(np.int16) - np.array(bg_color, np.int16).reshape(1, 1, 3)
    ).astype(np.int32)
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)  # 0..~441
    a_soft = _smoothstep(dist, key_lo, key_hi)
    alpha = np.minimum(a_in, a_soft)

    # 2) tiny morphology on a *hard* proxy mask to kill specks/holes
    hard = alpha > hard_threshold
    for _ in range(open_iters):
        hard = _count_neighbors(hard) >= 4  # erode
        hard = _count_neighbors(hard) > 0  # dilate
    for _ in range(close_iters):
        hard = _count_neighbors(hard) > 0  # dilate
        hard = _count_neighbors(hard) >= 4  # erode
    alpha *= hard.astype(np.float32)  # keep alpha soft

    # 3) de-matte: C = a*F + (1-a)*B  => F = (C - (1-a)B)/a
    rgb01 = rgb / 255.0
    bg01 = np.array(bg_color, np.float32).reshape(1, 1, 3) / 255.0
    a = np.clip(alpha[..., None], 0.0, 1.0)
    F = (rgb01 - (1.0 - a) * bg01) / np.maximum(a, 1e-5)
    F = np.clip(F, 0.0, 1.0)

    # 4) magenta despill near edges
    F = _despill_vec(F, alpha, bg_dir=(1.0, 0.0, 1.0), strength=despill_strength)

    # 5) optional green-spill suppression (common opposite magenta)
    if green_spill_strength > 0:
        r, g, b = F[..., 0], F[..., 1], F[..., 2]
        green_dom = (g > r * 1.15) & (g > b * 1.15)
        k = green_spill_strength * (1.0 - alpha)  # stronger where alpha is thin
        target_g = (r + b) / 2.0
        g = np.where(green_dom, g * (1 - k) + target_g * k, g)
        F = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)

    out = np.empty_like(arr)
    out[..., :3] = (F * 255.0 + 0.5).astype(np.uint8)
    out[..., 3] = (alpha * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


if __name__ == "__main__":
    img = Image.open("output_postprocessing/pixelized.png_8_upscaled.png")
    cleaned = remove_magenta_bleed(
        img,
        bg_color=(255, 0, 255),
        key_lo=45,
        key_hi=200,
        open_iters=1,
        close_iters=1,
        green_spill_strength=1,
    )
    cleaned.save("output_postprocessing/clean.png_8_upscaled.png")
