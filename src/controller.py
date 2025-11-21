#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

"""Plan a smooth droplet path to reach a high-density colony region (Bezier)."""

def cubic_bezier(p0, p1, p2, p3, T=140):
    """
    Generate T points along a cubic Bezier curve with control points p0..p3 (2D).
    Uses Bernstein basis: B(t) = (1-t)^3 p0 + 3(1-t)^2 t p1 + 3(1-t) t^2 p2 + t^3 p3
    """
    t = np.linspace(0.0, 1.0, T)

    # Bernstein polynomials (scalar vectors of length T)
    b0 = (1.0 - t)**3
    b1 = 3.0 * (1.0 - t)**2 * t
    b2 = 3.0 * (1.0 - t) * t**2
    b3 = t**3

    # Broadcast to (T,2) and sum
    B = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2 + b3[:, None] * p3
    return B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True,
                    help='Path to RGB colony image (the corresponding _depth image is inferred).')
    ap.add_argument('--target_density', type=float, required=True,
                    help='Target density threshold (0..1) to reach with the droplet.')
    ap.add_argument('--out', type=str, default='results/ctrl',
                    help='Output prefix for overlay and metrics images.')
    args = ap.parse_args()

    # Load image and depth
    img = Image.open(args.image).convert('RGB')
    W, H = img.size
    depth_path = args.image.replace('_image', '_depth')
    depth = np.array(Image.open(depth_path).convert('L')).astype(np.float32) / 255.0

    # Find a target pixel with density >= target (fallback to global max)
    mask = depth >= args.target_density
    if mask.sum() == 0:
        peak_idx = np.unravel_index(np.argmax(depth), depth.shape)
    else:
        ys, xs = np.nonzero(mask)
        idx = np.argmax(depth[ys, xs])
        peak_idx = (ys[idx], xs[idx])

    # Start and goal (image coordinates)
    start = np.array([W * 0.10, H * 0.90], dtype=float)
    goal  = np.array([float(peak_idx[1]), float(peak_idx[0])], dtype=float)

    # Control points to create a gentle path
    p0 = start
    p3 = goal
    p1 = p0 + 0.5  * (p3 - p0) + np.array([ 20.0, -40.0])
    p2 = p0 + 0.85 * (p3 - p0) + np.array([-10.0, -20.0])

    # Generate path
    path = cubic_bezier(p0, p1, p2, p3, T=140)

    # Sample density along the path
    dens = []
    for x, y in path:
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))
        dens.append(depth[yi, xi])
    dens = np.array(dens)

    # Overlay path on image
    overlay = img.copy()
    dr = ImageDraw.Draw(overlay)
    for x, y in path.astype(int):
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))
    # start & goal markers
    dr.ellipse([int(p0[0]) - 4, int(p0[1]) - 4, int(p0[0]) + 4, int(p0[1]) + 4], fill=(255, 0, 0))
    dr.ellipse([int(p3[0]) - 5, int(p3[1]) - 5, int(p3[0]) + 5, int(p3[1]) + 5], fill=(255, 255, 0))
    overlay.save(f"{args.out}_overlay.png")

    # Plot density along path
    plt.figure(figsize=(5, 3))
    plt.plot(dens, label='density along path')
    plt.axhline(args.target_density, color='r', ls='--', label='target')
    plt.legend()
    plt.xlabel('path sample')
    plt.tight_layout()
    plt.savefig(f"{args.out}_metrics.png", dpi=200)

    print(f"Saved: {args.out}_overlay.png, {args.out}_metrics.png (peak~{dens.max():.2f})")

if __name__ == '__main__':
    main()