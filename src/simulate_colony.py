#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""Gray-Scott reaction-diffusion simulation (U,V) on a 2D grid.
We render V as 'colony density' and export an RGB image + depth map.
"""

def laplacian(a):
    return (np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1) - 4*a)

def simulate(size=256, steps=1200, Du=0.16, Dv=0.08, F=0.055, k=0.062):
    H, W = size, size
    U = np.ones((H,W), dtype=np.float32)
    V = np.zeros((H,W), dtype=np.float32)
    # initial seeding: a square of V in the center
    r = size//12
    U[H//2-r:H//2+r, W//2-r:W//2+r] = 0.50
    V[H//2-r:H//2+r, W//2-r:W//2+r] = 0.25

    for t in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        UVV = U*V*V
        U += (Du*Lu - UVV + F*(1.0-U))
        V += (Dv*Lv + UVV - (F+k)*V)
        # small noise to break symmetry
        if t % 80 == 0:
            U += (np.random.rand(H,W).astype(np.float32)-0.5)*0.002
            V += (np.random.rand(H,W).astype(np.float32)-0.5)*0.002
        U = np.clip(U, 0.0, 1.0)
        V = np.clip(V, 0.0, 1.0)
    return U, V

def render(U,V):
    D = V  # visualize V as colony density
    Dn = (D - D.min())/(D.max()-D.min()+1e-8)
    rgb = (255*plt.cm.plasma(Dn)[...,:3]).astype(np.uint8)
    img = Image.fromarray(rgb)
    depth = (255*Dn).astype(np.uint8)
    depth_img = Image.fromarray(depth)
    return img, depth_img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--F', type=float, default=0.055)
    ap.add_argument('--k', type=float, default=0.062)
    ap.add_argument('--save', type=str, default='results/col')
    args = ap.parse_args()

    U,V = simulate(size=args.size, steps=args.steps, F=args.F, k=args.k)
    img, depth = render(U,V)

    img.save(f"{args.save}_image.png")
    depth.save(f"{args.save}_depth.png")

    # simple metric: coverage of V>0.2
    coverage = (V>0.2).mean()
    plt.figure(figsize=(5,3))
    plt.title(f"Colony coverage (V>0.2): {coverage*100:.1f}%")
    plt.imshow(img); plt.axis('off'); plt.tight_layout()
    plt.savefig(f"{args.save}_preview.png", dpi=200)
    print(f"Saved: {args.save}_image.png, {args.save}_depth.png, {args.save}_preview.png")

if __name__ == '__main__':
    main()