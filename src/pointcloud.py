#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""Depth → 3D point cloud + Kabsch alignment demo."""

def depth_to_points(depth_img, scale=1.0):
    D = np.array(depth_img).astype(np.float32)
    h, w = D.shape
    ys, xs = np.nonzero(D > D.mean())
    zs = D[ys, xs] / 255.0 * 40.0
    xs = xs.astype(np.float32) * scale
    ys = ys.astype(np.float32) * scale
    pts = np.stack([xs, ys, zs], axis=1)
    return pts

def kabsch(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:,-1] *= -1
        R = V @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=str, required=True)
    ap.add_argument('--out', type=str, default='results/pc')
    args = ap.parse_args()

    depth = Image.open(args.depth).convert('L')
    P = depth_to_points(depth, scale=1.0)

    theta = np.deg2rad(8.0)
    R_true = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    t_true = np.array([6.0, -3.0, 2.0])
    Q = (R_true @ P.T).T + t_true

    R_est, t_est = kabsch(P, Q)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], s=2, c='b', label='P')
    ax.scatter(Q[:,0], Q[:,1], Q[:,2], s=2, c='r', label='Q')
    ax.set_title('Colony point cloud (synthetic)')
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{args.out}_scatter.png", dpi=200)

    P_aligned = (R_est @ P.T).T + t_est
    err = np.mean(np.linalg.norm(P_aligned - Q, axis=1))
    with open(f"{args.out}_report.txt", 'w') as f:
        f.write(f"Estimated R:\n{R_est}\n\nEstimated t:\n{t_est}\n\nMean alignment error: {err:.3f}\n")
    print(f"Saved: {args.out}_scatter.png, {args.out}_report.txt (error={err:.3f})")

if __name__ == '__main__':
    main()