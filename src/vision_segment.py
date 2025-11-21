#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""K-means segmentation on colony image; compute density & area metrics."""

def kmeans(img_arr, K=4, iters=12):
    h, w, c = img_arr.shape
    X = img_arr.reshape(-1, c).astype(np.float32)
    rng = np.random.default_rng(0)
    centers = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        d = ((X[:,None,:] - centers[None,:,:])**2).sum(axis=2)
        labels = d.argmin(axis=1)
        for k in range(K):
            pts = X[labels==k]
            if len(pts)>0:
                centers[k] = pts.mean(axis=0)
    return labels.reshape(h,w), centers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--out', type=str, default='results/seg')
    args = ap.parse_args()

    img = Image.open(args.image).convert('RGB')
    arr = np.array(img)
    labels, centers = kmeans(arr, K=4, iters=12)

    # depth for density proxy
    depth_path = args.image.replace('_image', '_depth')
    depth = np.array(Image.open(depth_path).convert('L')).astype(np.float32)/255.0
    mean_density = depth.mean()
    high_density_area = (depth > 0.7).mean()

    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(img); plt.title('Input'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(labels, cmap='tab20'); plt.title('k-means labels'); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(depth, cmap='inferno'); plt.title(f'Depth~density\nmean={mean_density:.2f}, area>0.7={high_density_area*100:.1f}%'); plt.axis('off')
    plt.tight_layout(); plt.savefig(f"{args.out}_overlay.png", dpi=200)
    print(f"Saved: {args.out}_overlay.png")

if __name__ == '__main__':
    main()