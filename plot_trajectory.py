"""3D trajectory comparison: GT vs VGGT vs Ours, adapted from compare_colmap_trajectories."""
import argparse, os, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.colmap as colmap_utils
import pycolmap

sys.path.insert(0, os.path.dirname(__file__))


def camera_center_from_w2c(R, t):
    return -(R.T @ t)


def umeyama_alignment(src, dst):
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: {src.shape} vs {dst.shape}")
    mu_src = src.mean(axis=0); mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src; dst_centered = dst - mu_dst
    cov = dst_centered.T @ src_centered / src.shape[0]
    u, singular_values, vh = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0: correction[-1, -1] = -1
    rotation = u @ correction @ vh
    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.trace(np.diag(singular_values) @ correction) / var_src
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def load_model(path):
    """Load COLMAP .bin or pycolmap format."""
    try:
        _, images, _ = colmap_utils.read_model(path, ext='.bin')
        poses = {}
        for img_id, img in images.items():
            R = colmap_utils.qvec2rotmat(img.qvec)
            t = np.asarray(img.tvec, dtype=np.float64)
            poses[img.name] = (R, t)
        return poses
    except Exception:
        r = pycolmap.Reconstruction(); r.read_binary(path)
        poses = {}
        for img_id in range(1, r.num_images() + 1):
            if r.exists_image(img_id) and r.image(img_id).has_pose:
                cfw = r.image(img_id).cam_from_world()
                poses[r.image(img_id).name] = (cfw.rotation.matrix(), cfw.translation)
        return poses


def load_vggt(npz_path):
    d = np.load(npz_path); ext = d['extrinsics'][0]
    # Need names — read from images directory
    img_dir = Path(npz_path).parent / 'images'
    names = sorted([p.name for p in img_dir.glob("*.[jJ][pP][gG]")])
    poses = {}
    for i, name in enumerate(names):
        if i < len(ext):
            poses[name] = (ext[i, :3, :3], ext[i, :3, 3])
    return poses


def save_plot(gt_centers, pred_centers, pred_aligned, name, out_path, ate_rmse):
    """4-panel plot: 3D + XY + per-frame ATE + histogram."""
    colors = np.linspace(0, 1, len(gt_centers))
    step = max(1, len(gt_centers) // 30)
    errs = np.linalg.norm(gt_centers - pred_aligned, axis=1)
    max_idx = int(np.argmax(errs))

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(*gt_centers.T, color='#1f77b4', lw=2.5, label='GT')
    ax1.plot(*pred_aligned.T, color='#d62728', lw=2, label=name, alpha=0.8, ls='--')
    ax1.scatter(*gt_centers.T, c=colors, cmap='Blues', s=12, alpha=0.7)
    ax1.scatter(*pred_aligned.T, c=colors, cmap='Reds', s=10, alpha=0.6)
    for i in range(0, len(gt_centers), step):
        ax1.plot([gt_centers[i,0], pred_aligned[i,0]],
                 [gt_centers[i,1], pred_aligned[i,1]],
                 [gt_centers[i,2], pred_aligned[i,2]],
                 color='gray', alpha=0.15, lw=0.5)
    ax1.set_title('3D Trajectory'); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(gt_centers[:,0], gt_centers[:,1], color='#1f77b4', lw=2.5, label='GT')
    ax2.plot(pred_aligned[:,0], pred_aligned[:,1], color='#d62728', lw=2, label=name, ls='--')
    ax2.scatter(gt_centers[0,0], gt_centers[0,1], c='green', s=80, marker='o', label='Start', zorder=5)
    ax2.scatter(gt_centers[-1,0], gt_centers[-1,1], c='purple', s=80, marker='s', label='End', zorder=5)
    ax2.set_title('Top View (XY)'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.axis('equal')

    ax3 = fig.add_subplot(223)
    ax3.plot(errs, color='#2ca02c', lw=1.5)
    ax3.fill_between(range(len(errs)), errs, alpha=0.3, color='#2ca02c')
    ax3.axhline(errs.mean(), color='red', ls='--', lw=1.5, label=f'Mean: {errs.mean():.2f}m')
    ax3.set_title(f'ATE per frame (RMSE={ate_rmse:.2f}m)'); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.hist(errs, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax4.axvline(errs.mean(), color='red', ls='--', lw=2, label=f'Mean: {errs.mean():.2f}m')
    ax4.set_title('Error Distribution'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3, axis='y')

    fig.suptitle(f'GT vs {name} Trajectory (ATE RMSE={ate_rmse:.2f}m)', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='test')
    args = parser.parse_args()
    s = args.scene

    gt_poses = load_model(f'input/{s}/sparse/1')
    vggt_poses = load_vggt(f'input/{s}/predictions.npz')
    ours_poses = load_model(f'output/{s}/sparse/0')

    common = sorted(set(gt_poses) & set(vggt_poses) & set(ours_poses))
    print(f'Matched images: {len(common)}')

    def centers(poses, names):
        pts = []
        for n in names:
            R, t = poses[n]; pts.append(camera_center_from_w2c(R, t))
        return np.array(pts)

    cg = centers(gt_poses, common)
    cv = centers(vggt_poses, common)
    co = centers(ours_poses, common)

    # Align to GT
    sv, Rv, tv = umeyama_alignment(cv, cg)
    cv_a = apply_sim3(cv, sv, Rv, tv)
    so, Ro, to = umeyama_alignment(co, cg)
    co_a = apply_sim3(co, so, Ro, to)

    out = f'output/{s}/sparse/0'
    ate_v = np.sqrt(np.mean(np.linalg.norm(cg - cv_a, axis=1)**2))
    ate_o = np.sqrt(np.mean(np.linalg.norm(cg - co_a, axis=1)**2))

    save_plot(cg, cv, cv_a, 'VGGT', f'{out}/traj_gt_vggt.png', ate_v)
    save_plot(cg, co, co_a, 'Ours', f'{out}/traj_gt_ours.png', ate_o)

    # 3-way plot
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(*cg.T, color='#1f77b4', lw=2.5, label='GT')
    ax1.plot(*cv_a.T, color='#d62728', lw=1.5, label='VGGT', alpha=0.7, ls='--')
    ax1.plot(*co_a.T, color='#2ca02c', lw=2.5, label='Ours')
    ax1.set_title('3D Trajectory: GT vs VGGT vs Ours'); ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(122)
    ax2.plot(cg[:,0], cg[:,1], color='#1f77b4', lw=2.5, label='GT')
    ax2.plot(cv_a[:,0], cv_a[:,1], color='#d62728', lw=1.5, label='VGGT', ls='--')
    ax2.plot(co_a[:,0], co_a[:,1], color='#2ca02c', lw=2.5, label='Ours')
    ax2.scatter(cg[0,0], cg[0,1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(cg[-1,0], cg[-1,1], c='purple', s=100, marker='s', label='End', zorder=5)
    ax2.set_title('Top View (XY)'); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.axis('equal')

    fig.suptitle('Camera Trajectory Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{out}/traj_all.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out}/traj_all.png')
    print(f'VGGT ATE={ate_v:.2f}m, Ours ATE={ate_o:.2f}m')


if __name__ == '__main__':
    main()
