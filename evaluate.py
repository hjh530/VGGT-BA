"""Evaluate COLMAP model against GT: ATE, RRE, RTE, RRA, RTA, AUC.
Uses name-matching + Umeyama Sim(3) alignment (same as compare_colmap_trajectories)."""
import numpy as np, torch, sys, os
from pathlib import Path
from utils.metrics import compute_metrics
import pycolmap, utils.colmap as colmap_utils


def camera_center(R, t):
    return -(R.T @ t)


def umeyama(src, dst):
    """Sim(3) alignment: find s, R, t s.t. s*R*src + t ≈ dst."""
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s; dc = dst - mu_d
    cov = dc.T @ sc / src.shape[0]
    u, s_vals, vh = np.linalg.svd(cov)
    c = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0: c[-1, -1] = -1
    rot = u @ c @ vh
    scale = np.trace(np.diag(s_vals) @ c) / np.mean(np.sum(sc**2, axis=1))
    trans = mu_d - scale * (rot @ mu_s)
    return scale, rot, trans


def apply_sim3(pts, s, R, t):
    return (s * (R @ pts.T)).T + t


def apply_sim3_se3(se3_arr, s, R, t):
    """Apply Sim(3) to SE3 matrices."""
    N = len(se3_arr)
    result = np.zeros_like(se3_arr)
    for i in range(N):
        M = se3_arr[i]
        c = camera_center(M[:3, :3], M[:3, 3])
        c_new = s * (R @ c) + t.flatten()
        R_new = M[:3, :3] @ R.T
        t_new = -R_new @ c_new
        result[i, :3, :3] = R_new
        result[i, :3, 3] = t_new
        result[i, 3, 3] = 1
    return result


def load_poses(path):
    """Load poses keyed by image name. Handles both COLMAP .bin and pycolmap."""
    # Try COLMAP format first
    try:
        _, images, _ = colmap_utils.read_model(path, ext='.bin')
        poses = {}
        for img in images.values():
            R = colmap_utils.qvec2rotmat(img.qvec)
            t = np.asarray(img.tvec, dtype=np.float64)
            poses[img.name] = (R, t)
        if poses: return poses
    except Exception:
        pass
    # Try pycolmap format
    r = pycolmap.Reconstruction(); r.read_binary(path)
    poses = {}
    for img_id in range(1, r.num_images() + 1):
        if r.exists_image(img_id) and r.image(img_id).has_pose:
            cfw = r.image(img_id).cam_from_world()
            poses[r.image(img_id).name] = (cfw.rotation.matrix(), cfw.translation)
    return poses


def load_vggt(npz_path):
    """Load VGGT poses keyed by image name (matched from images directory)."""
    d = np.load(npz_path); ext = d['extrinsics'][0]
    img_dir = Path(npz_path).parent / 'images'
    names = sorted([p.name for p in img_dir.glob('*.[jJ][pP][gG]')])
    poses = {}
    for i, name in enumerate(names):
        if i < len(ext):
            poses[name] = (ext[i, :3, :3], ext[i, :3, 3])
    return poses


def to_se3_array(poses_dict, names):
    """Convert name-keyed poses to SE3 array in names order."""
    se3 = np.zeros((len(names), 4, 4)); se3[:, 3, 3] = 1
    for i, n in enumerate(names):
        if n in poses_dict:
            R, t = poses_dict[n]
            se3[i, :3, :3] = R; se3[i, :3, 3] = t
    return se3


def evaluate(pred_se3, gt_se3, label):
    N = len(pred_se3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Camera centers
    c_pred = np.array([camera_center(M[:3, :3], M[:3, 3]) for M in pred_se3])
    c_gt = np.array([camera_center(M[:3, :3], M[:3, 3]) for M in gt_se3])

    # Umeyama alignment (our umeyama takes (N,3))
    s, R_u, t_u = umeyama(c_pred, c_gt)
    pred_aligned = apply_sim3_se3(pred_se3, s, R_u, t_u)
    c_aligned = apply_sim3(c_pred, s, R_u, t_u)

    # ATE
    ate = np.sqrt(np.mean(np.linalg.norm(c_aligned - c_gt, axis=1) ** 2))

    # RRE, RTE
    rre, rte = compute_metrics(
        torch.tensor(pred_aligned, dtype=torch.float32, device=device),
        torch.tensor(gt_se3, dtype=torch.float32, device=device))
    rre_np = rre.cpu().numpy() if hasattr(rre, 'cpu') else np.asarray(rre)
    rte_np = rte.cpu().numpy() if hasattr(rte, 'cpu') else np.asarray(rte)

    def recall(err, thresh):
        return (err < thresh).mean() * 100
    def auc(err, max_thresh):
        bins = np.arange(int(max_thresh) + 1)
        hist, _ = np.histogram(err, bins=bins)
        return np.cumsum(hist / len(err)).mean()

    print(f'\n{"="*50}')
    print(f'  {label}')
    print(f'{"="*50}')
    print(f'  ATE:  {ate:.4f} m  (scale={s:.3f})')
    print(f'  RRE:  {rre_np.mean():.2f}° mean | {np.median(rre_np):.2f}° median')
    print(f'  RTE:  {rte_np.mean():.2f}° mean | {np.median(rte_np):.2f}° median')
    print(f'  RRA@5°/15°/30°:  {recall(rre_np,5):.1f}% / {recall(rre_np,15):.1f}% / {recall(rre_np,30):.1f}%')
    print(f'  RTA@5°/15°/30°:  {recall(rte_np,5):.1f}% / {recall(rte_np,15):.1f}% / {recall(rte_np,30):.1f}%')
    print(f'  AUC@5° R/T:  {auc(rre_np,5):.4f} / {auc(rte_np,5):.4f}')
    print(f'  AUC@30° R/T: {auc(rre_np,30):.4f} / {auc(rte_np,30):.4f}')

    return {
        'ATE': ate, 'scale': s,
        'RRE_mean': rre_np.mean(), 'RRE_median': np.median(rre_np),
        'RTE_mean': rte_np.mean(), 'RTE_median': np.median(rte_np),
        'RRA5': recall(rre_np, 5), 'RRA15': recall(rre_np, 15), 'RRA30': recall(rre_np, 30),
        'RTA5': recall(rte_np, 5), 'RTA15': recall(rte_np, 15), 'RTA30': recall(rte_np, 30),
        'AUC5_rot': auc(rre_np, 5), 'AUC5_trans': auc(rte_np, 5),
        'AUC30_rot': auc(rre_np, 30), 'AUC30_trans': auc(rte_np, 30),
    }


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else 'test'
    gt = load_poses(f'input/{scene}/sparse/1')
    vggt = load_vggt(f'input/{scene}/predictions.npz')
    ours = load_poses(f'output/{scene}/sparse/0')

    # Match by name
    common = sorted(set(gt) & set(vggt) & set(ours))
    print(f'Matched images: {len(common)}')

    gt_se3 = to_se3_array(gt, common)
    vggt_se3 = to_se3_array(vggt, common)
    ours_se3 = to_se3_array(ours, common)

    rv = evaluate(vggt_se3, gt_se3, 'VGGT')
    ro = evaluate(ours_se3, gt_se3, 'GA Optimized')

    print(f'\n{"="*50}')
    print(f'  Improvement Summary')
    print(f'{"="*50}')
    for k in rv:
        if k in ('ATE', 'scale'):
            v, p = rv[k], ro[k]
            rel = (v - p) / v * 100 if v > 0 else 0
            print(f'  {k:20s}: {v:8.4f} → {p:8.4f}  ({rel:+.1f}%)')
        elif 'mean' in k or 'median' in k:
            v, p = rv[k], ro[k]
            rel = (v - p) / v * 100 if v > 0 else 0
            print(f'  {k:20s}: {v:6.2f}° → {p:6.2f}°  ({rel:+.1f}%)')
        elif k.startswith('AUC'):
            v, p = rv[k], ro[k]
            print(f'  {k:20s}: {v:.4f} → {p:.4f}  ({p-v:+.4f})')
        else:
            v, p = rv[k], ro[k]
            print(f'  {k:20s}: {v:5.1f}% → {p:5.1f}%  ({p-v:+.1f}pp)')


if __name__ == '__main__':
    main()
