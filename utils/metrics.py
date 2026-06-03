"""Quantitative VGGT-X style AUC metrics — with proper Umeyama pre-alignment."""
import torch, numpy as np
import utils.colmap as colmap_utils


def read_se3(path):
    cameras, images, _ = colmap_utils.read_model(path, ext='.bin')
    items = sorted(images.items(), key=lambda x: x[1].name)
    se3 = []
    for _, img in items:
        R = colmap_utils.qvec2rotmat(img.qvec); t = img.tvec.reshape(3)
        m = np.eye(4); m[:3, :3] = R; m[:3, 3] = t
        se3.append(m)
    return np.stack(se3)


def umeyama(X, Y):
    """Sim(3) alignment: find s, R, t s.t. s*R*X + t ≈ Y."""
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    Xc = X - mu_x; Yc = Y - mu_y
    var_x = np.mean(Xc**2)
    if var_x < 1e-12: return 1.0, np.eye(X.shape[0]), mu_y - mu_x
    cov = (Yc @ Xc.T) / Xc.shape[1]
    U, D, Vh = np.linalg.svd(cov)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vh) < 0: S[-1, -1] = -1
    R = U @ S @ Vh
    s = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - s * R @ mu_x
    return s, R, t


def compute_metrics(pred_se3, gt_se3):
    """Pairwise relative pose errors (VGGT-X style)."""
    N = len(pred_se3)
    device = pred_se3.device

    # Generate all pair indices (batched for memory)
    i1, i2 = torch.combinations(torch.arange(N), 2).unbind(-1)

    def closed_form_inv(se3):
        R = se3[:, :3, :3]; T = se3[:, :3, 3:4]
        Rt = R.transpose(1, 2)
        return torch.cat([Rt, -Rt @ T], dim=2)  # (B, 3, 4)

    def so3_angle(R):
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        return torch.acos(torch.clamp((trace - 1) * 0.5, -1 + 1e-7, 1 - 1e-7))

    # Batch process pairs to avoid OOM
    batch_size = 4096
    r_errors = []; t_errors = []
    for start in range(0, len(i1), batch_size):
        end = min(start + batch_size, len(i1))
        ii1, ii2 = i1[start:end], i2[start:end]

        # Forward: i1→i2
        rel_gt_fw = closed_form_inv(gt_se3[ii1]).bmm(gt_se3[ii2])
        rel_pred_fw = closed_form_inv(pred_se3[ii1]).bmm(pred_se3[ii2])
        # Backward: i2→i1
        rel_gt_bw = closed_form_inv(gt_se3[ii2]).bmm(gt_se3[ii1])
        rel_pred_bw = closed_form_inv(pred_se3[ii2]).bmm(pred_se3[ii1])

        rel_gt = torch.cat([rel_gt_fw, rel_gt_bw])
        rel_pred = torch.cat([rel_pred_fw, rel_pred_bw])

        R_pred_rel = rel_pred[:, :3, :3] @ rel_gt[:, :3, :3].transpose(1, 2)
        r_err = so3_angle(R_pred_rel)
        t_pred = rel_pred[:, :3, 3]; t_gt = rel_gt[:, :3, 3]
        t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + 1e-12)
        t_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + 1e-12)
        t_err = torch.acos(torch.clamp((t_pred * t_gt).sum(dim=1), -1, 1))

        r_errors.append(r_err.cpu())
        t_errors.append(t_err.cpu())

    r_err = torch.cat(r_errors) * 180 / np.pi
    t_err = torch.cat(t_errors) * 180 / np.pi
    return r_err.numpy(), t_err.numpy()


def evaluate_one(pred, gt, label):
    """Compute and print metrics for a single prediction."""
    N = len(pred)
    # Align with Umeyama
    c_pred = -np.einsum('aij,aj->ai', pred[:, :3, :3].transpose(0, 2, 1), pred[:, :3, 3])
    c_gt = -np.einsum('aij,aj->ai', gt[:, :3, :3].transpose(0, 2, 1), gt[:, :3, 3])
    s, R_u, t_u = umeyama(c_pred.T, c_gt.T)
    pred_aligned = np.zeros_like(pred)
    for i in range(N):
        c_new = s * (R_u @ c_pred[i]) + t_u.flatten()
        R_new = pred[i, :3, :3] @ R_u.T
        t_new = -R_new @ c_new
        pred_aligned[i, :3, :3] = R_new; pred_aligned[i, :3, 3] = t_new

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_err, t_err = compute_metrics(
        torch.tensor(pred_aligned, dtype=torch.float32, device=device),
        torch.tensor(gt, dtype=torch.float32, device=device))

    print(f"\n=== {label} ({N} cameras, Umeyama s={s:.2f}x) ===")
    print(f"  Rel Rot Error:  {r_err.mean():8.2f}° mean")
    print(f"  Rel Trans Error:{t_err.mean():8.2f}° mean")
    print(f"  R_acc@5°:       {(r_err<5).mean()*100:8.1f}%")
    print(f"  R_acc@15°:      {(r_err<15).mean()*100:8.1f}%")
    print(f"  T_acc@5°:       {(t_err<5).mean()*100:8.1f}%")
    print(f"  T_acc@15°:      {(t_err<15).mean()*100:8.1f}%")
    for tag, err in [("Rotation", r_err), ("Translation", t_err), ("Combined (max)", np.maximum(r_err, t_err))]:
        bins = np.arange(31)
        hist, _ = np.histogram(err, bins=bins)
        auc = np.cumsum(hist / len(err)).mean()
        print(f"  AUC@30°({tag:15s}): {auc:8.4f}")
    return r_err, t_err


def main():
    gt = read_se3('input/test/sparse/1')
    vggt = read_se3('input/test/sparse/0')
    ga = read_se3('output/sparse/0')

    r_v, t_v = evaluate_one(vggt, gt, "VGGT (sparse/0)")
    r_g, t_g = evaluate_one(ga, gt, "GA optimized (output/sparse/0)")

    print(f"\n=== Summary ===")
    print(f"  Rot: VGGT={r_v.mean():.1f}° → GA={r_g.mean():.1f}° (Δ={r_g.mean()-r_v.mean():+.1f}°)")
    print(f"  Trans: VGGT={t_v.mean():.1f}° → GA={t_g.mean():.1f}° (Δ={t_g.mean()-t_v.mean():+.1f}°)")
    print(f"  AUC rot: VGGT={np.cumsum(np.histogram(r_v, bins=np.arange(31))[0]/len(r_v)).mean():.4f} → "
          f"GA={np.cumsum(np.histogram(r_g, bins=np.arange(31))[0]/len(r_g)).mean():.4f}")


if __name__ == "__main__":
    main()
