"""run_ba — VGGT pose + per-image intrinsics + camera-disjoint UF + Ceres BA."""
import torch, numpy as np, os, sys, sqlite3, argparse, tempfile, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import utils.colmap as colmap_utils
import pycolmap


# ── 3D helpers ────────────────────────────────────────────────────────────────

def unproject(uv, d, K, R, t):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[0] - cx) / fx * d
    y = (uv[1] - cy) / fy * d
    return R.T @ (np.array([x, y, d]) - t)


# ── Database export ──────────────────────────────────────────────────────────

def export_db(db_path, sp_kp, raw_match_data, cameras_in,
              Wd, Hd, N, img_names, img_dir, group_ids, group_cams):
    import os as _os
    from utils.database import COLMAPDatabase
    db_p = Path(db_path)
    if db_p.exists(): db_p.unlink()
    db = COLMAPDatabase.connect(db_p)
    db.create_tables()
    cam0 = list(cameras_in.values())[0]
    model_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2}.get(cam0.model, 1)
    for gid, (fx, fy, cx, cy) in group_cams.items():
        db.add_camera(model_id, cam0.width, cam0.height, np.array([fx, fy, cx, cy]), camera_id=gid)
    for i, name in enumerate(img_names):
        db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (i + 1, name, group_ids[i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sx_kp = cam0.width / Wd; sy_kp = cam0.height / Hd
    for cam in range(N):
        kp = sp_kp[cam].copy(); kp[:, 0] *= sx_kp; kp[:, 1] *= sy_kp
        db.execute("INSERT INTO keypoints VALUES (?,?,?,?)",
                   (cam + 1, len(kp), 2, kp.astype(np.float32).tobytes()))
    MAX_IMG_ID = 2147483647
    for (ci, cj), pr in raw_match_data.items():
        if ci > cj: ci, cj = cj, ci
        db.execute("INSERT INTO matches VALUES (?,?,?,?)",
                   (int((ci + 1) * MAX_IMG_ID + (cj + 1)), len(pr), 2, pr.astype(np.uint32).tobytes()))
    db.commit(); db.close()
    pairs_path = db_p.with_suffix('.pairs.txt')
    with open(pairs_path, 'w') as f:
        for ci, cj in raw_match_data.keys():
            f.write(f'{img_names[ci]} {img_names[cj]}\n')
    with pycolmap.ostream():
        pycolmap.verify_matches(str(db_p.absolute()), str(pairs_path.absolute()),
                                 options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)))


# ── Camera grouping (EXIF) ───────────────────────────────────────────────────

def group_cameras_by_exif(img_names, img_dir, K_all, Wd, Hd, cam0):
    from PIL import Image
    groups = {}; gids = []
    for name in img_names:
        exif = Image.open(str(img_dir / name)).getexif()
        key = f"{exif.get(271, '?')}_{exif.get(272, '?')}"
        if key not in groups: groups[key] = len(groups) + 1
        gids.append(groups[key])
    group_cams = {}
    for gid in set(gids):
        idxs = [i for i in range(len(gids)) if gids[i] == gid]
        fx = np.median(K_all[idxs, 0, 0]) * (cam0.width / Wd)
        fy = np.median(K_all[idxs, 1, 1]) * (cam0.height / Hd)
        cx = np.median(K_all[idxs, 0, 2]) * (cam0.width / Wd)
        cy = np.median(K_all[idxs, 1, 2]) * (cam0.height / Hd)
        group_cams[gid] = (fx, fy, cx, cy)
        gname = list(groups.keys())[list(groups.values()).index(gid)]
        print(f"  Camera {gid} ({gname}): fx={fx:.1f} fy={fy:.1f}  ({len(idxs)} images)")
    return gids, group_cams


# ── 3D init (depth weighted avg) ─────────────────────────────────────────────

def init_3d_points(tracks, K_sp, ext_all, depth_sp, conf_sp, Wsp, Hsp):
    track_data = {}
    for tid, (root, obs_list) in enumerate(tqdm(tracks.items(), desc="3D init")):
        M = len(obs_list)
        cams = np.array([cam for cam, _ in obs_list])
        uvs = np.array([uv for _, uv in obs_list])
        uv_int = np.round(uvs).astype(int)
        uv_int[:, 0] = uv_int[:, 0].clip(0, Wsp - 1)
        uv_int[:, 1] = uv_int[:, 1].clip(0, Hsp - 1)
        c_vals = conf_sp[cams, uv_int[:, 1], uv_int[:, 0]]
        d_vals = depth_sp[cams, uv_int[:, 1], uv_int[:, 0]]
        valid = d_vals > 0
        if valid.sum() < 2: continue
        X_sum = np.zeros(3); C_sum = 0.0
        for idx in range(M):
            if not valid[idx]: continue
            Ki = K_sp[cams[idx]]; Ri = ext_all[cams[idx], :3, :3]; ti = ext_all[cams[idx], :3, 3]
            X_sum += c_vals[idx] * unproject(uvs[idx], d_vals[idx], Ki, Ri, ti)
            C_sum += c_vals[idx]
        if C_sum > 0:
            track_data[tid] = {'X': X_sum / C_sum, 'obs': [(cam, uv) for idx, (cam, uv) in enumerate(obs_list) if valid[idx]]}
    # Front-check
    ok_tracks = {}
    for tid, td in track_data.items():
        Xw = td['X']; ok = True
        for cam, _ in td['obs']:
            if (ext_all[cam, :3, :3] @ Xw + ext_all[cam, :3, 3])[2] <= 0.01:
                ok = False; break
        if ok: ok_tracks[tid] = td
    print(f"Tracks after front-check: {len(ok_tracks)}")
    return ok_tracks


# ── Camera-disjoint Union-Find ───────────────────────────────────────────────

def camera_disjoint_uf(obs_pairs, kp_store):
    parent = {}; root_cams = {}
    def find(x):
        parent.setdefault(x, x); root_cams.setdefault(x, {kp_store[x][0]})
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return True
        cams_a = root_cams.setdefault(ra, set()); cams_b = root_cams.setdefault(rb, set())
        if cams_a & cams_b: return False
        parent[rb] = ra; cams_a |= cams_b; del root_cams[rb]
        return True
    merged = rejected = 0
    for ki, kj, _, _ in obs_pairs:
        if union(ki, kj): merged += 1
        else: rejected += 1
    print(f"UF: merged={merged:,} rejected={rejected:,} ({100 * rejected / max(1, merged + rejected):.1f}%)")
    all_nodes = set()
    for ki, kj, _, _ in obs_pairs: all_nodes.add(ki); all_nodes.add(kj)
    tracks = defaultdict(list)
    for kid in all_nodes:
        root = find(kid)
        cam, uv = kp_store[kid]
        tracks[root].append((cam, uv))
    return tracks, find


# ── Angular filter ───────────────────────────────────────────────────────────

def angular_filter(pair_obs, kp_store, K_sp, ext_all, depth_sp, Wsp, Hsp,
                   img_orient, is_ground, tau_aa, tau_ga):
    obs_pairs = []
    stats = {'aa_total': 0, 'aa_pass': 0, 'gg_total': 0, 'gg_pass': 0,
             'ga_total': 0, 'ga_pass': 0, 'o6_kept': 0}
    K_inv = np.linalg.inv(K_sp)
    for (ci, cj), items in tqdm(pair_obs.items(), desc="Angular"):
        ki_list, kj_list = zip(*items)
        is_ga = (is_ground[ci] != is_ground[cj])
        is_o6 = (img_orient[ci] == 6 or img_orient[cj] == 6)
        tau = tau_ga if is_ga else (10.0 if is_o6 else tau_aa)
        uvi = np.array([kp_store[ki][1] for ki in ki_list])
        uvj = np.array([kp_store[kj][1] for kj in kj_list])
        n = len(ki_list)
        ui, vi = np.round(uvi[:, 0]).astype(int), np.round(uvi[:, 1]).astype(int)
        uj, vj = np.round(uvj[:, 0]).astype(int), np.round(uvj[:, 1]).astype(int)
        valid = (ui >= 0) & (ui < Wsp) & (vi >= 0) & (vi < Hsp) & (uj >= 0) & (uj < Wsp) & (vj >= 0) & (vj < Hsp)
        if not valid.any(): continue
        di, dj = depth_sp[ci, vi, ui], depth_sp[cj, vj, uj]
        valid &= (di > 0) & (dj > 0)
        if not valid.any(): continue
        Ri, Rj = ext_all[ci, :3, :3], ext_all[cj, :3, :3]
        ti, tj = ext_all[ci, :3, 3], ext_all[cj, :3, 3]
        cam_i, cam_j = -Ri.T @ ti, -Rj.T @ tj
        uv_hi = np.stack([uvi[:, 0] * di, uvi[:, 1] * di, di], axis=1)
        Xi = np.einsum('ij,nj->ni', Ri.T, (np.einsum('ij,nj->ni', K_inv[ci], uv_hi) - ti))
        r_proj = Xi - cam_j; r_proj /= np.linalg.norm(r_proj, axis=1, keepdims=True)
        uv_obs = np.stack([uvj[:, 0], uvj[:, 1], np.ones(n)], axis=1)
        r_obs = np.einsum('ij,nj->ni', Rj.T, np.einsum('ij,nj->ni', K_inv[cj], uv_obs))
        r_obs /= np.linalg.norm(r_obs, axis=1, keepdims=True)
        err_fwd = np.arccos(np.clip(np.sum(r_proj * r_obs, axis=1), -1, 1)) * 180 / np.pi
        uv_hj = np.stack([uvj[:, 0] * dj, uvj[:, 1] * dj, dj], axis=1)
        Xj = np.einsum('ij,nj->ni', Rj.T, (np.einsum('ij,nj->ni', K_inv[cj], uv_hj) - tj))
        r_proj2 = Xj - cam_i; r_proj2 /= np.linalg.norm(r_proj2, axis=1, keepdims=True)
        uv_obs2 = np.stack([uvi[:, 0], uvi[:, 1], np.ones(n)], axis=1)
        r_obs2 = np.einsum('ij,nj->ni', Ri.T, np.einsum('ij,nj->ni', K_inv[ci], uv_obs2))
        r_obs2 /= np.linalg.norm(r_obs2, axis=1, keepdims=True)
        err_bwd = np.arccos(np.clip(np.sum(r_proj2 * r_obs2, axis=1), -1, 1)) * 180 / np.pi
        inlier = valid & (np.maximum(err_fwd, err_bwd) <= tau)
        if is_ga:
            stats['ga_total'] += n; stats['ga_pass'] += inlier.sum()
        elif is_o6:
            stats['o6_kept'] += inlier.sum()
        elif is_ground[ci]:  # both ground
            stats['gg_total'] += n; stats['gg_pass'] += inlier.sum()
        else:  # both air
            stats['aa_total'] += n; stats['aa_pass'] += inlier.sum()
        for k in np.where(inlier)[0]:
            obs_pairs.append((ki_list[k], kj_list[k], ci, cj))
    print(f"  AA (τ={tau_aa}°): {stats['aa_total']:,} → {stats['aa_pass']:,} "
          f"({100 * stats['aa_pass'] / max(1, stats['aa_total']):.1f}%)")
    print(f"  GG (τ={tau_aa}°): {stats['gg_total']:,} → {stats['gg_pass']:,} "
          f"({100 * stats['gg_pass'] / max(1, stats['gg_total']):.1f}%)")
    print(f"  GA (τ={tau_ga}°): {stats['ga_total']:,} → {stats['ga_pass']:,} "
          f"({100 * stats['ga_pass'] / max(1, stats['ga_total']):.1f}%)")
    print(f"  o6 (τ=10°): kept {stats['o6_kept']:,}")
    return obs_pairs


# ── Top-K filter ─────────────────────────────────────────────────────────────

def topk_filter(obs_pairs, tracks, kp_store, conf_sp, Wsp, Hsp, topk, find_fn):
    track_len = {}
    for kid in kp_store:
        root = find_fn(kid)
        track_len[kid] = len(tracks.get(root, []))
    groups = defaultdict(list)
    conf_cache = {}
    for idx, (ki, kj, ci, cj) in enumerate(obs_pairs):
        tl = max(track_len.get(ki, 0), track_len.get(kj, 0))
        for kid in (ki, kj):
            if kid not in conf_cache:
                cam, uv = kp_store[kid]
                uvi = np.round(uv).astype(int)
                uvi[0] = np.clip(uvi[0], 0, Wsp - 1); uvi[1] = np.clip(uvi[1], 0, Hsp - 1)
                conf_cache[kid] = conf_sp[cam, uvi[1], uvi[0]]
        avg = (conf_cache[ki] + conf_cache[kj]) / 2
        groups[(ci, cj)].append((idx, (tl, avg)))
    filtered = []
    for items in groups.values():
        items.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in items[:topk]:
            ki, kj, _, _ = obs_pairs[idx]; filtered.append((ki, kj))
    return filtered


# ── Build COLMAP model ───────────────────────────────────────────────────────

def build_colmap_model(track_data, group_ids, group_cams, ext_all, img_names,
                       K_all, Wd, Hd, Wsp, Hsp, cameras_in):
    cam0 = list(cameras_in.values())[0]
    sx2 = cam0.width / Wsp; sy2 = cam0.height / Hsp
    cams_out = {}
    for gid, (fx, fy, cx, cy) in group_cams.items():
        cams_out[gid] = colmap_utils.Camera(id=gid, model=cam0.model,
                                             width=cam0.width, height=cam0.height,
                                             params=np.array([fx, fy, cx, cy], dtype=np.float64))
    img_xys = defaultdict(list); img_p3d = defaultdict(list)
    pt_imgs = defaultdict(list); pt_p2ds = defaultdict(list)
    for pt_id, (tid, td) in enumerate(track_data.items()):
        pid = pt_id + 1
        for cam, uv in td['obs']:
            iid = int(cam) + 1; p2d = len(img_xys[iid])
            img_xys[iid].append(uv.astype(np.float64)); img_p3d[iid].append(pid)
            pt_imgs[pid].append(iid); pt_p2ds[pid].append(p2d)
    imgs_out = {}
    for iid in range(1, len(img_names) + 1):
        xys = np.array(img_xys[iid], dtype=np.float64) if img_xys[iid] else np.zeros((0, 2))
        if len(xys) > 0: xys[:, 0] *= sx2; xys[:, 1] *= sy2
        p3d = np.array(img_p3d[iid], dtype=np.int64) if img_p3d[iid] else np.zeros(0, dtype=np.int64)
        i = iid - 1; Ri, ti = ext_all[i, :3, :3], ext_all[i, :3, 3]
        imgs_out[iid] = colmap_utils.Image(
            id=iid, qvec=colmap_utils.rotmat2qvec(Ri.astype(np.float64)),
            tvec=ti.astype(np.float64), camera_id=group_ids[i],
            name=img_names[i], xys=xys, point3D_ids=p3d)
    pts_out = {}
    for pt_id, (tid, td) in enumerate(track_data.items()):
        pid = pt_id + 1
        pts_out[pid] = colmap_utils.Point3D(
            id=pid, xyz=td['X'].astype(np.float64),
            rgb=np.array([128, 128, 128], dtype=np.uint8), error=np.float64(0.0),
            image_ids=np.array(pt_imgs.get(pid, []), dtype=np.int32),
            point2D_idxs=np.array(pt_p2ds.get(pid, []), dtype=np.int32))
    return cams_out, imgs_out, pts_out


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_ate(out_dir, ext_all, img_names, scene):
    _, gt_imgs, _ = colmap_utils.read_model(f'{scene}/sparse/1', ext='.bin')
    gt = {img.name: (colmap_utils.qvec2rotmat(img.qvec), np.asarray(img.tvec)) for img in gt_imgs.values()}
    vggt = {img_names[i]: (ext_all[i, :3, :3], ext_all[i, :3, 3]) for i in range(len(img_names))}
    _, ba_imgs, _ = colmap_utils.read_model(out_dir, ext='.bin')
    pred = {img.name: (colmap_utils.qvec2rotmat(img.qvec), np.asarray(img.tvec)) for img in ba_imgs.values()}

    def cc(R, t): return -(R.T @ t)
    def umeyama(s, d):
        mu_s, mu_d = s.mean(0), d.mean(0); sc, dc = s - mu_s, d - mu_d
        cov = dc.T @ sc / len(sc); u, sv, vh = np.linalg.svd(cov); cm = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vh) < 0: cm[-1, -1] = -1
        R = u @ cm @ vh; sca = np.trace(np.diag(sv) @ cm) / np.mean(np.sum(sc ** 2, 1))
        t = mu_d - sca * (R @ mu_s); return sca, R, t
    common = sorted(set(gt) & set(vggt) & set(pred))
    cg = np.array([cc(*gt[n]) for n in common])
    cv = np.array([cc(*vggt[n]) for n in common])
    cp = np.array([cc(*pred[n]) for n in common])
    s, R, t = umeyama(cv, cg); ate_v = np.sqrt(np.mean(np.linalg.norm((s * (R @ cv.T)).T + t - cg, axis=1) ** 2))
    s, R, t = umeyama(cp, cg); ate_b = np.sqrt(np.mean(np.linalg.norm((s * (R @ cp.T)).T + t - cg, axis=1) ** 2))
    print(f"\n=== Results ===\nVGGT ATE = {ate_v:.4f}m\nBA ATE   = {ate_b:.4f}m")
    print(f"Images: {len(img_names)}")
    return ate_b


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VGGT-BA: Bundle Adjustment with VGGT predictions")
    parser.add_argument("--scene", default="input/lib")
    parser.add_argument("--sp_res", type=int, default=1600)
    parser.add_argument("--tau_aa", type=float, default=1.0, help="Angular filter for same-type pairs (°)")
    parser.add_argument("--tau_ga", type=float, default=10.0, help="Angular filter for air-ground pairs (°)")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()

    try: sys.stdout.reconfigure(line_buffering=True)
    except: pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    scene = args.scene

    # 1. Load data ─────────────────────────────────────────────────────────────
    npz_path = list(Path(scene).glob('*.npz'))[0]
    d = np.load(str(npz_path))
    depth_all = d['depth'][0, ..., 0]; conf_all = d['depth_conf'][0]
    ext_all = d['extrinsics'][0]; K_all = d['intrinsics'][0]
    img_names = [str(n) for n in d['image_names']]
    N, Hd, Wd = depth_all.shape
    is_ground = [n.startswith('Terr_') for n in img_names]
    print(f"Loaded: {N} images ({Hd}x{Wd})")

    cameras_in, _, _ = colmap_utils.read_model(f'{scene}/sparse/0', ext='.bin')
    img_dir = Path(scene) / 'images'

    # EXIF camera groups
    cam0 = list(cameras_in.values())[0]
    group_ids, group_cams = group_cameras_by_exif(img_names, img_dir, K_all, Wd, Hd, cam0)

    # 2. SP load ───────────────────────────────────────────────────────────────
    sys.path.insert(0, '/mnt/hdd1/hjh/vggt-omega')
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from PIL import Image as PILImage
    SP_RES = args.sp_res
    image_paths = [str(img_dir / name) for name in img_names]
    imgs_tensor = load_and_preprocess_images(image_paths, image_resolution=SP_RES, mode="max_size", patch_size=16)
    _, _, Hsp, Wsp = imgs_tensor.shape
    imgs_sp = [imgs_tensor[i].mean(dim=0, keepdim=True).unsqueeze(0).float() for i in range(N)]
    del imgs_tensor

    depth_t = torch.tensor(depth_all[:, None, :, :], dtype=torch.float32)
    depth_sp = F.interpolate(depth_t, size=(Hsp, Wsp), mode='bilinear', align_corners=False).squeeze(1).numpy()
    conf_t = torch.tensor(conf_all[:, None, :, :], dtype=torch.float32)
    conf_sp = F.interpolate(conf_t, size=(Hsp, Wsp), mode='bilinear', align_corners=False).squeeze(1).numpy()
    del depth_t, conf_t
    sx, sy = Wsp / Wd, Hsp / Hd
    K_sp = K_all.copy(); K_sp[:, 0, 0] *= sx; K_sp[:, 0, 2] *= sx; K_sp[:, 1, 1] *= sy; K_sp[:, 1, 2] *= sy

    img_orient = [PILImage.open(p).getexif().get(0x0112, 1) for p in image_paths]

    # 3. SP extraction ────────────────────────────────────────────────────────
    from lightglue.superpoint import SuperPoint as HLOC_SP
    from lightglue.models.matchers.lightglue import LightGlue as HLOC_LG
    ckpt = 'weights/sp_lg_100h.ckpt'
    sp = HLOC_SP({'max_num_keypoints': 8192, 'force_num_keypoints': True,
                  'detection_threshold': 0.0, 'nms_radius': 3, 'trainable': False}).eval().to(device)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    for k in list(sd.keys()):
        if k.startswith('model.'): del sd[k]
        elif k.startswith('superpoint.'): sd[k.replace('superpoint.', '', 1)] = sd.pop(k)
    sp.load_state_dict(sd, strict=False)
    lg = HLOC_LG({'filter_threshold': 0.1, 'flash': True, 'checkpointed': True}).eval().to(device)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    for k in list(sd.keys()):
        if k.startswith('superpoint.'): del sd[k]
        elif k.startswith('model.'): sd[k.replace('model.', '', 1)] = sd.pop(k)
    lg.load_state_dict(sd, strict=False)

    sp_kp = [None] * N; sp_desc = [None] * N
    for i in tqdm(range(N), desc="SP"):
        with torch.no_grad():
            raw = sp({"image": imgs_sp[i].to(device)})
        kp = raw['keypoints'][0].cpu().numpy(); valid = kp[:, 0] >= 0
        sp_kp[i] = kp[valid]
        desc = raw['descriptors'][0].cpu().numpy()
        sp_desc[i] = desc[:, valid] if desc.shape[0] != len(valid) else desc[valid]
    del sp

    # 4. Dedup ────────────────────────────────────────────────────────────────
    RADIUS = 1.5; kp_store = {}; cam_cell = {}; dedup_map = {}
    for cam in range(N):
        for ki, uv in enumerate(sp_kp[cam]):
            cx, cy = int(uv[0] / RADIUS), int(uv[1] / RADIUS); found = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (cam, cx + dx, cy + dy)
                    if nk in cam_cell and np.linalg.norm(uv - kp_store[cam_cell[nk]][1]) < RADIUS:
                        found = cam_cell[nk]; break
                if found is not None: break
            if found is None:
                kid = len(kp_store); kp_store[kid] = (cam, uv)
                cam_cell[(cam, cx, cy)] = kid; found = kid
            dedup_map[(cam, ki)] = found

    # 5. LG matching ──────────────────────────────────────────────────────────
    from utils.pairs import generate_pairs
    pairs = generate_pairs(img_names, ext_all)
    raw_match_data = {}
    for ci, cj in tqdm(pairs, desc="Match"):
        with torch.no_grad():
            sz0 = torch.tensor(imgs_sp[ci].shape[-2:][::-1], device=device).unsqueeze(0)
            sz1 = torch.tensor(imgs_sp[cj].shape[-2:][::-1], device=device).unsqueeze(0)
            out = lg({"image0": imgs_sp[ci].to(device),
                      "keypoints0": torch.from_numpy(sp_kp[ci]).unsqueeze(0).to(device),
                      "descriptors0": torch.from_numpy(sp_desc[ci]).unsqueeze(0).to(device),
                      "resize0": sz0,
                      "image1": imgs_sp[cj].to(device),
                      "keypoints1": torch.from_numpy(sp_kp[cj]).unsqueeze(0).to(device),
                      "descriptors1": torch.from_numpy(sp_desc[cj]).unsqueeze(0).to(device),
                      "resize1": sz1})
        m = out["matches"][0].cpu().numpy()
        if m.ndim == 2: pr = m
        else: pr = np.array([[j, m[j]] for j in range(len(m)) if m[j] > -1])
        if len(pr) > 0: raw_match_data[(ci, cj)] = pr
    del lg, sp_desc

    # 6. RANSAC ────────────────────────────────────────────────────────────────
    out_dir = f'output/{Path(scene).name}/sparse/0'
    os.makedirs(out_dir, exist_ok=True)
    export_db(os.path.join(out_dir, 'database.db'), sp_kp, raw_match_data, cameras_in,
              Wsp, Hsp, N, img_names, str(img_dir), group_ids, group_cams)

    db_r = sqlite3.connect(os.path.join(out_dir, 'database.db'))
    ransac_matches = []; MAX_IMG_ID = 2147483647
    for pair_id, rows, cols, blob in db_r.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries'):
        if blob is None or rows == 0: continue
        inliers = np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols)
        ci = int((pair_id // MAX_IMG_ID) - 1); cj = int((pair_id % MAX_IMG_ID) - 1)
        if ci < 0 or cj < 0 or ci >= N or cj >= N: continue
        for k in range(len(inliers)):
            ki_r, kj_r = int(inliers[k, 0]), int(inliers[k, 1])
            if (ci, ki_r) in dedup_map and (cj, kj_r) in dedup_map:
                ransac_matches.append((dedup_map[(ci, ki_r)], dedup_map[(cj, kj_r)], ci, cj))
    db_r.close()
    print(f"RANSAC inliers: {len(ransac_matches):,}")

    # 7. Angular filter ───────────────────────────────────────────────────────
    pair_obs = defaultdict(list)
    for ki, kj, ci, cj in ransac_matches: pair_obs[(ci, cj)].append((ki, kj))
    obs_pairs = angular_filter(pair_obs, kp_store, K_sp, ext_all, depth_sp, Wsp, Hsp,
                               img_orient, is_ground, args.tau_aa, args.tau_ga)
    if len(obs_pairs) == 0: print("No matches!"); return

    # 8. Camera-disjoint UF ───────────────────────────────────────────────────
    tracks, find_fn = camera_disjoint_uf(obs_pairs, kp_store)
    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    print(f"Tracks (≥2): {len(tracks)}")

    # 9. Top-K filter ─────────────────────────────────────────────────────────
    obs_pairs = topk_filter(obs_pairs, tracks, kp_store, conf_sp, Wsp, Hsp, args.topk, find_fn)
    print(f"Top-K (K={args.topk}): {len(obs_pairs):,} obs")

    tracks, _ = camera_disjoint_uf([(ki, kj, 0, 0) for ki, kj in obs_pairs], kp_store)
    tracks = {k: v for k, v in tracks.items() if len(v) >= 3}
    print(f"Tracks (≥3): {len(tracks)}")

    # 10. 3D init ─────────────────────────────────────────────────────────────
    track_data = init_3d_points(tracks, K_sp, ext_all, depth_sp, conf_sp, Wsp, Hsp)
    if len(track_data) == 0: print("No valid tracks!"); return

    # 11. BA ───────────────────────────────────────────────────────────────────
    tmp = tempfile.mkdtemp(prefix='ba_')
    model_in = os.path.join(tmp, 'input'); os.makedirs(model_in)
    cams_out, imgs_out, pts_out = build_colmap_model(
        track_data, group_ids, group_cams, ext_all, img_names,
        K_all, Wd, Hd, Wsp, Hsp, cameras_in)
    colmap_utils.write_cameras_binary(cams_out, os.path.join(model_in, 'cameras.bin'))
    colmap_utils.write_images_binary(imgs_out, os.path.join(model_in, 'images.bin'))
    colmap_utils.write_points3D_binary(pts_out, os.path.join(model_in, 'points3D.bin'))

    recon = pycolmap.Reconstruction(); recon.read_binary(model_in)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.ceres.solver_options.max_num_iterations = args.max_iter
    opts.ceres.solver_options.function_tolerance = 1e-6
    opts.ceres.solver_options.min_relative_decrease = 1e-4
    opts.ceres.solver_options.initial_trust_region_radius = 1e4
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = 0.5
    opts.refine_focal_length = False; opts.refine_extra_params = False
    opts.refine_principal_point = False; opts.refine_points3D = True
    opts.min_track_length = 3
    pycolmap.bundle_adjustment(recon, opts)
    print(f"BA: {recon.num_points3D()} points, {recon.num_reg_images()} images")

    # Point cloud coloring
    from PIL import Image as PILImg
    img_cache = {}
    for pid, pt3d in recon.points3D.items():
        els = pt3d.track.elements
        if len(els) == 0: continue
        img_id = els[0].image_id; p2d_idx = els[0].point2D_idx
        img_obj = recon.image(img_id)
        if p2d_idx >= img_obj.num_points3D: continue
        kp_x, kp_y = img_obj.points2D[p2d_idx].xy
        i = img_id - 1
        if i not in img_cache:
            img_cache[i] = np.array(PILImg.open(str(img_dir / img_names[i])).convert('RGB'))
        h, w = img_cache[i].shape[:2]
        x = np.clip(int(round(kp_x)), 0, w - 1)
        y = np.clip(int(round(kp_y)), 0, h - 1)
        pt3d.color = img_cache[i][y, x].astype(np.uint8)

    recon.write(out_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved to {out_dir}/")

    # 12. Evaluate ─────────────────────────────────────────────────────────────
    evaluate_ate(out_dir, ext_all, img_names, scene)


if __name__ == "__main__":
    main()
