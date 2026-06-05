"""run_ba_full pipeline with Ceres BA via pycolmap (replaces Adam with second-order LM)."""
import torch, numpy as np, os, sys, roma
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import kornia
# (SP/LG imported inside main())
import utils.colmap as colmap_utils


def load_sp_image_with_exif(path, Wd, Hd):
    from PIL import Image, ImageOps
    raw = Image.open(path)
    exif = raw.getexif()
    orientation = exif.get(274, 1) if exif else 1
    oriented = ImageOps.exif_transpose(raw)
    sp_w, sp_h = (Hd, Wd) if orientation in (5, 6, 7, 8) else (Wd, Hd)
    img = oriented.convert('L').resize((sp_w, sp_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0), orientation, raw.size, (sp_w, sp_h)


def exif_keypoints_to_depth_grid(kpts, orientation, Wd, Hd):
    pts = np.asarray(kpts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    xmax, ymax = Wd - 1.0, Hd - 1.0
    if orientation == 2: out = np.stack([xmax - x, y], axis=1)
    elif orientation == 3: out = np.stack([xmax - x, ymax - y], axis=1)
    elif orientation == 4: out = np.stack([x, ymax - y], axis=1)
    elif orientation == 5: out = np.stack([y, x], axis=1)
    elif orientation == 6: out = np.stack([y, ymax - x], axis=1)
    elif orientation == 7: out = np.stack([xmax - y, ymax - x], axis=1)
    elif orientation == 8: out = np.stack([xmax - y, x], axis=1)
    else: out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], 0, xmax)
    out[:, 1] = np.clip(out[:, 1], 0, ymax)
    return out


def unproject(uv, d, K, R, t):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (uv[0] - cx) / fx * d
    y = (uv[1] - cy) / fy * d
    return R.T @ (np.array([x, y, d]) - t)


def project(X, K, R, t):
    xc = R @ X + t
    u = K[0,0] * xc[0] / xc[2] + K[0,2]
    v = K[1,1] * xc[1] / xc[2] + K[1,2]
    return np.array([u, v])


def _export_db(db_path, sp_kp=None, sp_desc=None, raw_match_data=None,
               cameras_in=None, Wd=None, Hd=None, N=None, img_names=None, img_dir=None):
    import os as _os
    from pathlib import Path as _Path
    from utils.database import COLMAPDatabase
    _os.makedirs(_os.path.dirname(db_path) or '.', exist_ok=True)
    db_path_p = _Path(db_path)
    if db_path_p.exists(): db_path_p.unlink()
    db = COLMAPDatabase.connect(db_path_p)
    db.create_tables()
    cam0 = cameras_in[list(cameras_in.keys())[0]]
    model_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2}.get(cam0.model, 1)
    db.add_camera(model_id, cam0.width, cam0.height, cam0.params, camera_id=1)
    for i, name in enumerate(img_names):
        db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (i + 1, name, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sx_kp, sy_kp = cam0.width / Wd, cam0.height / Hd
    for cam in range(N):
        kp = sp_kp[cam].copy()
        kp[:, 0] *= sx_kp; kp[:, 1] *= sy_kp
        db.execute("INSERT INTO keypoints VALUES (?,?,?,?)",
                   (cam + 1, len(kp), 2, kp.astype(np.float32).tobytes()))
    MAX_IMG_ID = 2147483647
    for (ci, cj), pr in raw_match_data.items():
        if ci > cj: ci, cj = cj, ci
        pair_id = (ci + 1) * MAX_IMG_ID + (cj + 1)
        db.execute("INSERT INTO matches VALUES (?,?,?,?)",
                   (pair_id, len(pr), 2, pr.astype(np.uint32).tobytes()))
    db.commit()
    db.close()
    pairs_path = db_path_p.with_suffix('.pairs.txt')
    with open(pairs_path, 'w') as f:
        for ci, cj in raw_match_data.keys():
            f.write(f'{img_names[ci]} {img_names[cj]}\n')
    import pycolmap
    with pycolmap.ostream():
        pycolmap.verify_matches(
            str(db_path_p), str(pairs_path),
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)))
    print(f"Exported database: {db_path} ({len(raw_match_data)} pairs, geometry verified)")


def main():
    # Force unbuffered output
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    scene = sys.argv[1] if len(sys.argv) > 1 else 'input/test'
    npz_path = list(Path(scene).glob('*.npz'))
    if not npz_path: print("No .npz found!"); return
    d = np.load(str(npz_path[0]))
    depth_all = d['depth'][0, ..., 0]
    conf_all = d['depth_conf'][0]
    ext_all = d['extrinsics'][0]
    K_all = d['intrinsics'][0]
    N, Hd, Wd = depth_all.shape
    print(f"Loaded {npz_path[0].name}: {N}x{Hd}x{Wd}")

    cameras_in, images, _ = colmap_utils.read_model(f'{scene}/sparse/0', ext='.bin')
    items = sorted(images.items(), key=lambda x: x[1].name)
    img_names = [img.name for _, img in items]
    img_dir = Path(scene) / 'images'

    import sys as _sys; _sys.path.insert(0, '/mnt/hdd1/hjh/vggt-omega')
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from PIL import Image as PILImage

    SP_RES = 1600

    print(f"Loading {N} images for SP (VGGT-omega max_size, res={SP_RES})...")
    image_paths = [str(img_dir / name) for name in img_names]
    images_sp_tensor = load_and_preprocess_images(
        image_paths, image_resolution=SP_RES, mode="max_size", patch_size=16)
    _, _, Hsp, Wsp = images_sp_tensor.shape
    imgs_sp = []
    for i in range(N):
        gray = images_sp_tensor[i].mean(dim=0, keepdim=True)
        imgs_sp.append(gray.unsqueeze(0).float())
    del images_sp_tensor
    print(f"SP resolution: {Wsp}x{Hsp} (depth: {Wd}x{Hd})")

    import torch.nn.functional as tnf
    depth_t = torch.tensor(depth_all[:, None, :, :], dtype=torch.float32)
    depth_sp = tnf.interpolate(depth_t, size=(Hsp, Wsp), mode='bilinear',
                                align_corners=False).squeeze(1).numpy()
    conf_t = torch.tensor(conf_all[:, None, :, :], dtype=torch.float32)
    conf_sp = tnf.interpolate(conf_t, size=(Hsp, Wsp), mode='bilinear',
                               align_corners=False).squeeze(1).numpy()
    del depth_t, conf_t
    sx_sp = Wsp / Wd; sy_sp = Hsp / Hd
    K_sp = K_all.copy()
    K_sp[:, 0, 0] *= sx_sp; K_sp[:, 0, 2] *= sx_sp
    K_sp[:, 1, 1] *= sy_sp; K_sp[:, 1, 2] *= sy_sp
    K_inv_sp = np.linalg.inv(K_sp)

    img_orient = [PILImage.open(p).getexif().get(0x0112, 1) for p in image_paths]

    from utils.pairs import generate_pairs
    pairs_to_match = generate_pairs(img_names, ext_all)

    from lightglue.superpoint import SuperPoint as HLOC_SP
    from lightglue.models.matchers.lightglue import LightGlue as HLOC_LG

    ckpt_path = 'weights/sp_lg_100h.ckpt'
    print(f"Loading SP/LG models from {ckpt_path}...")
    sp = HLOC_SP({'max_num_keypoints': 8192, 'force_num_keypoints': True,
                'detection_threshold': 0.0, 'nms_radius': 3, 'trainable': False}).eval().to(device)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    for k in list(sd.keys()):
        if k.startswith('model.'): del sd[k]
        elif k.startswith('superpoint.'): sd[k.replace('superpoint.', '', 1)] = sd.pop(k)
    sp.load_state_dict(sd, strict=False)
    lg = HLOC_LG({'filter_threshold': 0.1, 'flash': True, 'checkpointed': True}).eval().to(device)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    for k in list(sd.keys()):
        if k.startswith('superpoint.'): del sd[k]
        elif k.startswith('model.'): sd[k.replace('model.', '', 1)] = sd.pop(k)
    lg.load_state_dict(sd, strict=False)

    TAU_O6 = 10.0
    TAU_O1 = 1.0
    TOPK = 500

    obs_pairs = []

    def batch_filter(ci, cj, uvi_arr, uvj_arr, Ki, Kj, Ri, Rj, ti, tj, tau):
        n = len(uvi_arr)
        ui = np.round(uvi_arr[:, 0]).astype(int); vi = np.round(uvi_arr[:, 1]).astype(int)
        uj = np.round(uvj_arr[:, 0]).astype(int); vj = np.round(uvj_arr[:, 1]).astype(int)
        valid = (ui>=0)&(ui<Wsp)&(vi>=0)&(vi<Hsp)&(uj>=0)&(uj<Wsp)&(vj>=0)&(vj<Hsp)
        if not valid.any(): return np.zeros(n, dtype=bool), uvi_arr, uvj_arr
        d_i = depth_sp[ci, vi, ui]; d_j = depth_sp[cj, vj, uj]
        valid &= (d_i > 0) & (d_j > 0)
        if not valid.any(): return np.zeros(n, dtype=bool), uvi_arr, uvj_arr
        cam_i = -Ri.T @ ti; cam_j = -Rj.T @ tj
        uv_h = np.stack([uvi_arr[:,0]*d_i, uvi_arr[:,1]*d_i, d_i], axis=1)
        X_i = np.einsum('ij,nj->ni', Ri.T, (np.einsum('ij,nj->ni', K_inv_sp[ci], uv_h) - ti))
        r_proj = X_i - cam_j
        r_proj /= np.linalg.norm(r_proj, axis=1, keepdims=True)
        uv_h_obs = np.stack([uvj_arr[:,0], uvj_arr[:,1], np.ones(n)], axis=1)
        r_obs = np.einsum('ij,nj->ni', Rj.T, np.einsum('ij,nj->ni', K_inv_sp[cj], uv_h_obs))
        r_obs /= np.linalg.norm(r_obs, axis=1, keepdims=True)
        cos_i = np.clip(np.sum(r_proj * r_obs, axis=1), -1, 1)
        err_i = np.arccos(cos_i) * 180 / np.pi
        uv_h_j = np.stack([uvj_arr[:,0]*d_j, uvj_arr[:,1]*d_j, d_j], axis=1)
        X_j = np.einsum('ij,nj->ni', Rj.T, (np.einsum('ij,nj->ni', K_inv_sp[cj], uv_h_j) - tj))
        r_proj2 = X_j - cam_i
        r_proj2 /= np.linalg.norm(r_proj2, axis=1, keepdims=True)
        uv_h_obs2 = np.stack([uvi_arr[:,0], uvi_arr[:,1], np.ones(n)], axis=1)
        r_obs2 = np.einsum('ij,nj->ni', Ri.T, np.einsum('ij,nj->ni', K_inv_sp[ci], uv_h_obs2))
        r_obs2 /= np.linalg.norm(r_obs2, axis=1, keepdims=True)
        cos_j = np.clip(np.sum(r_proj2 * r_obs2, axis=1), -1, 1)
        err_j = np.arccos(cos_j) * 180 / np.pi
        inlier = valid & (np.maximum(err_i, err_j) <= tau)
        return inlier, uvi_arr, uvj_arr

    print(f"Extracting SP for {N} images...")
    sp_kp = [None]*N; sp_desc = [None]*N
    for i in tqdm(range(N), desc="SP extract"):
        with torch.no_grad():
            raw = sp({"image": imgs_sp[i].to(device)})
        kp = raw['keypoints'][0].cpu().numpy()
        desc = raw['descriptors'][0].cpu().numpy()
        valid = kp[:, 0] >= 0
        kp = kp[valid]
        desc = desc[:, valid] if desc.shape[0] != len(valid) else desc[valid]
        sp_kp[i] = kp
        sp_desc[i] = desc

    RADIUS = 1.5
    kp_store = {}
    cam_cell = {}
    dedup_map = {}
    for cam in tqdm(range(N), desc="SP dedup"):
        for ki, uv in enumerate(sp_kp[cam]):
            cx, cy = int(uv[0]/RADIUS), int(uv[1]/RADIUS)
            found = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (cam, cx+dx, cy+dy)
                    if nk in cam_cell:
                        existing_uv = kp_store[cam_cell[nk]][1]
                        if np.linalg.norm(uv - existing_uv) < RADIUS:
                            found = cam_cell[nk]; break
                if found is not None: break
            if found is None:
                kid = len(kp_store)
                kp_store[kid] = (cam, uv)
                cam_cell[(cam, cx, cy)] = kid
                found = kid
            dedup_map[(cam, ki)] = found

    raw_match_data = {}
    print("LG matching...")
    for ci, cj in tqdm(pairs_to_match, desc="Match"):
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
        if m.ndim == 2: pairs_raw = m
        else: pairs_raw = np.array([[j, m[j]] for j in range(len(m)) if m[j] > -1])
        if len(pairs_raw) == 0: continue
        raw_match_data[(ci, cj)] = pairs_raw

    out_dir = f'output/{Path(scene).name}/sparse/0'
    os.makedirs(out_dir, exist_ok=True)

    # === Step 1: COLMAP RANSAC geometric verification (C++, fast) ===
    _export_db(os.path.join(out_dir, 'database.db'),
               sp_kp=sp_kp, sp_desc=sp_desc, raw_match_data=raw_match_data,
               cameras_in=cameras_in, Wd=Wsp, Hd=Hsp, N=N, img_names=img_names, img_dir=str(img_dir))

    # Read RANSAC inliers from two_view_geometries
    import sqlite3
    print("Reading COLMAP RANSAC results...")
    db_r = sqlite3.connect(os.path.join(out_dir, 'database.db'))
    ransac_matches = []  # (ki, kj, ci, cj) — dedup_ids for RANSAC inliers
    MAX_IMG_ID = 2147483647
    n_raw = 0; n_ransac = 0
    for pair_id, rows, cols, blob in db_r.execute(
            'SELECT pair_id, rows, cols, data FROM two_view_geometries'):
        if blob is None or rows == 0: continue
        inliers = np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols)
        ci = int((pair_id // MAX_IMG_ID) - 1)
        cj = int((pair_id % MAX_IMG_ID) - 1)
        if ci < 0 or cj < 0 or ci >= N or cj >= N: continue
        if sp_kp[ci] is None or sp_kp[cj] is None: continue
        # Get raw LG count for this pair
        if (ci, cj) in raw_match_data:
            n_raw += len(raw_match_data[(ci, cj)])
        n_ransac += len(inliers)
        for k in range(len(inliers)):
            ki_raw = int(inliers[k, 0]); kj_raw = int(inliers[k, 1])
            if (ci, ki_raw) in dedup_map and (cj, kj_raw) in dedup_map:
                ransac_matches.append((dedup_map[(ci, ki_raw)], dedup_map[(cj, kj_raw)], ci, cj))
    db_r.close()
    print(f"  LG raw: {n_raw:,} → RANSAC inliers: {n_ransac:,} → mapped: {len(ransac_matches):,}")

    # === Step 2: Angular filter only on o1-o1 pairs; o6 pairs keep all RANSAC inliers ===
    pair_obs = defaultdict(list)
    for ki, kj, ci, cj in tqdm(ransac_matches, desc="Group by pair"):
        pair_obs[(ci, cj)].append((ki, kj))

    obs_pairs = []
    n_angular_total = 0; n_angular_pass = 0; n_o6_keep = 0
    for (ci, cj), items in tqdm(pair_obs.items(), desc="Angular filter"):
        ki_list, kj_list = zip(*items)
        is_o6 = (img_orient[ci] == 6 or img_orient[cj] == 6)
        tau = TAU_O6 if is_o6 else TAU_O1
        uvi_arr = np.array([kp_store[ki][1] for ki in ki_list])
        uvj_arr = np.array([kp_store[kj][1] for kj in kj_list])
        inlier, _, _ = batch_filter(ci, cj, uvi_arr, uvj_arr,
                                     K_all[ci], K_all[cj],
                                     ext_all[ci,:3,:3], ext_all[cj,:3,:3],
                                     ext_all[ci,:3,3], ext_all[cj,:3,3], tau=tau)
        if is_o6:
            n_o6_keep += inlier.sum()
        else:
            n_angular_total += len(ki_list)
            n_angular_pass += inlier.sum()
        for k in np.where(inlier)[0]:
            obs_pairs.append((ki_list[k], kj_list[k], ci, cj))

    print(f"  o6 pairs (angular τ={TAU_O6}°): kept {n_o6_keep:,} obs")
    print(f"  o1 pairs (angular τ={TAU_O1}°): {n_angular_total:,} → {n_angular_pass:,} "
          f"({100*n_angular_pass/max(1,n_angular_total):.1f}%)")

    if len(obs_pairs) == 0:
        print("No verified matches! Lower τ.")
        return

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for ki, kj, _, _ in obs_pairs:
        union(ki, kj)

    all_nodes = set()
    for ki, kj, _, _ in obs_pairs:
        all_nodes.add(ki)
        all_nodes.add(kj)

    tracks = defaultdict(list)
    for kid in all_nodes:
        root = find(kid)
        cam, uv = kp_store[kid]
        tracks[root].append((cam, uv))

    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    print(f"Tracks: {len(tracks)} (≥2 views)")

    # Top-K filtering by track length + depth confidence
    track_len = {}
    for kid, (cam, uv) in kp_store.items():
        root = find(kid)
        track_len[kid] = len(tracks.get(root, []))
    pair_groups = defaultdict(list)
    conf_by_kid = {}
    for idx, (ki, kj, ci, cj) in enumerate(obs_pairs):
        tl = max(track_len.get(ki, 0), track_len.get(kj, 0))
        if ki not in conf_by_kid:
            cam_i, uv_i = kp_store[ki]
            uvi = np.round(uv_i).astype(int)
            uvi[0] = uvi[0].clip(0, Wsp - 1); uvi[1] = uvi[1].clip(0, Hsp - 1)
            conf_by_kid[ki] = conf_sp[cam_i, uvi[1], uvi[0]]
        if kj not in conf_by_kid:
            cam_j, uv_j = kp_store[kj]
            uvj = np.round(uv_j).astype(int)
            uvj[0] = uvj[0].clip(0, Wsp - 1); uvj[1] = uvj[1].clip(0, Hsp - 1)
            conf_by_kid[kj] = conf_sp[cam_j, uvj[1], uvj[0]]
        avg_conf = (conf_by_kid[ki] + conf_by_kid[kj]) / 2
        pair_groups[(ci, cj)].append((idx, (tl, avg_conf)))
    obs_pairs_filtered = []
    n_before = len(obs_pairs)
    for (ci, cj), items in pair_groups.items():
        items.sort(key=lambda x: x[1], reverse=True)
        keep = items[:TOPK]
        for idx, _ in keep:
            ki, kj, _, _ = obs_pairs[idx]
            obs_pairs_filtered.append((ki, kj))
    obs_pairs = obs_pairs_filtered
    n_after = len(obs_pairs)
    print(f"Top-K filter (K={TOPK}): {n_before:,} obs → {n_after:,} obs ({100*n_after/max(1,n_before):.1f}%)")

    parent = {}
    for ki, kj in obs_pairs:
        ra, rb = find(ki), find(kj)
        if ra != rb: parent[ra] = rb
    all_nodes = set()
    for ki, kj in obs_pairs:
        all_nodes.add(ki); all_nodes.add(kj)
    tracks = defaultdict(list)
    for kid in all_nodes:
        root = find(kid)
        cam, uv = kp_store[kid]
        tracks[root].append((cam, uv))
    tracks = {k: v for k, v in tracks.items() if len(v) >= 3}
    print(f"Tracks (after top-K): {len(tracks)} (≥3 views)")

    track_data = {}
    for tid, (root, obs_list) in enumerate(tqdm(tracks.items(), desc="Track 3D init")):
        M = len(obs_list)
        cams = np.array([cam for cam, _ in obs_list])
        uvs  = np.array([uv for _, uv in obs_list])
        uv_int = np.round(uvs).astype(int)
        uv_int[:, 0] = uv_int[:, 0].clip(0, Wsp - 1)
        uv_int[:, 1] = uv_int[:, 1].clip(0, Hsp - 1)
        c_vals = conf_sp[cams, uv_int[:, 1], uv_int[:, 0]]
        d_vals = depth_sp[cams, uv_int[:, 1], uv_int[:, 0]]
        valid = d_vals > 0
        if valid.sum() < 2: continue
        best = int(np.argmax(c_vals * valid.astype(float)))
        Ki = K_sp[cams[best]]; Ri = ext_all[cams[best],:3,:3]; ti = ext_all[cams[best],:3,3]
        X = unproject(uvs[best], d_vals[best], Ki, Ri, ti)
        C = c_vals[best]
        track_data[tid] = {'X': X, 'C': C,
                           'obs': [(cam, uv) for cam, uv in obs_list]}
    # Filter: remove points behind any camera
    track_data_filt = {}
    for tid, td in track_data.items():
        Xw = td['X']
        ok = True
        for cam, _ in td['obs']:
            Ri = ext_all[cam, :3, :3]; ti = ext_all[cam, :3, 3]
            z = (Ri @ Xw + ti)[2]
            if z <= 0.01: ok = False; break
        if ok:
            track_data_filt[tid] = td
    print(f"Tracks after front-check: {len(track_data_filt)}")
    track_data = track_data_filt

    if len(track_data) == 0:
        print("No valid tracks!"); return

    # === Step 4: Ceres BA via pycolmap ===
    import pycolmap, tempfile, shutil
    n_obs_ba = sum(len(td['obs']) for td in track_data.values())
    print(f"BA (Ceres-pycolmap): {n_obs_ba:,} obs, {len(track_data):,} tracks, {N} cameras")

    tmp = tempfile.mkdtemp(prefix='colmap_ceres_')
    model_in = os.path.join(tmp, 'input')
    os.makedirs(model_in)

    img_xys = defaultdict(list)
    img_p3d_ids = defaultdict(list)
    pt_img_ids = defaultdict(list)
    pt_p2d_idxs = defaultdict(list)

    for pt_id, (tid, td) in enumerate(track_data.items()):
        pid = pt_id + 1
        for cam, uv in td['obs']:
            img_id = int(cam) + 1
            p2d_idx = len(img_xys[img_id])
            img_xys[img_id].append(uv.astype(np.float64))
            img_p3d_ids[img_id].append(pid)
            pt_img_ids[pid].append(img_id)
            pt_p2d_idxs[pid].append(p2d_idx)

    # Use original camera model (6000×4000) — scale keypoints from SP to match
    cam0_in = list(cameras_in.values())[0]
    sx2orig = cam0_in.width / Wsp
    sy2orig = cam0_in.height / Hsp
    cams_out = {}
    for cid, cam in cameras_in.items():
        cams_out[cid] = colmap_utils.Camera(
            id=cam.id, model=cam.model, width=cam.width, height=cam.height,
            params=cam.params)

    imgs_out = {}
    for img_id in range(1, N + 1):
        xys_raw = np.array(img_xys[img_id], dtype=np.float64) if img_xys[img_id] else np.zeros((0, 2))
        if len(xys_raw) > 0:
            xys_raw[:, 0] *= sx2orig
            xys_raw[:, 1] *= sy2orig
        p3d_arr = np.array(img_p3d_ids[img_id], dtype=np.int64) if img_p3d_ids[img_id] else np.zeros(0, dtype=np.int64)
        i = img_id - 1
        R_i = ext_all[i, :3, :3]; t_i = ext_all[i, :3, 3]
        qvec = colmap_utils.rotmat2qvec(R_i.astype(np.float64))
        cam_ref_id = list(cameras_in.keys())[0]  # all images share one camera
        imgs_out[img_id] = colmap_utils.Image(
            id=img_id, qvec=qvec, tvec=t_i.astype(np.float64), camera_id=cam_ref_id,
            name=img_names[i], xys=xys_raw, point3D_ids=p3d_arr)

    pts_out = {}
    for pt_id, (tid, td) in enumerate(track_data.items()):
        pid = pt_id + 1
        pts_out[pid] = colmap_utils.Point3D(
            id=pid, xyz=td['X'].astype(np.float64),
            rgb=np.array([128, 128, 128], dtype=np.uint8),
            error=np.float64(0.0),
            image_ids=np.array(pt_img_ids.get(pid, []), dtype=np.int32),
            point2D_idxs=np.array(pt_p2d_idxs.get(pid, []), dtype=np.int32))

    colmap_utils.write_cameras_binary(cams_out, os.path.join(model_in, 'cameras.bin'))
    colmap_utils.write_images_binary(imgs_out, os.path.join(model_in, 'images.bin'))
    colmap_utils.write_points3D_binary(pts_out, os.path.join(model_in, 'points3D.bin'))

    recon = pycolmap.Reconstruction()
    recon.read_binary(model_in)
    before_err = float(recon.compute_mean_reprojection_error())
    print(f"  Before Ceres: {before_err:.3f}px, {recon.num_reg_images()} images registered")

    opts = pycolmap.BundleAdjustmentOptions()
    opts.ceres.solver_options.max_num_iterations = 100
    opts.ceres.solver_options.function_tolerance = 1e-6
    opts.ceres.solver_options.min_relative_decrease = 1e-4
    opts.ceres.solver_options.initial_trust_region_radius = 1e4
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = 0.5
    opts.refine_focal_length = False
    opts.refine_extra_params = False
    opts.refine_principal_point = False
    opts.refine_points3D = True  # Ceres optimizes both poses and points
    opts.min_track_length = 3

    pycolmap.bundle_adjustment(recon, opts)
    after_err = float(recon.compute_mean_reprojection_error())
    print(f"  After Ceres:  {after_err:.3f}px (Δ={100*(before_err-after_err)/max(1e-6,before_err):.1f}%)")

    model_out = os.path.join(tmp, 'output')
    os.makedirs(model_out, exist_ok=True)
    recon.write(model_out)
    _, opt_imgs, _ = colmap_utils.read_model(model_out, ext='.bin')

    R_out = np.zeros((N, 3, 3))
    t_out = np.zeros((N, 3))
    for img_id, img in opt_imgs.items():
        if img_id - 1 < N:
            R_out[img_id - 1] = colmap_utils.qvec2rotmat(img.qvec)
            t_out[img_id - 1] = img.tvec
    for i in range(N):
        if np.allclose(R_out[i], 0):
            R_out[i] = ext_all[i, :3, :3]
            t_out[i] = ext_all[i, :3, 3]
    # === Add colors before writing (keypoints already at original resolution) ===
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
            img_cache[i] = np.array(PILImg.open(
                str(img_dir / img_names[i])).convert('RGB'))
        h, w = img_cache[i].shape[:2]
        x = np.clip(int(round(kp_x)), 0, w - 1)
        y = np.clip(int(round(kp_y)), 0, h - 1)
        pt3d.color = img_cache[i][y, x].astype(np.uint8)

    # === Output ===
    recon.write(out_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved Ceres model to {out_dir}/ ({recon.num_points3D()} points)")


if __name__ == "__main__":
    main()

