"""
Pipeline: SP+LG matching → RANSAC → Angular filter → Tracks → BA:
1. Resize RGB to depth resolution, SuperPoint + LightGlue matching
2. Bidirectional reprojection filter (τ=8px at depth resolution)
3. Union-Find track building → confidence-weighted 3D positions
4. Global BA: pure reprojection L0.5 norm, optimize R,t,K,P
"""
import torch, numpy as np, os, sys, roma
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import kornia
# (SP/LG imported inside main())
import utils.colmap as colmap_utils


def load_sp_image_with_exif(path, Wd, Hd):
    """Load EXIF-oriented image for SP while keeping its resize tied to raw depth."""
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
    """Map keypoints from EXIF-oriented SP grid back to raw VGGT depth grid."""
    pts = np.asarray(kpts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    xmax, ymax = Wd - 1.0, Hd - 1.0

    if orientation == 2:
        out = np.stack([xmax - x, y], axis=1)
    elif orientation == 3:
        out = np.stack([xmax - x, ymax - y], axis=1)
    elif orientation == 4:
        out = np.stack([x, ymax - y], axis=1)
    elif orientation == 5:
        out = np.stack([y, x], axis=1)
    elif orientation == 6:
        out = np.stack([y, ymax - x], axis=1)
    elif orientation == 7:
        out = np.stack([xmax - y, ymax - x], axis=1)
    elif orientation == 8:
        out = np.stack([xmax - y, x], axis=1)
    else:
        out = pts.copy()

    out[:, 0] = np.clip(out[:, 0], 0, xmax)
    out[:, 1] = np.clip(out[:, 1], 0, ymax)
    return out


def match_pair(sp, lg, img_i, img_j, device):
    with torch.no_grad():
        raw_i = sp({"image": img_i.to(device)})
        raw_j = sp({"image": img_j.to(device)})
        out = lg({
            "image0": img_i.to(device), "keypoints0": raw_i["keypoints"][0].unsqueeze(0),
            "descriptors0": raw_i["descriptors"][0].unsqueeze(0),
            "image1": img_j.to(device), "keypoints1": raw_j["keypoints"][0].unsqueeze(0),
            "descriptors1": raw_j["descriptors"][0].unsqueeze(0),
        })
    kp_i = raw_i["keypoints"][0].cpu().numpy()
    kp_j = raw_j["keypoints"][0].cpu().numpy()
    if "matches" in out:
        m = out["matches"][0].cpu().numpy()
        if m.ndim == 2: pairs = m
        else: pairs = np.array([[j, m[j]] for j in range(len(m)) if m[j] > -1])
    else:
        pairs = np.zeros((0, 2), dtype=int)
    return kp_i, kp_j, pairs


def unproject(uv, d, K, R, t):
    """Unproject pixel+ depth → world 3D point."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (uv[0] - cx) / fx * d
    y = (uv[1] - cy) / fy * d
    return R.T @ (np.array([x, y, d]) - t)


def project(X, K, R, t):
    """Project world 3D → pixel."""
    xc = R @ X + t
    u = K[0,0] * xc[0] / xc[2] + K[0,2]
    v = K[1,1] * xc[1] / xc[2] + K[1,2]
    return np.array([u, v])


def _export_db(db_path, sp_kp=None, sp_desc=None, raw_match_data=None,
               cameras_in=None, Wd=None, Hd=None, N=None,
               img_names=None, img_dir=None):
    """Export raw LG matches + SP features as COLMAP database."""
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(42)

    # === Load VGGT data ===
    scene = sys.argv[1] if len(sys.argv) > 1 else 'input/test'
    npz_path = list(Path(scene).glob('*.npz'))
    if not npz_path: print("No .npz found!"); return
    d = np.load(str(npz_path[0]))
    depth_all = d['depth'][0, ..., 0]  # (N, Hd, Wd)
    conf_all = d['depth_conf'][0]      # (N, Hd, Wd)
    ext_all = d['extrinsics'][0]       # (N, 3, 4)
    K_all = d['intrinsics'][0]         # (N, 3, 3)
    N, Hd, Wd = depth_all.shape
    print(f"Loaded {npz_path[0].name}: {N}x{Hd}x{Wd}")

    cameras_in, images, _ = colmap_utils.read_model(f'{scene}/sparse/0', ext='.bin')
    items = sorted(images.items(), key=lambda x: x[1].name)
    img_names = [img.name for _, img in items]
    img_dir = Path(scene) / 'images'

    # === Load images: VGGT-omega official preprocessing ===
    import sys as _sys; _sys.path.insert(0, '/mnt/hdd1/hjh/vggt-omega')
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from PIL import Image as PILImage

    SP_RES = 1600  # SP/LG extraction resolution (depth upsampled to match)

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

    # Upsample depth to SP resolution (bilinear)
    import torch.nn.functional as tnf
    depth_t = torch.tensor(depth_all[:, None, :, :], dtype=torch.float32)  # (N,1,H,W)
    depth_sp = tnf.interpolate(depth_t, size=(Hsp, Wsp), mode='bilinear',
                                align_corners=False).squeeze(1).numpy()  # (N,Hsp,Wsp)
    conf_t = torch.tensor(conf_all[:, None, :, :], dtype=torch.float32)
    conf_sp = tnf.interpolate(conf_t, size=(Hsp, Wsp), mode='bilinear',
                               align_corners=False).squeeze(1).numpy()
    del depth_t, conf_t

    # Scale K matrices to SP resolution
    sx_sp = Wsp / Wd; sy_sp = Hsp / Hd
    K_sp = K_all.copy()
    K_sp[:, 0, 0] *= sx_sp; K_sp[:, 0, 2] *= sx_sp  # fx, cx
    K_sp[:, 1, 1] *= sy_sp; K_sp[:, 1, 2] *= sy_sp  # fy, cy
    K_inv_sp = np.linalg.inv(K_sp)

    # Track EXIF
    img_orient = [PILImage.open(p).getexif().get(0x0112, 1) for p in image_paths]

    # === Pair selection ===
    is_ground = [n.startswith('Terr_') for n in img_names]
    air_idx = [i for i in range(N) if not is_ground[i]]
    gnd_idx = [i for i in range(N) if is_ground[i]]
    centers = -np.einsum('aij,aj->ai', ext_all[:,:3,:3].transpose(0,2,1), ext_all[:,:3,3])

    pairs_to_match = []
    for i in range(N):
        if not is_ground[i]: continue
        for j in range(i+1, min(i+21, N)):
            if is_ground[j]: pairs_to_match.append((i, j))
    for ai in air_idx:
        dists = np.linalg.norm(centers[air_idx] - centers[ai], axis=1)
        for aj_idx in np.argsort(dists)[1:16]:
            aj = air_idx[aj_idx]
            if ai < aj: pairs_to_match.append((ai, aj))
    for ai in air_idx:
        dists = np.linalg.norm(centers[gnd_idx] - centers[ai], axis=1)
        for gj_idx in np.argsort(dists)[:5]:
            pairs_to_match.append((ai, gnd_idx[gj_idx]))
    pairs_to_match = list(set(pairs_to_match))
    print(f"Pairs: {len(pairs_to_match)}")

    # === Step 1: Matching (SP/LG models + checkpoint) ===
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

    TAU_O6 = 5.0  # angular threshold for pairs involving o6 (ground) images
    TAU_O1 = 1.0  # angular threshold for o1-o1 (air-air) pairs
    TOPK = 500

    obs_pairs = [] # [(dedup_id_i, dedup_id_j, ci, cj)]

    def batch_filter(ci, cj, uvi_arr, uvj_arr, Ki, Kj, Ri, Rj, ti, tj, tau=TAU):
        """Angular reprojection filter (depth-independent, tau in degrees)."""
        n = len(uvi_arr)
        ui = np.round(uvi_arr[:, 0]).astype(int); vi = np.round(uvi_arr[:, 1]).astype(int)
        uj = np.round(uvj_arr[:, 0]).astype(int); vj = np.round(uvj_arr[:, 1]).astype(int)
        valid = (ui>=0)&(ui<Wsp)&(vi>=0)&(vi<Hsp)&(uj>=0)&(uj<Wsp)&(vj>=0)&(vj<Hsp)
        if not valid.any(): return np.zeros(n, dtype=bool), uvi_arr, uvj_arr
        d_i = depth_sp[ci, vi, ui]; d_j = depth_sp[cj, vj, uj]
        valid &= (d_i > 0) & (d_j > 0)
        if not valid.any(): return np.zeros(n, dtype=bool), uvi_arr, uvj_arr

        # Camera centers in world
        cam_i = -Ri.T @ ti; cam_j = -Rj.T @ tj
        # Unproject i→3D world
        uv_h = np.stack([uvi_arr[:,0]*d_i, uvi_arr[:,1]*d_i, d_i], axis=1)
        X_i = np.einsum('ij,nj->ni', Ri.T, (np.einsum('ij,nj->ni', K_inv_sp[ci], uv_h) - ti))
        # Angular err i→j: angle between X_i→cam_j and obs_ray_j
        r_proj = X_i - cam_j
        r_proj /= np.linalg.norm(r_proj, axis=1, keepdims=True)
        # Observed ray direction from cam j
        uv_h_obs = np.stack([uvj_arr[:,0], uvj_arr[:,1], np.ones(n)], axis=1)
        r_obs = np.einsum('ij,nj->ni', Rj.T, np.einsum('ij,nj->ni', K_inv_sp[cj], uv_h_obs))
        r_obs /= np.linalg.norm(r_obs, axis=1, keepdims=True)
        cos_i = np.clip(np.sum(r_proj * r_obs, axis=1), -1, 1)
        err_i = np.arccos(cos_i) * 180 / np.pi

        # Unproject j→3D world, angular err j→i
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

    # Pre-extract SP (filter padded kp)
    print(f"Extracting SP for {N} images...")
    sp_kp = [None]*N; sp_desc = [None]*N
    for i in tqdm(range(N), desc="SP extract"):
        with torch.no_grad():
            raw = sp({"image": imgs_sp[i].to(device)})
        kp = raw['keypoints'][0].cpu().numpy()
        desc = raw['descriptors'][0].cpu().numpy()
        valid = kp[:, 0] >= 0  # pads to max_kps with x=-2
        kp = kp[valid]
        desc = desc[:, valid] if desc.shape[0] != len(valid) else desc[valid]
        sp_kp[i] = kp
        sp_desc[i] = desc

    # Spatial keypoint dedup per camera (1.5px radius, O(K) with cell grid)
    RADIUS = 1.5
    kp_store = {}          # dedup_id -> (cam, uv)
    cam_cell = {}          # (cam, cell_x, cell_y) -> dedup_id
    dedup_map = {}         # (cam, sp_idx) -> dedup_id

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
                            found = cam_cell[nk]
                            break
                if found is not None: break
            if found is None:
                kid = len(kp_store)
                kp_store[kid] = (cam, uv)
                cam_cell[(cam, cx, cy)] = kid
                found = kid
            dedup_map[(cam, ki)] = found

    raw_match_data = {}
    for ci, cj in tqdm(pairs_to_match, desc="Match+Filter"):
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
        kp_i, kp_j = sp_kp[ci], sp_kp[cj]
        uvi_arr = kp_i[pairs_raw[:, 0].astype(int)]
        uvj_arr = kp_j[pairs_raw[:, 1].astype(int)]
        tau_pair = TAU_O6 if (img_orient[ci] == 6 or img_orient[cj] == 6) else TAU_O1
        inlier, _, _ = batch_filter(ci, cj, uvi_arr, uvj_arr, K_all[ci], K_all[cj],
                                     ext_all[ci,:3,:3], ext_all[cj,:3,:3], ext_all[ci,:3,3], ext_all[cj,:3,3], tau=tau_pair)
        n_in = inlier.sum()
        print(f"  [{ci:3d},{cj:3d}] LG={len(pairs_raw):5d} → filter={n_in:5d} ({100*n_in/max(1,len(pairs_raw)):.1f}%)  "
              f"τ={tau_pair}°{' o6' if tau_pair==TAU_O6 else ''}", flush=True)
        for k in np.where(inlier)[0]:
            ki_raw = int(pairs_raw[k, 0])
            kj_raw = int(pairs_raw[k, 1])
            obs_pairs.append((dedup_map[(ci, ki_raw)], dedup_map[(cj, kj_raw)], ci, cj))

    print(f"Verified matches: {len(obs_pairs)} (τ={TAU}px)")

    # Export raw LG matches as COLMAP database
    out_dir = f'output/{Path(scene).name}/sparse/0'
    os.makedirs(out_dir, exist_ok=True)
    _export_db(os.path.join(out_dir, 'database.db'),
               sp_kp=sp_kp, sp_desc=sp_desc, raw_match_data=raw_match_data,
               cameras_in=cameras_in, Wd=Wsp, Hd=Hsp, N=N, img_names=img_names, img_dir=str(img_dir))

    if len(obs_pairs) == 0:
        print("No verified matches! Lower τ.")
        return

    # === Step 2: Union-Find track building (on dedup IDs) ===
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

    # Collect all unique observed dedup IDs
    all_nodes = set()
    for ki, kj, _, _ in obs_pairs:
        all_nodes.add(ki)
        all_nodes.add(kj)

    tracks = defaultdict(list)  # track_root -> [(cam_idx, uv)]
    for kid in all_nodes:
        root = find(kid)
        cam, uv = kp_store[kid]
        tracks[root].append((cam, uv))

    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    print(f"Tracks: {len(tracks)} (≥2 views)")

    # === Step 2.5: Top-K per pair filtering (track length + depth confidence) ===
    # Compute track length for each dedup_id
    track_len = {}  # dedup_id -> track length
    for root, obs_list in tracks.items():
        L = len(obs_list)
        for cam, uv in obs_list:
            # find the dedup_id for this (cam, uv)
            pass  # need reverse lookup — compute from kp_store
    # Build reverse map: (cam, rounded_uv) -> dedup_id... too slow.
    # Instead: iterate over kp_store to get track lengths
    for kid, (cam, uv) in kp_store.items():
        root = find(kid)
        if root in tracks:
            track_len[kid] = len(tracks[root])
        else:
            track_len[kid] = 0

    # Group obs_pairs by (ci, cj) and rank
    pair_groups = defaultdict(list)  # (ci, cj) -> [(idx, track_len, conf), ...]
    conf_by_kid = {}  # dedup_id -> depth_conf (cache)
    for idx, (ki, kj, ci, cj) in enumerate(obs_pairs):
        tl = max(track_len.get(ki, 0), track_len.get(kj, 0))
        # Compute avg depth confidence (cache per kid)
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
        # Sort key: first by track_len (0 if no track), then by depth_conf
        # Larger is better for both — concatenate into a single composite key
        sort_key = (tl, avg_conf)
        pair_groups[(ci, cj)].append((idx, sort_key))

    # Filter: keep top-K per pair, rebuild obs_pairs
    obs_pairs_filtered = []
    n_before = len(obs_pairs)
    for (ci, cj), items in pair_groups.items():
        items.sort(key=lambda x: x[1], reverse=True)
        keep = items[:TOPK]
        for idx, _ in keep:
            ki, kj, *_ = obs_pairs[idx]
            obs_pairs_filtered.append((ki, kj))
    obs_pairs = obs_pairs_filtered
    n_after = len(obs_pairs)
    print(f"Top-K filter (K={TOPK}): {n_before:,} obs → {n_after:,} obs "
          f"({100*n_after/max(1,n_before):.1f}%)")

    # Rebuild UF tracks from filtered obs_pairs
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
    tracks = {k: v for k, v in tracks.items() if len(v) >= 2}
    print(f"Tracks (after top-K): {len(tracks)} (≥2 views)")

    # === Step 3: Confidence-weighted track 3D init ===
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

        X_sum = np.zeros(3); C_sum = 0.0
        for idx in range(M):
            if not valid[idx]: continue
            Ki = K_sp[cams[idx]]; Ri = ext_all[cams[idx],:3,:3]; ti = ext_all[cams[idx],:3,3]
            Xk = unproject(uvs[idx], d_vals[idx], Ki, Ri, ti)
            X_sum += c_vals[idx] * Xk; C_sum += c_vals[idx]
        if C_sum > 0:
            track_data[tid] = {'X': X_sum / C_sum, 'C': C_sum / M,
                               'obs': [(cam, uv) for cam, uv in obs_list]}

    print(f"Tracks with 3D: {len(track_data)}")

    if len(track_data) == 0:
        print("No valid tracks!"); return

    # === Step 4: Global BA (pure reprojection, ) ===
    ext_t = torch.tensor(ext_all, dtype=torch.float32, device=device)
    qvec = roma.rotmat_to_unitquat(ext_t[:, :3, :3])
    tvec = ext_t[:, :3, 3].clone()
    q0 = qvec[0:1].detach(); t0 = tvec[0:1].detach()
    q_opt = torch.nn.Parameter(qvec[1:].detach())
    t_opt = torch.nn.Parameter(tvec[1:].detach())

    # 3D point parameters
    X_tensor = torch.tensor(np.stack([td['X'] for td in track_data.values()]),
                            dtype=torch.float32, device=device)
    X_opt = torch.nn.Parameter(X_tensor)
    C_weight = torch.tensor([td['C'] for td in track_data.values()],
                            dtype=torch.float32, device=device)

    K_t = torch.tensor(K_sp, dtype=torch.float32, device=device)

    # Build observations: (cam_idx, uv_x, uv_y, track_id)
    tid_to_idx = {tid: i for i, tid in enumerate(track_data.keys())}
    obs_list = []
    for tid, td in track_data.items():
        for cam, uv in td['obs']:
            obs_list.append([cam, uv[0], uv[1], tid])
    obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
    ci_all = obs_t[:, 0].long()
    uv_all = obs_t[:, 1:3]
    tid_all = obs_t[:, 3].long()
    X_idx_all = torch.tensor([tid_to_idx[int(t)] for t in tid_all], device=device)
    n_obs = len(obs_list)
    print(f"BA: {n_obs} obs, {len(track_data)} tracks")

    params = [
        {"params": [q_opt], "lr": 1e-3},
        {"params": [t_opt], "lr": 1e-3},
        {"params": [X_opt], "lr": 1e-3},
    ]
    opt = torch.optim.Adam(params)
    N_ITER = 2000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ITER, 1e-5)
    loss_hist = []
    best_loss = float("inf")
    best_state = (q_opt.detach().clone(), t_opt.detach().clone(), X_opt.detach().clone())

    for epoch in tqdm(range(N_ITER), desc="BA"):
        opt.zero_grad()
        q_full = torch.cat([q0, q_opt]); t_full = torch.cat([t0, t_opt])
        R_all = roma.unitquat_to_rotmat(F.normalize(q_full, dim=-1))

        # Reprojection with per-camera intrinsics. Only positive-depth finite
        # observations enter the loss; negative z otherwise creates huge
        # residuals and quickly poisons Adam with NaNs.
        Rc = R_all[ci_all]; tc = t_full[ci_all]
        Xw = X_opt[X_idx_all]
        Xc = torch.bmm(Rc, Xw.unsqueeze(2)).squeeze(2) + tc
        z = Xc[:, 2]
        valid_z = z > 1e-4
        if valid_z.sum() < max(16, int(0.05 * n_obs)):
            print(f"BA stopped at iter {epoch}: too few positive-depth observations ({int(valid_z.sum())}/{n_obs}).")
            break
        K_obs = K_t[ci_all]
        fx, fy = K_obs[:, 0, 0], K_obs[:, 1, 1]
        cx, cy = K_obs[:, 0, 2], K_obs[:, 1, 2]
        u = fx * Xc[:,0] / z.clamp(min=1e-4) + cx
        v = fy * Xc[:,1] / z.clamp(min=1e-4) + cy
        err_px = torch.stack([u, v], dim=1).sub(uv_all).norm(dim=1)
        finite = torch.isfinite(err_px) & valid_z
        if finite.sum() < max(16, int(0.05 * n_obs)):
            print(f"BA stopped at iter {epoch}: too few finite observations ({int(finite.sum())}/{n_obs}).")
            break

        # Confidence-weighted L0.5 norm (per )
        cw = C_weight[X_idx_all][finite].clamp(min=0.0)
        loss = (cw * torch.sqrt(err_px[finite].clamp(max=100.0) + 1e-6)).mean()
        if not torch.isfinite(loss):
            print(f"BA stopped at iter {epoch}: non-finite loss.")
            break
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = (q_opt.detach().clone(), t_opt.detach().clone(), X_opt.detach().clone())

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([q_opt, t_opt, X_opt], max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"BA stopped at iter {epoch}: non-finite gradient.")
            break
        opt.step()
        loss_hist.append(loss.item())
        if not all(torch.isfinite(p).all() for p in (q_opt, t_opt, X_opt)):
            print(f"BA stopped at iter {epoch}: non-finite parameter after optimizer step.")
            break
        sched.step()

    if not loss_hist:
        print("BA did not complete any finite optimization step; using initial poses.")
    else:
        print(f"BA loss: {loss_hist[0]:.3f} → {loss_hist[-1]:.3f} → {min(loss_hist):.3f} (Δ={100*(loss_hist[0]-min(loss_hist))/loss_hist[0]:.1f}%)")
        # Plot loss curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_hist, linewidth=0.5, alpha=0.6, color='steelblue')
        # Moving average smoothing
        win = max(1, len(loss_hist) // 50)
        if len(loss_hist) > win:
            smooth = np.convolve(loss_hist, np.ones(win)/win, mode='valid')
            ax.plot(np.arange(win-1, len(loss_hist)), smooth, linewidth=1.5, color='darkred')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (L0.5)')
        ax.set_title(f'BA Convergence (TAU={TAU}°, lr=1e-3, N={len(loss_hist)})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'ba_loss.png'), dpi=100)
        plt.close(fig)
        print(f"Loss curve saved to {out_dir}/ba_loss.png")
    with torch.no_grad():
        q_opt.copy_(best_state[0])
        t_opt.copy_(best_state[1])
        X_opt.copy_(best_state[2])

    # === Output: optimized poses + new point cloud from depth ===
    with torch.no_grad():
        q_full = torch.cat([q0, q_opt]); t_full = torch.cat([t0, t_opt])
        R_out = roma.unitquat_to_rotmat(F.normalize(q_full, dim=-1)).cpu().numpy()
        t_out = t_full.cpu().numpy()

    # Cameras (from original COLMAP)
    new_cameras = {}
    for cid, cam in cameras_in.items():
        new_cameras[cid] = colmap_utils.Camera(
            id=cam.id, model=cam.model, width=cam.width, height=cam.height, params=cam.params)

    # Generate new 3D points from depth + optimized poses (VGGT-omega style: stride-based grid sampling)
    STRIDE = 8
    CONF_THRESH = 3.0
    print(f"Generating point cloud (stride={STRIDE}, conf>{CONF_THRESH})...")
    next_pid = 1
    new_points3D = {}
    new_images_dict = {}

    grid_ys, grid_xs = np.meshgrid(
        np.arange(0, Hd, STRIDE), np.arange(0, Wd, STRIDE), indexing='ij')
    grid_ys = grid_ys.ravel(); grid_xs = grid_xs.ravel()

    from PIL import Image

    img_items = sorted(images.items(), key=lambda x: x[1].name)
    for i, (img_id, img) in enumerate(tqdm(img_items, desc="Unproject")):
        R, t = R_out[i].copy(), t_out[i].copy()
        if np.isnan(R).any():
            R, t = np.eye(3), ext_all[i, :3, 3]
        Ki = K_all[i]

        # Filter: valid depth + confidence threshold
        d_vals = depth_all[i, grid_ys, grid_xs]
        c_vals = conf_all[i, grid_ys, grid_xs]
        valid = (d_vals > 0) & (c_vals > CONF_THRESH)
        ys, xs = grid_ys[valid], grid_xs[valid]
        d_vals = d_vals[valid]
        if len(xs) == 0: continue

        # Unproject batch
        fx, fy, cx, cy = Ki[0,0], Ki[1,1], Ki[0,2], Ki[1,2]
        x_cam = (xs.astype(np.float32) - cx) / fx * d_vals
        y_cam = (ys.astype(np.float32) - cy) / fy * d_vals
        X_w = (R.T @ (np.stack([x_cam, y_cam, d_vals]) - t.reshape(3,1))).T

        # Color from raw image coordinates, matching VGGT depth/K/extrinsics.
        img_full = np.array(Image.open(str(img_dir / img_names[i])).convert("RGB"))
        sy, sx = img_full.shape[0]/Hd, img_full.shape[1]/Wd
        H_full, W_full = img_full.shape[:2]
        y_img = np.round(ys * sy).astype(int).clip(0, H_full - 1)
        x_img = np.round(xs * sx).astype(int).clip(0, W_full - 1)
        colors = img_full[y_img, x_img]

        # Create Point3D and Image entries
        n_pts = len(xs)
        point_ids = np.arange(next_pid, next_pid + n_pts)
        next_pid += n_pts
        xys = np.stack([xs.astype(np.float64) * sx, ys.astype(np.float64) * sy], axis=1)

        for k in range(n_pts):
            pid = int(point_ids[k])
            new_points3D[pid] = colmap_utils.Point3D(
                id=pid, xyz=X_w[k].astype(np.float64),
                rgb=colors[k].astype(np.uint8), error=np.float64(0.0),
                image_ids=np.array([img_id]), point2D_idxs=np.array([k]))

        qvec = colmap_utils.rotmat2qvec(R.astype(np.float64))
        new_images_dict[img_id] = colmap_utils.Image(
            id=img.id, qvec=qvec, tvec=t.astype(np.float64), camera_id=img.camera_id,
            name=img.name, xys=xys.astype(np.float64), point3D_ids=point_ids.astype(np.int64))

    colmap_utils.write_cameras_binary(new_cameras, os.path.join(out_dir, 'cameras.bin'))
    colmap_utils.write_images_binary(new_images_dict, os.path.join(out_dir, 'images.bin'))
    colmap_utils.write_points3D_binary(new_points3D, os.path.join(out_dir, 'points3D.bin'))
    print(f"Saved to {out_dir}/ ({len(new_points3D)} points)")


if __name__ == "__main__":
    main()
