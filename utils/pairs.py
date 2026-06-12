"""Matching pair generation: exhaustive (normal scenes) or air-ground rules."""
import numpy as np


def generate_pairs(image_names, extrinsics, mode='auto'):
    """Generate image pairs for matching.

    Args:
        image_names: list of image filenames
        extrinsics: (N, 3, 4) array of camera extrinsics
        mode: 'auto' — detect if ground images exist, use rules if yes, exhaustive if no
              'exhaustive' — all possible pairs N*(N-1)/2
              'rules' — air/ground distance-based pairing

    Returns:
        list of (i, j) tuples
    """
    N = len(image_names)
    is_ground = [n.startswith('Terr_') for n in image_names]
    has_ground = any(is_ground)

    if mode == 'exhaustive' or (mode == 'auto' and not has_ground):
        # All-vs-all — suitable for small normal scenes
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        print(f"Pairs: {len(pairs)} (exhaustive, {N} images)")
        return pairs

    # Air-ground rules
    air_idx = [i for i in range(N) if not is_ground[i]]
    gnd_idx = [i for i in range(N) if is_ground[i]]
    centers = -np.einsum('aij,aj->ai',
                         extrinsics[:, :3, :3].transpose(0, 2, 1),
                         extrinsics[:, :3, 3])

    pairs = []

    # Ground-ground: sequential window + loop closure
    for i in range(N):
        if not is_ground[i]:
            continue
        for j in range(i + 1, min(i + 21, N)):
            if is_ground[j]:
                pairs.append((i, j))

    # Air-air: 15 nearest by camera center
    for ai in air_idx:
        dists = np.linalg.norm(centers[air_idx] - centers[ai], axis=1)
        for aj_idx in np.argsort(dists)[1:16]:
            aj = air_idx[aj_idx]
            if ai < aj:
                pairs.append((ai, aj))

    # Air-ground: 10 nearest ground per air + 10 nearest air per ground, dedup
    for ai in air_idx:
        dists = np.linalg.norm(centers[gnd_idx] - centers[ai], axis=1)
        for gj_idx in np.argsort(dists)[:10]:
            pairs.append((ai, gnd_idx[gj_idx]))
    for gi in gnd_idx:
        dists = np.linalg.norm(centers[air_idx] - centers[gi], axis=1)
        for aj_idx in np.argsort(dists)[:10]:
            pairs.append((air_idx[aj_idx], gi))

    pairs = list(set(pairs))
    n_air, n_gnd = len(air_idx), len(gnd_idx)
    print(f"Pairs: {len(pairs)} (air={n_air}, ground={n_gnd}, rules mode)")
    return pairs
