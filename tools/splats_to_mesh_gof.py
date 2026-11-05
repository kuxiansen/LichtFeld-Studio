#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract a vertex-colored mesh from a trained splat PLY with a GOF-style tetrahedral field.

This is the Gaussian-Opacity-Fields route: candidate points are the splat centers plus their box
corners, a Delaunay tetrahedralization connects them, and every vertex gets a signed value from the
multi-view minimum of the accumulated occlusion along its ray. The surface is then extracted with
marching tetrahedra (optionally GOF's 8-step bisection refinement) and colored from the photos.

Unlike tools/splats_to_mesh.py it needs no depth maps and no LiDAR priors: only the PLY, the COLMAP
cameras and the images. The fisheye/PINHOLE projection, the fragment generation and the PLY writing
are shared with tools/splat_vs_las.py and tools/splats_to_mesh.py, both of which stay untouched.

Needs numpy, scipy, opencv and torch (CUDA when available). Example:

    python tools/splats_to_mesh_gof.py --ply splat.ply --dataset work/scene --out mesh_gof
    python tools/splats_to_mesh_gof.py --ply splat.ply --dataset work/scene --out mesh_gof \\
        --refine bisect --binary-steps 8
    python tools/splats_to_mesh_gof.py --selftest
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splat_vs_las as sv
import splats_to_mesh as sm

STAGES = ("load", "fragments", "tetra", "delaunay", "alpha", "refine", "filter", "components",
          "color", "write")


def emit(kind, *parts):
    print("[gof]", kind, *parts, flush=True)


def stage(name, done, total):
    emit("stage", name, int(done), int(total))


class FragmentIndex:
    """Per-pixel, depth-sorted splat fragments with prefix sums of log transmittance.

    Equivalent to GOF's integrateCUDA: the accumulated occlusion in front of a point is
    1 - exp(sum of log(1 - alpha) over the fragments nearer to the camera than it).
    """

    __slots__ = ("width", "height", "depths", "cum", "start", "count", "steps", "total")

    def __init__(self, width, height, depths, cum, start, count):
        self.width, self.height = width, height
        self.depths, self.cum = depths, cum
        self.start, self.count = start, count
        self.total = int(depths.numel())
        self.steps = 1 if self.total == 0 else max(1, int(math.ceil(math.log2(int(count.max().item()) + 1))))

    @classmethod
    def build(cls, pixels, depths, log_t, width, height):
        """Pixels/depths/log_t are unordered fragment triples; groups end up sorted by depth."""
        device = depths.device
        count_pixels = width * height
        if depths.numel() == 0:
            return cls(width, height, depths, depths.double(), torch.zeros(count_pixels, device=device, dtype=torch.long),
                       torch.zeros(count_pixels, device=device, dtype=torch.long))
        order = torch.argsort(depths)
        order = order[torch.argsort(pixels[order], stable=True)]
        pixels, depths, log_t = pixels[order], depths[order], log_t[order].double()
        is_first = torch.ones_like(pixels, dtype=torch.bool)
        is_first[1:] = pixels[1:] != pixels[:-1]
        group = torch.cumsum(is_first, dim=0) - 1
        running = torch.cumsum(log_t, dim=0)
        first = torch.nonzero(is_first).squeeze(1)
        cum = running - (running[first] - log_t[first])[group]
        start = torch.zeros(count_pixels, device=device, dtype=torch.long)
        count = torch.zeros(count_pixels, device=device, dtype=torch.long)
        first_pixels = pixels[first]
        start[first_pixels] = first
        count[first_pixels] = torch.diff(torch.cat([first, torch.tensor([depths.numel()], device=device)]))
        return cls(width, height, depths.float(), cum.float(), start, count)

    def opacity_before(self, pixel_ids, depths):
        """Occlusion accumulated by the fragments strictly in front of `depths` in those pixels."""
        lo = self.start[pixel_ids]
        hi = lo + self.count[pixel_ids]
        if self.total == 0:
            return torch.zeros_like(depths)
        last = self.total - 1
        for _ in range(self.steps):
            mid = (lo + hi) >> 1
            active = mid < hi
            probe = self.depths[mid.clamp(0, last)]
            go_right = active & (probe < depths)
            lo = torch.where(go_right, mid + 1, lo)
            hi = torch.where(go_right, hi, mid)
        before = (lo - 1).clamp_min(0)
        has = lo > self.start[pixel_ids]
        return torch.where(has, 1.0 - torch.exp(self.cum[before]), torch.zeros_like(depths))


def build_fragment_index(splats, lens, R, t, kind, chunk):
    """The fragment stream of sv.render_depths, kept as (pixel, depth, log_t) instead of composited."""
    device = lens.device
    R = torch.as_tensor(R, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    p = splats["means"] @ R.T + t
    uv, depth, jac = lens.project(p)
    keep = (depth > sv.MIN_RANGE_M) & lens.in_field_of_view(p)
    margin = sv.MAX_HALF_WIDTH_PX
    keep &= (uv[:, 0] > -margin) & (uv[:, 0] < lens.width + margin)
    keep &= (uv[:, 1] > -margin) & (uv[:, 1] < lens.height + margin)
    index = torch.nonzero(keep).squeeze(1)
    uv, depth, jac = uv[index], depth[index], jac[index]

    axes = jac @ (R @ splats["rotations"][index]) * splats["scales"][index][:, None, :]
    cov = axes @ axes.transpose(1, 2)
    a = cov[:, 0, 0] + 1e-4
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 1e-4
    det = (a * c - b * b).clamp_min(1e-10)
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    tilt_u = cov[:, 2, 0] * inv_a + cov[:, 2, 1] * inv_b
    tilt_v = cov[:, 2, 0] * inv_b + cov[:, 2, 1] * inv_c
    sigma_depth = torch.sqrt(cov[:, 2, 2].clamp_min(0))
    opacity = splats["opacity"][index]

    half_w = (3 * torch.sqrt(a)).ceil().clamp_max(sv.MAX_HALF_WIDTH_PX).long()
    half_h = (3 * torch.sqrt(c)).ceil().clamp_max(sv.MAX_HALF_WIDTH_PX).long()
    u_min = (torch.floor(uv[:, 0]).long() - half_w).clamp_min(0)
    u_max = (torch.floor(uv[:, 0]).long() + half_w).clamp_max(lens.width - 1)
    v_min = torch.floor(uv[:, 1]).long() - half_h
    v_max = torch.floor(uv[:, 1]).long() + half_h

    pixels_out, depth_out, log_t_out = [], [], []
    for band_start in range(0, lens.height, sv.BAND_ROWS):
        band_stop = min(band_start + sv.BAND_ROWS, lens.height)
        band = torch.nonzero((u_max >= u_min) & (v_max >= band_start) & (v_min < band_stop)).squeeze(1)
        if band.numel() == 0:
            continue
        row_min = v_min[band].clamp_min(band_start)
        columns = u_max[band] - u_min[band] + 1
        counts = columns * (v_max[band].clamp_max(band_stop - 1) - row_min + 1)
        start = 0
        while start < band.numel():
            running = torch.cumsum(counts[start:], dim=0)
            stop = start + max(1, int(torch.searchsorted(running, chunk).item()))
            local = torch.repeat_interleave(torch.arange(start, stop, device=device), counts[start:stop])
            ends = torch.cumsum(counts[start:stop], dim=0)
            offset = torch.arange(local.numel(), device=device) - torch.repeat_interleave(
                ends - counts[start:stop], counts[start:stop]
            )
            rep = band[local]
            px = u_min[rep] + offset % columns[local]
            py = row_min[local] + offset // columns[local]
            du = px.float() + 0.5 - uv[rep, 0]
            dv = py.float() + 0.5 - uv[rep, 1]
            mahalanobis = inv_a[rep] * du * du + 2 * inv_b[rep] * du * dv + inv_c[rep] * dv * dv
            alpha = opacity[rep] * torch.exp(-0.5 * mahalanobis)
            solid = alpha > sv.MIN_FRAGMENT_ALPHA
            rep, du, dv, px, py, alpha = rep[solid], du[solid], dv[solid], px[solid], py[solid], alpha[solid]
            offset_depth = tilt_u[rep] * du + tilt_v[rep] * dv
            limit = 3 * sigma_depth[rep]
            pixels_out.append(py * lens.width + px)
            depth_out.append(depth[rep] + torch.maximum(torch.minimum(offset_depth, limit), -limit)
                             if kind == "tilted" else depth[rep])
            log_t_out.append(torch.log1p(-alpha.clamp_max(0.9999)))
            start = stop

    if not pixels_out:
        return FragmentIndex.build(torch.empty(0, device=device, dtype=torch.long),
                                   torch.empty(0, device=device), torch.empty(0, device=device),
                                   lens.width, lens.height)
    return FragmentIndex.build(torch.cat(pixels_out), torch.cat(depth_out), torch.cat(log_t_out),
                               lens.width, lens.height)


def view_opacity(lens, index, valid, p, R, t):
    """Occlusion accumulated in front of each point, or 1.0 where this view cannot see it.

    1.0 is GOF's "no vote": out_alpha_integrated starts at 1.0 for points a view did not cover,
    so the multi-view minimum ignores them instead of reading them as fully exposed.
    """
    device = lens.device
    R = torch.as_tensor(R, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    pc = p @ R.T + t
    uv, r, _ = lens.project(pc)
    x, y = uv[:, 0], uv[:, 1]
    inside = lens.in_field_of_view(pc) & (r > sv.MIN_RANGE_M)
    inside &= (x >= 0) & (x < lens.width) & (y >= 0) & (y < lens.height)
    out = torch.ones_like(r)
    ids = torch.nonzero(inside).squeeze(1)
    if ids.numel() == 0:
        return out
    x0 = torch.floor(x[ids]).long()
    y0 = torch.floor(y[ids]).long()
    fx, fy, rr = x[ids] - x0.float(), y[ids] - y0.float(), r[ids]
    acc = torch.zeros_like(rr)
    weight = torch.zeros_like(rr)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx = (x0 + dx).clamp(0, lens.width - 1)
        yy = (y0 + dy).clamp(0, lens.height - 1)
        pixels = yy * lens.width + xx
        usable = valid[pixels]
        occlusion = index.opacity_before(pixels, rr)
        acc += torch.where(usable, w * occlusion, torch.zeros_like(acc))
        weight += torch.where(usable, w, torch.zeros_like(weight))
    out[ids] = torch.where(weight > 0, acc / weight.clamp_min(1e-8), torch.ones_like(acc))
    return out


def evaluate_alpha(points, contexts, splats, kind, chunk, progress=None):
    """Multi-view minimum of the accumulated occlusion, EXACTLY as GOF's rasterize_points.cu.

    out_alpha_integrated starts at 1.0 there, so a view that does not see the point (outside its
    frustum, outside the image or masked out) does not vote at all: it must not be read as "outside".
    """
    n = points.shape[0]
    minimum = torch.ones(n, device=points.device)
    if n == 0:
        return minimum
    for i, ctx in enumerate(contexts):
        index = build_fragment_index(splats, ctx["lens"], ctx["R"], ctx["t"], kind, chunk)
        for start in range(0, n, chunk):
            part = points[start:start + chunk]
            occlusion = view_opacity(ctx["lens"], index, ctx["valid"], part, ctx["R"], ctx["t"])
            minimum[start:start + chunk] = torch.minimum(minimum[start:start + chunk], occlusion)
        if progress is not None:
            progress(i + 1, len(contexts))
    return 1.0 - minimum


def voxel_dedupe_first(points, scales, voxel):
    """One point (and its scale) per voxel, so the Delaunay input does not follow splat density."""
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    key = keys[:, 0] + dims[0] * (keys[:, 1] + dims[1] * keys[:, 2])
    _, first = np.unique(key, return_index=True)
    return points[first], scales[first]


def tetra_points(splats, contexts, min_opacity, scale_mult, point_voxel, max_points, margin):
    """GOF's get_tetra_points: 8 box corners plus the center, culled by frustum and deduplicated."""
    device = splats["means"].device
    visible = torch.zeros(splats["means"].shape[0], device=device, dtype=torch.bool)
    for ctx in contexts:
        lens = ctx["lens"]
        p = splats["means"] @ torch.as_tensor(ctx["R"], device=device, dtype=torch.float32).T \
            + torch.as_tensor(ctx["t"], device=device, dtype=torch.float32)
        uv, r, _ = lens.project(p)
        seen = lens.in_field_of_view(p) & (r > sv.MIN_RANGE_M)
        seen &= (uv[:, 0] > -margin) & (uv[:, 0] < lens.width + margin)
        seen &= (uv[:, 1] > -margin) & (uv[:, 1] < lens.height + margin)
        visible |= seen
    keep = visible & (splats["opacity"] >= min_opacity)
    kept = int(keep.sum().item())
    if kept == 0:
        raise RuntimeError("no splat survived the opacity and frustum filter; lower --min-opacity")
    means = splats["means"][keep]
    axes = scale_mult * splats["scales"][keep]

    signs = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
                         device=device, dtype=torch.float32)
    offset = signs[None] * axes[:, None, :]
    corners = (splats["rotations"][keep] @ offset.transpose(1, 2)).transpose(1, 2) + means[:, None, :]
    corner_scale = axes.max(dim=1).values[:, None].repeat(1, 8)

    points = torch.cat([corners.reshape(-1, 3), means])
    scales = torch.cat([corner_scale.reshape(-1), axes.max(dim=1).values])
    points, scales = voxel_dedupe_first(points.cpu().numpy(), scales.cpu().numpy(), point_voxel)
    if points.shape[0] > max_points:
        raise RuntimeError(f"{points.shape[0]} tetra points exceed --max-points {max_points}; "
                           f"raise --point-voxel (now {point_voxel})")
    stats = {"splat_total": int(splats["means"].shape[0]), "splat_kept": kept,
             "point_count": int(points.shape[0])}
    return (torch.as_tensor(points, device=device, dtype=torch.float32),
            torch.as_tensor(scales, device=device, dtype=torch.float32), stats)


def marching_tetrahedra(points, tets, sdf, point_scale):
    """Marching tetrahedra on a Delaunay cell set. Vertices are the crossings, one per unique edge."""
    device = points.device
    occ = sdf[tets] > 0
    positive = occ.sum(dim=1)
    valid = (positive > 0) & (positive < 4)
    valid_ids = torch.nonzero(valid).squeeze(1)
    if valid_ids.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, empty, empty, empty.reshape(0, 3)

    occ_v = occ[valid_ids]
    count = positive[valid_ids]
    two_two = count == 2
    one_three = ~two_two

    o8 = occ_v.to(torch.uint8)
    n8 = (~occ_v).to(torch.uint8)
    first_pos = o8.argmax(dim=1)
    last_pos = 3 - o8.flip(1).argmax(dim=1)
    first_neg = n8.argmax(dim=1)
    last_neg = 3 - n8.flip(1).argmax(dim=1)

    # 1-3: the three crossings all start at the lone corner.
    lone = torch.where(count == 1, first_pos, first_neg)[one_three]
    k = (torch.arange(4, device=device)[None] < lone[:, None]).to(torch.uint8).sum(dim=1)
    j = torch.arange(3, device=device)[None]
    others = torch.where(j < k[:, None], j, j + 1)
    tetra_a = tets[valid_ids[one_three]]
    edges_a = torch.stack([tetra_a.gather(1, lone[:, None]).expand(-1, 3), tetra_a.gather(1, others)], dim=-1)

    # 2-2: the four crossings form a quad, split along one diagonal.
    p0, p1 = first_pos[two_two], last_pos[two_two]
    n0, n1 = first_neg[two_two], last_neg[two_two]
    tetra_b = tets[valid_ids[two_two]]
    index_p0 = tetra_b.gather(1, p0[:, None]).squeeze(1)
    index_p1 = tetra_b.gather(1, p1[:, None]).squeeze(1)
    index_n0 = tetra_b.gather(1, n0[:, None]).squeeze(1)
    index_n1 = tetra_b.gather(1, n1[:, None]).squeeze(1)
    edges_b = torch.stack([
        torch.stack([index_p0, index_n0], dim=1),
        torch.stack([index_p1, index_n0], dim=1),
        torch.stack([index_p1, index_n1], dim=1),
        torch.stack([index_p0, index_n1], dim=1),
    ], dim=1)

    all_edges = torch.cat([edges_a.reshape(-1, 2), edges_b.reshape(-1, 2)])
    lo = all_edges.min(dim=1).values
    hi = all_edges.max(dim=1).values
    n_points = points.shape[0]
    unique_keys, inverse = torch.unique(lo * n_points + hi, return_inverse=True)
    edge_a = unique_keys // n_points
    edge_b = unique_keys % n_points

    faces_a = inverse[: edges_a.shape[0] * 3].reshape(-1, 3)
    faces_b = inverse[edges_a.shape[0] * 3:].reshape(-1, 4)
    faces = torch.cat([faces_a, faces_b[:, [0, 1, 2]], faces_b[:, [0, 2, 3]]])

    tri_tet = torch.cat([valid_ids[one_three],
                         valid_ids[two_two].repeat_interleave(2)])
    corners = points[tets[tri_tet]]
    positive_corners = occ[tri_tet]
    pos_mean = (corners * positive_corners[..., None]).sum(1) / positive_corners.sum(1, keepdim=True)
    neg_mean = (corners * (~positive_corners)[..., None]).sum(1) / (~positive_corners).sum(1, keepdim=True)
    outward = pos_mean - neg_mean
    mid = 0.5 * (points[edge_a] + points[edge_b])
    normal = torch.cross(mid[faces[:, 1]] - mid[faces[:, 0]], mid[faces[:, 2]] - mid[faces[:, 0]], dim=1)
    flip = (normal * outward).sum(dim=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]

    return edge_a, edge_b, sdf[edge_a], sdf[edge_b], faces


def span_keep(distance, scale, span_mult):
    """GOF's filter_mesh: a crossing edge longer than its endpoint scales is a sliver between blobs."""
    return distance <= span_mult * scale


def refine_vertices(points, edge_a, edge_b, s_a, s_b, sdf_of, mode, steps):
    pa, pb = points[edge_a], points[edge_b]
    if mode == "linear":
        # The endpoints straddle the surface, so s_a - s_b never vanishes.
        t = (s_a / (s_a - s_b)).clamp(0.0, 1.0)
        return pa + t[:, None] * (pb - pa)
    left_p, right_p = pa.clone(), pb.clone()
    left_s, right_s = s_a.clone(), s_b.clone()
    for _ in range(max(0, steps)):
        mid = 0.5 * (left_p + right_p)
        mid_s = sdf_of(mid)
        ind_low = ((mid_s < 0) & (left_s < 0)) | ((mid_s > 0) & (left_s > 0))
        left_s[ind_low] = mid_s[ind_low]
        right_s[~ind_low] = mid_s[~ind_low]
        left_p[ind_low] = mid[ind_low]
        right_p[~ind_low] = mid[~ind_low]
    return 0.5 * (left_p + right_p)


def view_mask(image, lens, masks_root):
    valid = lens.valid_pixels()
    if masks_root is None:
        return valid
    path = masks_root / f"{image['name']}.png"
    if not path.exists():
        return valid
    mask = sv.read_image_any(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = cv2.resize(mask, (lens.width, lens.height), interpolation=cv2.INTER_NEAREST) > 127
    return valid & torch.as_tensor(mask, device=lens.device)


def render_color_depths(splats, images, cameras, root, masks_root, cache, kind, device, progress=None):
    """Median depth per color view, in the uint16 mm PNG format sm.read_cached_depth expects."""
    cache.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = cache / f"{image['index']:05d}.png"
        if path.exists():
            continue
        camera = cameras[image["camera_id"]]
        lens = sv.Lens(camera, camera["width"], camera["height"], device)
        out = sv.render_depths(splats, lens, image["R"], image["t"], kinds=(kind,))
        depth = torch.nan_to_num(out[f"{kind}_median"], nan=0.0)
        depth[~view_mask(image, lens, masks_root)] = 0
        millimeters = (depth * 1000).clamp(0, 65535).round().cpu().numpy().astype(np.uint16)
        alpha = (out["alpha"] * 255).clamp(0, 255).round().byte().cpu().numpy()
        written, encoded = cv2.imencode(".png", alpha)
        assert written
        encoded.tofile(str(cache / f"{image['index']:05d}_alpha.png"))
        written, encoded = cv2.imencode(".png", millimeters)
        assert written
        encoded.tofile(str(path))
        if progress is not None:
            progress(i + 1, len(images))


def color_vertices_gof(vertices, images, cameras, root, cache, device, min_alpha, tolerance):
    """sm.color_vertices with an optional mask (already folded into the cached depth)."""
    total = torch.zeros((vertices.shape[0], 3), device=device)
    count = torch.zeros(vertices.shape[0], device=device)
    for image in images:
        camera = cameras[image["camera_id"]]
        lens = sv.Lens(camera, camera["width"], camera["height"], device)
        depth = sm.read_cached_depth(cache, image, device, min_alpha)
        photo = sv.read_image_any(root / "images" / image["name"])
        photo = cv2.resize(photo, (lens.width, lens.height), interpolation=cv2.INTER_AREA)
        photo = torch.as_tensor(photo[..., ::-1].copy(), device=device, dtype=torch.float32)
        for start in range(0, vertices.shape[0], 5_000_000):
            part = vertices[start:start + 5_000_000]
            p = part @ torch.as_tensor(image["R"], device=device, dtype=torch.float32).T \
                + torch.as_tensor(image["t"], device=device, dtype=torch.float32)
            uv, r, _ = lens.project(p)
            px, py = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()
            ok = lens.in_field_of_view(p) & (px >= 0) & (px < lens.width) & (py >= 0) & (py < lens.height)
            ids = torch.nonzero(ok).squeeze(1)
            px, py = px[ids], py[ids]
            seen = (depth[py, px] > 0) & ((depth[py, px] - r[ids]).abs() < tolerance)
            ids, px, py = ids[seen] + start, px[seen], py[seen]
            total[ids] += photo[py, px]
            count[ids] += 1
    colors = total / count.clamp_min(1)[:, None]
    colors[count == 0] = 128
    return colors.round().clamp(0, 255).byte(), float((count > 0).float().mean())


def build_contexts(images, cameras, device, scale, masks_root):
    """Alpha-resolution lenses plus their valid-pixel masks, one per view."""
    contexts = []
    for image in images:
        camera = cameras[image["camera_id"]]
        width = max(1, int(round(camera["width"] * scale)))
        height = max(1, int(round(camera["height"] * scale)))
        lens = sv.Lens(camera, width, height, device)
        contexts.append({"image": image, "lens": lens, "R": image["R"], "t": image["t"],
                         "valid": view_mask(image, lens, masks_root)})
    return contexts


def build_mesh(args):
    begin = time.time()
    timings = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sparse = Path(args.sparse) if args.sparse else root / "sparse" / "0"
    images_root = Path(args.images) if args.images else root / "images"
    masks_root = Path(args.masks) if args.masks else root / "masks"
    if not masks_root.is_dir():
        masks_root = None

    step = time.time()
    cameras = sv.read_cameras(sparse / "cameras.bin")
    images = sv.read_images(sparse / "images.bin")
    for index, image in enumerate(images):
        image["index"] = index
    training = [image for image in images if image["index"] % sm.HELD_OUT_EVERY != 0][::args.every]
    if not training:
        raise RuntimeError("no training views left after the held-out and --every selection")
    splats = sm.load_splats(args.ply, device)
    contexts = build_contexts(training, cameras, device, args.alpha_scale, masks_root)
    emit("info", f"{splats['means'].shape[0]} splats, {len(training)} alpha views, "
                 f"masks {'on' if masks_root else 'off'}")
    stage("load", 1, 1)
    timings["load"] = time.time() - step

    step = time.time()
    points, scales, stats = tetra_points(splats, contexts, args.min_opacity, args.scale_mult,
                                         args.point_voxel, args.max_points, sv.MAX_HALF_WIDTH_PX)
    stage("tetra", 1, 1)
    emit("info", f"{stats['point_count']} tetra points from {stats['splat_kept']} splats")
    timings["tetra"] = time.time() - step

    step = time.time()
    array = points.cpu().numpy()
    tets = None
    cache_path = Path(args.cells_cache) if args.cells_cache else None
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        if data["points"].shape == array.shape:
            tets = data["tets"]
            emit("info", f"loaded {tets.shape[0]} cells from {cache_path}")
    if tets is None:
        try:
            tets = Delaunay(array).simplices.astype(np.int32)
        except QhullError as exc:
            raise RuntimeError(f"Delaunay failed on a degenerate point set ({exc}); raise --point-voxel")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, tets=tets, points=array)
    tets = torch.as_tensor(tets.astype(np.int64), device=device)
    stage("delaunay", 1, 1)
    emit("info", f"{tets.shape[0]} tetrahedra")
    timings["delaunay"] = time.time() - step

    step = time.time()
    stage("alpha", 0, len(contexts))
    alpha = evaluate_alpha(points, contexts, splats, args.fragment_depth, args.chunk_points,
                           progress=lambda done, total: stage("alpha", done, total))
    sdf = alpha - 0.5
    timings["alpha"] = time.time() - step
    emit("info", "alpha p10/p50/p90 {:.3f} {:.3f} {:.3f}".format(*np.percentile(
        alpha.cpu().numpy(), [10, 50, 90])))

    step = time.time()
    edge_a, edge_b, s_a, s_b, faces = marching_tetrahedra(points, tets, sdf, scales)
    emit("info", f"{edge_a.numel()} crossing edges, {faces.shape[0]} raw faces")
    if faces.shape[0] == 0:
        raise RuntimeError("no surface found: the alpha field never crosses 0.5; raise --point-voxel "
                           "or lower --min-opacity")
    distance = (points[edge_a] - points[edge_b]).norm(dim=1)
    edge_scale = scales[edge_a] + scales[edge_b]
    total_steps = max(0, args.binary_steps) if args.refine == "bisect" else 1
    stage("refine", 0, total_steps)
    done = [0]

    def sdf_of(sample):
        done[0] += 1
        stage("refine", done[0], total_steps)
        return evaluate_alpha(sample, contexts, splats, args.fragment_depth, args.chunk_points) - 0.5

    vertices = refine_vertices(points, edge_a, edge_b, s_a, s_b, sdf_of, args.refine, args.binary_steps)
    stage("refine", total_steps, total_steps)
    timings["refine"] = time.time() - step

    step = time.time()
    keep_vertex = span_keep(distance, edge_scale, args.span_mult)
    keep_face = keep_vertex[faces].all(dim=1)
    faces = faces[keep_face]
    if faces.shape[0] == 0:
        raise RuntimeError("the span filter removed every face; raise --point-voxel or --span-mult")
    used, remap = torch.unique(faces, return_inverse=True)
    vertices = vertices[used]
    faces = remap.reshape(-1, 3)
    stage("filter", 1, 1)
    emit("info", f"{faces.shape[0]} faces after the span filter ({int((~keep_vertex).sum())} vertices dropped)")
    timings["filter"] = time.time() - step

    step = time.time()
    raw_vertices = int(vertices.shape[0])
    raw_faces = int(faces.shape[0])
    vertices, faces, kept, total = sm.remove_small_components(vertices, faces, args.min_component_faces)
    normals, area = sm.vertex_normals(vertices, faces)
    stage("components", 1, 1)
    emit("info", f"{vertices.shape[0]} vertices, {faces.shape[0]} faces, {kept}/{total} components")
    timings["components"] = time.time() - step

    step = time.time()
    color_views = training[::args.color_every]
    kind = args.fragment_depth
    cache = out / "depth_cache" / f"{kind}_median"
    stage("color", 0, len(color_views))
    render_color_depths(splats, color_views, cameras, images_root, masks_root, cache, kind, device,
                        progress=lambda done, total: stage("color", done, total))
    colors, colored = color_vertices_gof(vertices, color_views, cameras, images_root, cache, device,
                                         args.min_alpha, args.color_tolerance)
    timings["color"] = time.time() - step

    step = time.time()
    mesh_path = out / "mesh.ply"
    sm.write_ply(mesh_path, vertices, normals, colors, faces)
    if args.preview:
        for image in images[:1]:
            camera = cameras[image["camera_id"]]
            lens = sv.Lens(camera, camera["width"], camera["height"], device)
            photo = sv.read_image_any(images_root / image["name"])
            sm.write_preview(out / f"preview_{image['index']:05d}.jpg", vertices, faces, colors, normals,
                             lens, lens.pixel_rays(), torch.as_tensor(image["R"], device=device, dtype=torch.float32),
                             torch.as_tensor(image["t"], device=device, dtype=torch.float32), photo)
    stage("write", 1, 1)
    timings["write"] = time.time() - step

    report = {
        "route": "gof-tetrahedra",
        "ply": str(Path(args.ply)),
        "dataset": str(root),
        "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
        "raw_vertices": raw_vertices, "raw_faces": raw_faces,
        "area_m2": float(area.sum()),
        "alpha_views": len(contexts), "color_views": len(color_views),
        "point_count": stats["point_count"], "splat_total": stats["splat_total"],
        "splat_kept": stats["splat_kept"],
        "edge_count": int(edge_a.numel()), "tetra_count": int(tets.shape[0]),
        "refine": args.refine, "binary_steps": int(args.binary_steps),
        "point_voxel": args.point_voxel, "scale_mult": args.scale_mult,
        "alpha_scale": args.alpha_scale, "fragment_depth": args.fragment_depth,
        "span_mult": args.span_mult, "components_kept": kept, "components_total": total,
        "colored_vertex_share": colored, "masks": bool(masks_root),
        "alpha_p10": float(np.percentile(alpha.cpu().numpy(), 10)),
        "alpha_p50": float(np.percentile(alpha.cpu().numpy(), 50)),
        "alpha_p90": float(np.percentile(alpha.cpu().numpy(), 90)),
        "seconds": timings, "seconds_total": time.time() - begin,
    }
    (out / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    emit("done", str(mesh_path))


def selftest():
    device = "cpu"
    points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tets = torch.tensor([[0, 1, 2, 3]])
    scale = torch.ones(4)

    edge_a, edge_b, _, _, faces = marching_tetrahedra(points, tets, torch.tensor([-1.0, 0.5, 0.5, 0.5]), scale)
    assert faces.shape == (1, 3), "1-3 tetrahedron must give one triangle"
    mid = 0.5 * (points[edge_a] + points[edge_b])
    normal = torch.cross(mid[faces[0, 1]] - mid[faces[0, 0]], mid[faces[0, 2]] - mid[faces[0, 0]])
    outward = torch.tensor([1.0, 0, 0]) + torch.tensor([0.0, 1, 0]) + torch.tensor([0.0, 0, 1]) - points[0]
    assert (normal * outward).sum() > 0, "the triangle must face the positive corners"

    edge_a, edge_b, _, _, faces = marching_tetrahedra(points, tets, torch.tensor([-1.0, -1.0, 0.5, 0.5]), scale)
    assert faces.shape == (2, 3), "2-2 tetrahedron must give two triangles"
    edges_of_faces = [tuple(sorted(edge)) for face in faces.tolist()
                      for edge in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2]))]
    shared = len(edges_of_faces) - len(set(edges_of_faces))
    assert shared == 1, "the two triangles must share exactly the diagonal edge"
    assert len(set(edges_of_faces)) == 5, "a split quad has four border edges and one diagonal"

    edge_a, edge_b, _, _, faces = marching_tetrahedra(points, tets, torch.tensor([0.5, 0.5, 0.5, 0.5]), scale)
    assert faces.shape[0] == 0, "a tetrahedron with no sign change must give no face"

    distance = torch.tensor([1.0, 0.5])
    scale = torch.tensor([0.1, 0.5])
    assert span_keep(distance, scale, 1.0).tolist() == [False, True], "long crossing edges are dropped"

    index = FragmentIndex.build(torch.tensor([0, 0]), torch.tensor([1.0, 2.0]),
                                torch.tensor([math.log(0.5), math.log(0.5)]), 1, 1)
    occlusion = index.opacity_before(torch.tensor([0, 0, 0]), torch.tensor([0.5, 1.5, 2.5]))
    assert np.allclose(occlusion.tolist(), [0.0, 0.5, 0.75]), "prefix sums must reproduce 1 - transmittance"

    points = np.zeros((9, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.001, 0.019, 9)
    deduped, _ = voxel_dedupe_first(points, np.ones(9, dtype=np.float32), 0.02)
    assert deduped.shape[0] == 1, "nine co-located corners must collapse to one tetra point"

    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", help="trained 3DGS PLY")
    parser.add_argument("--dataset", help="dataset root holding sparse/0, images and optionally masks")
    parser.add_argument("--out", help="output directory; receives mesh.ply and mesh_report.json")
    parser.add_argument("--sparse", help="override <dataset>/sparse/0")
    parser.add_argument("--images", help="override <dataset>/images")
    parser.add_argument("--masks", help="override <dataset>/masks; missing masks make every pixel valid")
    parser.add_argument("--every", type=int, default=2, help="use every Nth training view for alpha")
    parser.add_argument("--alpha-scale", type=float, default=0.5, help="resolution factor of the alpha field")
    parser.add_argument("--fragment-depth", default="center", choices=("center", "tilted"))
    parser.add_argument("--point-voxel", type=float, default=0.02, help="candidate point spacing in meters")
    parser.add_argument("--scale-mult", type=float, default=3.0, help="box corner offset as a multiple of the splat scale")
    parser.add_argument("--min-opacity", type=float, default=0.1, help="drop splats below this opacity")
    parser.add_argument("--max-points", type=int, default=3000000, help="refuse a Delaunay larger than this")
    parser.add_argument("--cells-cache", help="reuse/save the tetrahedralization as a .npz")
    parser.add_argument("--refine", default="bisect", choices=("bisect", "linear"))
    parser.add_argument("--binary-steps", type=int, default=8)
    parser.add_argument("--span-mult", type=float, default=1.0, help="sliver filter: |left-right| <= span_mult * (sa+sb)")
    parser.add_argument("--min-component-faces", type=int, default=100)
    parser.add_argument("--color-every", type=int, default=2)
    parser.add_argument("--color-tolerance", type=float, default=0.05)
    parser.add_argument("--min-alpha", type=float, default=0.5)
    parser.add_argument("--chunk-points", type=int, default=1000000)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.ply and args.dataset and args.out):
        parser.error("--ply, --dataset and --out are required")
    try:
        with torch.no_grad():
            build_mesh(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        emit("error", str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
