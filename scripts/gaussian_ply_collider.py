#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from collider_repair import repair_small_holes, topology


PRESETS = {
    "fast": dict(spacing_factor=2.0, max_depth=9, max_points=200000,
                 max_triangles=60000, error_voxels=1.5, samples=2000, steps=2),
    "balanced": dict(spacing_factor=1.25, max_depth=11, max_points=500000,
                     max_triangles=150000, error_voxels=0.75, samples=5000, steps=3),
    "high": dict(spacing_factor=0.75, max_depth=12, max_points=1000000,
                 max_triangles=300000, error_voxels=0.4, samples=10000, steps=4),
}


def measured_spacing(points):
    """Sample queries against the complete XYZ tree, not a sparse sample tree."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(points), min(len(points), 20000), replace=False)
    tree = cKDTree(points)
    distances = tree.query(points[indices], k=min(8, len(points)), workers=-1)[0]
    distances[distances <= np.finfo(float).eps * 32] = np.inf
    nearest = distances.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) < 10:
        raise ValueError("Too few distinct positions to estimate point spacing.")
    return {"median": float(np.median(nearest)),
            "p90": float(np.quantile(nearest, 0.9)), "queries": len(indices)}


def prepare_cloud(points, voxel_size, config, max_points, max_depth, outlier_std):
    """Choose relative resolution; coarsen auto voxels to a reconstruction cap."""
    spacing = measured_spacing(points)
    automatic = voxel_size is None
    if automatic:
        voxel_size = max(spacing["median"] * config["spacing_factor"],
                         1.05 / (2 ** max_depth))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    initial_voxel = voxel_size
    for attempt in range(12):
        cloud = source.voxel_down_sample(voxel_size)
        cloud.remove_duplicated_points()
        count = len(cloud.points)
        if count <= max_points:
            break
        if not automatic:
            raise ValueError(
                f"Explicit voxel leaves {count:,} points above --max-points "
                f"{max_points:,}. Increase --max-points or choose a larger voxel."
            )
        voxel_size *= min(2.5, max(1.1, math.sqrt(count / max_points) * 1.05))
    else:
        raise ValueError("Could not meet the point cap. Crop with --bounds.")
    del source
    if outlier_std > 0 and count > 25:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=outlier_std)
    if len(cloud.points) < 100:
        raise ValueError("Too few reconstruction points. Use a finer voxel / weaker filtering.")
    return cloud, voxel_size, {
        "voxel_mode": "auto" if automatic else "explicit",
        "initial_voxel_normalized": initial_voxel,
        "point_cap_adjustments": attempt,
        "input_spacing_normalized": spacing,
    }


def load_points(path, bounds, opacity_mode, min_opacity):
    """Read XYZ without treating placeholder PLY normals as surface normals."""
    ply = PlyData.read(str(path), mmap="c")
    vertex = ply["vertex"]
    if not {"x", "y", "z"}.issubset(vertex.data.dtype.names):
        raise ValueError(
            "The PLY vertex element needs explicit x, y and z. For a packed "
            "or compressed export, export/decompress to standard XYZ PLY first."
        )
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    keep = np.isfinite(points).all(axis=1)
    original_count = len(points)

    if bounds is not None:
        lower, upper = np.asarray(bounds, dtype=float).reshape(2, 3)
        if not np.isfinite(bounds).all() or np.any(upper <= lower):
            raise ValueError("Every bounds maximum must exceed its minimum.")
        keep &= np.all((points >= lower) & (points <= upper), axis=1)

    if opacity_mode != "ignore":
        if "opacity" not in vertex.data.dtype.names:
            raise ValueError("Opacity filtering requested, but opacity is missing.")
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        finite_opacity = np.isfinite(opacity)
        if opacity_mode == "logit":
            # sigmoid(logit) >= minimum, without overflow or a full sigmoid copy.
            threshold = math.log(min_opacity / (1.0 - min_opacity))
            keep &= finite_opacity & (opacity >= threshold)
        else:
            if np.any(finite_opacity & ((opacity < 0) | (opacity > 1))):
                raise ValueError("Alpha opacity must be in [0, 1]; try logit mode.")
            keep &= finite_opacity & (opacity >= min_opacity)

    if not keep.all():
        points = points[keep]
    if len(points) < 100:
        raise ValueError("Fewer than 100 points remain. Check bounds/filtering.")
    print(f"PLY points: {original_count:,}; retained: {len(points):,}", flush=True)
    return points


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("No triangles remain. Reduce trimming or inspect the input.")
    return mesh


def trim_unsupported(mesh, tree, distance, batch_size=50000):

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_distances = tree.query(vertices, workers=-1)[0]
    remove = np.any(vertex_distances[triangles] > distance, axis=1)

    for start in range(0, len(triangles), batch_size):
        stop = min(start + batch_size, len(triangles))
        xyz = vertices[triangles[start:stop]]
        samples = np.concatenate(
            [
                xyz.mean(axis=1),
                (xyz[:, 0] + xyz[:, 1]) * 0.5,
                (xyz[:, 1] + xyz[:, 2]) * 0.5,
                (xyz[:, 2] + xyz[:, 0]) * 0.5,
            ],
            axis=0,
        )
        distances = tree.query(samples, workers=-1)[0].reshape(4, -1)
        remove[start:stop] |= np.any(distances > distance, axis=0)

    removed = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    clean_mesh(mesh)
    print(f"Unsupported faces removed: {removed:,}", flush=True)
    return mesh


def sampled_deviation(reference, collider, samples):
    if samples == 0:
        return None
    result = {}
    o3d.utility.random.seed(42)
    for label, source, destination in (
        ("reference_to_collider", reference, collider),
        ("collider_to_reference", collider, reference),
    ):
        sampled_points = np.asarray(source.sample_points_uniformly(samples).points)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(destination))
        distances = scene.compute_distance(
            o3d.core.Tensor(sampled_points.astype(np.float32))
        ).numpy()
        result[label] = {
            "samples": samples,
            "median": float(np.median(distances)),
            "p95": float(np.quantile(distances, 0.95)),
            "p99": float(np.quantile(distances, 0.99)),
            "sampled_max": float(distances.max()),
        }
        del scene
    return result


def rescale_deviation(deviation, factor):
    if deviation is None:
        return None
    return {label: {key: value if key == "samples" else value * factor
                    for key, value in stats.items()}
            for label, stats in deviation.items()}


def within_tolerance(deviation, tolerance):
    return bool(deviation) and all(
        stats["p95"] <= tolerance and stats["p99"] <= 2 * tolerance
        and stats["sampled_max"] <= 5 * tolerance
        for stats in deviation.values()
    )


def choose_collider(reference, tree, support_distance, target, cap, tolerance,
                    samples, steps, repair_callback=None):
    first_target = min(len(reference.triangles), target if target is not None else cap)
    selected, selected_deviation, selected_target = None, None, None
    trials = []
    for attempt in range(1 if target is not None else steps + 1):
        requested = max(4, first_target // (2 ** attempt))
        print(f"Trying collider budget {requested:,} triangles...", flush=True)
        if requested >= len(reference.triangles):
            candidate = copy.deepcopy(reference)
        else:
            candidate = reference.simplify_quadric_decimation(
                target_number_of_triangles=requested, boundary_weight=3.0)
            clean_mesh(candidate)
            try:
                trim_unsupported(candidate, tree, support_distance)
            except ValueError:
                trials.append({"target": requested, "status": "empty_after_trimming"})
                break
        repair_stats = repair_callback(candidate) if repair_callback else None
        deviation = sampled_deviation(reference, candidate, samples)
        passed = within_tolerance(deviation, tolerance) if samples else None
        trials.append({"target": requested, "triangles": len(candidate.triangles),
                       "tolerance_met": passed, "deviation_normalized": deviation, "hole_repair": repair_stats})
        if passed or samples == 0:
            selected, selected_deviation, selected_target = candidate, deviation, requested
        if target is not None or not passed or requested == 4:
            break
    if selected is None:
        # Preserve accuracy if no candidate meets the measured tolerance. Report a
        # budget miss explicitly instead of silently accepting an inaccurate mesh.
        selected = copy.deepcopy(reference)
        selected_deviation = sampled_deviation(reference, selected, samples)
        selected_target = len(selected.triangles)
        trials.append({"status": "reference_fallback_no_candidate_met_tolerance"})
    return selected, selected_deviation, selected_target, trials


def save_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".obj":
        # Preserve small features even when original coordinates have a large offset.
        vertices, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
        with path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("# Adaptive Gaussian/point PLY collider; original coordinate frame\n")
            for start in range(0, len(vertices), 10000):
                stream.writelines(f"v {x:.17g} {y:.17g} {z:.17g}\n"
                                  for x, y, z in vertices[start:start + 10000])
            has_normals = len(normals) == len(vertices)
            if has_normals:
                for start in range(0, len(normals), 10000):
                    stream.writelines(f"vn {x:.9g} {y:.9g} {z:.9g}\n"
                                      for x, y, z in normals[start:start + 10000])
            for start in range(0, len(mesh.triangles), 10000):
                faces = np.asarray(mesh.triangles)[start:start + 10000] + 1
                if has_normals:
                    stream.writelines(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in faces)
                else:
                    stream.writelines(f"f {a} {b} {c}\n" for a, b, c in faces)
        return
    if not o3d.io.write_triangle_mesh(
        str(path), mesh, write_vertex_colors=False, write_triangle_uvs=False
    ):
        raise OSError(f"Could not write mesh: {path}")


def create_collider_from_ply(
    input_ply_path,
    output_obj_path,
    voxel_size=None,
    poisson_depth=None,
    target_triangles=None,
    *,
    bounds=None,
    opacity_mode="ignore",
    min_opacity=0.05,
    outlier_std=2.5,
    density_quantile=0.0,
    support_distance=None,
    normal_radius=None,
    normal_direction=None,
    reference_path=None,
    flip_winding=False,
    error_samples=None,
    threads=-1,
    metres_per_unit=None,
    quality="balanced",
    max_points=None,
    max_depth=None,
    max_triangles=None,
    simplification_tolerance=None,
    analyze_only=False,
    repair_holes=True,
    max_hole_diameter=None,
    max_hole_planarity_error=None,
    max_hole_vertices=128,
):
    start_time = time.perf_counter()
    input_path, output_path = Path(input_ply_path), Path(output_obj_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() != ".ply":
        raise ValueError("Export this format to a standard XYZ .ply first.")
    if quality not in PRESETS:
        raise ValueError("Quality must be fast, balanced or high.")
    config = PRESETS[quality]
    max_points = config["max_points"] if max_points is None else max_points
    max_depth = config["max_depth"] if max_depth is None else max_depth
    max_triangles = config["max_triangles"] if max_triangles is None else max_triangles
    error_samples = config["samples"] if error_samples is None else error_samples
    if output_path.suffix.lower() != ".obj":
        raise ValueError("Use an .obj filename for the Unity collider.")
    output_paths = [output_path, output_path.with_suffix(".report.json")]
    if reference_path:
        if Path(reference_path).suffix.lower() != ".ply":
            raise ValueError("Use a .ply filename for the optional reference mesh.")
        output_paths.append(Path(reference_path))
    resolved_paths = [input_path.resolve()] + [p.resolve() for p in output_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Input, collider, report and reference paths must differ.")
    numeric = [min_opacity, outlier_std, density_quantile]
    numeric += [x for x in [voxel_size, support_distance, normal_radius,
                           metres_per_unit, simplification_tolerance, max_hole_diameter, max_hole_planarity_error] if x is not None]
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric settings must be finite.")
    if (voxel_size is not None and voxel_size <= 0) or (
        target_triangles is not None and target_triangles < 4
    ) or (poisson_depth is not None and not 5 <= poisson_depth <= 14):
        raise ValueError("Use voxel > 0, triangles >= 4 and depth between 5 and 14.")
    if max_points < 100 or max_triangles < 4 or not 5 <= max_depth <= 14:
        raise ValueError("Use max-points >= 100, max-triangles >= 4, max-depth in [5,14].")
    if not 0 <= density_quantile < 1 or not 0 < min_opacity < 1:
        raise ValueError("Use density quantile in [0, 1) and opacity in (0, 1).")
    if outlier_std < 0 or error_samples < 0 or threads == 0 or threads < -1:
        raise ValueError("Invalid outlier, error-sample or thread setting.")
    if any(x is not None and x <= 0 for x in [support_distance, normal_radius,
                                           metres_per_unit, simplification_tolerance, max_hole_diameter, max_hole_planarity_error]):
        raise ValueError("Optional distances must be positive.")
    if error_samples == 0 and target_triangles is None and not analyze_only:
        raise ValueError("Automatic simplification needs error samples; use --triangles for a fixed budget.")
    if max_hole_vertices < 3:
        raise ValueError("Max hole vertices must be at least 3.")
    if opacity_mode not in {"ignore", "alpha", "logit"}:
        raise ValueError("Opacity mode must be ignore, alpha or logit.")

    print(f"Reading {input_path.name} ({input_path.stat().st_size / 1024**2:.1f} MiB)...", flush=True)
    points = load_points(input_path, bounds, opacity_mode, min_opacity)
    input_count = len(points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    normalization_scale = float(np.max(upper - lower))
    if normalization_scale <= 0 or not math.isfinite(normalization_scale):
        raise ValueError("Input positions have no usable spatial extent.")
    offset = lower + (upper - lower) * 0.5
    # Normalize every scene to the same scale for stable, unit-independent math.
    points -= offset
    points /= normalization_scale
    explicit_voxel = voxel_size
    normalized_voxel = voxel_size / normalization_scale if voxel_size is not None else None
    pcd, voxel_size, adaptation = prepare_cloud(
        points, normalized_voxel, config, max_points, max_depth, outlier_std)
    del points
    print(f"Reconstruction points: {len(pcd.points):,}", flush=True)

    nearest = np.asarray(pcd.compute_nearest_neighbor_distance())
    p90_spacing = float(np.quantile(nearest, 0.90))
    if normal_radius is None:
        normal_radius = max(4 * voxel_size, 3 * p90_spacing)
    else:
        normal_radius /= normalization_scale
    if support_distance is None:
        support_distance = 3 * max(voxel_size, p90_spacing)
    else:
        support_distance /= normalization_scale
    span = float(np.max(pcd.get_max_bound() - pcd.get_min_bound()))
    suggested_depth = max(5, math.ceil(math.log2(1.05 * span / voxel_size)))
    depth_mode = "auto" if poisson_depth is None else "explicit"
    if poisson_depth is None:
        poisson_depth = min(max_depth, suggested_depth)
    cell_width = 1.05 * span / (2**poisson_depth)
    tolerance = (voxel_size * config["error_voxels"] if simplification_tolerance is None
                 else simplification_tolerance / normalization_scale)
    print(f"Quality: {quality}; voxel mode: {adaptation['voxel_mode']}")
    print(f"Voxel: {voxel_size * normalization_scale:.6g} input units; depth: {poisson_depth}")
    print(f"Longest scene extent: {span * normalization_scale:.6g} input units", flush=True)
    print(f"Finest Poisson cell ~ {cell_width * normalization_scale:.6g} input units", flush=True)
    print(f"Normal radius: {normal_radius * normalization_scale:.6g}; support distance: {support_distance * normalization_scale:.6g}")
    if metres_per_unit is not None:
        print(
            f"Metric voxel: {voxel_size * normalization_scale * metres_per_unit:.4g} m; "
            f"finest Poisson cell ~ {cell_width * normalization_scale * metres_per_unit:.4g} m",
            flush=True,
        )

    hole_diameter = (4 * voxel_size if max_hole_diameter is None
                     else max_hole_diameter / normalization_scale)
    hole_planarity = (0.25 * voxel_size if max_hole_planarity_error is None
                     else max_hole_planarity_error / normalization_scale)
    if metres_per_unit is not None:
        if max_hole_diameter is None:
            hole_diameter = min(hole_diameter, 0.5 / metres_per_unit / normalization_scale)
        if max_hole_planarity_error is None:
            hole_planarity = min(hole_planarity, 0.05 / metres_per_unit / normalization_scale)

    report = {
        "input": str(input_path), "output": str(output_path),
        "input_format": "standard_xyz_ply", "quality_preset": quality,
        "units": "Input coordinate units; metres cannot be inferred from XYZ alone",
        "metres_per_unit": metres_per_unit,
        "coordinate_frame": "Original input coordinates, axis order and scale preserved",
        "normalization_center": offset.tolist(),
        "normalization_scale": normalization_scale,
        "input_bounds": [lower.tolist(), upper.tolist()],
        "retained_input_point_count": input_count,
        "method": "Poisson on Gaussian centers / point samples; approximate surface",
        "voxel_size": voxel_size * normalization_scale,
        "explicit_voxel_size": explicit_voxel,
        "poisson_depth": poisson_depth, "depth_mode": depth_mode,
        "finest_poisson_cell_width_estimate": cell_width * normalization_scale,
        "normal_radius": normal_radius * normalization_scale,
        "normal_direction": list(normal_direction) if normal_direction is not None else None,
        "support_distance": support_distance * normalization_scale,
        "opacity_mode": opacity_mode, "min_opacity": min_opacity,
        "outlier_std": outlier_std, "density_quantile": density_quantile,
        "bounds": list(bounds) if bounds is not None else None,
        "flip_winding": flip_winding, "adaptation": adaptation,
        "reconstruction_point_count": len(pcd.points),
        "limits": {"max_reconstruction_points": max_points,
                   "max_auto_depth": max_depth, "max_auto_triangles": max_triangles},
        "simplification_mode": "auto" if target_triangles is None else "explicit",
        "requested_triangle_count": target_triangles,
        "simplification_tolerance": tolerance * normalization_scale,
        "tolerance_note": "Sampled p95 target; p99 <= 2x and sampled max <= 5x. Not a guaranteed bound.",
        "error_note": "Deviation from reconstructed reference, not scene ground truth.",
        "analyze_only": analyze_only,
        "hole_repair_settings": {
            "enabled": repair_holes,
            "max_diameter_input_units": hole_diameter * normalization_scale,
            "max_planarity_error_input_units": hole_planarity * normalization_scale,
            "max_vertices": max_hole_vertices,
            "source_support_distance_input_units": support_distance * normalization_scale,
            "note": "Repairs only small supported planar loops; never guarantees a watertight mesh.",
        },
    }
    report_path = output_path.with_suffix(".report.json")
    if analyze_only:
        report["elapsed_seconds"] = time.perf_counter() - start_time
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Analysis only. Settings report: {report_path}", flush=True)
        return None, report
    if cell_width > voxel_size:
        print(
            f"NOTE: depth {suggested_depth} would be needed for a finest cell "
            "as small as the voxel. Prefer a smaller --bounds region before "
            "raising depth; depth can substantially increase RAM/time.",
            flush=True,
        )

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
    )
    pcd.normalize_normals()
    if normal_direction is not None:
        direction = np.asarray(normal_direction, dtype=float)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError("Normal direction must be a finite nonzero vector.")
        # Useful for a ground-only crop; inappropriate for a full building scene.
        pcd.orient_normals_to_align_with_direction(direction / np.linalg.norm(direction))
        report["normal_orientation"] = "explicit_direction"
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
            report["normal_orientation"] = "consistent_tangent_plane"
        except RuntimeError as exc:
            _, covariance = pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if eigenvalues[0] <= eigenvalues[-1] * 1e-8:
                direction = eigenvectors[:, 0]
                if direction[np.argmax(np.abs(direction))] < 0:
                    direction = -direction
                pcd.orient_normals_to_align_with_direction(direction)
                report["normal_orientation"] = "planar_pca_arbitrary_sign"
                print("Planar cloud: normal sign is arbitrary; inspect facing on import.", flush=True)
            else:
                raise RuntimeError(
                    "Normal orientation failed. Inspect positions or crop the cloud. "
                    "For a planar crop, --normal-direction can specify its facing."
                ) from exc

    print("Reconstructing Poisson surface...", flush=True)
    reference, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.05, linear_fit=False, n_threads=threads
    )
    raw_triangles = len(reference.triangles)
    densities = np.asarray(densities)
    if density_quantile > 0:
        reference.remove_vertices_by_mask(
            densities < np.quantile(densities, density_quantile)
        )
    box = o3d.geometry.AxisAlignedBoundingBox(
        pcd.get_min_bound() - support_distance,
        pcd.get_max_bound() + support_distance,
    )
    reference = reference.crop(box)
    clean_mesh(reference)
    tree = cKDTree(np.asarray(pcd.points))
    reference = trim_unsupported(reference, tree, support_distance)
    repair_records = []
    def repair(mesh):
        if not repair_holes:
            return None
        triangles, stats = repair_small_holes(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles), tree,
            max_diameter=hole_diameter, max_planarity_error=hole_planarity,
            max_support_distance=support_distance, max_vertices=max_hole_vertices,
            max_added_triangles=max(0, max_triangles - len(mesh.triangles)),
        )
        if stats["added_triangles"]:
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
        stats["coordinate_units"] = "normalized reconstruction units"
        repair_records.append(stats)
        return stats
    repair(reference)
    print(f"Trimmed reference triangles: {len(reference.triangles):,}", flush=True)

    collider, normalized_deviation, selected_target, trials = choose_collider(
        reference, tree, support_distance, target_triangles, max_triangles,
        tolerance, error_samples, config["steps"], repair_callback=repair)
    deviation = rescale_deviation(normalized_deviation, normalization_scale)
    triangles = np.asarray(collider.triangles)
    edges = np.sort(np.concatenate([
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]
    ]), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    topology = {
        "boundary_edges": int(np.sum(incidence == 1)),
        "nonmanifold_edges": int(np.sum(incidence > 2)),
    }
    tolerance_met = within_tolerance(normalized_deviation, tolerance) if error_samples else None
    report.update({
        "raw_triangle_count": raw_triangles,
        "reference_triangle_count": len(reference.triangles),
        "selected_triangle_target": selected_target,
        "collider_triangle_count": len(collider.triangles),
        "topology": topology,
        "sampled_simplification_deviation": deviation,
        "simplification_tolerance_met": tolerance_met,
        "simplification_trials": trials,
        "hole_repair_passes": repair_records,
        "triangle_budget_met": len(collider.triangles) <= (target_triangles if target_triangles is not None else max_triangles),
    })
    if metres_per_unit is not None and deviation:
        report["sampled_simplification_deviation_m"] = rescale_deviation(deviation, metres_per_unit)

    for mesh in (reference, collider):
        if flip_winding:
            mesh.triangles = o3d.utility.Vector3iVector(
                np.asarray(mesh.triangles)[:, ::-1].copy()
            )
        mesh.scale(normalization_scale, center=np.zeros(3))
        mesh.translate(offset)
        mesh.compute_vertex_normals()
    save_mesh(output_path, collider)
    if reference_path:
        save_mesh(reference_path, reference)
    report["elapsed_seconds"] = time.perf_counter() - start_time
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Collider: {output_path} ({len(collider.triangles):,} triangles)")
    print(f"Report: {report_path}")
    print(f"Boundary edges: {topology['boundary_edges']:,}; non-manifold edges: {topology['nonmanifold_edges']:,}")
    if tolerance_met is False:
        print("Sampled deviation target was not met at this budget. Increase --max-triangles or use --triangles.")
    if deviation:
        print("Sampled simplification deviation, in PLY units:")
        for label, stats in deviation.items():
            print(f"  {label}: p95={stats['p95']:.6g}, max={stats['sampled_max']:.6g}")
        if metres_per_unit is not None:
            print("Sampled simplification deviation, in metres:")
            for label, stats in report["sampled_simplification_deviation_m"].items():
                print(f"  {label}: p95={stats['p95']:.6g}, max={stats['sampled_max']:.6g}")
    print(f"Finished in {report['elapsed_seconds']:.2f} seconds", flush=True)
    return collider, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Standard XYZ point / Gaussian-splat PLY")
    parser.add_argument("output", help="Unity collider OBJ")
    parser.add_argument("--quality", choices=PRESETS, default="balanced")
    parser.add_argument("--analyze-only", action="store_true", help="Write a small settings report; skip reconstruction")
    voxel_group = parser.add_mutually_exclusive_group()
    voxel_group.add_argument("--voxel", type=float, help="Override automatic voxel width, in PLY units")
    voxel_group.add_argument("--voxel-m", type=float, help="Voxel width in metres; requires a scale below")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--transform-json", help="Read colmap_to_enu.scale; output stays in the PLY frame")
    scale_group.add_argument("--metres-per-unit", type=float, help="Alternative to reading a transform JSON")
    parser.add_argument("--depth", type=int, help="Override automatic Poisson depth")
    parser.add_argument("--triangles", type=int, help="Fixed budget; default searches smaller meshes using sampled deviation")
    parser.add_argument("--max-depth", type=int, help="Ceiling for automatically selected depth")
    parser.add_argument("--max-points", type=int, help="Reconstruction point cap; not a total RAM limit")
    parser.add_argument("--max-triangles", type=int, help="Ceiling for automatic triangle budgets")
    parser.add_argument("--simplification-tolerance", type=float, help="Sampled p95 target in input units; default relative to voxel")
    parser.add_argument("--bounds", type=float, nargs=6)
    parser.add_argument("--opacity-mode", choices=["ignore", "logit", "alpha"], default="ignore")
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--outlier-std", type=float, default=2.5, help="0 disables statistical filtering")
    parser.add_argument("--density-quantile", type=float, default=0.0, help="Optional; 0 avoids deleting a fixed fraction of supported surface")
    parser.add_argument("--support-distance", type=float)
    parser.add_argument("--normal-radius", type=float)
    parser.add_argument("--normal-direction", type=float, nargs=3, help="Only for a ground-only/planar crop")
    parser.add_argument("--no-hole-repair", action="store_true", help="Disable conservative small-hole repair")
    parser.add_argument("--max-hole-diameter", type=float, help="Maximum repaired loop bounding diagonal, in PLY units")
    parser.add_argument("--max-hole-planarity-error", type=float, help="Maximum plane deviation, in PLY units")
    parser.add_argument("--max-hole-diameter-m", type=float, help="Metric hole limit; requires a scale")
    parser.add_argument("--max-hole-planarity-error-m", type=float, help="Metric plane error limit; requires a scale")
    parser.add_argument("--max-hole-vertices", type=int, default=128)
    parser.add_argument("--reference", help="Optional reference mesh PLY before simplification")
    parser.add_argument("--flip-winding", action="store_true")
    parser.add_argument("--error-samples", type=int, help="Default set by quality; 0 requires a fixed --triangles budget")
    parser.add_argument("--threads", type=int, default=-1, help="Poisson threads; -1 = automatic")
    args = parser.parse_args()
    try:
        metres_per_unit = args.metres_per_unit
        if args.transform_json:
            transform = json.loads(Path(args.transform_json).read_text(encoding="utf-8-sig"))
            metres_per_unit = float(transform["colmap_to_enu"]["scale"])
        if metres_per_unit is not None and (
            not math.isfinite(metres_per_unit) or metres_per_unit <= 0
        ):
            raise ValueError("Metres per PLY unit must be finite and positive.")
        voxel = args.voxel
        if args.voxel_m is not None:
            if metres_per_unit is None:
                raise ValueError("--voxel-m requires --transform-json or --metres-per-unit.")
            voxel = args.voxel_m / metres_per_unit
        hole_diameter, hole_planarity = args.max_hole_diameter, args.max_hole_planarity_error
        for metric, original in [(args.max_hole_diameter_m,hole_diameter),(args.max_hole_planarity_error_m,hole_planarity)]:
            if metric is not None and (original is not None or metres_per_unit is None):
                raise ValueError("Use either metric or PLY-unit hole limits; metric limits require a scale.")
        if args.max_hole_diameter_m is not None:
            hole_diameter = args.max_hole_diameter_m / metres_per_unit
        if args.max_hole_planarity_error_m is not None:
            hole_planarity = args.max_hole_planarity_error_m / metres_per_unit
        create_collider_from_ply(
            args.input, args.output, voxel, args.depth, args.triangles,
            bounds=args.bounds, opacity_mode=args.opacity_mode,
            min_opacity=args.min_opacity, outlier_std=args.outlier_std,
            density_quantile=args.density_quantile, support_distance=args.support_distance,
            normal_radius=args.normal_radius, normal_direction=args.normal_direction,
            reference_path=args.reference, flip_winding=args.flip_winding,
            error_samples=args.error_samples, threads=args.threads,
            metres_per_unit=metres_per_unit,
            quality=args.quality, max_points=args.max_points, max_depth=args.max_depth,
            max_triangles=args.max_triangles, simplification_tolerance=args.simplification_tolerance,
            analyze_only=args.analyze_only,
            repair_holes=not args.no_hole_repair,
            max_hole_diameter=hole_diameter, max_hole_planarity_error=hole_planarity,
            max_hole_vertices=args.max_hole_vertices,
        )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
