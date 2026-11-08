"""Bounded repair of small planar holes; original vertex positions stay unchanged."""
from collections import Counter, defaultdict
import math
import numpy as np


def boundary_data(triangles):
    triangles = np.asarray(triangles, dtype=np.int64)
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    indices = np.flatnonzero(counts[inverse] == 1)
    return directed[indices], indices % len(triangles), unique[counts > 2]


def topology(triangles):
    boundary, _, nonmanifold = boundary_data(triangles)
    return {'boundary_edges': len(boundary), 'nonmanifold_edges': len(nonmanifold)}


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _triangulate(points):
    """Ear clip a simple loop; reject crossings and degenerate triangles."""
    extent = max(float(np.ptp(points, axis=0).max()), np.finfo(float).tiny)
    eps = extent * extent * 1e-12
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == (i + 1) % n or (j + 1) % n == i:
                continue
            c, d = points[j], points[(j + 1) % n]
            s = [_cross(b-a, c-a), _cross(b-a, d-a), _cross(d-c, a-c), _cross(d-c, b-c)]
            if s[0]*s[1] <= eps*eps and s[2]*s[3] <= eps*eps:
                if np.all(np.maximum(np.minimum(a,b), np.minimum(c,d)) <= np.minimum(np.maximum(a,b), np.maximum(c,d)) + math.sqrt(eps)):
                    return None
    area = sum(_cross(points[i], points[(i+1) % n]) for i in range(n))
    if abs(area) <= eps:
        return None
    sign = 1 if area > 0 else -1
    remaining, result = list(range(n)), []
    while len(remaining) > 3:
        found = False
        for k, current in enumerate(remaining):
            prev, nxt = remaining[k-1], remaining[(k+1) % len(remaining)]
            a,b,c = points[[prev,current,nxt]]
            if sign * _cross(b-a,c-a) <= eps:
                continue
            others = [i for i in remaining if i not in (prev,current,nxt)]
            if any(min(sign*_cross(b-a,points[i]-a), sign*_cross(c-b,points[i]-b), sign*_cross(a-c,points[i]-c)) >= -eps for i in others):
                continue
            result.append([prev,current,nxt]); remaining.pop(k); found = True; break
        if not found:
            return None
    if sign * _cross(points[remaining[1]]-points[remaining[0]],points[remaining[2]]-points[remaining[0]]) <= eps:
        return None
    result.append(remaining)
    return np.asarray(result, dtype=np.int64)


def repair_small_holes(vertices, triangles, support_tree, *, max_diameter,
                       max_planarity_error, max_support_distance, max_vertices=128,
                       max_added_triangles=None):
    """Patch only bounded, supported, planar loops with consistent face orientation.

    All distances use the units of vertices and support_tree. Patches are inferred
    geometry, not measured surface. A large outside boundary is never forced shut.
    """
    if not all(math.isfinite(x) and x > 0 for x in (max_diameter,max_planarity_error,max_support_distance)) or max_vertices < 3:
        raise ValueError('Repair distances must be finite and positive; max_vertices >= 3.')
    if support_tree is None:
        raise ValueError('Repair requires a source support tree.')
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if not len(triangles):
        raise ValueError('Cannot repair an empty mesh.')
    edges, faces, nonmanifold = boundary_data(triangles)
    outgoing, incoming = defaultdict(list), Counter()
    for edge, face in zip(edges, faces):
        a,b = map(int,edge); outgoing[a].append((b,int(face))); incoming[b] += 1
    bad = {v for v in set(outgoing) | set(incoming) if len(outgoing[v]) != 1 or incoming[v] != 1}
    bad.update(map(int,nonmanifold.ravel()))
    seen, patches, rejected = set(), [], Counter()
    existing = {tuple(sorted(map(int,t))) for t in triangles}
    max_distance, max_gap, added_area, loops, added_count = 0.,0.,0.,0,0
    normals = np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
    for start in outgoing:
        if start in seen or start in bad:
            continue
        ring, adjacent, current = [], [], start
        while current not in seen and current not in bad:
            seen.add(current); ring.append(current)
            current, face = outgoing[current][0]; adjacent.append(face)
        if current != start or len(ring) < 3 or current in bad:
            rejected['ambiguous_boundary'] += 1; continue
        loops += 1
        if len(ring) > max_vertices:
            rejected['too_many_vertices'] += 1; continue
        xyz = vertices[ring]
        diameter = float(np.linalg.norm(np.ptp(xyz,axis=0)))
        if diameter > max_diameter:
            rejected['too_large'] += 1; continue
        centre = xyz.mean(axis=0)
        _,_,basis = np.linalg.svd(xyz-centre, full_matrices=False)
        if np.max(np.abs((xyz-centre) @ basis[2])) > max_planarity_error:
            rejected['nonplanar'] += 1; continue
        ring = np.asarray(ring[::-1],dtype=np.int64)  # Oppose the old boundary edges.
        xyz = vertices[ring]
        local = _triangulate((xyz-centre) @ basis[:2].T)
        if local is None:
            rejected['invalid_polygon'] += 1; continue
        patch = ring[local]
        if any(tuple(sorted(map(int,t))) in existing for t in patch):
            rejected['duplicate_faces'] += 1; continue
        new_normals = np.cross(vertices[patch[:,1]]-vertices[patch[:,0]],vertices[patch[:,2]]-vertices[patch[:,0]])
        near = normals[np.unique(adjacent)]
        reference_normal = near.sum(axis=0)
        length = np.linalg.norm(reference_normal)
        if length < 1e-15:
            rejected['ambiguous_normals'] += 1; continue
        reference_normal /= length
        near_lengths, new_lengths = np.linalg.norm(near,axis=1), np.linalg.norm(new_normals,axis=1)
        if np.any(near_lengths <= 1e-15) or np.any(new_lengths <= 1e-15) or np.any((near @ reference_normal)/near_lengths < .707) or np.any((new_normals @ reference_normal)/new_lengths < .866):
            rejected['outside_border_or_curved_surface'] += 1; continue
        samples = vertices[patch]
        samples = np.concatenate([samples.mean(axis=1),(samples[:,0]+samples[:,1])*.5,(samples[:,1]+samples[:,2])*.5,(samples[:,2]+samples[:,0])*.5])
        distance = float(np.max(support_tree.query(samples, workers=1)[0]))
        if distance > max_support_distance:
            rejected['unsupported'] += 1; continue
        if max_added_triangles is not None and added_count + len(patch) > max_added_triangles:
            rejected['triangle_budget'] += 1; continue
        patches.append(patch); added_count += len(patch)
        existing.update(tuple(sorted(map(int,t))) for t in patch)
        max_distance = max(max_distance,distance); max_gap = max(max_gap,diameter)
        added_area += float(new_lengths.sum()*.5)
    output = np.concatenate([triangles,*patches]) if patches else triangles.copy()
    return output, {'closed_loops_examined':loops,'holes_filled':len(patches),
        'ambiguous_boundary_vertices':len(bad),'added_triangles':added_count,
        'inferred_patch_area':added_area,'max_filled_diameter':max_gap,
        'max_patch_sample_distance_to_source':max_distance,'rejected':dict(rejected),
        'before':topology(triangles),'after':topology(output),
        'note':'Only bounded supported planar holes are patched; inferred geometry is not ground truth.'}
