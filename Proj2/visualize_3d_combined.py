import numpy as np
import os
import plotly.graph_objects as go

def fit_plane(points):
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) == 0:
        return None
    a, b, c, d = *normal, -np.dot(normal, p1)
    return a, b, c, d

def ransac_plane(points, threshold=0.01, max_iterations=2000):
    n_points = points.shape[0]
    best_inliers = []
    best_plane = None
    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        sample = points[idx]
        plane = fit_plane(sample)
        if plane is None:
            continue
        distances = np.abs((points @ plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])
        inliers = np.where(distances < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane
    return best_plane, best_inliers

def load_and_transform_icosahedron(filename, scene_centroid, scene_R, scale=0.35):
    verts = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts)
    verts_centered = verts.copy() * scale
    verts_centered[:,2] += 4.0
    ico_vertices_scene = verts_centered @ scene_R.T + scene_centroid
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('f '):
                idx = [int(x)-1 for x in line.strip().split()[1:]]
                faces.append(idx)
    return ico_vertices_scene, faces

def visualize_combined(run_dir, icosahedron_path='icosahedron.txt'):
    colmap_dir = os.path.join('colmap', run_dir)
    out_dir = os.path.join('output', run_dir)
    pts = []
    with open(os.path.join(colmap_dir, 'points3D.txt'), 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pts = np.array(pts)
    np.random.seed(42)
    if len(pts) < 10:
        return
    plane, inliers_local = ransac_plane(pts, threshold=0.01, max_iterations=2000)
    if plane is None:
        return
    inliers = pts[inliers_local]
    centroid = np.mean(inliers, axis=0)
    normal = np.array(plane[:3])
    z_axis = normal / np.linalg.norm(normal)
    ref = np.array([1,0,0]) if abs(z_axis[0]) < 0.9 else np.array([0,1,0])
    x_axis = np.cross(ref, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    R_scene = np.stack([x_axis, y_axis, z_axis], axis=1)
    ico_scene, faces = load_and_transform_icosahedron(icosahedron_path, centroid, R_scene, scale=0.35)
    # Plotly traces
    all_points = go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.2),
        name='All Points'
    )
    inlier_points = go.Scatter3d(
        x=inliers[:,0], y=inliers[:,1], z=inliers[:,2],
        mode='markers',
        marker=dict(size=6, color='red', line=dict(width=2, color='black')),
        name='Inliers'
    )
    # Plane mesh
    a, b, c, d = plane
    plane_mesh = None
    if c != 0:
        xx, yy = np.meshgrid(
            np.linspace(np.min(inliers[:,0]), np.max(inliers[:,0]), 10),
            np.linspace(np.min(inliers[:,1]), np.max(inliers[:,1]), 10)
        )
        zz = (-a * xx - b * yy - d) / c
        plane_mesh = go.Surface(x=xx, y=yy, z=zz, opacity=0.7, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, name='Fitted Plane')
    # Icosahedron mesh
    ico_meshes = []
    for face in faces:
        tri = ico_scene[face]
        mesh = go.Mesh3d(
            x=tri[:,0], y=tri[:,1], z=tri[:,2],
            i=[0], j=[1], k=[2],
            color='lime', opacity=0.7, name='Icosahedron',
            flatshading=True, showscale=False
        )
        ico_meshes.append(mesh)
    data = [all_points, inlier_points]
    if plane_mesh:
        data.append(plane_mesh)
    data.extend(ico_meshes)
    layout = go.Layout(
        title=f'Run: {run_dir} - Inliers, Plane, Icosahedron',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        legend=dict(x=0.02, y=0.98)
    )
    fig = go.Figure(data=data, layout=layout)
    out_html = os.path.join(out_dir, f'combined_3d_{run_dir}.html')
    fig.write_html(out_html)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_dirs = sys.argv[1:]
    else:
        run_dirs = [f'run_{i}' for i in range(1,8)]
    for run_dir in run_dirs:
        visualize_combined(run_dir)
