import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


def fit_plane(points):
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    if np.linalg.norm(n) == 0:
        return None
    a, b, c = n
    d = -np.dot(n, p1)
    return a, b, c, d


def point_to_plane_distance(point, plane):
    a, b, c, d = plane
    x, y, z = point
    return abs(a * x + b * y + c * z + d) / np.linalg.norm([a, b, c])


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

def local_coordinate(inlier_points, plane):
    centroid = np.mean(inlier_points, axis=0)
    normal = np.array(plane[:3])
    z_axis = normal / np.linalg.norm(normal)

    arbitrary_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(arbitrary_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    local_points = (inlier_points - centroid) @ R
    global_points = local_points @ R.T + centroid
    return local_points, global_points


def load_and_transform_icosahedron(filename, scene_centroid, scene_R, scale=1.0, anchor='face_center'):
    verts = []
    verts = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts)
    z0 = np.isclose(verts[:,2], 0)
    z0 = np.isclose(verts[:,2], 0)
    bottom_indices = np.where(z0)[0]
    bottom_face_verts = verts[bottom_indices]
    bottom_face_verts = verts[bottom_indices]
    if anchor == 'face_center':
        anchor_point = np.mean(bottom_face_verts, axis=0)
    else:
        anchor_point = bottom_face_verts[0]
    verts_centered = (verts - anchor_point) * scale
    verts_centered = (verts - anchor_point) * scale
    ico_vertices_scene = verts_centered @ scene_R.T + scene_centroid
    ico_vertices_scene = verts_centered @ scene_R.T + scene_centroid
    return ico_vertices_scene


def parse_colmap_camera(filename):
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            fx, cx, cy = float(parts[4]), float(parts[5]), float(parts[6])
            w, h = int(parts[2]), int(parts[3])
            return {'fx': fx, 'cx': cx, 'cy': cy, 'w': w, 'h': h}
    return None

def parse_colmap_images(filename):
    images = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 9 or not parts[-1].lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            q = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            t = [float(parts[5]), float(parts[6]), float(parts[7])]
            image_name = parts[-1]
            images.append({'q': q, 't': t, 'image_name': image_name})
    return images

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])
    return R

def project_points_pinhole(XYZ, K, R, t):
    XYZ_cam = (R @ XYZ.T + t.reshape(3,1)).T
    x = XYZ_cam[:,0] / XYZ_cam[:,2]
    y = XYZ_cam[:,1] / XYZ_cam[:,2]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]
    return np.stack([u, v], axis=1)

def overlay_mesh_on_image(image, verts2d, faces, depths, color=(0,255,0), alpha=0.5):
    import cv2
    img = image.copy()
    order = np.argsort(depths)[::-1]
    for idx in order:
        pts = verts2d[faces[idx]].astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return img

def save_images_as_video(image_paths, output_path, fps=10):
    import cv2
    img = cv2.imread(image_paths[0])
    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            out.write(img)
    out.release()

# --- MAIN PIPELINE EXAMPLE ---


def run_ar_pipeline_video():
    import glob
    pts = []
    with open('points3D.txt', 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pts = np.array(pts)
    plane, inlier_idx = ransac_plane(pts)
    inliers = pts[inlier_idx]
    centroid = np.mean(inliers, axis=0)
    normal = np.array(plane[:3])

    z_axis = normal / np.linalg.norm(normal)
    ref = np.array([1,0,0]) if abs(z_axis[0]) < 0.9 else np.array([0,1,0])
    x_axis = np.cross(ref, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    R_scene = np.stack([x_axis, y_axis, z_axis], axis=1)
    ico_scene = load_and_transform_icosahedron('icosahedron.txt', centroid, R_scene, scale=0.5, anchor='face_center')
    faces = []
    with open('icosahedron.txt', 'r') as f:
        for line in f:
            if line.startswith('f '):
                idx = [int(x)-1 for x in line.strip().split()[1:]]
                faces.append(idx)
    faces = np.array(faces)
    cam = parse_colmap_camera('cameras.txt')
    images = parse_colmap_images('images.txt')
    K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fx'], cam['cy']], [0,0,1]])
    overlay_paths = []
    for img_info in images:
        q = img_info['q']
        t = np.array(img_info['t'])
        R = quaternion_to_rotation_matrix(q)
        img_path = f"Images/{img_info['image_name']}"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue
        verts2d = project_points_pinhole(ico_scene, K, R, t)
        ico_cam = (R @ ico_scene.T + t.reshape(3,1)).T
        depths = np.min(ico_cam[faces,2], axis=1)
        img_overlay = overlay_mesh_on_image(img, verts2d, faces, depths, color=(0,255,0), alpha=0.5)
        out_path = f"overlay_{img_info['image_name']}"
        cv2.imwrite(out_path, img_overlay)
        overlay_paths.append(out_path)
    save_images_as_video(overlay_paths, 'output.mp4', fps=5)

if __name__ == "__main__":
    run_ar_pipeline_video()