import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ransac_plane import ransac_plane

def load_colmap_points(filename):
    """
    Load COLMAP points3D.txt file, return Nx3 numpy array of 3D points.
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            try:
                x, y, z = map(float, tokens[1:4])
                points.append([x, y, z])
            except Exception:
                continue
    return np.array(points)

if __name__ == "__main__":
    # Load COLMAP 3D points
    points = load_colmap_points("../Proj2/points3D.txt")
    print(f"Loaded {points.shape[0]} points from COLMAP.")
    # Run RANSAC
    best_plane, best_inliers = ransac_plane(points, threshold=0.05, max_iterations=2000)
    print("Best plane:", best_plane)
    print("Number of inliers:", len(best_inliers))

    # Plot point cloud and inliers
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=5, label='All points', alpha=0.5)
    if len(best_inliers) > 0:
        inlier_pts = points[best_inliers]
        ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], c='red', s=10, label='Inliers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('COLMAP 3D Points with RANSAC Plane Inliers')
    plt.show()
