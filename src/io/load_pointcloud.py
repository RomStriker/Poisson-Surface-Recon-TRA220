import numpy as np
import open3d as o3d


def load_ply_points_normals(path):
    """
    Load point cloud with consistently oriented normals.

    Args:
        path: Path to PLY file
        radius: Radius for normal estimation
        max_nn: Maximum neighbors for normal estimation
        orient_normals: If True, ensure normals are consistently oriented
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(path)
    print(f"Loaded {len(pcd.points)} points")

    if not pcd.has_normals():
        pcd.estimate_normals()

    # It makes all normals consistent (either all inward OR all outward)
    pcd.orient_normals_consistent_tangent_plane(k=15)

    return np.asarray(pcd.points), np.asarray(pcd.normals)