import numpy as np
import open3d as o3d


def load_ply_points_normals(path, radius=0.05, max_nn=30):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(path)

    # If normals are missing, estimate them
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )

    # Convert to numpy arrays
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    return points, normals
