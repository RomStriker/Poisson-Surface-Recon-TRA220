import copy
import open3d as o3d


def show_mesh_comparison(original_path, reconstructed_mesh):
    """
    Show side-by-side comparison of the original point cloud and reconstructed mesh.

    Args:
        original_path: path to original PLY point cloud
        reconstructed_mesh: open3d TriangleMesh
    """
    print("\n" + "="*70)
    print("MESH COMPARISON (Point Cloud vs. Surface)")
    print("="*70)

    # Load original as point cloud
    original_pcd = o3d.io.read_point_cloud(original_path)
    if not original_pcd.has_points():
        print("✗ Could not load original point cloud")
        return

    print(f"Original: {len(original_pcd.points):,} points")
    print(f"Reconstructed: {len(reconstructed_mesh.vertices):,} vertices, "
          f"{len(reconstructed_mesh.triangles):,} faces")

    # Create copies for visualization
    original_viz = copy.deepcopy(original_pcd)
    reconstructed_viz = copy.deepcopy(reconstructed_mesh)

    # Color for distinction
    original_viz.paint_uniform_color([0.1, 0.3, 0.8])      # Blue points
    reconstructed_viz.paint_uniform_color([0.8, 0.3, 0.1])  # Orange mesh

    # Compute bounding boxes for positioning
    bbox_original = original_viz.get_axis_aligned_bounding_box()
    bbox_reconstructed = reconstructed_viz.get_axis_aligned_bounding_box()
    bbox_size = max(bbox_original.get_extent().max(),
                    bbox_reconstructed.get_extent().max())

    # Offset reconstructed mesh to the right
    offset = bbox_size * 1.5
    reconstructed_viz.translate([offset, 0, 0])

    # Add a center coordinate frame for reference
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=bbox_size*0.2)
    axis.translate([offset/2, -bbox_size*0.3, 0])

    # Draw the geometries
    o3d.visualization.draw_geometries(
        [original_viz, reconstructed_viz, axis],
        window_name="Point Cloud vs Reconstructed Surface",
        width=1600,
        height=800,
        mesh_show_wireframe=True
    )