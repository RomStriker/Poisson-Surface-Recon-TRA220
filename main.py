import sys
import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import open3d as o3d
from src.utils.visualization import show_mesh_comparison
from src.io.load_pointcloud import load_ply_points_normals
from src.poisson_builder.regular_grid_builder import RegularGridPoissonBuilder
from src.solvers.benchmark_solvers import SolverBenchmark


def trilinear_eval(builder, solution, point):
    p = (point - builder.bbox_min) / builder.voxel_size
    i0 = np.floor(p).astype(int)
    i0 = np.clip(i0, 0, builder.grid_res - 2)
    tx, ty, tz = p - i0

    val = 0.0
    for di in (0, 1):
        wx = (1 - tx) if di == 0 else tx
        for dj in (0, 1):
            wy = (1 - ty) if dj == 0 else ty
            for dk in (0, 1):
                wz = (1 - tz) if dk == 0 else tz
                gi = i0[0] + di
                gj = i0[1] + dj
                gk = i0[2] + dk
                idx = builder.get_grid_index(gi, gj, gk)
                val += solution[idx] * (wx * wy * wz)
    return val


def compute_iso_from_samples(builder, solution):
    """Compute iso-value using trilinear interpolation at the sample points."""

    sample_vals = np.array([trilinear_eval(builder, solution, p) for p in builder.points])
    return np.median(sample_vals)


def compute_iso_centroid_outside(builder, solution):
    def tri(point):
        return trilinear_eval(builder, solution, point)

    centroid = builder.points.mean(axis=0)
    outside = centroid + builder.bbox_size  # guaranteed outside domain

    cval = tri(centroid)
    oval = tri(outside)

    # iso halfway between "inside" and "far outside"
    return 0.5 * (cval + oval)


# Main execution
if __name__ == "__main__":
    # Configuration
    model = 'armadillo'
    grid_resolution = 32
    data_path = f"./data/{model}.ply"
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    A_path = output_dir / f'{model}_{str(grid_resolution)}_poisson_A.npz'
    b_path = output_dir / f'{model}_{str(grid_resolution)}_poisson_b.npy'
    metadata_path = output_dir / f'{model}_{str(grid_resolution)}_poisson_metadata.npy'
    mesh_path = output_dir / f'{model}_{str(grid_resolution)}_reconstructed.ply'

    reuse_system = False
    reuse_mesh = False
    visualize_normals = False  # Set to True to debug normal orientation

    print("=" * 70)
    print("POISSON SURFACE RECONSTRUCTION")
    print("=" * 70)

    # 1. Load and preprocess point cloud
    print("\n[1] Loading point cloud...")
    try:
        points, normals = load_ply_points_normals(data_path)
        print(f"  Loaded {len(points):,} points")
        print(f"  Bounding box: [{points.min(axis=0)}] to [{points.max(axis=0)}]")
    except FileNotFoundError:
        print(f" Error: Could not find file {data_path}")
        sys.exit(1)

    # Optional: Visualize normals for debugging
    if visualize_normals:
        print("\n  Visualizing normals for debugging...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        # Color by normal direction
        colors = np.abs(normals)  # RGB based on normal components
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # 2. Create Poisson reconstruction problem
    print("\n[2] Setting up Poisson reconstruction...")
    print(f"  Grid resolution: {grid_resolution}^3")

    builder = RegularGridPoissonBuilder(
        points=points,
        normals=normals,
        grid_resolution=grid_resolution,
        screen_weight=0.01,  # Small regularization
        bbox_expand=0.1  # Add 10% padding around points
    )

    # 3. Load or build linear system
    print("\n[3] Building linear system...")
    if all(p.exists() for p in [A_path, b_path, metadata_path]) and reuse_system:
        print("  Loading precomputed system from disk...")
        A = sp.load_npz(A_path)
        b = np.load(b_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        print(f"  Loaded: {A.shape[0]:,} unknowns, {A.nnz:,} non-zeros")
    else:
        print("  Computing FEM discretization...")
        A, b, metadata = builder.build_poisson_problem()
        print(f"  Built: {A.shape[0]:,} unknowns, {A.nnz:,} non-zeros")
        print(f"  Condition estimate: {metadata.get('condition_estimate', 'N/A')}")

        # Save for future use
        print("\n  Saving system to disk...")
        sp.save_npz(A_path, A)
        np.save(b_path, b)
        np.save(metadata_path, metadata)
        print(f"  Saved to {output_dir}/")

    # 4. Check for existing mesh (optional reuse)
    if mesh_path.exists() and reuse_mesh:
        print(f"\n[4] Loading existing mesh from {mesh_path}...")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        if len(mesh.vertices) > 0:
            print(f"  ✓ Loaded mesh with {len(mesh.vertices):,} vertices")

            # Create comparison visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.1, 0.3, 0.8])  # Blue
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.8, 0.3, 0.1])  # Orange

            print("\n  Visualizing existing reconstruction...")
            o3d.visualization.draw_geometries(
                [pcd, mesh],
                window_name='Existing Reconstruction',
                width=1200,
                height=800,
                mesh_show_wireframe=True
            )

            # Compare original and reconstructed mesh
            show_mesh_comparison(data_path, mesh)

            sys.exit(0)  # Exit after showing existing mesh

    # 5. Benchmark solvers
    print("\n[5] Running solver benchmark...")
    print("-" * 50)

    grid_shape = (builder.nx, builder.ny, builder.nz)
    benchmark = SolverBenchmark(grid_shape=grid_shape, tol=1e-6)
    results = benchmark.run_benchmark(A, b)
    # Save benchmark results
    benchmark_results_path = output_dir / f'{model}_{str(grid_resolution)}_benchmark_results.pkl'
    with open(benchmark_results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Benchmark results saved to {benchmark_results_path}")

    benchmark.print_summary()
    benchmark.plot_results()
    best_solver, best_solution = benchmark.get_best_solution()
    print(f"\n  Selected solver: {best_solver}")

    # 6. Extract surface mesh
    print("\n[6] Extracting surface mesh...")
    # Compute iso-value (choose method based on what works best)
    print("  Computing iso-value...")

    # Option 1: Centroid method (usually works well for closed surfaces)
    # iso_value = compute_iso_centroid_outside(builder, best_solution)
    # Option 2: Median of samples
    iso_value = compute_iso_from_samples(builder, best_solution)

    print(f"  Using iso-value: {iso_value:.6f}")
    # Extract mesh using marching cubes
    vertices, faces = builder.solution_to_mesh(best_solution, iso_value)
    if len(vertices) == 0:
        print(" Could not extract mesh with given iso-value")
        sys.exit(1)

    # 7. Save and visualize results
    print("\n[7] Saving and visualizing results...")

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Save mesh
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"  Mesh saved to {mesh_path}")

    # Create visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.1, 0.3, 0.8])  # Blue for point cloud
    mesh.paint_uniform_color([0.8, 0.3, 0.1])  # Orange for mesh
    mesh.compute_vertex_normals()

    print("\n" + "=" * 70)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"Original points: {len(points):,}")
    print(f"Reconstructed mesh: {len(vertices):,} vertices, {len(faces):,} faces")
    print(f"Best solver: {best_solver}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Show comparison
    print("\nVisualizing comparison (close window to exit)...")
    o3d.visualization.draw_geometries(
        [pcd, mesh],
        window_name=f'Poisson Reconstruction - {best_solver}',
        width=1200,
        height=800,
        mesh_show_wireframe=True,
        point_show_normal=False
    )

    # Compare original and reconstructed mesh
    show_mesh_comparison(data_path, mesh)
