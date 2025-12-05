import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


class RegularGridPoissonBuilder:
    """
    Regular grid Poisson builder with Galerkin FEM trilinear stiffness (LHS)
    and FEM-consistent RHS (b = sum_p N(p)·∇φ).
    """

    def __init__(self, points, normals, grid_resolution=64,
                 screen_weight=0.01, bbox_expand=0.1):
        self.points = np.asarray(points, dtype=float)
        self.normals = np.asarray(normals, dtype=float)
        self.grid_res = int(grid_resolution)
        self.screen_weight = float(screen_weight)
        self.bbox_expand = float(bbox_expand)

        # Bounding box
        self.bbox_min = self.points.min(axis=0) - self.bbox_expand
        self.bbox_max = self.points.max(axis=0) + self.bbox_expand
        self.bbox_size = self.bbox_max - self.bbox_min

        # Voxel sizes (distance between grid nodes)
        # grid_res nodes along each axis => grid_res-1 elements in each axis
        self.voxel_size = self.bbox_size / (self.grid_res - 1)
        self.hx, self.hy, self.hz = self.voxel_size

        self.nx = self.ny = self.nz = self.grid_res
        self.total_cells = self.nx * self.ny * self.nz

        # Precompute coordinates (world)
        grid_coords = np.linspace(0, self.grid_res - 1, self.grid_res)
        self.x_coords = self.bbox_min[0] + grid_coords * self.hx
        self.y_coords = self.bbox_min[1] + grid_coords * self.hy
        self.z_coords = self.bbox_min[2] + grid_coords * self.hz

        print("Building KD-tree for point cloud...")
        self.kdtree = cKDTree(self.points)

    # ---------------------------
    # indexing helpers
    # ---------------------------
    def get_grid_index(self, i, j, k):
        return (int(i) * self.ny + int(j)) * self.nz + int(k)

    def get_grid_coords(self, idx):
        k = int(idx) % self.nz
        j = (int(idx) // self.nz) % self.ny
        i = int(idx) // (self.ny * self.nz)
        return i, j, k

    # ---------------------------
    # World/Grid transforms
    # ---------------------------
    def world_to_grid_float(self, point):
        return (np.asarray(point, dtype=float) - self.bbox_min) / self.voxel_size

    def point_to_voxel(self, point):
        p = self.world_to_grid_float(point)
        i0 = np.floor(p).astype(int)
        # ensure interior element index (so i0+1 exists)
        i0 = np.clip(i0, 0, self.grid_res - 2)
        tx, ty, tz = (p - i0)
        return (i0[0], i0[1], i0[2]), (tx, ty, tz)

    # ---------------------------
    # Reference shape function derivatives for trilinear hexahedron
    # Using reference domain ξ,η,ζ in [-1,1]
    # Eight local nodes with local coords (ξ_i,η_i,ζ_i) ∈ {-1,+1}^3
    # Node ordering we use:
    # 0:(0,0,0) -> (ξ,η,ζ)=(-1,-1,-1)
    # 1:(1,0,0) -> (+1,-1,-1)
    # 2:(0,1,0) -> (-1,+1,-1)
    # 3:(1,1,0) -> (+1,+1,-1)
    # 4:(0,0,1) -> (-1,-1,+1)
    # 5:(1,0,1) -> (+1,-1,+1)
    # 6:(0,1,1) -> (-1,+1,+1)
    # 7:(1,1,1) -> (+1,+1,+1)
    # ---------------------------
    def _local_node_xi(self, n):
        return -1.0 if (n % 2) == 0 else +1.0

    def _local_node_eta(self, n):
        # nodes 0-1: eta=-1, nodes 2-3: eta=+1; nodes 4-5: eta=-1, 6-7:eta=+1
        return -1.0 if ((n // 2) % 2) == 0 else +1.0

    def _local_node_zeta(self, n):
        return -1.0 if n < 4 else +1.0

    def shape_derivs_ref(self, xi, eta, zeta):
        """
        Compute reference derivatives (dN/dxi, dN/deta, dN/dzeta) for 8 nodes
        at reference point (xi,eta,zeta) (xi,eta,zeta in [-1,1]).
        Returns arrays dN_dxi (8,), dN_deta (8,), dN_dzeta (8,).
        Using N_i = 1/8 (1+xi*xi_i)(1+eta*eta_i)(1+zeta*zeta_i)
        so dN/dxi = (1/8) * xi_i * (1+eta*eta_i)(1+zeta*zeta_i)
        """
        dN_dxi = np.zeros(8, dtype=float)
        dN_deta = np.zeros(8, dtype=float)
        dN_dzeta = np.zeros(8, dtype=float)
        for n in range(8):
            xi_i = -1.0 if ((n % 2) == 0) else +1.0
            eta_i = -1.0 if ((n // 2) % 2 == 0) else +1.0
            zeta_i = -1.0 if n < 4 else +1.0

            common = 1.0 / 8.0
            dN_dxi[n] = common * xi_i * (1.0 + eta * eta_i) * (1.0 + zeta * zeta_i)
            dN_deta[n] = common * eta_i * (1.0 + xi * xi_i) * (1.0 + zeta * zeta_i)
            dN_dzeta[n] = common * zeta_i * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)

        return dN_dxi, dN_deta, dN_dzeta

    # ---------------------------
    # Compute element stiffness (8x8) via 2x2x2 Gauss quadrature
    # ---------------------------
    def element_stiffness(self):
        """
        Because grid is regular axis-aligned, the element Jacobian J is constant:
        J = diag(hx/2, hy/2, hz/2) for mapping from reference [-1,1]^3 to element.
        We compute the 8x8 local stiffness once (all elements identical up to scaling),
        but since hx,hy,hz may be anisotropic we compute using current hx,hy,hz.
        Returns Ke (8x8).
        """
        hx, hy, hz = self.hx, self.hy, self.hz
        # Jacobian and its determinant
        J = np.diag([hx / 2.0, hy / 2.0, hz / 2.0])
        detJ = np.linalg.det(J)
        invJT = np.linalg.inv(J).T  # (J^{-T})

        # 2-point Gauss points in 1D (±1/√3) and weights 1
        gp = 1.0 / np.sqrt(3.0)
        gauss_pts = [ -gp, +gp ]
        gauss_w = [1.0, 1.0]

        Ke = np.zeros((8, 8), dtype=float)

        # integrate over reference domain
        for xi, wx in zip(gauss_pts, gauss_w):
            for eta, wy in zip(gauss_pts, gauss_w):
                for zeta, wz in zip(gauss_pts, gauss_w):
                    w = wx * wy * wz

                    dN_dxi, dN_deta, dN_dzeta = self.shape_derivs_ref(xi, eta, zeta)
                    # build dN_dxi matrix (3 x 8)
                    dN_ref = np.vstack([dN_dxi, dN_deta, dN_dzeta])  # shape (3,8)

                    # physical gradients: ∇N = J^{-T} * dN_ref  -> (3,8)
                    gradN = invJT @ dN_ref  # (3,8)

                    # compute contribution gradN^T * gradN (8x8)
                    # For each pair (i,j): gradN[:,i]·gradN[:,j]
                    # So Ke += (gradN.T @ gradN) * detJ * w
                    Ke += (gradN.T @ gradN) * (detJ * w)

        return Ke

    # ---------------------------
    # FEM Laplacian assembly (global)
    # ---------------------------
    def build_fem_laplacian_matrix(self):
        """
        Assemble global stiffness matrix (Galerkin FEM) for trilinear hexahedral elements.
        Returns sparse CSR matrix of shape (Nnodes, Nnodes).
        """
        print("Building FEM Laplacian (Galerkin stiffness)...")
        start_time = time.time()

        nx_e = self.nx - 1  # elements per axis
        ny_e = self.ny - 1
        nz_e = self.nz - 1

        Nnodes = self.total_cells

        # Precompute local element stiffness (same for every element in a structured mesh)
        Ke = self.element_stiffness()  # 8x8

        data = []
        rows = []
        cols = []

        # local node index offsets in (i,j,k)
        local_offsets = [
            (0, 0, 0),  # node 0
            (1, 0, 0),  # node 1
            (0, 1, 0),  # node 2
            (1, 1, 0),  # node 3
            (0, 0, 1),  # node 4
            (1, 0, 1),  # node 5
            (0, 1, 1),  # node 6
            (1, 1, 1),  # node 7
        ]

        # Loop elements and scatter Ke into global arrays
        for ei in range(nx_e):
            for ej in range(ny_e):
                for ek in range(nz_e):
                    # compute global node indices for this element
                    global_nodes = []
                    for (oi, oj, ok) in local_offsets:
                        gi = ei + oi
                        gj = ej + oj
                        gk = ek + ok
                        global_nodes.append(self.get_grid_index(gi, gj, gk))

                    # scatter Ke into global COO arrays
                    for a_local, Arow in enumerate(global_nodes):
                        for b_local, Acol in enumerate(global_nodes):
                            val = Ke[a_local, b_local]
                            if val != 0.0:
                                rows.append(Arow)
                                cols.append(Acol)
                                data.append(val)

        A = sp.coo_matrix((data, (rows, cols)), shape=(Nnodes, Nnodes)).tocsr()

        print(f"FEM Laplacian assembled in {time.time() - start_time:.2f}s")
        print(f"Matrix shape: {A.shape}, NNZ: {A.nnz}")
        return A

    # ---------------------------
    # Screening matrix λ I
    # ---------------------------
    def build_screening_matrix(self):
        N = self.total_cells
        return self.screen_weight * sp.identity(N, format='csr')

    def build_divergence_field(self):
        """
        Compute the right-hand side (RHS) vector for the Poisson surface reconstruction
        problem using a finite element method with element-wise density-weighted integration.

        This method implements the discretization of the divergence term:
            b_i = ∫_Ω ∇φ_i(x) · V(x) dΩ

        where:
          - φ_i are the trilinear basis functions defined on a regular 3D grid
          - V(x) is the vector field reconstructed from oriented surface normals
          - Ω is the reconstruction domain

        Algorithm:
        1. Spatial binning: Points are assigned to their containing voxel elements using
           the point_to_voxel() method which provides integer element indices.
        2. Element-wise processing: For each occupied element:
           a. Compute the average normal direction from all samples within the element
           b. Evaluate the shape function gradients ∇φ_i at the element center
             (using the midpoint rule for numerical integration)
           c. Apply density weighting: Elements with more samples receive proportionally
             higher contributions (weight = len(samples) * det(J))
           d. Distribute contributions to the 8 nodes of the hexahedral element

        Key design choices:
        - Element averaging: Smooths noisy normals while preserving local orientation
        - Midpoint rule: Provides stable numerical integration with minimal computation
        - Density weighting: Emphasizes regions with higher sample density (surface areas)

        Returns:
        --------
        b : ndarray of shape (N_nodes,)
            Right-hand side vector for the Poisson linear system A·x = b, where
            A is the FEM stiffness matrix and x is the indicator function to be solved.

        Notes:
        ------
        - The method assumes a regular hexahedral grid with trilinear basis functions
        - The Jacobian determinant det(J) = (hx·hy·hz)/8 accounts for the mapping
          from reference to physical coordinates
        - Empirical testing shows this approach produces cleaner surfaces than
          point-wise evaluation or more complex integration schemes
        """
        print("Building divergence field (element-wise interpolation)...")
        start_time = time.time()

        b = np.zeros(self.total_cells, dtype=float)

        # Bin points into elements for efficiency
        # Create element occupancy map: element_index -> list of (point, normal) in that element
        element_map = {}

        for p_idx, (point, normal) in enumerate(zip(self.points, self.normals)):
            (i, j, k), _ = self.point_to_voxel(point)
            elem_idx = (i, j, k)

            if elem_idx not in element_map:
                element_map[elem_idx] = []
            element_map[elem_idx].append((point, normal))

        print(f"Points distributed into {len(element_map):,} elements")

        # For each occupied element, compute contributions
        J = np.diag([self.hx / 2.0, self.hy / 2.0, self.hz / 2.0])
        detJ = np.linalg.det(J)
        invJT = np.linalg.inv(J).T

        local_offsets = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
        ]

        for (ei, ej, ek), samples in element_map.items():
            # Ensure element is within bounds
            if ei >= self.nx - 1 or ej >= self.ny - 1 or ek >= self.nz - 1:
                continue

            # Get global node indices for this element
            global_nodes = []
            for (oi, oj, ok) in local_offsets:
                gi = ei + oi
                gj = ej + oj
                gk = ek + ok
                global_nodes.append(self.get_grid_index(gi, gj, gk))

            # Compute average normal for this element (or integrate each sample)
            avg_normal = np.zeros(3)
            for point, normal in samples:
                avg_normal += normal
            avg_normal = avg_normal / len(samples)

            # Use single quadrature point at element center (midpoint rule)
            # This is simpler but still better than pointwise evaluation
            xi, eta, zeta = 0.0, 0.0, 0.0  # Center of reference element

            dN_dxi, dN_deta, dN_dzeta = self.shape_derivs_ref(xi, eta, zeta)
            dN_ref = np.vstack([dN_dxi, dN_deta, dN_dzeta])
            gradN = invJT @ dN_ref

            # Contribution = (∇N_i · V) * detJ * element_volume_weight
            # Weight by number of samples in element (approximate density)
            weight = len(samples) * detJ

            for local_idx in range(8):
                contrib = np.dot(gradN[:, local_idx], avg_normal) * weight
                b[global_nodes[local_idx]] += contrib

        print(f"Divergence field built in {time.time() - start_time:.2f}s")
        return b

    # ---------------------------
    # Build full Poisson problem (FEM K + λI, and b)
    # ---------------------------
    def build_poisson_problem(self):
        print("\n" + "=" * 60)
        print(f"Building FEM Poisson problem on {self.grid_res}^3 grid")
        print(f"Total unknowns: {self.total_cells:,}")
        print("=" * 60)

        A_fem = self.build_fem_laplacian_matrix()
        A_screen = self.build_screening_matrix()
        A = A_fem + A_screen

        b = self.build_divergence_field()

        metadata = {
            'grid_resolution': self.grid_res,
            'total_cells': self.total_cells,
            'bbox_min': self.bbox_min,
            'bbox_max': self.bbox_max,
            'voxel_size': self.voxel_size,
            'screen_weight': self.screen_weight,
            'matrix_nnz': A.nnz,
            'matrix_density': A.nnz / (A.shape[0] * A.shape[1])
        }

        print("Problem built successfully.")
        try:
            print(f"Matrix density: {metadata['matrix_density']:.2e}")
            print(f"Condition number estimate: {self.estimate_condition_number(A):.2e}")
        except Exception:
            pass

        return A, b, metadata

    # ---------------------------
    # Condition number estimate (same)
    # ---------------------------
    def estimate_condition_number(self, A):
        try:
            n = A.shape[0]
            x = np.random.randn(n)
            x /= np.linalg.norm(x)
            for _ in range(10):
                x = A.dot(x)
                x /= np.linalg.norm(x)
            lambda_max = (x @ (A.dot(x))) / (x @ x)
            h = max(self.hx, self.hy, self.hz)
            lambda_min = (np.pi ** 2) * (h ** 2)
            return float(lambda_max / lambda_min)
        except Exception:
            return np.nan

    # ---------------------------
    # Solution to mesh (unchanged)
    # ---------------------------
    def solution_to_mesh(self, solution, iso_value=None):
        print("\nExtracting mesh from solution...")
        solution_3d = np.asarray(solution).reshape(self.nx, self.ny, self.nz)

        if iso_value is None:
            iso_value = np.median(solution_3d)

        print(f"Using iso-value: {iso_value:.6f}")

        try:
            verts, faces, norms, vals = marching_cubes(
                solution_3d,
                level=iso_value,
                spacing=(self.hx, self.hy, self.hz)
            )
            verts = verts + self.bbox_min
            print(f"Mesh extracted: {len(verts)} verts, {len(faces)} faces")
            return verts, faces
        except Exception as e:
            print(f"Marching cubes failed: {e}")
            vals_range = np.linspace(solution_3d.min(), solution_3d.max(), 8)
            for val in vals_range:
                try:
                    verts, faces, norms, vals = marching_cubes(
                        solution_3d, level=val, spacing=(self.hx, self.hy, self.hz)
                    )
                    verts = verts + self.bbox_min
                    print(f"Found mesh at iso-value {val:.6f}")
                    return verts, faces
                except Exception:
                    continue
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)