import numpy as np
import time
import scipy.sparse as sp
import pyamg
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.sparse.linalg import spsolve
import warnings


class PoissonSolver(ABC):
    """Abstract base class for Poisson solvers"""

    def __init__(self, name, max_iter=1000, tol=1e-6):
        self.name = name
        self.max_iter = max_iter
        self.tol = tol
        self.iterations = 0
        self.solve_time = 0.0
        self.residual_history = []

    @abstractmethod
    def solve(self, A, b):
        """Solve A*x = b, return solution x"""
        pass

    def get_stats(self):
        """Return solver statistics"""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'time': self.solve_time,
            'converged': len(self.residual_history) > 0 and
                         self.residual_history[-1] < self.tol
        }

    def reset(self):
        """Reset solver state"""
        self.iterations = 0
        self.solve_time = 0.0
        self.residual_history = []


class GaussSeidelSolver(PoissonSolver):
    """Gauss-Seidel iterative solver"""

    def __init__(self, max_iter=1000, tol=1e-6, omega=1.0, use_sor=False):
        super().__init__("Gauss-Seidel" + (" (SOR)" if use_sor else ""),
                         max_iter, tol)
        self.omega = omega  # Relaxation parameter (1.0 = standard GS)
        self.use_sor = use_sor  # Successive Over-Relaxation

    def solve(self, A, b, x0=None):
        """Solve using Gauss-Seidel iteration"""
        print(f"Running {self.name}...")
        start_time = time.time()

        n = len(b)
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        self.residual_history = []

        # Precompute diagonal indices for efficiency
        diag_vals = A.diagonal()

        for iteration in tqdm(range(self.max_iter), desc="Iteration"):
            x_old = x.copy()

            # Gauss-Seidel iteration
            for i in range(n):
                # Get row data
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]

                # Extract row values and columns
                cols = A.indices[row_start:row_end]
                vals = A.data[row_start:row_end]

                # Sum contributions from neighbors
                sum_neighbors = 0.0
                diag_val = diag_vals[i]

                for col, val in zip(cols, vals):
                    if col != i:  # Off-diagonal
                        sum_neighbors += val * x[col]

                # Update solution
                if diag_val != 0:
                    if self.use_sor:
                        # Successive Over-Relaxation
                        gs_update = (b[i] - sum_neighbors) / diag_val
                        x[i] = x_old[i] + self.omega * (gs_update - x_old[i])
                    else:
                        # Standard Gauss-Seidel
                        x[i] = (b[i] - sum_neighbors) / diag_val

            # Compute residual
            residual = np.linalg.norm(A @ x - b)
            self.residual_history.append(residual)

            # Check convergence
            if residual < self.tol:
                print(f"  Converged in {iteration + 1} iterations")
                break

            # Optional: check for stagnation
            if iteration > 10:
                rel_change = np.linalg.norm(x - x_old) / np.linalg.norm(x)
                if rel_change < 1e-10:
                    warnings.warn(f"{self.name}: Solution stagnated")
                    break

        self.iterations = iteration + 1
        self.solve_time = time.time() - start_time

        return x


class ConjugateGradientSolver(PoissonSolver):
    """Conjugate Gradient solver for symmetric positive definite systems"""

    def __init__(self, max_iter=1000, tol=1e-6, preconditioner=None):
        super().__init__("Conjugate Gradient", max_iter, tol)
        self.preconditioner = preconditioner

    def solve(self, A, b, x0=None):
        print(f"Running {self.name}...")
        start_time = time.time()

        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        # Store BOTH preconditioned and true residuals
        self.residual_history = []  # True residuals for fair comparison
        self.prec_residual_history = []  # Preconditioned residuals for CG

        # Initial residual
        r = b - A @ x
        true_residual = np.linalg.norm(r)
        self.residual_history.append(true_residual)

        # Preconditioner setup
        if self.preconditioner == 'jacobi':
            diag = A.diagonal()
            diag = np.where(np.abs(diag) < 1e-12, 1.0, diag)
            M_inv = 1.0 / diag
            z = M_inv * r
        else:
            z = r.copy()

        p = z.copy()
        rz_old = r @ z

        # Preconditioned initial residual
        prec_residual = np.sqrt(abs(rz_old))
        self.prec_residual_history.append(prec_residual)

        print(f"  Initial residual: {true_residual:.2e}")

        for iteration in range(self.max_iter):
            Ap = A @ p

            # Breakdown check
            pAp = p @ Ap
            if abs(pAp) < 1e-14:
                print(f"  CG breakdown: p·Ap = {pAp:.2e}")
                break

            alpha = rz_old / pAp

            # Update
            x = x + alpha * p
            r = r - alpha * Ap

            # Compute TRUE residual (for fair comparison)
            true_residual = np.linalg.norm(r)
            self.residual_history.append(true_residual)

            # Precondition
            if self.preconditioner == 'jacobi':
                z = M_inv * r
            else:
                z = r.copy()

            rz_new = r @ z

            # Preconditioned residual (for CG convergence)
            prec_residual = np.sqrt(abs(rz_new))
            self.prec_residual_history.append(prec_residual)

            # Convergence check - use TRUE residual for fair comparison
            if true_residual < self.tol:
                print(f"  Converged in {iteration + 1} iterations")
                print(f"  True residual: {true_residual:.2e}, "
                      f"Prec residual: {prec_residual:.2e}")
                break

            # Update search direction
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        self.iterations = iteration + 1
        self.solve_time = time.time() - start_time

        return x


class DirectSolver(PoissonSolver):
    """Direct solver using SciPy's sparse direct methods"""

    def __init__(self, method='spsolve', use_umfpack=True):
        super().__init__(f"Direct ({method})", max_iter=1, tol=1e-15)
        self.method = method
        self.use_umfpack = use_umfpack

    def solve(self, A, b):
        """Solve using direct method"""
        print(f"Running {self.name}...")
        start_time = time.time()

        try:
            if self.method == 'spsolve':
                # Use SuperLU by default
                x = spsolve(A, b, use_umfpack=self.use_umfpack)
            elif self.method == 'factorized':
                # Factorize once, solve multiple times
                from scipy.sparse.linalg import factorized
                solve_func = factorized(A.tocsc())
                x = solve_func(b)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            residual = np.linalg.norm(A @ x - b)
            self.residual_history = [residual]
            self.iterations = 1
            self.solve_time = time.time() - start_time

            print(f"  Direct solve completed, residual: {residual:.2e}")

        except Exception as e:
            print(f"  Direct solver failed: {e}")
            # Fallback to zeros
            x = np.zeros_like(b)
            self.residual_history = [np.inf]
            self.iterations = 0
            self.solve_time = time.time() - start_time

        return x


class GeometricMultigridSolver(PoissonSolver):
    """Geometric multigrid for 3D regular grids"""

    def __init__(self, grid_shape, max_levels=4, max_iter=20, tol=1e-6,
                 pre_smooths=2, post_smooths=2, cycle_type='V'):
        """
        Parameters:
        -----------
        grid_shape : tuple (nx, ny, nz)
            Fine grid dimensions
        max_levels : int
            Maximum number of coarsening levels
        cycle_type : 'V', 'W', or 'F'
            Multigrid cycle type
        """
        super().__init__(f"Geometric MG ({cycle_type}-cycle)", max_iter, tol)
        self.grid_shape = grid_shape
        self.max_levels = max_levels
        self.pre_smooths = pre_smooths
        self.post_smooths = post_smooths
        self.cycle_type = cycle_type

        # Precompute grid hierarchy
        self.grid_hierarchy = self._build_grid_hierarchy()

    def _build_grid_hierarchy(self):
        """Build list of grid sizes from fine to coarsest"""
        hierarchy = [self.grid_shape]
        nx, ny, nz = self.grid_shape

        for level in range(1, self.max_levels):
            # Coarsen each dimension by factor 2
            nx = max(2, (nx + 1) // 2)  # Ensure at least 2 points
            ny = max(2, (ny + 1) // 2)
            nz = max(2, (nz + 1) // 2)

            if nx * ny * nz < 8:  # Too small
                break

            hierarchy.append((nx, ny, nz))

        print(f"Multigrid hierarchy: {len(hierarchy)} levels")
        for i, (nx, ny, nz) in enumerate(hierarchy):
            print(f"  Level {i}: {nx}×{ny}×{nz} = {nx * ny * nz:,} unknowns")

        return hierarchy

    def _build_restriction_3d(self, fine_shape, coarse_shape):
        """Build 3D full-weighting restriction operator"""
        nxf, nyf, nzf = fine_shape
        nxc, nyc, nzc = coarse_shape

        n_fine = nxf * nyf * nzf
        n_coarse = nxc * nyc * nzc

        # Full weighting stencil: average of 8 fine cells
        data, rows, cols = [], [], []

        for ic in range(nxc):
            for jc in range(nyc):
                for kc in range(nzc):
                    coarse_idx = (ic * nyc + jc) * nzc + kc

                    # Corresponding fine indices (2×2×2 block)
                    i_start = ic * 2
                    j_start = jc * 2
                    k_start = kc * 2

                    # Weight contributions from 8 fine cells
                    for di in (0, 1):
                        for dj in (0, 1):
                            for dk in (0, 1):
                                if (i_start + di < nxf and
                                        j_start + dj < nyf and
                                        k_start + dk < nzf):
                                    fine_idx = ((i_start + di) * nyf + (j_start + dj)) * nzf + (k_start + dk)
                                    data.append(1.0 / 8.0)
                                    rows.append(coarse_idx)
                                    cols.append(fine_idx)

        return sp.csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))

    def _build_prolongation_3d(self, fine_shape, coarse_shape):
        """Build 3D linear interpolation prolongation operator"""
        # For regular grids, prolongation = restriction.T * 8
        R = self._build_restriction_3d(fine_shape, coarse_shape)
        return 8.0 * R.T  # Scale to preserve constants

    def _build_coarse_operator(self, A_fine, fine_shape, coarse_shape):
        """Build coarse operator using Galerkin projection"""
        R = self._build_restriction_3d(fine_shape, coarse_shape)
        P = self._build_prolongation_3d(fine_shape, coarse_shape)
        A_coarse = R @ A_fine @ P
        return A_coarse, R, P

    def _v_cycle(self, A_level, b_level, x_level, level):
        """Execute V-cycle at given level"""
        if level == len(self.grid_hierarchy) - 1:
            # Coarsest level - solve directly
            try:
                return spsolve(A_level, b_level)
            except:
                # Fallback to CG
                solver = ConjugateGradientSolver(max_iter=200, tol=1e-6)
                return solver.solve(A_level, b_level)

        # Pre-smoothing
        smoother = GaussSeidelSolver(max_iter=self.pre_smooths, tol=1e-3)
        x_level = smoother.solve(A_level, b_level, x0=x_level)

        # Compute residual
        r_level = b_level - A_level @ x_level

        # Restrict residual to coarser level
        fine_shape = self.grid_hierarchy[level]
        coarse_shape = self.grid_hierarchy[level + 1]
        R = self._build_restriction_3d(fine_shape, coarse_shape)
        r_coarse = R @ r_level

        # Recursive coarse grid correction
        A_coarse = self.coarse_operators[level + 1]
        e_coarse = np.zeros_like(r_coarse)
        e_coarse = self._v_cycle(A_coarse, r_coarse, e_coarse, level + 1)

        # Prolongate and correct
        P = self._build_prolongation_3d(fine_shape, coarse_shape)
        x_level = x_level + P @ e_coarse

        # Post-smoothing
        smoother = GaussSeidelSolver(max_iter=self.post_smooths, tol=1e-3)
        x_level = smoother.solve(A_level, b_level, x0=x_level)

        return x_level

    def solve(self, A, b):
        """Solve using geometric multigrid"""
        print(f"Running {self.name}...")
        start_time = time.time()

        n = len(b)
        x = np.zeros_like(b)
        self.residual_history = []

        # Build coarse operators for all levels
        print("Building multigrid hierarchy...")
        self.coarse_operators = [A]  # Level 0 is fine operator

        for level in range(len(self.grid_hierarchy) - 1):
            fine_shape = self.grid_hierarchy[level]
            coarse_shape = self.grid_hierarchy[level + 1]

            A_coarse, _, _ = self._build_coarse_operator(
                self.coarse_operators[-1], fine_shape, coarse_shape
            )
            self.coarse_operators.append(A_coarse)

        # Main multigrid iteration
        initial_residual = np.linalg.norm(b - A @ x)
        self.residual_history.append(initial_residual)
        print(f"  Initial residual: {initial_residual:.2e}")

        for cycle in range(self.max_iter):
            # Execute V-cycle
            x = self._v_cycle(A, b, x, level=0)

            # Compute residual
            residual = np.linalg.norm(b - A @ x)
            self.residual_history.append(residual)

            # Check convergence
            if residual / max(1.0, initial_residual) < self.tol:
                print(f"  Converged in {cycle + 1} cycles")
                break

            # Progress
            if cycle % 2 == 0:
                reduction = residual / self.residual_history[-2] if cycle > 0 else 1.0
                print(f"    Cycle {cycle + 1}: residual={residual:.2e}, "
                      f"reduction={reduction:.3f}")

        self.iterations = cycle + 1
        self.solve_time = time.time() - start_time

        print(f"  Final residual: {residual:.2e}")
        print(f"  Total time: {self.solve_time:.2f}s")

        return x


class PyAMGSolver(PoissonSolver):
    """PyAMG algebraic multigrid solver optimized for 3D Poisson"""

    def __init__(self, max_iter=20, tol=1e-6):
        super().__init__("PyAMG", max_iter, tol)

    def solve(self, A, b):
        print(f"Running {self.name}...")
        start_time = time.time()

        # Just use defaults from your PyAMG version
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength='classical',  # Change from default 'symmetric'
            max_coarse=300,  # Increase from default 10
            max_levels=25  # Increase from default 10
        )

        # Solve
        residuals = []
        x = ml.solve(b, tol=self.tol, maxiter=self.max_iter,
                     residuals=residuals)

        # Results
        self.residual_history = residuals
        self.iterations = len(residuals) - 1
        self.solve_time = time.time() - start_time

        print(f"  {self.iterations} cycles, {self.solve_time:.2f}s")

        return x


class CustomSolver(PoissonSolver):
    """Template for custom solver implementations"""

    def __init__(self, max_iter=1000, tol=1e-6):
        super().__init__("Custom Solver", max_iter, tol)

    def solve(self, A, b):
        """Implement your custom solver here"""
        print(f"Running {self.name}...")
        start_time = time.time()

        n = len(b)
        x = np.zeros_like(b)
        self.residual_history = []

        # TODO: Implement your custom solver algorithm here

        # For now, just return zeros
        self.iterations = 0
        self.solve_time = time.time() - start_time

        print(f"  Custom solver not implemented yet")

        return x