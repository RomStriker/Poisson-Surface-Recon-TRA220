import numpy as np
import matplotlib.pyplot as plt
from src.solvers.cpu_solvers import (
    GaussSeidelSolver,
    ConjugateGradientSolver,
    DirectSolver,
    GeometricMultigridSolver,
    PyAMGSolver)


# Solver comparison framework
class SolverBenchmark:
    """Benchmark framework for comparing Poisson solvers"""

    def __init__(self, grid_shape=None, tol=1e-9):
        self.solvers = {}
        self.results = {}
        self.grid_shape = grid_shape
        self.problem_info = None
        self.tol = tol
        self.create_default_solvers()

    def register_solver(self, name, solver):
        """Register a solver for benchmarking"""
        self.solvers[name] = solver

    def create_default_solvers(self):
        """Create a set of default solvers"""
        self.solvers = {
            # 'Gauss-Seidel': GaussSeidelSolver(max_iter=100, tol=self.tol),
            # 'Gauss-Seidel (SOR)': GaussSeidelSolver(max_iter=100, tol=self.tol,
            #                                         use_sor=True, omega=1.2),
            'Conjugate Gradient': ConjugateGradientSolver(max_iter=100, tol=self.tol),
            # 'CG with Jacobi': ConjugateGradientSolver(max_iter=1000, tol=self.tol,
            #                                           preconditioner='jacobi'),
            # 'Direct (SuperLU)': DirectSolver(method='spsolve'), # Can be used for smaller grid sizes
            # 'Multigrid': GeometricMultigridSolver(grid_shape=self.grid_shape, max_levels=4,
            #                                       max_iter=20, tol=self.tol, pre_smooths=2, post_smooths=2),
            # 'PyAMG': PyAMGSolver(max_iter=50, tol=self.tol),
            # 'Custom': CustomSolver(max_iter=1000, tol=self.tol),
        }

    def run_benchmark(self, A, b):
        """Run all registered solvers on the given problem"""
        print("\n" + "=" * 60)
        print("Running Solver Benchmark")
        print("=" * 60)

        self.problem_info = {
            'matrix_size': A.shape[0],
            'matrix_nnz': A.nnz,
            'matrix_density': A.nnz / (A.shape[0] * A.shape[1]),
            'rhs_norm': np.linalg.norm(b)
        }

        self.results = {}

        # Run benchmarks
        print("\n" + "-" * 60)
        print("Benchmark Results")
        print("-" * 60)

        for name, solver in self.solvers.items():
            print(f"\n>>> Testing: {name}")

            try:
                # Solve
                solution = solver.solve(A, b)

                # Compute final stats
                final_residual = np.linalg.norm(A @ solution - b)
                stats = solver.get_stats()

                # Store results
                self.results[name] = {
                    'solution': solution,
                    'stats': stats,
                    'final_residual': final_residual,
                    'residual_history': solver.residual_history.copy()
                }

                print(f"    Time: {stats['time']:.4f}s, "
                      f"Iterations: {stats['iterations']}, "
                      f"Residual: {final_residual:.2e}")

            except Exception as e:
                print(f"    FAILED: {e}")
                self.results[name] = {
                    'solution': None,
                    'stats': {'name': name, 'time': np.inf, 'iterations': 0, 'converged': False},
                    'final_residual': np.inf,
                    'residual_history': []
                }

        return self.results

    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if self.problem_info:
            print(f"\nProblem Information:")
            print(f"  Matrix size: {self.problem_info['matrix_size']}")
            print(f"  Non-zeros: {self.problem_info['matrix_nnz']:,}")
            print(f"  Density: {self.problem_info['matrix_density']:.2e}")
            print(f"  RHS norm: {self.problem_info['rhs_norm']:.4f}")

        print(f"\nSolver Performance:")
        print("-" * 60)
        print(f"{'Solver':<25} {'Time (s)':>10} {'Iterations':>12} "
              f"{'Residual':>12} {'Converged':>10}")
        print("-" * 60)

        # Sort by time
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['stats']['time']
        )

        for name, result in sorted_results:
            stats = result['stats']
            time_str = f"{stats['time']:.4f}" if stats['time'] < 100 else ">100"
            iter_str = f"{stats['iterations']}" if stats['iterations'] < 10000 else ">10k"
            res_str = f"{result['final_residual']:.2e}" if result['final_residual'] < 1e10 else "FAILED"
            conv_str = "✓" if stats.get('converged', False) else "✗"

            print(f"{name:<25} {time_str:>10} {iter_str:>12} "
                  f"{res_str:>12} {conv_str:>10}")

    def plot_results(self):
        """Plot comparison of solver performance with improved layout and spacing"""
        # Create 3x3 grid with more vertical space
        fig = plt.figure(figsize=(18, 16))  # Increased height
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, height_ratios=[1, 1, 0.8])

        # Create subplots with additional bottom padding
        ax_time = fig.add_subplot(gs[0, 0])
        ax_iters = fig.add_subplot(gs[0, 1])
        ax_resid = fig.add_subplot(gs[0, 2])
        ax_speed = fig.add_subplot(gs[1, 0])
        ax_conv = fig.add_subplot(gs[1, 1])
        ax_summary = fig.add_subplot(gs[2, :])  # Full width for summary

        # Extract data
        names = list(self.results.keys())
        times = [self.results[name]['stats']['time'] for name in names]
        iterations = [self.results[name]['stats']['iterations'] for name in names]
        residuals = [self.results[name]['final_residual'] for name in names]
        converged = [self.results[name]['stats'].get('converged', False) for name in names]

        # Filter out failed solvers
        valid_idx = [i for i, r in enumerate(residuals) if r < 1e10]
        valid_names = [names[i] for i in valid_idx]
        valid_times = [times[i] for i in valid_idx]
        valid_iters = [iterations[i] for i in valid_idx]
        valid_residuals = [residuals[i] for i in valid_idx]
        valid_converged = [converged[i] for i in valid_idx]

        # Color scheme
        colors = ['#2E8B57' if conv else '#DC143C' for conv in valid_converged]  # Green/Red

        # Helper function to adjust bar plot layout
        def setup_bar_plot(ax, data, ylabel, title, value_format='.2f',
                           value_offset=0.01, value_fontsize=8,
                           value_rotation=0, show_values=True):
            """Setup a bar plot with consistent styling"""
            bars = ax.bar(range(len(valid_names)), data, color=colors,
                          edgecolor='black', linewidth=0.5, alpha=0.8)

            # X-axis labels with more space
            ax.set_xticks(range(len(valid_names)))
            ax.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=9)

            # Add padding between x-axis and labels
            ax.tick_params(axis='x', pad=10)

            # Titles and labels with padding
            ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=15)

            # Grid
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels if requested
            if show_values:
                for bar, val in zip(bars, data):
                    height = bar.get_height()
                    # Position text above bar with small offset
                    text_y = height + (max(data) * value_offset)
                    ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                            f'{val:{value_format}}', ha='center', va='bottom',
                            fontsize=value_fontsize, rotation=value_rotation)

            # Adjust ylim to make room for value labels
            if len(data) > 0:
                current_ylim = ax.get_ylim()
                ax.set_ylim(current_ylim[0], current_ylim[1] * 1.15)

            return bars

        # 1. Time comparison
        setup_bar_plot(ax_time, valid_times, 'Time (s)',
                       'Solution Time\n(lower is better)',
                       value_format='.3f', value_offset=0.02)

        # 2. Iterations comparison
        setup_bar_plot(ax_iters, valid_iters, 'Iterations',
                       'Iterations to Converge\n(lower is better)',
                       value_format='d', value_offset=0.02)

        # 3. Residual comparison (log scale)
        log_residuals = np.log10(valid_residuals)
        bars_resid = setup_bar_plot(ax_resid, log_residuals, 'log₁₀(Residual)',
                                    'Final Residual\n(lower is better)',
                                    value_format='.1f', value_offset=0.05,
                                    show_values=False)  # Don't show log values

        # Add actual residual values (rotated)
        for bar, res in zip(bars_resid, valid_residuals):
            height = bar.get_height()
            ax_resid.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                          f'{res:.1e}', ha='center', va='bottom',
                          fontsize=7, rotation=45)

        # 4. Speed comparison (iterations per second)
        speed = [valid_iters[i] / max(valid_times[i], 1e-6) for i in range(len(valid_times))]
        setup_bar_plot(ax_speed, speed, 'Iterations/second',
                       'Solver Speed\n(higher is better)',
                       value_format='.1f', value_offset=0.02)

        # 5. Convergence history with better spacing
        legend_handles = []
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for idx, (name, result) in enumerate(self.results.items()):
            if len(result['residual_history']) > 0 and result['final_residual'] < 1e10:
                residuals_hist = result['residual_history']
                line_style = line_styles[idx % len(line_styles)]
                marker = markers[idx % len(markers)]

                line, = ax_conv.semilogy(range(len(residuals_hist)), residuals_hist,
                                         linestyle=line_style, marker=marker,
                                         markersize=5, linewidth=2, alpha=0.8,
                                         markevery=max(1, len(residuals_hist) // 20))
                legend_handles.append((name, line))

        ax_conv.set_xlabel('Iteration', fontsize=10, labelpad=10)
        ax_conv.set_ylabel('Residual', fontsize=10, labelpad=10)
        ax_conv.set_title('Convergence History', fontsize=11, fontweight='bold', pad=15)
        ax_conv.grid(True, alpha=0.3)
        ax_conv.set_axisbelow(True)  # Grid behind data

        if legend_handles:
            names, lines = zip(*legend_handles)  # unpack names and matplotlib Line2D objects
            ax_conv.legend(lines, names,
                           loc='upper right',
                           fontsize=9,
                           ncol=1,
                           framealpha=0.9,
                           fancybox=True,
                           borderaxespad=0.5)

        # 6. Performance summary table with more vertical space
        ax_summary.axis('off')

        # Prepare table data
        table_data = []
        headers = ['Solver', 'Time (s)', 'Iterations', 'Residual', 'Converged', 'Rate (it/s)']

        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['stats']['time'])

        for name, result in sorted_results:
            if result['final_residual'] < 1e10:
                stats = result['stats']
                time_val = stats['time']
                iters = stats['iterations']
                resid = result['final_residual']
                conv = '✓' if stats.get('converged', False) else '✗'
                rate = iters / max(time_val, 1e-6)

                table_data.append([
                    name,
                    f'{time_val:.3f}',
                    f'{iters}',
                    f'{resid:.2e}',
                    conv,
                    f'{rate:.1f}'
                ])

        # Create table with better vertical spacing
        if table_data:
            table = ax_summary.table(cellText=table_data,
                                     colLabels=headers,
                                     cellLoc='center',
                                     loc='center',
                                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.1, 0.15])

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.0)  # Increased vertical scaling

            # Color header row
            for j, cell in enumerate(table.get_celld().values()):
                if j < len(headers):  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2E8B57')  # Green header
                    cell.set_height(0.15)  # Header height
                else:  # Data rows
                    row = (j - len(headers)) // len(headers)
                    col = (j - len(headers)) % len(headers)
                    cell.set_height(0.12)  # Data row height
                    if col == 4:  # Converged column
                        cell.get_text().set_color('green' if table_data[row][col] == '✓' else 'red')
                    elif row % 2 == 0:  # Alternate row colors
                        cell.set_facecolor('#F5F5F5')

        # Add problem info as annotation with more space
        if hasattr(self, 'problem_info'):
            prob_text = (f"Problem: {self.problem_info['matrix_size']:,} unknowns | "
                         f"{self.problem_info['matrix_nnz']:,} non-zeros | "
                         f"Density: {self.problem_info['matrix_density']:.2e}")
            fig.text(0.5, 0.01, prob_text, ha='center', fontsize=10, style='italic')

        # Adjust overall layout
        plt.suptitle('Poisson Solver Benchmark Results', fontsize=16, fontweight='bold', y=0.98)

        # Use constrained_layout for better automatic spacing
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0.3, wspace=0.3)

        # Manual adjustment if needed
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08,
                            hspace=0.5, wspace=0.4)

        plt.show()

    def get_best_solution(self):
        """Get the solution with the smallest residual"""
        valid_results = [(name, result) for name, result in self.results.items()
                         if result['solution'] is not None and
                         result['final_residual'] < 1e10]

        if not valid_results:
            return None, None

        best_name, best_result = min(valid_results,
                                     key=lambda x: x[1]['final_residual'])

        return best_name, best_result['solution']