# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis

from mpi4py import MPI

import dolfinx
import time
from utils import stdout, CSVWriter
from petsc_interface import SNESSolver


class AlternateMinimizer:
    def __init__(
        self,
        comm: MPI.Comm,
        solver_uη: SNESSolver,
        solver_α: SNESSolver,
        E_pot_form: dolfinx.fem.form,
        E_frac_form: dolfinx.fem.form,
        tol_Rnorm: float = 1e-6,
        max_iter: int = 100000,
        monitoring: bool = True,
    ):
        # store MPI communicator
        self.comm = comm

        # store the separate solvers
        self.solver_uη = solver_uη
        self.solver_α = solver_α

        # store forms for energy contributions
        self.E_pot_form = E_pot_form
        self.E_frac_form = E_frac_form

        # store all parameters
        self.tol_Rnorm = tol_Rnorm
        self.max_iter = max_iter
        self.monitoring = monitoring

        # initialize the monitoring output
        if self.monitoring:
            self.csv_staggered_convergence = CSVWriter(
                "staggered_convergence.csv",
                "step\titeration\tE_pot\tE_frac\tE_tot\tR_uη_norm"
                "\tNR_iter_uη\tfuncevals_uη\tNR_iter_α\tfuncevals_α\ttime_staggered\n",
            )

    def solve(self, step: int):
        # start timer
        t_staggered_start = time.time()

        # loop over the staggered iterations
        for iteration in range(self.max_iter):
            if iteration > 0:
                stdout("_" * 75)

            stdout(f"staggered iteration: {iteration:3d}")

            # minimize wrt u
            stdout("\nminimizing wrt uη")
            solver_uη_iterations, solver_uη_funcevals = self.solver_uη.solve(
                step, iteration
            )

            # minimize wrt α
            stdout("\nminimizing wrt α")
            solver_α_iterations, solver_α_funcevals = self.solver_α.solve(
                step, iteration
            )

            # re-evaluate the uη residual
            R_uη_norm = self.solver_uη.residual_norm()

            # compute the energies
            E_pot_iter = dolfinx.fem.assemble_scalar(self.E_pot_form)
            E_frac_iter = dolfinx.fem.assemble_scalar(self.E_frac_form)

            stdout(f"\n||R_uη||_2 = {R_uη_norm}")

            # output staggered solver monitoring
            if self.monitoring:
                self.csv_staggered_convergence.write(
                    f"{step:04d}\t{iteration:04d}\t{E_pot_iter:.8e}\t{E_frac_iter:.8e}\t{E_pot_iter + E_frac_iter:.8e}\t{R_uη_norm:.8e}"
                    f"\t{solver_uη_iterations:04d}\t{solver_uη_funcevals:04d}\t{solver_α_iterations:04d}\t{solver_α_funcevals:04d}"
                    f"\t{time.time() - t_staggered_start:.8e}\n",
                )

            # check convergence criteria and if fulfilled exit the staggered loop
            if R_uη_norm <= self.tol_Rnorm:
                break

        else:
            raise RuntimeError(
                f"convergence not reached after {iteration:3d} iterations"
            )

        # return the energies of the last converged state
        return E_pot_iter, E_frac_iter
