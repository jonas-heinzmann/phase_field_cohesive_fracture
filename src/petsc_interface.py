# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis

import ufl
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import dolfinx.nls.petsc
import typing

from utils import CSVWriter


# =======================================================================================
# S N E S   P R O B L E M
# =======================================================================================
class SNESProblem:
    """
    based on the code in the file
    https://github.com/FEniCS/dolfinx/blob/v0.9.0/python/test/unit/nls/test_newton.py
    which is part of DOLFINx (https://www.fenicsproject.org), Garth N. Wells (2018)
    """

    def __init__(
        self,
        F: ufl.form.Form,
        J: ufl.form.Form,
        u: dolfinx.fem.Function,
        bcs: typing.List[dolfinx.fem.dirichletbc],
        objective: typing.Optional[ufl.form.Form] = None,
    ) -> None:
        self.L = F
        self.a = J
        self.bcs = bcs
        self.u = u

        # set callback for objective function
        self.objective = objective

    def f(self, snes: PETSc.SNES, x_: PETSc.Vec) -> float:
        """
        assemble the objective function and return its value

        snes    snes object
        x_      vector containing the current solution
        """

        # update the solution vector
        x_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x_.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # compute the objective function
        obj = MPI.COMM_WORLD.allreduce(
            dolfinx.fem.assemble_scalar(self.obj), op=MPI.SUM
        )

        return obj

    def F(self, snes: PETSc.SNES, x_: PETSc.Vec, b_: PETSc.Vec) -> None:
        """
        assemble the residual F into the vector b

        snes    snes object
        x_      vector containing the current solution
        b_      vector to assemble the residual into
        """

        # update the solution vector
        x_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x_.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # initialize zero residual vector
        with b_.localForm() as b_local:
            b_local.set(0.0)

        # apply Dirichlet BCs to the solution vector
        dolfinx.fem.petsc.assemble_vector(b_, self.L)
        dolfinx.fem.petsc.apply_lifting(
            b_, [self.a], bcs=[self.bcs], x0=[x_], alpha=-1.0
        )
        b_.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b_, self.bcs, x_, -1.0)

    def J(self, snes, x_: PETSc.Vec, A_: PETSc.Mat, P_: PETSc.Mat) -> None:
        """
        assemble the Jacobian matrix

        x_   vector containing the current solution
        A_   matrix to assemble the Jacobian into
        """

        # update the solution vector
        x_.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x_.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # assemble Jacobian
        A_.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A_, self.a, bcs=self.bcs)
        A_.assemble()


# =======================================================================================
# S N E S   S O L V E R
# =======================================================================================
class SNESSolver:
    """
    solver class used for the above SNESProblem class
    """

    def __init__(
        self,
        snes_problem: SNESProblem,
        comm: MPI.Comm,
        options_prefix: str = "",
        convergence_monitoring: bool = False,
    ) -> None:
        self.snes_problem = snes_problem
        self.comm = comm
        self.options_prefix = options_prefix
        self.convergence_monitoring = convergence_monitoring

        # create SNES object
        self.petsc_snes = PETSc.SNES().create(comm=self.comm)

        # create PETSc vectors and matrices
        self.x_vec = self.snes_problem.u.x.petsc_vec.copy()
        self.b_vec = dolfinx.fem.petsc.create_vector(self.snes_problem.L)
        self.J_mat = dolfinx.fem.petsc.create_matrix(self.snes_problem.a)

        # set function and jacobian
        if self.snes_problem.objective is not None:
            self.petsc_snes.setObjective(self.snes_problem.f)
        self.petsc_snes.setFunction(self.snes_problem.F, self.b_vec)
        self.petsc_snes.setJacobian(self.snes_problem.J, self.J_mat)

        # set options prefix
        self.petsc_snes.setOptionsPrefix(options_prefix)

        # set options
        self.petsc_snes.setFromOptions()
        self.petsc_snes.setErrorIfNotConverged(
            True
        )  # PETSc might not raise an error otherwise
        self.petsc_snes.logConvergenceHistory(norm=2)

        # initialize convergence history file
        if self.convergence_monitoring:
            self.csv_convergence = CSVWriter(
                f"snes_{self.options_prefix}convergence.csv",
                "step\tstagg_iter\tNR_iter\t||R||_2\n",
            )

    def solve(self, step: int = 0, iteration: int = 0) -> tuple[int, int]:
        # reset convergence history
        self.petsc_snes.setConvergenceHistory()

        # solve the problem
        self.petsc_snes.solve(None, self.x_vec)
        iterations = self.petsc_snes.getIterationNumber()
        funcevals = self.petsc_snes.getFunctionEvaluations()

        # output convergence history
        if self.convergence_monitoring:
            (R_norm_hist, _) = self.petsc_snes.getConvergenceHistory()
            for i in range(len(R_norm_hist)):
                self.csv_convergence.write(
                    f"{step:04d}\t{iteration:04d}\t{i:04d}\t{R_norm_hist[i]:.8e}\n",
                )

        return iterations, funcevals

    def residual_norm(self) -> float:
        # compute residual of the displacement problem with PETSc
        # note: the convergence check must include the constraints
        # which is difficult and inefficient to do through the Python interface,
        # hence we instead 'trick' PETSc to compute the residual norm internally
        # without performing any iterations by setting max_it=0 temporarily,
        # and then resetting it afterwards to its original value

        self.petsc_snes.setTolerances(max_it=0)
        self.petsc_snes.setErrorIfNotConverged(False)
        self.petsc_snes.solve(None, self.x_vec)
        R_norm = self.petsc_snes.getFunctionNorm()
        self.petsc_snes.setTolerances(
            max_it=PETSc.Options().getInt(f"{self.options_prefix}snes_max_it")
        )
        self.petsc_snes.setErrorIfNotConverged(True)

        return R_norm
