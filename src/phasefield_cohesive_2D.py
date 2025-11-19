# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis
#
# note: we solve the displacement and non-linear strain problem monolithically
# using a similar approach as in the DOLFINx demo (non-blocked direct solver):
# https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_stokes.html#non-blocked-direct-solver

from mpi4py import MPI
import petsc4py

comm = MPI.COMM_WORLD

petsc4py.init(["-options_file", "./petsc_options.yaml"], comm=comm)
from petsc4py import PETSc

import numpy as np
import dolfinx
import dolfinx.fem.petsc
from dolfinx.io import VTXWriter
from dolfinx.cpp.io import VTXMeshPolicy
import ufl
import basix
import time
import argparse

from petsc_interface import SNESProblem, SNESSolver
from solver import AlternateMinimizer
from utils import stdout, CSVWriter, MixedVTXWriter
import problems
from model import CohesivePhaseField, elastic_constants

# command line arguments
parser = argparse.ArgumentParser(
    description="2D/ 2D plane strain implementation of the phase-field cohesive model"
)
parser.add_argument(
    "--problem",
    type=str,
    help="problem to be computed, either ElasticDomainBlock (default) or CohesiveForcesBlockRigid",
    default="ElasticDomainBlock",
)
parser.add_argument(
    "--ell_ellch",
    type=float,
    help="target ratio of ell/ell_ch (default: 1/4)",
    default=1 / 4,
)
parser.add_argument(
    "--rnorm",
    type=str,
    help="norm for the strength potential, either 1, 2 (default) or inf",
    default="2",
)
parser.add_argument(
    "--theta",
    type=float,
    help="loading angle (default: 0.0)",
    default=0.0,
)
parser.add_argument(
    "--maxalpha",
    type=float,
    help="maximum alpha after which to stop the computations (default: 1e-5)",
    default=1e-5,
)
args = parser.parse_args()


# define solver parameters
solver_setup = {
    "tol_R": 1e-8,
    "nonzeroα_output": False,
    "εσ_output": False,
    "αmax_stop": args.maxalpha,
    "outdir": "./",
}

# =======================================================================================
# P R O B L E M   S E T U P
# =======================================================================================

# material parameters
E = 100
ν = 0.3
μ, _, κ = elastic_constants(3, E, ν)
Gc = 0.2
ell = 0.025

# ratio pc/τc
pc_τc = 1.0

# determine p_c and τ_c based on strain hardening limit
if args.ell_ellch == 1 / 4:
    args.ell_ellch -= 1e-6  # make sure we are slightly below the limit

if (args.rnorm == "2") or (args.rnorm == "inf"):
    w_1 = Gc / (2 * ell)
    p_c_lim = 2 * np.sqrt(args.ell_ellch) * np.sqrt(κ * w_1 / 2.0)
    τ_c_lim = 2 * np.sqrt(args.ell_ellch) * np.sqrt(μ * w_1)
    if pc_τc <= 1.0:
        p_c = pc_τc * min(p_c_lim, τ_c_lim)
        τ_c = min(p_c_lim, τ_c_lim)
    else:
        p_c = min(p_c_lim, τ_c_lim)
        τ_c = 1 / pc_τc * min(p_c_lim, τ_c_lim)

elif args.rnorm == "1":
    p_c_τ_c_lim = np.sqrt(2 * μ * κ * Gc / ((2 * μ + κ) * ell / args.ell_ellch))
    if pc_τc <= 1.0:
        p_c = pc_τc * p_c_τ_c_lim
        τ_c = p_c_τ_c_lim
    else:
        p_c = p_c_τ_c_lim
        τ_c = 1 / pc_τc * p_c_τ_c_lim


if args.problem == "ElasticDomainBlock":
    Problem = problems.ElasticDomainBlock(comm=comm, h=ell / 5, Θ=args.theta)
elif args.problem == "CohesiveForcesBlockRigid":
    Problem = problems.CohesiveForcesBlockRigid(comm=comm, h=ell / 5, Θ=args.theta)
else:
    raise ValueError(f"Unknown problem type: {args.problem}")

# =======================================================================================

# create model
model = CohesivePhaseField(
    Problem.domain,
    E=E,
    ν=ν,
    Gc=Gc,
    ell=ell,
    p_c=p_c,
    τ_c=τ_c,
    stress_state="plane_strain",
    r_norm=args.rnorm,
)

# create elements
u_element = basix.ufl.element("CG", Problem.domain.basix_cell(), 1, shape=(2,))
ηtr_element = basix.ufl.element("DG", Problem.domain.basix_cell(), 0)
ηdev_element = basix.ufl.element("DG", Problem.domain.basix_cell(), 0)
α_element = basix.ufl.element("CG", Problem.domain.basix_cell(), 1)

uη_element = basix.ufl.mixed_element([u_element, ηtr_element, ηdev_element])

# create function spaces and functions
uη_space = dolfinx.fem.functionspace(Problem.domain, uη_element)
uη = dolfinx.fem.Function(uη_space)
(u, ηtr, ηdev) = ufl.split(uη)

α_space = dolfinx.fem.functionspace(Problem.domain, α_element)
α = dolfinx.fem.Function(α_space, name="α")

# define upper/ lower bound for damage and iteration state
α_lowerbound = dolfinx.fem.Function(α.function_space)
α_upperbound = dolfinx.fem.Function(α.function_space)

α_lowerbound.x.array[:] = 0.0
α_upperbound.x.array[:] = 1.0

# boundary conditions
bcs_uη, bcs_α = Problem.create_bcs_mixed(uη_space, α_space)


# =======================================================================================
# V A R I A T I O N A L   F O R M U L A T I O N
# =======================================================================================

# measure for integration
dx = ufl.Measure("dx", Problem.domain)

# energies
E_pot = model.ψ_el(model.ε(u), α, ηtr, ηdev) * dx
E_frac = model.ψ_frac(α) * dx
E = E_pot + E_frac

# displacement problem
E_uη = ufl.derivative(E, uη)
E_uηuη = ufl.derivative(E_uη, uη)

# phase-field problem
E_α = ufl.derivative(E, α)
E_αα = ufl.derivative(E_α, α)

# energy forms
E_pot_form = dolfinx.fem.form(E_pot)
E_frac_form = dolfinx.fem.form(E_frac)

# create specific test function as well as a form of the action to compute reaction forces
uη_reaction = dolfinx.fem.Function(uη_space)
virtual_work_form = dolfinx.fem.form(ufl.action(E_uη, uη_reaction))

# =======================================================================================
# S O L V E R
# =======================================================================================

opts = PETSc.Options()

# solver for the displacement problem
problem_uη = SNESProblem(
    dolfinx.fem.form(E_uη),
    dolfinx.fem.form(E_uηuη),
    uη,
    bcs_uη,
    objective=E_pot_form,
)
solver_uη = SNESSolver(problem_uη, comm, "untrndev_", convergence_monitoring=True)


# lower bound of uη solution
ηtr_subspacedofs = uη_space.sub(1).dofmap.list
ηdev_subspacedofs = uη_space.sub(2).dofmap.list

uη_lowerbound = solver_uη.x_vec.copy()
uη_lowerbound.array[:] = PETSc.NINFINITY  # displacement DOFs
uη_lowerbound.array[ηtr_subspacedofs] = 0.0  # ηtr DOFs
uη_lowerbound.array[ηdev_subspacedofs] = 0.0  # ηdev DOFs

# upper bound of uη solution
uη_upperbound = solver_uη.x_vec.copy()
uη_upperbound.array[:] = PETSc.INFINITY

# set upper bounds for solver of uη problem
solver_uη.petsc_snes.setVariableBounds(uη_lowerbound, uη_upperbound)

# non-linear solver for the phase-field problem
problem_α = SNESProblem(dolfinx.fem.form(E_α), dolfinx.fem.form(E_αα), α, bcs_α)
solver_α = SNESSolver(problem_α, comm, "a_", convergence_monitoring=True)
solver_α.petsc_snes.setVariableBounds(
    α_lowerbound.x.petsc_vec, α_upperbound.x.petsc_vec
)

# create staggered solver object
alternate_minimizer = AlternateMinimizer(
    comm,
    solver_uη,
    solver_α,
    E_pot_form,
    E_frac_form,
    tol_Rnorm=solver_setup["tol_R"],
)

# =======================================================================================
# P O S T P R O C E S S I N G
# =======================================================================================


# open ADIOS2 files
vtx_uη = MixedVTXWriter(
    comm,
    [
        f"{solver_setup['outdir']}/solution_u.bp",
        f"{solver_setup['outdir']}/solution_η.bp",
    ],
    uη,
    engine="BP4",
    mesh_policy=VTXMeshPolicy.reuse,
)
vtx_α = VTXWriter(
    comm,
    f"{solver_setup['outdir']}/solution_α.bp",
    α,
    engine="BP4",
    mesh_policy=VTXMeshPolicy.reuse,
)

# strain and stress output
if solver_setup["εσ_output"]:
    if model.stress_state == "plane_strain":
        σε_dim = 3
    else:
        σε_dim = model.dim

    # stress projection
    σ_space = dolfinx.fem.functionspace(
        Problem.domain, ("DG", 0, (σε_dim, σε_dim), True)
    )
    σ_nodes = dolfinx.fem.Function(σ_space, name="σ")
    ε_var = ufl.variable(model.ε(u))
    σ_expr = dolfinx.fem.Expression(
        model.σ(model.ε(u), α, ηtr, ηdev), σ_space.element.interpolation_points()
    )

    # strain projection
    ε_space = dolfinx.fem.functionspace(
        Problem.domain, ("DG", 0, (σε_dim, σε_dim), True)
    )
    ε_nodes = dolfinx.fem.Function(ε_space, name="ε")
    ε_expr = dolfinx.fem.Expression(model.ε(u), ε_space.element.interpolation_points())

    # open ADIOS2 file
    vtx_εσ = VTXWriter(
        comm,
        f"{solver_setup['outdir']}/solution_εσ.bp",
        [ε_nodes, σ_nodes],
        engine="BP4",
        mesh_policy=VTXMeshPolicy.reuse,
    )

# initialize monitoring output
csv_energies = CSVWriter(
    f"{solver_setup['outdir']}/energies.csv", "step\tE_pot\tE_frac\tE_tot\n"
)
csv_reaction = CSVWriter(
    f"{solver_setup['outdir']}/reaction.csv", "step\tu_x\tu_y\tF_x\tF_y\n"
)
csv_damage = CSVWriter(
    f"{solver_setup['outdir']}/damage.csv", "step\tα_max\tηtr_max\tηdev_max\n"
)

# =======================================================================================
# I N C R E M E N T A L   S T A G G E R E D   S O L U T I O N
# =======================================================================================

# initial condition on the phase field
Problem.initial_α(solver_α, α, α_lowerbound)

# start timer
t_solution_start = time.time()

# loop over the load step increments
step = 0
while step <= Problem.steps:
    stdout("=" * 75 + f"\nstep: {step:3d} / {Problem.steps - 1:3d}\n" + "=" * 75)

    # update Dirichlet BCs
    u_x, u_y = Problem.update_bcs(step)

    # update the lower bound of the phase field
    α_lowerbound.x.array[:] = α.x.array

    # staggered solution
    E_pot_step, E_frac_step = alternate_minimizer.solve(step)

    # write energies of converged state
    csv_energies.write(
        f"{step:04d}\t{E_pot_step:.8e}\t{E_frac_step:.8e}\t{E_pot_step + E_frac_step:.8e}\n",
    )

    # reaction forces
    F_x, F_y = Problem.compute_reaction(uη_reaction, virtual_work_form)
    csv_reaction.write(
        f"{step:04d}\t{u_x:.8e}\t{u_y:.8e}\t{F_x:.8e}\t{F_y:.8e}\n",
    )

    # get maximum of the phase field
    α_max = comm.allreduce(α.x.petsc_vec.max()[1], op=MPI.MAX)

    # output fields
    if (
        (not solver_setup["nonzeroα_output"] and α_max > 0.0)
        or (solver_setup["nonzeroα_output"])
        or (step == 1)
        or (step == Problem.steps - 1)
    ):
        vtx_uη.write(step)
        vtx_α.write(step)

        if solver_setup["εσ_output"]:
            σ_nodes.interpolate(σ_expr)
            ε_nodes.interpolate(ε_expr)
            vtx_εσ.write(step)

    # get maximum of non-linear strains
    ηtr_max = comm.allreduce(uη.sub(1).collapse().x.array.max(), op=MPI.MAX)
    ηdev_max = comm.allreduce(uη.sub(2).collapse().x.array.max(), op=MPI.MAX)
    csv_damage.write(
        f"{step:04d}\t{α_max:.8e}\t{ηtr_max:.8e}\t{ηdev_max:.8e}\n",
    )

    # problem-dependent post-processing of the current step
    Problem.postprocess_step(step)

    # stop the computations once crack as nucleated
    if α_max > solver_setup["αmax_stop"]:
        stdout(
            f"\nstopping due to α_max = {α_max:.3e} > {solver_setup['αmax_stop']:.3e}"
        )
        break

    # increment step
    step += 1


# stop timer
stdout(f"\ntotal time for solution: {time.time() - t_solution_start:.2f} s")

# close the output file
vtx_uη.close()
vtx_α.close()
if solver_setup["εσ_output"]:
    vtx_εσ.close()
