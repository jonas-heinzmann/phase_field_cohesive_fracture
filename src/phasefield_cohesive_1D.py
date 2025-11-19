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
from model import CohesivePhaseField

# command line arguments
parser = argparse.ArgumentParser(
    description="1D implementation of the phase-field cohesive model"
)
parser.add_argument(
    "--narrow_element",
    action="store_true",
    help="insert a narrow element with ell=h/25 at the center of the domain, as mentioned in the caption of figure 20",
    default=False,
)
args = parser.parse_args()

# define solver parameters
solver_setup = {
    "tol_R": 1e-8,
    "nonzeroα_output": True,
    "εσ_output": True,
    "outdir": "./",
}

# =======================================================================================
# P R O B L E M   S E T U P
# =======================================================================================

# material parameters
L = 1.0
E = 10000.0
Gc = 0.001
ell = 0.05
σ_c = 5.0

ell_ch = Gc * E / (σ_c**2)
B = 2 * L / ell_ch

stdout(f"ell_ch = {ell_ch:.4e}, B = {B:.4e}")
stdout(f"ell/ ell_ch = {ell / ell_ch:.4e}")
stdout(f"σ_c = {σ_c:.4e}, E = {E:.4e}, Gc = {Gc:.4e}, ell = {ell:.4e}, L = {L:.4e}")
stdout(f"u_max = {5 * σ_c / E * 2 * L}")

Problem = problems.Bar(
    comm=comm,
    L=L,
    h=ell / 5,
    steps=500,
    u_max=5 * σ_c / E * 2 * L,
    narrow_element=args.narrow_element,
)

# create model
model = CohesivePhaseField(
    Problem.domain,
    E=E,
    Gc=Gc,
    ell=ell,
    p_c=σ_c,
)

# create elements
u_element = basix.ufl.element("CG", Problem.domain.basix_cell(), 1, shape=())
η_element = basix.ufl.element("DG", Problem.domain.basix_cell(), 0, shape=())
uη_element = basix.ufl.mixed_element([u_element, η_element])

α_element = basix.ufl.element("CG", Problem.domain.basix_cell(), 1, shape=())


# create function spaces
uη_space = dolfinx.fem.functionspace(Problem.domain, uη_element)
uη = dolfinx.fem.Function(uη_space)
(u, η) = ufl.split(uη)

α_space = dolfinx.fem.functionspace(Problem.domain, α_element)
α = dolfinx.fem.Function(α_space, name="α")

# define upper/ lower bound for damage and iteration state
α_lowerbound = dolfinx.fem.Function(α.function_space)
α_upperbound = dolfinx.fem.Function(α.function_space)

α_lowerbound.x.array[:] = 0.0
α_upperbound.x.array[:] = 1.0

# define Dirichlet boundary conditions
bcs_uη, bcs_α = Problem.create_bcs_mixed(uη_space, α_space)

# =======================================================================================
# V A R I A T I O N A L   F O R M U L A T I O N
# =======================================================================================

# measure for integration
dx = ufl.Measure("dx", Problem.domain)

# energies
E_pot = model.ψ_el(model.ε(u), α, η) * dx
E_frac = model.ψ_frac(α) * dx
E = E_pot + E_frac


# displacement problem
E_uη = ufl.derivative(E, uη)
E_uηuη = ufl.derivative(E_uη, uη)

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
solver_uη = SNESSolver(problem_uη, comm, "un_", convergence_monitoring=True)

# lower bound of uη solution
η_subspacedofs = uη_space.sub(1).dofmap.list

uη_lowerbound = solver_uη.x_vec.copy()
uη_lowerbound.array[:] = PETSc.NINFINITY  # displacement DOFs
uη_lowerbound.array[η_subspacedofs] = 0.0  # non-linear strain DOFs

# upper bound of uη solution
# (without loss of generality, we allow for non-zero η only in a single
# element to facilitate the comparison with the analytical solution)
uη_upperbound = solver_uη.x_vec.copy()
uη_upperbound.array[:] = PETSc.INFINITY
uη_upperbound.array[η_subspacedofs[0 : int(len(η_subspacedofs) / 2)]] = 0.0
uη_upperbound.array[η_subspacedofs[int(len(η_subspacedofs) / 2 + 1) :]] = 0.0

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
    # stress projection
    σ_space = dolfinx.fem.functionspace(Problem.domain, ("DG", 0))
    σ_nodes = dolfinx.fem.Function(σ_space, name="σ")
    σ_expr = dolfinx.fem.Expression(
        model.σ(model.ε(u), α, η), σ_space.element.interpolation_points()
    )

    # strain projection
    ε_space = dolfinx.fem.functionspace(Problem.domain, ("DG", 0))
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
csv_reaction = CSVWriter(f"{solver_setup['outdir']}/reaction.csv", "step\tu\tF\n")
csv_damage = CSVWriter(f"{solver_setup['outdir']}/damage.csv", "step\tα_max\tη_max\n")

# =======================================================================================
# I N C R E M E N T A L   S T A G G E R E D   S O L U T I O N
# =======================================================================================

# initial condition on the phase field
Problem.initial_α(solver_α, α, α_lowerbound)

# start timer
t_solution_start = time.time()

# incremental solution
step = 0
while step <= Problem.steps:
    stdout("=" * 75 + f"\nstep: {step:3d} / {Problem.steps:3d}\n" + "=" * 75)

    # update Dirichlet BCs
    u_bar = Problem.update_bcs(step)

    # update the lower bound of the phase field
    α_lowerbound.x.array[:] = α.x.array

    # staggered solution
    E_pot_step, E_frac_step = alternate_minimizer.solve(step)

    # write energies of converged state
    csv_energies.write(
        f"{step:04d}\t{E_pot_step:.8e}\t{E_frac_step:.8e}\t{E_pot_step + E_frac_step:.8e}\n",
    )

    # reaction forces
    F_react = Problem.compute_reaction(uη_reaction, virtual_work_form)
    csv_reaction.write(
        f"{step:04d}\t{u_bar:.8e}\t{F_react:.8e}\n",
    )

    # get maximum of the phase field
    α_max = comm.allreduce(α.x.petsc_vec.max()[1], op=MPI.MAX)

    # output fields
    if (
        (not solver_setup["nonzeroα_output"] and α_max > 0.0)
        or (solver_setup["nonzeroα_output"])
        or (step == 1)
    ):
        vtx_uη.write(step)
        vtx_α.write(step)

        if solver_setup["εσ_output"]:
            σ_nodes.interpolate(σ_expr)
            ε_nodes.interpolate(ε_expr)
            vtx_εσ.write(step)

    # get maximum of non-linear strains
    η_max = comm.allreduce(uη.sub(1).collapse().x.array.max(), op=MPI.MAX)
    csv_damage.write(
        f"{step:04d}\t{α_max:.8e}\t{η_max:.8e}\n",
    )

    # problem-dependent post-processing of the current step
    Problem.postprocess_step(step)

    # increment step
    step += 1

# stop timer
stdout(f"\ntotal time for solution: {time.time() - t_solution_start:.2f} s")

# close the output files
vtx_uη.close()
vtx_α.close()
if solver_setup["εσ_output"]:
    vtx_εσ.close()
