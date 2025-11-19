# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis

import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from utils import CSVWriter


# =======================================================================================
# O N E   D I M E N S I O N A L   B A R
# =======================================================================================
class Bar:
    """
    1D bar which is clamped on the left end, and has a prescribed displacement on the right end,
    Section 5.2 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.005,
        steps: int = 500,
        u_max: float = 0.001,
        narrow_element: bool = False,
    ) -> None:
        # save geometric and mesh parameters
        self.L = L
        self.h = h

        # define mesh
        n = int(2 * L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at x=0

        if narrow_element:
            # insert one more element in the middle, which is then shrunk
            n += 1

        # note: the bar has a length of 2L, as in the theoretical explanations
        self.domain = dolfinx.mesh.create_interval(
            comm,
            nx=n,
            points=[np.array([-self.L]), np.array([self.L])],
            ghost_mode=dolfinx.mesh.GhostMode.none,
        )

        # locally change a single element to be much smaller
        if narrow_element:
            assert comm.size == 1, (
                "narrow element mesh adjustment only works in serial with the current implementation"
            )

            # create array with original spacing without the small element
            x_h = np.linspace(-self.L, self.L, n)

            # replace the nodes with the original spacing one except the middle one
            self.domain.geometry.x[: int(n / 2), 0] = x_h[: int(n / 2)]
            self.domain.geometry.x[int(n / 2 + 1) :, 0] = x_h[int(n / 2) :]

            # adjust one coordinate of the middle element to have size h/5
            self.domain.geometry.x[int(n / 2), 0] = (
                self.domain.geometry.x[int(n / 2 + 1), 0] - h / 5
            )

        # initialize load stepping
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

    def create_bcs_mixed(self, uη_space, α_space):
        # Get subspace of mixed space
        u_subspace = uη_space.sub(0)

        # left edge: fixed (zero) displacement
        facets_left = dolfinx.mesh.locate_entities_boundary(
            self.domain,
            self.domain.topology.dim - 1,
            lambda x: np.isclose(x[0], -self.L),
        )
        dofs_u_left = dolfinx.fem.locate_dofs_topological(
            u_subspace, self.domain.topology.dim - 1, facets_left
        )
        self.bc_u_left = dolfinx.fem.Constant(self.domain, 0.0)
        bc_u_left = dolfinx.fem.dirichletbc(self.bc_u_left, dofs_u_left, u_subspace)

        # right edge: prescribed displacement
        facets_right = dolfinx.mesh.locate_entities_boundary(
            self.domain,
            self.domain.topology.dim - 1,
            lambda x: np.isclose(x[0], self.L),
        )
        dofs_u_right = dolfinx.fem.locate_dofs_topological(
            u_subspace, self.domain.topology.dim - 1, facets_right
        )
        self.bc_u_right = dolfinx.fem.Constant(self.domain, 0.0)
        bc_u_right = dolfinx.fem.dirichletbc(self.bc_u_right, dofs_u_right, u_subspace)

        bcs_uη = [bc_u_left, bc_u_right]
        self.bcs_uη = bcs_uη

        # no Dirichlet BCs on the phase field
        bcs_α = []

        return bcs_uη, bcs_α

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # set everything to zero except considered BC
        uη_reaction.x.petsc_vec.set(0.0)
        self.bc_u_left.value = 1.0
        self.bc_u_right.value = 0.0
        dolfinx.fem.set_bc(uη_reaction.x.petsc_vec, self.bcs_uη)
        F_x_local = dolfinx.fem.assemble_scalar(virtual_work_form)
        F_x = MPI.COMM_WORLD.allreduce(np.sum(F_x_local), op=MPI.SUM)

        # reset BCs to their original values
        self.bc_u_left.value = 0.0

        return F_x

    def update_bcs(self, step):
        u_bar = self.loadsteps[step]
        self.bc_u_right.value = u_bar

        return u_bar

    def postprocess_step(self, step):
        pass


# =======================================================================================
# E L A S T I C   D O M A I N   B L O C K
# =======================================================================================
class ElasticDomainBlock:
    """
    quadratic block with a homogeneous deformation state, used to validate the strength surface,
    Section 5.3 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.01,
        Θ: float = 25.0,
        steps: int = 2500,
        u_max: float = 0.25,
    ):
        # save parameters
        self.L = L

        # create mesh
        n = int(self.L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at L/2

        self.domain = dolfinx.mesh.create_rectangle(
            comm,
            [np.array([-self.L / 2, -self.L / 2]), np.array([self.L / 2, self.L / 2])],
            [n, n],
            dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.none,
        )

        # initialize load stepping
        self.Θ = Θ
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

        # prepare post-processing of each step (output of deformation state)
        self.csv_deformation = CSVWriter(
            "deformation.csv", "step\tε_xx\tε_yy\tεtr\tεdev\n"
        )

    def create_bcs_mixed(self, uη_space, α_space):
        ux_space = uη_space.sub(0).sub(0).collapse()[0]
        self.bc_ux = dolfinx.fem.Function(ux_space)

        dofs_ux = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(0), ux_space),
            lambda x: np.isclose(np.abs(x[0]), self.L / 2),
        )
        bc_ux = dolfinx.fem.dirichletbc(self.bc_ux, dofs_ux, uη_space)

        uy_space = uη_space.sub(0).sub(1).collapse()[0]
        self.bc_uy = dolfinx.fem.Function(uy_space)

        dofs_uy = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(1), uy_space),
            lambda x: np.isclose(np.abs(x[1]), self.L / 2),
        )
        bc_uy = dolfinx.fem.dirichletbc(self.bc_uy, dofs_uy, uη_space)

        bcs_upq = [bc_ux, bc_uy]

        # Dirichlet BCs for the phase field
        dofs_α = dolfinx.fem.locate_dofs_geometrical(
            α_space,
            lambda x: np.logical_or(
                np.isclose(np.abs(x[0]), (self.L / 2)),
                np.isclose(np.abs(x[1]), (self.L / 2)),
            ),
        )
        zero_α = dolfinx.fem.Function(α_space)
        with zero_α.x.petsc_vec.localForm() as bc_local:
            bc_local.set(0.0)

        bc_α0 = dolfinx.fem.dirichletbc(zero_α, dofs_α)
        bcs_α = [bc_α0]

        return bcs_upq, bcs_α

    def update_bcs(self, step):
        loadstep_factor_x = self.loadsteps[step] * np.cos(np.pi * self.Θ / 180)
        loadstep_factor_y = self.loadsteps[step] * np.sin(np.pi * self.Θ / 180)

        self.bc_ux.interpolate(lambda x: loadstep_factor_x * np.sign(x[0]))
        self.bc_uy.interpolate(lambda x: loadstep_factor_y * np.sign(x[1]))

        return loadstep_factor_x, loadstep_factor_y

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # note: computing a reaction force does not make sense for this test since
        # it has a symmetric prescribed displacement on opposing edges

        return 0.0, 0.0

    def postprocess_step(self, step):
        # get strain state in xx-yy space
        ε_xx = 2 * np.cos(np.pi * self.Θ / 180) * self.loadsteps[step]
        ε_yy = 2 * np.sin(np.pi * self.Θ / 180) * self.loadsteps[step]

        # get strain in volumetric-deviatoric space
        εtr = ε_xx + ε_yy
        εdev = np.sqrt(2 / 3 * (ε_xx**2 + ε_yy**2 - ε_xx * ε_yy))

        self.csv_deformation.write(
            f"{step:04d}\t{ε_xx:.8e}\t{ε_yy:.8e}\t{εtr:.8e}\t{εdev:.8e}\n"
        )


# =======================================================================================
# C O H E S I V E   F O R C E S   B L O C K
# =======================================================================================
class CohesiveForcesBlockRigid:
    """
    quadratic block with a pre-defined deformation and crack location, used to validate the cohesive law,
    Section 5.4 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.01,
        Θ: float = 25.0,
        steps: int = 2500,
        u_max: float = 0.5,
        narrow_elementrow: bool = True,
    ):
        # save geometric parameters
        self.L = L

        # create mesh
        n = int(self.L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at L/2

        self.n = n

        self.domain = dolfinx.mesh.create_rectangle(
            comm,
            [np.array([-self.L / 2, -self.L / 2]), np.array([self.L / 2, self.L / 2])],
            [n, n],
            dolfinx.mesh.CellType.triangle,
            ghost_mode=dolfinx.mesh.GhostMode.none,
        )

        # locally change a single element to be much smaller
        if narrow_elementrow:
            assert comm.size == 1, (
                "narrow element mesh adjustment only works in serial with the current implementation"
            )

            # project nodal positions to rescaled coordinates
            h_orig = self.L / self.n
            h_narrow = h_orig / 5
            a = (h_orig - h_narrow) / (h_orig - self.L)
            b = -a * self.L / 2

            right_block = np.where(self.domain.geometry.x[:, 0] > 0)[0]
            self.domain.geometry.x[right_block, 0] = (1 - a) * self.domain.geometry.x[
                right_block, 0
            ] - b

            left_block = np.where(self.domain.geometry.x[:, 0] < 0)[0]
            self.domain.geometry.x[left_block, 0] = (
                self.domain.geometry.x[left_block, 0]
                - a * self.domain.geometry.x[left_block, 0]
                + b
            )

        # initialize load stepping
        self.Θ = Θ
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

    def create_bcs_mixed(self, uη_space, α_space):
        # Dirichlet BCs on ux
        ux_space = uη_space.sub(0).sub(1).collapse()[0]
        uy_space = uη_space.sub(0).sub(1).collapse()[0]

        # left half: ux zero
        self.bc_ux_fixed = dolfinx.fem.Function(ux_space)
        self.bc_ux_fixed.x.array[:] = 0.0
        dofs_ux_fixed = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(0), ux_space),
            lambda x: x[0] < 0,
        )
        bc_ux_fixed = dolfinx.fem.dirichletbc(self.bc_ux_fixed, dofs_ux_fixed, uη_space)

        # right half: ux prescribed
        self.bc_ux_prescribed = dolfinx.fem.Function(ux_space)
        self.bc_ux_prescribed.x.array[:] = 0.0
        dofs_ux_prescribed = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(0), ux_space),
            lambda x: x[0] > 0,
        )
        bc_ux_prescribed = dolfinx.fem.dirichletbc(
            self.bc_ux_prescribed, dofs_ux_prescribed, uη_space
        )

        # left half: uy fixed
        self.bc_uy_fixed = dolfinx.fem.Function(uy_space)
        self.bc_uy_fixed.x.array[:] = 0.0
        dofs_uy_fixed = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(1), uy_space),
            lambda x: x[0] < 0,
        )
        bc_uy_fixed = dolfinx.fem.dirichletbc(self.bc_uy_fixed, dofs_uy_fixed, uη_space)

        # right half: uy prescribed
        self.bc_uy_prescribed = dolfinx.fem.Function(uy_space)
        self.bc_uy_prescribed.x.array[:] = 0.0
        dofs_uy_prescribed = dolfinx.fem.locate_dofs_geometrical(
            (uη_space.sub(0).sub(1), uy_space),
            lambda x: x[0] > 0,
        )
        bc_uy_prescribed = dolfinx.fem.dirichletbc(
            self.bc_uy_prescribed, dofs_uy_prescribed, uη_space
        )

        # store all BCs on u
        bcs_upq = [bc_ux_fixed, bc_uy_fixed, bc_ux_prescribed, bc_uy_prescribed]
        self.bcs_upq = bcs_upq

        # Dirichlet BCs for the phase field
        bcs_α = []

        return bcs_upq, bcs_α

    def update_bcs(self, step):
        loadstep_factor_x = self.loadsteps[step] * np.cos(np.pi * self.Θ / 180)
        loadstep_factor_y = self.loadsteps[step] * np.sin(np.pi * self.Θ / 180)

        self.bc_ux_prescribed.x.array[:] = loadstep_factor_x
        self.bc_uy_prescribed.x.array[:] = loadstep_factor_y

        return loadstep_factor_x, loadstep_factor_y

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # set everything to zero except fixed ux
        uη_reaction.value = 0.0
        self.bc_ux_fixed.x.array[:] = 1.0
        self.bc_uy_fixed.x.array[:] = 0.0
        self.bc_ux_prescribed.x.array[:] = 0.0
        self.bc_uy_prescribed.x.array[:] = 0.0

        # assemble x-reaction
        dolfinx.fem.set_bc(uη_reaction.x.petsc_vec, self.bcs_upq)
        F_x_local = dolfinx.fem.assemble_scalar(virtual_work_form)
        F_x = MPI.COMM_WORLD.allreduce(np.sum(F_x_local), op=MPI.SUM)

        # reset fixed ux
        self.bc_ux_fixed.x.array[:] = 0.0

        # set everything to zero except fixed uy
        uη_reaction.x.array[:] = 0.0
        self.bc_ux_fixed.x.array[:] = 0.0
        self.bc_uy_fixed.x.array[:] = 1.0
        self.bc_ux_prescribed.x.array[:] = 0.0
        self.bc_uy_prescribed.x.array[:] = 0.0

        # assemble x-reaction
        dolfinx.fem.set_bc(uη_reaction.x.petsc_vec, self.bcs_upq)
        F_y_local = dolfinx.fem.assemble_scalar(virtual_work_form)
        F_y = MPI.COMM_WORLD.allreduce(np.sum(F_y_local), op=MPI.SUM)

        # reset fixed ux
        self.bc_uy_fixed.x.array[:] = 0.0

        return F_x, F_y

    def postprocess_step(self, step):
        pass
