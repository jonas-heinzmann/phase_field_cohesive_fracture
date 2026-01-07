# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis

import ufl
import dolfinx

from utils import stdout

from typing import Literal


def elastic_constants(dim_: int, E_: float, ν_: float) -> tuple[float, float, float]:
    """
    helper function to compute bulk modulus and Lamé parameters from Young's modulus and Poisson's ratio
    """
    μ = E_ / (2 * (1 + ν_))
    λ = E_ * ν_ / ((1 + ν_) * (1 - (float(dim_) - 1) * ν_))
    κ = E_ / (float(dim_) * (1 - (float(dim_) - 1) * ν_))

    return μ, λ, κ


# =======================================================================================
# L I N E A R    E L A S T I C I T Y
# =======================================================================================
class LinearElasticity:
    """base class for linear elasticity in small strain setting"""

    def __init__(
        self,
        domain,
        E: float = 210.0,
        ν: float = 0.3,
        stress_state: Literal[
            "plane_strain", None
        ] = None,  # plane stress is not implemented
    ) -> None:
        # save elastic parameters
        self.E = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(E))
        self.ν = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(ν))

        # save dimensionality (derived from domain) and stress state
        self.dim = domain.topology.dim
        if self.dim != 2 and stress_state is not None:
            raise ValueError("stress_state can only be set for 2D problems.")
        self.stress_state = stress_state

        # compute Lame parameters
        if stress_state == "plane_strain":
            μ, λ, κ = elastic_constants(3, E, ν)
        else:
            μ, λ, κ = elastic_constants(self.dim, E, ν)

        self.μ = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(μ))
        self.κ = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(κ))
        self.λ = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(λ))

    def ε(self, u_):
        """small strain tensor"""

        if self.dim == 1:
            # directly extract the first component, to work with scalars and not a 1x1 matrix
            return ufl.nabla_grad(u_)[0]

        elif self.dim == 2 and self.stress_state == "plane_strain":
            # embed 2D tensor in 3x3 matrix s.t. UFL operators like ufl.dev can be used correctly
            return ufl.sym(
                ufl.as_matrix(
                    [
                        [u_[0].dx(0), u_[0].dx(1), 0.0],
                        [u_[1].dx(0), u_[1].dx(1), 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            )

        else:
            return ufl.sym(ufl.grad(u_))

    def ψ_el(self, ε_):
        """strain energy density"""

        if self.dim == 1:
            return 1 / 2 * self.E * ε_**2

        else:
            return self.λ / 2 * ufl.tr(ε_) ** 2 + self.μ * ufl.inner(ε_, ε_)

    def σ(self, ε_):
        """Cauchy stress tensor"""
        ε_var = ufl.variable(ε_)
        return ufl.diff(self.ψ_el(ε_var), ε_var)


# =======================================================================================
# C O H E S I V E    P H A S E - F I E L D    F R A C T U R E
# =======================================================================================
class CohesivePhaseField(LinearElasticity):
    """model for cohesive phase-field fracture"""

    def __init__(
        self,
        domain,
        E: float = 210.0,
        ν: float = 0.3,
        Gc: float = 1.0,
        ell: float = 0.1,
        p_c: float = 1.0,  # this is used as σ_c in the 1D case
        τ_c: float = 1.0,
        stress_state: Literal[
            "plane_strain", None
        ] = None,  # plane stress is not implemented
        r_norm: Literal["1", "2", "inf", None] = None,
    ) -> None:
        # initialize the LinearElasticity base class
        super().__init__(domain, E, ν, stress_state)

        # save cohesive phase-field parameters
        self.Gc = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(Gc))
        self.ell = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(ell))
        self.p_c = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(p_c))
        self.τ_c = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(τ_c))
        self.r_norm = r_norm

        # print parameters of the model
        stdout(f"E={E}")
        stdout(f"ν={ν}")
        stdout(f"Gc={Gc}")
        stdout(f"ell={ell}")
        stdout(f"p_c={p_c}")
        stdout(f"τ_c={τ_c}")
        stdout(f"r={r_norm}")

        # make sure no r-norm is defined for 1D problems
        if self.dim == 1:
            assert r_norm is None, "r-norm must not be defined for 1D problems"

        # compute ell_ch
        if self.dim == 1:
            ell_ch = self.Gc.value * self.E.value / (self.p_c.value**2)

        else:
            if self.r_norm == "1":
                ell_ch = (
                    2
                    * self.μ.value
                    * self.κ.value
                    * self.Gc.value
                    / (
                        2 * self.μ.value * self.p_c.value**2
                        + self.κ.value * self.τ_c.value**2
                    )
                )
            elif self.r_norm == "2" or self.r_norm == "inf":
                ell_ch = min(
                    self.κ.value * self.Gc.value / (self.p_c.value**2),
                    2 * self.μ.value * self.Gc.value / (self.τ_c.value**2),
                )

        # make sure that strain hardening is fulfilled
        if self.ell.value <= ell_ch:
            stdout(
                f"strain hardening condition fulfilled with ell = {self.ell.value:.4e} <= {ell_ch / 4:.4e} = ell_ch/4; ell/ell_ch = {self.ell.value / ell_ch:.4e}"
            )
        else:
            raise ValueError(
                f"strain hardening condition NOT fulfilled with ell = {self.ell.value:.4e} > {ell_ch / 4:.4e} = ell_ch/4; ell/ell_ch = {self.ell.value / ell_ch:.4e}"
            )

    def a(self, α_):
        """degradation function"""
        return (1 - α_) ** 2

    def ψ_frac(self, α_):
        """surface energy density (AT2 regularization)"""

        return (
            self.Gc
            / 2.0
            * (α_**2 / self.ell + self.ell * ufl.dot(ufl.grad(α_), ufl.grad(α_)))
        )

    def ψ_el(self, ε_, α_, p_, q_=None):
        """
        strain energy density

        note:
        1D: p_ = η
        2D/3D: p_ = tr(η), q_ = ||dev(η)||
        """

        if self.dim == 1:
            return 1 / 2 * self.E * (ε_ - p_) ** 2 + self.a(α_) * self.p_c * p_

        else:
            # small number to avert issues with non-differentiability
            eps = 3.0e-16

            # purely elastic contribution
            ψ = (
                self.κ / 2 * (ufl.tr(ε_) - p_) ** 2
                + self.μ
                * (ufl.sqrt(ufl.inner(ufl.dev(ε_), ufl.dev(ε_)) + eps**2) - q_) ** 2
            )

            # strength potential
            if self.r_norm == "1":
                π = self.p_c * p_ + self.τ_c * q_

            elif self.r_norm == "2":
                π = ufl.sqrt(self.p_c**2 * p_**2 + self.τ_c**2 * q_**2 + eps**2)

            elif self.r_norm == "inf":
                a = self.p_c * p_
                b = self.τ_c * q_

                # originally, the identity is max(a,b) = 1/2 * (a + b) + 1/2 * |a - b|
                # however, this is not differentiable, leading to numerical issues of gradient-based solvers
                # instead, we use a smooth approximation for the latter, abs(a - b) ≈ sqrt((a - b)^2 + eps)
                # where the small numerical value eps is to avert issues with non-differentiability (see above)
                # note: this still poses challenges for gradient-based solvers, which is why we employ the
                # bisection line search (exploiting the convexity of the problem)
                # eps too low makes the system ill-conditioned, eps too high gives numerical artifacts in the solution
                π = 1 / 2 * (a + b) + 1 / 2 * ufl.sqrt((a - b) ** 2 + 1e-17)

            return ψ + self.a(α_) * π

    def σ(self, ε_, α_, p_, q_=None):
        """Cauchy stress tensor"""
        ε_var = ufl.variable(ε_)
        return ufl.diff(self.ψ_el(ε_var, α_, p_, q_), ε_var)
