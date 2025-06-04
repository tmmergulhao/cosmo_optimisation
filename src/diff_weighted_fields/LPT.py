# src/diff_weighted_fields/LPT.py
from __future__ import annotations
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
from .field import Field1D
from .grid import Grid1D
from .generators import GaussianFieldGenerator1D
import jax.numpy as jnp
from .utils import cic_paint_1d, tsc_paint_1d
import jax

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Zeldovich1D(Field1D):
    """
    1D Zeldovich approximation initializer.  Inherits from Field1D but
    uses the same `grid` (and `Pk_func`) coming from a GaussianFieldGenerator1D.
    """

    # We store only the GaussianFieldGenerator1D *factory* data: (grid, Pk_func).
    # In the constructor, we build a new GaussianFieldGenerator1D from those two.
    gaussian_gen: GaussianFieldGenerator1D

    def __init__(
        self,
        gaussian_gen: GaussianFieldGenerator1D,
        paint_grid: Grid1D | None = None,
        scheme: str = "cic"
    ):
        """
        If `paint_grid` is None, we use the same grid as `gaussian_gen` for painting.
        Otherwise, `paint_grid` is used for mass assignment and FFT, while
        displacements are still computed on `gaussian_gen.grid`.
        """
        self.gaussian_gen = gaussian_gen

        # The fine grid used for displacements:
        self.fine_grid = gaussian_gen.grid

        # The grid used for painting/FFT is either paint_grid or fine_grid:
        self.paint_grid = paint_grid if paint_grid is not None else self.fine_grid
        super().__init__(grid=self.paint_grid)

        # Lagrangian coordinates always on the fine grid:
        N_fine = self.fine_grid.shape[0]
        self._q = jnp.linspace(0.0, self.fine_grid.L, num=N_fine, endpoint=False)
        self.scheme = scheme

    def make_realization(self, D, theta, noise, displacement = False) -> Zeldovich1D:
        # 1. Generate linear density on the fine grid:
        linear_field = self.gaussian_gen.make_realization_from_noise(theta, noise)
        delta_lin_k = linear_field.delta_k   # shape: (N,) complex

        # 2. Compute displacement in Fourier space, but never divide by zero:
        kgrid     = self.fine_grid.kgrid           # shape: (N,)
        kgrid_abs = self.fine_grid.kgrid_abs       # shape: (N,)

        # Compute displacement field in Fourier space
        psi_k = 1j * kgrid / (kgrid_abs**2) * delta_lin_k
        psi_k = psi_k.at[0].set(0)
        psi_q = jnp.real(jnp.fft.ifft(psi_k) * self.fine_grid.norm_ifft)        
        if displacement:
            return psi_q
        x_eulerian = self._q + D * psi_q


        # …paint, FFT, normalize, etc.…
        self.paint_from_positions(x_eulerian, scheme=self.scheme)
        self.compute_fft()
        return self
    
    def ComputeBasis(self, Rsmooth) -> Zeldovich1D:
        """
        Smooth the field using a Gaussian kernel with standard deviation `sigma`.
        This modifies the field in place.
        """
        delta_k_smooth = self.delta_k * jnp.exp(-0.5 * (self.grid.kgrid_abs * Rsmooth)**2)
        delta_smooth = jnp.fft.ifft(delta_k_smooth).real * self.grid.norm_ifft
        self.m_array = jnp.stack([
                                jnp.ones_like(delta_smooth), 
                                delta_smooth,
                                delta_smooth**2,
                                delta_smooth**3,]
                                , axis=0)
        
    def WeightedChild(self, C):
        m = jnp.dot(C,self.m_array)
        rho_weighted = m*self.one_plus_delta
        mean_rho_marked = jnp.mean(rho_weighted)
        delta_marked = rho_weighted / mean_rho_marked - 1.0
        field_marked = Field1D(grid=self.grid)
        field_marked.assign_from_real_space(delta_marked)
        field_marked.compute_fft()
        field_marked.W = self.W
        return field_marked
    