# src/diff_weighted_fields/LPT.py
from __future__ import annotations
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
from .field import Field1D
from .grid import Grid1D
from .generators import GaussianFieldGenerator1D
import jax
import jax.numpy as jnp
from .utils import PowerSpectrum_batch
@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Zeldovich1D(Field1D):
    """
    1D Zeldovich approximation initializer.  Inherits from Field1D but
    uses the same `grid` (and `Pk_func`) coming from a GaussianFieldGenerator1D.
    """

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
    @jax.jit
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
    
    def make_realization_batch(self,D,theta,R_smooth,C,noise):
        gen = self.gaussian_gen            
        kgrid     = self.fine_grid.kgrid           # shape: (N,)
        kgrid_abs = self.fine_grid.kgrid_abs       # shape: (N,)
        kmesh_abs = self.grid.kgrid_abs
        def make_single(D,theta,noise):
            linear_field = gen.make_realization_from_noise(theta, noise)
            delta_lin_k = linear_field.delta_k   # shape: (N,) complex

            # Compute displacement field in Fourier space
            psi_k = 1j * kgrid / (kgrid_abs**2) * delta_lin_k
            psi_k = psi_k.at[0].set(0)
            psi_q = jnp.real(jnp.fft.ifft(psi_k) * self.fine_grid.norm_ifft)        

            x_eulerian = self._q + D * psi_q
            self.paint_from_positions(x_eulerian, scheme=self.scheme)
            self.compute_fft()
            return self.one_plus_delta, self.delta_k
        
        # Vectorize over the first axis of noise:
        one_plus_delta_batch, delta_k_batch = jax.vmap(make_single, in_axes=(None, None, 0))(D, theta, noise)
        delta_k_smooth_batch = delta_k_batch * jnp.exp(-0.5 * (kmesh_abs * R_smooth)**2)
        delta_smooth_batch = jnp.fft.ifft(delta_k_smooth_batch).real * self.grid.norm_ifft
        m_array = jnp.stack([delta_smooth_batch**0,
                             delta_smooth_batch**1,
                             delta_smooth_batch**2,
                             delta_smooth_batch**3,])
        
        m_batch = jnp.einsum('ij,jlm->ilm', C, m_array)
        rho_weighted = one_plus_delta_batch[None,:,:]*m_batch
        rho_bar = jnp.mean(rho_weighted, axis = -1)
        delta_weighted = rho_weighted/rho_bar[:,:,None] - 1
        delta_weighted_k = jnp.fft.fft(delta_weighted) * self.grid.norm_fft
        return PowerSpectrum_batch(delta_weighted_k, jnp.ones_like(delta_weighted),self.grid)
    
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
    
     # ---------------- PyTree registration ----------------
    def tree_flatten(self):
        # We want to flatten the following as leaves: delta, delta_k, one_plus_delta, W (from Field1D)
        # All other attributes are static: gaussian_gen, fine_grid, paint_grid, _q, scheme
        # Use the same presence flags as Field1D
        leaves = []
        is_present = [False, False, False, False]  # delta, delta_k, one_plus_delta, W
        if self.delta is not None:
            leaves.append(self.delta)
            is_present[0] = True
        if self.delta_k is not None:
            leaves.append(self.delta_k)
            is_present[1] = True
        if self.one_plus_delta is not None:
            leaves.append(self.one_plus_delta)
            is_present[2] = True
        if self.W is not None:
            leaves.append(self.W)
            is_present[3] = True
        # Aux data: gaussian_gen, fine_grid, paint_grid, _q, scheme, is_present
        aux_data = (self.gaussian_gen, self.fine_grid, self.paint_grid, self._q, self.scheme, tuple(is_present))
        return tuple(leaves), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        gaussian_gen, fine_grid, paint_grid, _q, scheme, is_present = aux_data
        # Construct instance without calling __init__
        obj = cls.__new__(cls)
        # Initialize base Field1D with paint_grid
        Field1D.__init__(obj, grid=paint_grid)
        # Set attributes
        obj.gaussian_gen = gaussian_gen
        obj.fine_grid = fine_grid
        obj.paint_grid = paint_grid
        obj._q = _q
        obj.scheme = scheme
        # Restore leaves
        idx = 0
        if is_present[0]:
            obj.delta = children[idx]
            idx += 1
        else:
            obj.delta = None
        if is_present[1]:
            obj.delta_k = children[idx]
            idx += 1
        else:
            obj.delta_k = None
        if is_present[2]:
            obj.one_plus_delta = children[idx]
            idx += 1
        else:
            obj.one_plus_delta = None
        if is_present[3]:
            obj.W = children[idx]
            idx += 1
        else:
            obj.W = None
        return obj