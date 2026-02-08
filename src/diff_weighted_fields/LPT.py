# src/diff_weighted_fields/LPT.py
from __future__ import annotations
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
from .field import Field1D, Field3D
from .grid import Grid1D, Grid3D
from .generators import GaussianFieldGenerator1D, GaussianFieldGenerator3D
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from .utils import PowerSpectrum_batch, MT2_pk_batch
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
        grid: Grid1D | None = None,
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
        self.grid = grid if grid is not None else self.fine_grid
        super().__init__(grid=self.grid)

        # Lagrangian coordinates always on the fine grid:
        N_fine = self.fine_grid.shape[0]
        self._q = jnp.linspace(0.0, self.fine_grid.L, num=N_fine, endpoint=False, dtype=jnp.float64)
        self.scheme = scheme

    def compute_displacement(self, theta, noise):
        # 1. Generate linear density on the fine grid:
        linear_field = self.gaussian_gen.make_realization_from_noise(theta, noise)
        delta_lin_k = linear_field.delta_k   # shape: (N,) complex

        # 2. Compute displacement in Fourier space, but never divide by zero:
        kgrid     = self.fine_grid.kgrid           # shape: (N,)
        kgrid_abs = self.fine_grid.kgrid_abs       # shape: (N,)

        # Compute displacement field in Fourier space
        psi_k = 1j * kgrid / (kgrid_abs**2) * delta_lin_k
        #psi_k = psi_k.at[0].set(0)
        psi_q = jnp.real(jnp.fft.ifft(psi_k) * self.fine_grid.norm_ifft)        
        return psi_q
    
    def make_realization(self, D, theta, noise) -> Zeldovich1D:
        #generate the displacement field and paint it onto the grid
        psi_q = self.compute_displacement(theta, noise)
        x_eulerian = self._q + D * psi_q
        self.paint_from_positions(x_eulerian, scheme=self.scheme)
        return self
    
    def make_realization_batch(self,D,theta,R_smooth,C,noise):
        gen = self.gaussian_gen            
        kgrid     = self.fine_grid.kgrid    
        kgrid_abs = self.fine_grid.kgrid_abs
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
            return self.delta+1, self.delta_k
        
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
        return PowerSpectrum_batch(delta_weighted_k,delta_weighted_k, self.W,self.grid,compensate=True)
    
    def make_realization_batch_2T(self,D,theta,R_smooth,C,noise):
        gen = self.gaussian_gen            
        kgrid     = self.fine_grid.kgrid    
        kgrid_abs = self.fine_grid.kgrid_abs
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
            return self.delta+1, self.delta_k
        
        # Vectorize over the first axis of noise:
        one_plus_delta_batch, delta_k_batch = jax.vmap(make_single, in_axes=(None, None, 0))(D, theta, noise)
        delta_k_smooth_batch = delta_k_batch * jnp.exp(-0.5 * (kmesh_abs * R_smooth)**2)
        delta_smooth_batch = jnp.fft.ifft(delta_k_smooth_batch).real * self.grid.norm_ifft
        m_array = jnp.stack([delta_smooth_batch**0,
                             delta_smooth_batch**1,
                             delta_smooth_batch**2,
                             delta_smooth_batch**3,])
        
        m_batch = jnp.einsum('tij,jlm->tilm', C, m_array)
        rho_weighted = one_plus_delta_batch[None,None,:,:]*m_batch
        rho_bar = jnp.mean(rho_weighted, axis = -1)
        delta_weighted = rho_weighted/rho_bar[:,:,:,None] - 1
        delta_weighted_k = jnp.fft.fft(delta_weighted) * self.grid.norm_fft
        delta_weighted_k = jnp.swapaxes(delta_weighted_k, 0, 1)
        return MT2_pk_batch(delta_weighted_k, self.grid)
    

    def ComputeBasis(self, Rsmooth) -> Zeldovich1D:
        """
        Smooth the field using a Gaussian kernel with standard deviation `sigma`.
        This modifies the field in place.
        """
        delta_k_smooth = self.delta_k * jnp.exp(-0.5 * (self.grid.kgrid_abs * Rsmooth)**2)
        delta_smooth = jnp.fft.ifft(delta_k_smooth).real * self.grid.norm_ifft
        m1 = jnp.ones_like(delta_smooth)
        m2 = delta_smooth
        m3 = delta_smooth**2
        m4 = delta_smooth**3
        self.m_array = jnp.stack([m1,m2,m3,m4], axis=0)
        
    def WeightedChild(self, C):
        m = jnp.dot(C,self.m_array)
        rho_weighted = m*(self.delta+1)
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


@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Zeldovich3D(Field3D):
    """
    3D Zeldovich approximation initializer for two-tracer marked fields.
    """

    def __init__(
        self,
        gaussian_gen: GaussianFieldGenerator3D,
        grid: Grid3D | None = None,
        scheme: str = "cic"
    ):
        self.gaussian_gen = gaussian_gen
        self.fine_grid = gaussian_gen.grid
        self.grid = grid if grid is not None else self.fine_grid
        super().__init__(grid=self.grid)

        Nx, Ny, Nz = self.fine_grid.shape
        qx = jnp.linspace(0.0, self.fine_grid.L, num=Nx, endpoint=False, dtype=jnp.float64)
        qy = jnp.linspace(0.0, self.fine_grid.L, num=Ny, endpoint=False, dtype=jnp.float64)
        qz = jnp.linspace(0.0, self.fine_grid.L, num=Nz, endpoint=False, dtype=jnp.float64)
        qx, qy, qz = jnp.meshgrid(qx, qy, qz, indexing="ij")
        self._q_flat = jnp.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

        kx, ky, kz = self.fine_grid.kgrid_components
        k2 = kx**2 + ky**2 + kz**2
        k2 = k2.at[0, 0, 0].set(jnp.float64(1e-12))
        self._k_over_k2 = (kx / k2, ky / k2, kz / k2)
        self.scheme = scheme

    def compute_displacement(self, theta, noise):
        linear_field = self.gaussian_gen.make_realization_from_noise(theta, noise)
        delta_lin_k = linear_field.delta_k
        kx_over_k2, ky_over_k2, kz_over_k2 = self._k_over_k2

        psi_kx = 1j * kx_over_k2 * delta_lin_k
        psi_ky = 1j * ky_over_k2 * delta_lin_k
        psi_kz = 1j * kz_over_k2 * delta_lin_k
        psi_kx = psi_kx.at[0, 0, 0].set(0)
        psi_ky = psi_ky.at[0, 0, 0].set(0)
        psi_kz = psi_kz.at[0, 0, 0].set(0)

        psi_qx = jnp.real(jnp.fft.ifftn(psi_kx) * self.fine_grid.norm_ifft)
        psi_qy = jnp.real(jnp.fft.ifftn(psi_ky) * self.fine_grid.norm_ifft)
        psi_qz = jnp.real(jnp.fft.ifftn(psi_kz) * self.fine_grid.norm_ifft)
        return psi_qx, psi_qy, psi_qz

    def make_realization(self, D, theta, noise) -> Zeldovich3D:
        psi_qx, psi_qy, psi_qz = self.compute_displacement(theta, noise)
        psi_flat = jnp.stack([psi_qx.ravel(), psi_qy.ravel(), psi_qz.ravel()], axis=1)
        x_eulerian = self._q_flat + D * psi_flat
        self.paint_from_positions(x_eulerian, scheme=self.scheme)
        return self

    def make_realization_batch_2T(self, D, theta, R_smooth, C, noise):
        gen = self.gaussian_gen
        kmesh_abs = self.grid.kgrid_abs

        def make_single(D, theta, noise):
            linear_field = gen.make_realization_from_noise(theta, noise)
            delta_lin_k = linear_field.delta_k
            kx_over_k2, ky_over_k2, kz_over_k2 = self._k_over_k2

            psi_kx = 1j * kx_over_k2 * delta_lin_k
            psi_ky = 1j * ky_over_k2 * delta_lin_k
            psi_kz = 1j * kz_over_k2 * delta_lin_k
            psi_kx = psi_kx.at[0, 0, 0].set(0)
            psi_ky = psi_ky.at[0, 0, 0].set(0)
            psi_kz = psi_kz.at[0, 0, 0].set(0)

            psi_qx = jnp.real(jnp.fft.ifftn(psi_kx) * self.fine_grid.norm_ifft)
            psi_qy = jnp.real(jnp.fft.ifftn(psi_ky) * self.fine_grid.norm_ifft)
            psi_qz = jnp.real(jnp.fft.ifftn(psi_kz) * self.fine_grid.norm_ifft)
            psi_flat = jnp.stack([psi_qx.ravel(), psi_qy.ravel(), psi_qz.ravel()], axis=1)

            x_eulerian = self._q_flat + D * psi_flat
            self.paint_from_positions(x_eulerian, scheme=self.scheme)
            self.compute_fft()
            return self.delta + 1, self.delta_k

        one_plus_delta_batch, delta_k_batch = jax.vmap(
            make_single, in_axes=(None, None, 0)
        )(D, theta, noise)

        delta_k_smooth_batch = delta_k_batch * jnp.exp(-0.5 * (kmesh_abs * R_smooth) ** 2)
        delta_smooth_batch = jnp.fft.ifftn(delta_k_smooth_batch, axes=(1, 2, 3)).real * self.grid.norm_ifft
        m_array = jnp.stack([
            delta_smooth_batch ** 0,
            delta_smooth_batch ** 1,
            delta_smooth_batch ** 2,
            delta_smooth_batch ** 3,
        ])

        m_batch = jnp.einsum('tc,cbxyz->tbxyz', C, m_array)
        rho_weighted = one_plus_delta_batch[None, ...] * m_batch
        rho_bar = jnp.mean(rho_weighted, axis=(2, 3, 4))
        delta_weighted = rho_weighted / rho_bar[:, :, None, None, None] - 1
        delta_weighted_k = jnp.fft.fftn(delta_weighted, axes=(2, 3, 4)) * self.grid.norm_fft
        delta_weighted_k = jnp.swapaxes(delta_weighted_k, 0, 1)
        return MT2_pk_batch(delta_weighted_k, self.grid)

    def tree_flatten(self):
        leaves = []
        is_present = [False, False, False, False]
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
        aux_data = (
            self.gaussian_gen,
            self.fine_grid,
            self.grid,
            self._q_flat,
            self._k_over_k2,
            self.scheme,
            tuple(is_present),
        )
        return tuple(leaves), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            gaussian_gen,
            fine_grid,
            grid,
            q_flat,
            k_over_k2,
            scheme,
            is_present,
        ) = aux_data
        obj = cls.__new__(cls)
        Field3D.__init__(obj, grid=grid)
        obj.gaussian_gen = gaussian_gen
        obj.fine_grid = fine_grid
        obj.grid = grid
        obj._q_flat = q_flat
        obj._k_over_k2 = k_over_k2
        obj.scheme = scheme

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
