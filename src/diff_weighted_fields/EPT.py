# src/diff_weighted_fields/EPT.py

from __future__ import annotations
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax
import jax.numpy as jnp

from .field import Field1D
from .generators import GaussianFieldGenerator1D
from .utils import PowerSpectrum_batch

@register_pytree_node_class
@dataclass(init=False)
class EPT1D(Field1D):

    def __init__(self, gaussian_gen: GaussianFieldGenerator1D):
        self.gaussian_gen = gaussian_gen
        # Use the same grid for FFTs
        self.grid = gaussian_gen.grid
        super().__init__(grid=self.grid)
        
    def make_realization(
        self,
        theta: jnp.ndarray,
        noise: jnp.ndarray,
        coeffs: jnp.ndarray,   
    ) -> None:
        
        #Generate the Gaussian Field
        field = self.gaussian_gen.make_realization_from_noise(theta, noise)
        field.compute_ifft()
        phi    = field.delta           
        phi_k  = field.delta_k         

        #Get the grid variables (already regulazired to avoid division by zero)
        k     = self.grid.kgrid_abs
        k2    = k**2

        #Spherical Collapse part
        phi2 = phi**2
        phi3 = phi**3
        phi2_k = jnp.fft.fft(phi2) * self.grid.norm_fft

        #Derivative operators
        lap_phi  = jnp.fft.ifft(-k2 * phi_k).real * self.grid.norm_ifft
        lap_phi2 = jnp.fft.ifft(-k2 * phi2_k).real * self.grid.norm_ifft
        inv_lap_phi = jnp.fft.ifft(-phi_k/k2).real * self.grid.norm_ifft

        #Stack all the 6 basis functions
        m_array = jnp.stack([
            phi,         # c0
            phi2,        # c1
            phi3,        # c2
            lap_phi,     # c3
            lap_phi2,    # c4
            inv_lap_phi, # c5
        ], axis=0)

        #Subtract the mean from the basis functions
        m_array = m_array - jnp.mean(m_array, axis=-1, keepdims=True)
        
        #Create the non-Gaussian realization
        delta_ng = jnp.einsum('j,ji->i', coeffs,m_array)
        field.assign_from_real_space(delta_ng)
        field.compute_fft()
        return field
    
    def make_realization_delta(
        self,
        theta: jnp.ndarray,
        noise: jnp.ndarray,
        coeffs: jnp.ndarray
    ) -> EPT1D:
        
        f = self.make_realization(theta, noise, coeffs)
        return f.delta

    def make_realization_batch_field(self,coeffs,theta,C,noise):
        delta_batch = jax.vmap(self.make_realization_delta, in_axes=(None, 0, None))(theta, noise,coeffs)
        m_array = jnp.stack([delta_batch**1,
                             delta_batch**2,
                             delta_batch**3])
        m_array = m_array - jnp.mean(m_array, axis=-1, keepdims=True)
        Omega_batch = jnp.einsum('ij,jlm->ilm', C, m_array)
        return Omega_batch
    
    def make_realization_batch(self,coeffs,theta,C,noise):
        delta_batch = jax.vmap(self.make_realization_delta, in_axes=(None, 0, None))(theta, noise,coeffs)
        m_array = jnp.stack([delta_batch**1,
                             delta_batch**2,
                             delta_batch**3])
        m_array = m_array - jnp.mean(m_array, axis=-1, keepdims=True)
        Omega_batch = jnp.einsum('ij,jlm->ilm', C, m_array)
        Omega_batch_k = jnp.fft.fft(Omega_batch) * self.grid.norm_fft
        return PowerSpectrum_batch(Omega_batch_k,Omega_batch_k, jnp.ones_like(Omega_batch_k),self.grid)
    
    def ComputeBasis(self, Rsmooth: float) -> EPT1D:
        """
        Exactly as in Zeldovich1D: smooth δ_k with a Gaussian of width Rsmooth
        and build m_array = [1, δ_smooth, δ_smooth^2, δ_smooth^3].
        """
        delta_k_smooth = self.delta_k * jnp.exp(
            -0.5 * (self.grid.kgrid_abs * Rsmooth)**2
        )
        delta_smooth = jnp.fft.ifft(delta_k_smooth).real * self.grid.norm_ifft

        m1 = jnp.ones_like(delta_smooth)
        m2 = delta_smooth
        m3 = delta_smooth**2
        m4 = delta_smooth**3
        self.m_array = jnp.stack([m1, m2, m3, m4], axis=0)
        return self

    def WeightedChild(self, C: jnp.ndarray) -> Field1D:
        """
        Apply weights C (shape (4,)) to the m_array, then re-weight the density:
          ρ_w(x) = [C⋅m_array(x)] * [1 + δ(x)], normalize, and return as a new Field1D.
        """
        # combine basis functions
        m = jnp.tensordot(C, self.m_array, axes=[0, 0])  # shape (N,)
        # weight the total density
        rho_weighted = m * (self.delta + 1)
        mean_rho = jnp.mean(rho_weighted)
        delta_marked = rho_weighted / mean_rho - 1.0

        # build child field
        child = Field1D(grid=self.grid)
        child.assign_from_real_space(delta_marked)
        child.compute_fft()
        # preserve any window if present (likely None here)
        child.W = self.W
        return child

    # ---------------- PyTree registration ----------------

    def tree_flatten(self):
        # Flatten leaves as in Field1D; treat gaussian_gen as static
        leaves, _ = super().tree_flatten()
        aux = (self.gaussian_gen,)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (gaussian_gen,) = aux_data
        obj = cls.__new__(cls)
        Field1D.__init__(obj, grid=gaussian_gen.grid)
        obj.gaussian_gen = gaussian_gen

        # restore leaves: delta, delta_k, one_plus_delta
        idx = 0
        obj.delta = children[idx] if len(children) > idx else None; idx += 1
        obj.delta_k = children[idx] if len(children) > idx else None; idx += 1
        obj.one_plus_delta = children[idx] if len(children) > idx else None
        obj.W = None
        return obj