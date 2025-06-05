# src/diff_weighted_fields/generators.py

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Any, Tuple

from .grid import Grid1D
from .field import Field1D

@jax.tree_util.register_pytree_node_class
@dataclass
class GaussianFieldGenerator1D:
    """
    Generates 1D Gaussian random‐field realizations from precomputed Hermitian‐symmetric noise.

    Attributes
    ----------
    grid : Grid1D
        The grid geometry (number of cells N, box length L, and k‐vectors).
    Pk_func : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A function that takes:
            - k_array: jnp.ndarray of wavenumbers (|k|), e.g. grid.kgrid_abs
            - theta:   jnp.ndarray of parameters (e.g. [A, n_s, ...])
        and returns a jnp.ndarray of length N giving P(k) at each mode.
    """

    grid: Grid1D
    Pk_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def make_realization_from_noise(
        self, theta: jnp.ndarray, noise_k: jnp.ndarray
    ) -> Field1D:
        """
        Create Gaussian‐field `delta_k` from provided Hermitian‐symmetric `noise_k`.

        Parameters
        ----------
        theta : jnp.ndarray
            1D array of parameters (e.g. amplitude, tilt) to feed into Pk_func.
        noise_k : jnp.ndarray
            Precomputed Hermitian‐symmetric complex noise in Fourier space.
            Must have shape = (N,) for a single realization.

        Returns
        -------
        Field1D
            A Field1D object whose `.delta_k` has been set according to P(k) and `noise_k`.
        """
        # Enforce exactly one realization at a time
        if noise_k.ndim != 1:
            raise ValueError(
                f"make_realization_from_noise only supports 1D noise arrays. "
                f"Got noise_k.ndim = {noise_k.ndim}. "
                "If you need to generate a batch, wrap this function in jax.vmap."
            )

        # Number of dimensions (1D)
        Ndim = self.grid.Ndim

        # 1) Compute P(k) at each FFT mode |k|. Always feed Pk the positive |k|.
        k_abs = self.grid.kgrid_abs       # shape: (N,)
        Pk_at_modes = self.Pk_func(k_abs, theta)  # shape: (N,)

        Hscalar = self.grid.H[0]

        # 3) Amplitude = sqrt(P(k) / (2 * H^Ndim)) * smoothing kernel
        amp = jnp.sqrt(Pk_at_modes / (2.0 * Hscalar**Ndim)) \
              * self.grid.GRID_SMOOTHING_KERNEL  # shape: (N,)

        # 4) Scale noise: delta_k = amp * noise_k
        delta_k = amp * noise_k  # shape: (N,)

        # 5) Wrap into a Field1D and return
        field = Field1D(grid=self.grid)
        field.assign_from_k(delta_k)
        return field
    
    # ---------------- PyTree registration ----------------
    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[Any, Any]]:
        """
        Since we never differentiate w.r.t. `grid`, keep it static.
        Leaves = ()  (no JAX arrays inside this object that we need gradients through)
        Aux_data = (grid, Pk_func)
        """
        leaves   = ()  
        aux_data = (self.grid, self.Pk_func)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Tuple[Any, Any], children: Tuple[jnp.ndarray, ...]
    ) -> "GaussianFieldGenerator1D":
        """
        Reconstruct from:
          - aux_data = (grid, Pk_func)
          - children = ()  (empty, because leaves=())
        """
        grid, Pk_func = aux_data
        return cls(grid=grid, Pk_func=Pk_func)