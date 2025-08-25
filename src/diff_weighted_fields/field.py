from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import jax.numpy as jnp
import jax
from .grid import Grid1D
from .utils import cic_paint_1d, tsc_paint_1d, cic_paint_batch_1d

@jax.tree_util.register_pytree_node_class
@dataclass()
class Field1D():
    """
    Base class for 1D fields.
    """
    grid: Grid1D
    delta: Optional[jnp.ndarray] = field(default=None, init=False)
    delta_k: Optional[jnp.ndarray] = field(default=None, init=False)
    one_plus_delta: Optional[jnp.ndarray] = field(default=None, init=False)  # density field
    W: Optional[jnp.ndarray] = field(default=None, init=False)  # CIC/TSC window function
    scheme: str = field(default="cic", init=True, repr=False)

    def assign_from_real_space(self, arr: jnp.ndarray) -> None:
        """
        Attach a real‐space overdensity (or density) array to this field.
        Clears any existing delta_k.
        """
        N = self.grid.shape[0]
        if arr.shape != (N,):
            raise ValueError(f"Expected real‐space array of shape {(N,)}, got {arr.shape}")
        arr = arr.astype(jnp.float64)
        self.delta = arr
        self.delta_k = None  # invalidate any previous Fourier‐space data

    def compute_fft(self) -> jnp.ndarray:
        """
        Compute delta_k = FFT[delta(x)] if not already set, then return delta_k.
        """
        if self.delta is None:
            raise ValueError("No real‐space field (self.delta) has been assigned.")
        # Compute the FFT:
        self.delta = self.delta.astype(jnp.float64)
        self.delta_k = jnp.fft.fft(self.delta) * self.grid.norm_fft

    def assign_from_k(self, arr_k: jnp.ndarray) -> None:
        """
        Attach a Fourier‐space array delta_k to this field.
        Clears any existing real‐space delta.
        """
        N = self.grid.shape[0]
        if arr_k.shape != (N,):
            raise ValueError(f"Expected Fourier‐space array of shape {(N,)}, got {arr_k.shape}")
        arr_k = arr_k.astype(jnp.complex128)
        self.delta_k = arr_k
        self.delta = None  # invalidate any previous real‐space data

    def compute_ifft(self) -> jnp.ndarray:
        """
        Compute delta(x) = IFFT[delta_k] if not already set, then return delta(x).
        """
        if self.delta_k is None:
            raise ValueError("No Fourier‐space field (self.delta_k) has been assigned.")
        # Inverse FFT and take the real part:
        self.delta = jnp.fft.ifft(self.delta_k).real.astype(jnp.float64) * self.grid.norm_ifft
        
    def paint_from_positions(
        self,
        positions: jnp.ndarray,
        scheme: str = "cic",
    ) -> jnp.ndarray:
        """
        Deposit particles onto the grid. `positions` is an array of shape (M,)
        giving each particle’s x ∈ [0, L).  `scheme` is one of: "cic", "tsc", "smooth_cic".
        If scheme=="smooth_cic", you must pass a small `sigma` (in physical units).
        Returns the real‐space density array of shape (N,) and caches it in self.delta.
        """
        if scheme == "cic":
            _one_plus_delta = cic_paint_1d(positions, self.grid)
            self.W = jnp.sinc(self.grid.kgrid * self.grid.H / (2 * jnp.pi)) ** 2
        elif scheme == "tsc":
            _one_plus_delta = tsc_paint_1d(positions, self.grid)
            self.W = jnp.sinc(self.grid.kgrid * self.grid.H / (2 * jnp.pi)) ** 3
        else:
            raise ValueError(f"Unknown painting scheme: {scheme}")
        
        _one_plus_delta = _one_plus_delta.astype(jnp.float64)
        _one_plus_delta = _one_plus_delta/jnp.mean(_one_plus_delta)
        _delta = _one_plus_delta - 1  # convert to overdensity

        #self.delta = _delta # convert to overdensity
        self.delta_k = jnp.fft.fft(_delta) * self.grid.norm_fft * self.grid.GRID_SMOOTHING_KERNEL
        self.delta = jnp.fft.ifft(self.delta_k).real * self.grid.norm_ifft
        
        #self.delta_k = jnp.fft.fft(_delta) * self.grid.norm_fft * self.grid.GRID_SMOOTHING_KERNEL
        #self.delta = jnp.fft.ifft(self.delta_k).real * self.grid.norm_ifft

    # ---------------- PyTree registration ----------------
    def tree_flatten(self):
        # Gather only non-None leaves
        leaves = []
        is_present = [False, False, False]  # flags for (delta, delta_k, one_plus_delta)
        if self.delta is not None:
            leaves.append(self.delta)
            is_present[0] = True
        if self.delta_k is not None:
            leaves.append(self.delta_k)
            is_present[1] = True
        if self.one_plus_delta is not None:
            leaves.append(self.one_plus_delta)
            is_present[2] = True
        # We do NOT store W as a leaf; the smoothing kernel is in grid
        aux_data = (self.grid, self.scheme, tuple(is_present))
        return tuple(leaves), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        grid, scheme, is_present = aux_data
        obj = cls(grid=grid)
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

        # We no longer store W as a leaf; the smoothing kernel is in grid
        obj.W = None
        return obj