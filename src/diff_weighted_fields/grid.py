# src/diff_weighted_fields/grid.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import jax.numpy as jnp
import jax.random as random

@dataclass(frozen=True)
class Grid1D:
    #---- Grid parameters ----
    shape: tuple[int, ...]
    L: float
    R_gauss: float = 0.0
    R_clip: float = 0.0

    #---- Power spectrum parameters ----
    kmin: Optional[float] = field(init=True, default=None)
    kmax: Optional[float] = field(init=True, default=None)
    dk: Optional[float] = field(init=True, default=None)

    #---- Derived grid attributes ----
    N: int = field(init=False, default=None)
    Ndim: Optional[int] = field(init=False, default=None)
    H: Optional[float] = field(init=False, default=None)
    Vol: Optional[float] = field(init=False, default=None)
    kf: Optional[float] = field(init=False, default=None)
    kNyq: Optional[float] = field(init=False, default=None)
    kgrid: Optional[jnp.ndarray] = field(init=False, default=None)
    kgrid2: Optional[jnp.ndarray] = field(init=False, default=None)
    kgrid_abs: Optional[jnp.ndarray] = field(init=False, default=None)
    GRID_SMOOTHING_KERNEL: Optional[jnp.ndarray] = field(init=False, default=None)
    q: Optional[jnp.ndarray] = field(init=False, default=None)

    #---- Derived pk attributes ----
    k_edges: Optional[jnp.ndarray] = field(init=False, default=None)
    k_ctrs: Optional[jnp.ndarray] = field(init=False, default=None)
    k_mapping: Optional[jnp.ndarray] = field(init=False, default=None)

    #---- FFT norms ----
    norm_fft: Optional[float] = field(init=False, default=None)
    norm_ifft: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, 'Ndim', len(self.shape))
        object.__setattr__(self, 'H', self.L / jnp.array(self.shape))
        object.__setattr__(self, 'Vol', self.L ** self.Ndim)
        object.__setattr__(self, 'kf', 2 * jnp.pi / self.L)
        object.__setattr__(self, 'kNyq', jnp.pi / self.H[0])
        object.__setattr__(self, 'N', self.shape[0])
        object.__setattr__(self, 'norm_fft', 1.0 / jnp.sqrt(self.N**self.Ndim))
        object.__setattr__(self, 'norm_ifft', jnp.sqrt(self.N**self.Ndim))

        # Initialize Lagrangian coordinates
        q = jnp.linspace(0, self.L, self.N, endpoint=False)
        object.__setattr__(self, 'q', q)

        # Initialize Fourier-space arrays
        kgrid = jnp.fft.fftfreq(self.shape[0], d=self.H) * 2 * jnp.pi
        kgrid = kgrid.at[0].set(1e-12)
        kgrid2 = kgrid**2
        kgrid2 = kgrid2.at[0].set(1e-12)
        kgrid_abs = jnp.abs(kgrid)
        kgrid_abs = kgrid_abs.at[0].set(1e-12)

        object.__setattr__(self, 'kgrid', kgrid)
        object.__setattr__(self, 'kgrid2', kgrid2)
        object.__setattr__(self, 'kgrid_abs', kgrid_abs)
        object.__setattr__(self, 'q', jnp.linspace(0, self.L, self.shape[0], endpoint=False))

        # Initialize the grid smoothing kernels
        if self.R_gauss > 0.0:
            gauss_kernel = jnp.exp(-0.5 * (kgrid * self.R_gauss * self.H)**2)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', gauss_kernel)
            print(f"Using Gaussian smoothing with R_gauss = {self.R_gauss}")
        elif self.R_clip > 0.0:
            # interpret R_clip as number of grid cells (physical length = R_clip * H[0])
            physical_R = float(self.R_clip * self.H[0])
            k_clip = 2 * jnp.pi / physical_R
            clip_kernel = jnp.where(self.kgrid_abs <= k_clip, 1, 0)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', clip_kernel)
            print(f"Using clipping smoothing with R_clip = {self.R_clip} cells (physical = {physical_R}); k_clip = {float(k_clip):.3e}")
        else:
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', jnp.ones_like(kgrid))
            print("Warning: No smoothing applied, GRID_SMOOTHING_KERNEL is set to ones.")

        if self.kmin is None:
            object.__setattr__(self, 'kmin', self.kf)
        else:
            object.__setattr__(self, 'kmin', max(self.kf, self.kmin * self.kf))

        if self.kmax is None:
            object.__setattr__(self, 'kmax', self.kNyq)
        else:
            object.__setattr__(self, 'kmax', self.kmax * self.kNyq)

        if self.dk is None:
            object.__setattr__(self, 'dk', 2 * self.kf)
        else:
            object.__setattr__(self, 'dk', max(self.dk * self.kf, 2 * self.kf))

        print(f"kmin: {self.kmin}")
        print(f"kmax: {self.kmax}")
        print(f"dk: {self.dk}")

        object.__setattr__(self, 'k_edges', jnp.arange(self.kmin, self.kmax + self.dk, self.dk))
        object.__setattr__(self, 'k_ctrs', (self.k_edges[:-1] + self.k_edges[1:]) / 2)
        bin_indices = jnp.searchsorted(self.k_edges, self.kgrid_abs, side='right') - 1
        valid = (bin_indices >= 0) & (bin_indices < len(self.k_edges) - 1)
        bin_assignment = jnp.where(valid, bin_indices, -1).reshape(-1)
        object.__setattr__(self, 'k_mapping', bin_assignment)

    def generate_hermitian_noise(self, key: jnp.ndarray) -> jnp.ndarray:
        """
        Generate a complex-valued array of length N with Hermitian symmetry,
        so that its inverse FFT is real.

        Parameters
        ----------
        key : jnp.ndarray
            A JAX PRNGKey (e.g. jax.random.PRNGKey(...)) used to draw Gaussian noise.

        Returns
        -------
        noise_k : jnp.ndarray (complex)
            A length‐N array of Hermitian-symmetric noise:
            - noise_k[0] is forced to be real
            - if N is even, noise_k[N//2] is forced to be real
            - for each 1 <= i < N/2, noise_k[N−i] = conj(noise_k[i])
        """
        N = self.shape[0]

        # 1) Draw independent standard normals for real and imaginary parts
        key_r, key_i = random.split(key)
        noise_real = random.normal(key_r, shape=(N,))
        noise_imag = random.normal(key_i, shape=(N,))

        # 2) Combine into a complex array
        noise_k = noise_real + 1j * noise_imag

        # 3) Enforce Hermitian symmetry:

        #   a) Zero‐mode (index 0) must be purely real
        noise_k = noise_k.at[0].set(jnp.real(noise_k[0]))

        #   b) Nyquist mode (index N//2) must be purely real if N is even
        if N % 2 == 0:
            noise_k = noise_k.at[N // 2].set(jnp.real(noise_k[N // 2]))

        #   c) For 1 <= i < N/2, set noise_k[N−i] = conj(noise_k[i])
        half = N // 2
        if N > 1:
            if N % 2 == 0:
                pos_idxs = jnp.arange(1, half)      # indices 1..(N/2−1)
            else:
                pos_idxs = jnp.arange(1, half + 1)  # indices 1..floor(N/2)
            neg_idxs = N - pos_idxs
            noise_k = noise_k.at[neg_idxs].set(jnp.conj(noise_k[pos_idxs]))

        return noise_k