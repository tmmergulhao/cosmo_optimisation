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
    size: Optional[int] = field(init=False, default=None)

    #---- Derived pk attributes ----
    k_edges: Optional[jnp.ndarray] = field(init=False, default=None)
    k_ctrs: Optional[jnp.ndarray] = field(init=False, default=None)
    k_mapping: Optional[jnp.ndarray] = field(init=False, default=None)

    #---- FFT norms ----
    norm_fft: Optional[float] = field(init=False, default=None)
    norm_ifft: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, 'Ndim', len(self.shape))
        object.__setattr__(self, 'H', (self.L / jnp.array(self.shape)).astype(jnp.float64))
        object.__setattr__(self, 'Vol', jnp.float64(self.L ** self.Ndim))
        object.__setattr__(self, 'kf', jnp.float64(2 * jnp.pi / self.L))
        object.__setattr__(self, 'kNyq', (jnp.pi / self.H[0]).astype(jnp.float64))
        object.__setattr__(self, 'N', self.shape[0])
        object.__setattr__(self, 'size', int(jnp.prod(jnp.array(self.shape))))
        object.__setattr__(self, 'norm_fft', jnp.float64(1.0 / jnp.sqrt(self.N**self.Ndim)))
        object.__setattr__(self, 'norm_ifft', jnp.float64(jnp.sqrt(self.N**self.Ndim)))

        # Initialize Lagrangian coordinates
        q = jnp.linspace(0, self.L, self.N, endpoint=False, dtype=jnp.float64)
        object.__setattr__(self, 'q', q)

        # Initialize Fourier-space arrays
        kgrid = jnp.fft.fftfreq(self.shape[0], d=self.H) * 2 * jnp.pi
        kgrid = kgrid.at[0].set(jnp.float64(1e-12))
        kgrid2 = kgrid**2
        kgrid2 = kgrid2.at[0].set(jnp.float64(1e-12))
        kgrid_abs = jnp.abs(kgrid)
        kgrid_abs = kgrid_abs.at[0].set(jnp.float64(1e-12))

        object.__setattr__(self, 'kgrid', kgrid.astype(jnp.float64))
        object.__setattr__(self, 'kgrid2', kgrid2.astype(jnp.float64))
        object.__setattr__(self, 'kgrid_abs', kgrid_abs.astype(jnp.float64))
        object.__setattr__(self, 'q', jnp.linspace(0, self.L, self.shape[0], endpoint=False, dtype=jnp.float64))

        # Initialize the grid smoothing kernels
        if self.R_gauss > 0.0:
            physical_R_smooth = float(self.R_gauss * self.H[0])
            k_smooth = jnp.pi/physical_R_smooth
            gauss_kernel = jnp.exp(-0.5 * (self.kgrid_abs /k_smooth)**2)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', gauss_kernel.astype(jnp.float64))
            print(f"Using smoothing smoothing with R_gauss = {self.R_gauss} cells (physical = {physical_R_smooth}); k_smooth = {float(k_smooth):.3e}")
        elif self.R_clip > 0.0:
            physical_R_smooth = float(self.R_clip * self.H[0])
            k_smooth = jnp.pi/physical_R_smooth

            clip_kernel = jnp.where(self.kgrid_abs <= k_smooth, 1, 0)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', clip_kernel.astype(jnp.float64))
            print(f"Using clipping smoothing with R_clip = {self.R_clip} cells (physical = {physical_R_smooth}); k_smooth = {float(k_smooth):.3e}")
        else:
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', jnp.ones_like(kgrid, dtype=jnp.float64))
            print("Warning: No smoothing applied, GRID_SMOOTHING_KERNEL is set to ones.")

        if self.kmin is None:
            object.__setattr__(self, 'kmin', float(self.kf))
        else:
            object.__setattr__(self, 'kmin', max(float(self.kf), float(self.kmin * self.kf)))

        if self.kmax is None:
            object.__setattr__(self, 'kmax', float(self.kNyq))
        else:
            object.__setattr__(self, 'kmax', float(self.kmax * self.kNyq))

        if self.dk is None:
            object.__setattr__(self, 'dk', float(2 * self.kf))
        else:
            object.__setattr__(self, 'dk', max(float(self.dk * self.kf), float(2 * self.kf)))

        object.__setattr__(self, 'k_edges', jnp.arange(self.kmin, self.kmax + self.dk, self.dk, dtype=jnp.float64))
        object.__setattr__(self, 'k_ctrs', ((self.k_edges[:-1] + self.k_edges[1:]) / 2).astype(jnp.float64))
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
        noise_real = random.normal(key_r, shape=(N,), dtype=jnp.float64)
        noise_imag = random.normal(key_i, shape=(N,), dtype=jnp.float64)

        # 2) Combine into a complex array
        noise_k = noise_real + 1j * noise_imag
        noise_k = noise_k.astype(jnp.complex128)

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


@dataclass(frozen=True)
class Grid3D:
    #---- Grid parameters ----
    shape: tuple[int, int, int]
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
    H: Optional[jnp.ndarray] = field(init=False, default=None)
    Vol: Optional[float] = field(init=False, default=None)
    kf: Optional[float] = field(init=False, default=None)
    kNyq: Optional[float] = field(init=False, default=None)
    kgrid_components: Optional[tuple[jnp.ndarray, ...]] = field(init=False, default=None)
    kgrid_abs: Optional[jnp.ndarray] = field(init=False, default=None)
    GRID_SMOOTHING_KERNEL: Optional[jnp.ndarray] = field(init=False, default=None)
    size: Optional[int] = field(init=False, default=None)

    #---- Derived pk attributes ----
    k_edges: Optional[jnp.ndarray] = field(init=False, default=None)
    k_ctrs: Optional[jnp.ndarray] = field(init=False, default=None)
    k_mapping: Optional[jnp.ndarray] = field(init=False, default=None)

    #---- FFT norms ----
    norm_fft: Optional[float] = field(init=False, default=None)
    norm_ifft: Optional[float] = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, 'Ndim', 3)
        H = (self.L / jnp.array(self.shape)).astype(jnp.float64)
        object.__setattr__(self, 'H', H)
        object.__setattr__(self, 'Vol', jnp.float64(self.L ** self.Ndim))
        object.__setattr__(self, 'kf', jnp.float64(2 * jnp.pi / self.L))
        object.__setattr__(self, 'kNyq', jnp.min(jnp.pi / H))
        object.__setattr__(self, 'N', self.shape[0])
        size = int(jnp.prod(jnp.array(self.shape)))
        object.__setattr__(self, 'size', size)
        object.__setattr__(self, 'norm_fft', jnp.float64(1.0 / jnp.sqrt(size)))
        object.__setattr__(self, 'norm_ifft', jnp.float64(jnp.sqrt(size)))

        kx = jnp.fft.fftfreq(self.shape[0], d=H[0]) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(self.shape[1], d=H[1]) * 2 * jnp.pi
        kz = jnp.fft.fftfreq(self.shape[2], d=H[2]) * 2 * jnp.pi
        kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")
        kx = kx.astype(jnp.float64)
        ky = ky.astype(jnp.float64)
        kz = kz.astype(jnp.float64)
        k2 = kx**2 + ky**2 + kz**2
        k2 = k2.at[0, 0, 0].set(jnp.float64(1e-12))
        k_abs = jnp.sqrt(k2)
        object.__setattr__(self, 'kgrid_components', (kx, ky, kz))
        object.__setattr__(self, 'kgrid_abs', k_abs)

        if self.R_gauss > 0.0:
            physical_R_smooth = float(self.R_gauss * H[0])
            k_smooth = jnp.pi / physical_R_smooth
            gauss_kernel = jnp.exp(-0.5 * (k_abs / k_smooth) ** 2)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', gauss_kernel.astype(jnp.float64))
        elif self.R_clip > 0.0:
            physical_R_smooth = float(self.R_clip * H[0])
            k_smooth = jnp.pi / physical_R_smooth
            clip_kernel = jnp.where(k_abs <= k_smooth, 1, 0)
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', clip_kernel.astype(jnp.float64))
        else:
            object.__setattr__(self, 'GRID_SMOOTHING_KERNEL', jnp.ones_like(k_abs, dtype=jnp.float64))

        if self.kmin is None:
            object.__setattr__(self, 'kmin', float(self.kf))
        else:
            object.__setattr__(self, 'kmin', max(float(self.kf), float(self.kmin * self.kf)))

        if self.kmax is None:
            object.__setattr__(self, 'kmax', float(self.kNyq))
        else:
            object.__setattr__(self, 'kmax', float(self.kmax * self.kNyq))

        if self.dk is None:
            object.__setattr__(self, 'dk', float(2 * self.kf))
        else:
            object.__setattr__(self, 'dk', max(float(self.dk * self.kf), float(2 * self.kf)))

        k_edges = jnp.arange(self.kmin, self.kmax + self.dk, self.dk, dtype=jnp.float64)
        object.__setattr__(self, 'k_edges', k_edges)
        object.__setattr__(self, 'k_ctrs', ((k_edges[:-1] + k_edges[1:]) / 2).astype(jnp.float64))
        flat_k = k_abs.reshape(-1)
        bin_indices = jnp.searchsorted(k_edges, flat_k, side='right') - 1
        valid = (bin_indices >= 0) & (bin_indices < len(k_edges) - 1)
        bin_assignment = jnp.where(valid, bin_indices, -1)
        object.__setattr__(self, 'k_mapping', bin_assignment)

    def __str__(self):
        smoothing_type = "Gaussian" if self.R_gauss > 0.0 else "Clipping" if self.R_clip > 0.0 else "None"
        smoothing_info = f"Using {smoothing_type} smoothing"
        if self.R_gauss > 0.0:
            smoothing_info += f" with R_gauss = {self.R_gauss}"
        elif self.R_clip > 0.0:
            smoothing_info += f" with R_clip = {self.R_clip}"

        return (
            f"{smoothing_info}\n"
            f"kmin: {self.kmin}\n"
            f"kmax: {self.kmax}\n"
            f"dk: {self.dk}\n"
            f"N: {self.N}\n"
            f"Shape: {self.shape}\n"
            f"L: {self.L}\n"
            f"Volume: {self.Vol}\n"
        )

    def generate_hermitian_noise(self, key: jnp.ndarray) -> jnp.ndarray:
        """
        Generate a Hermitian-symmetric 3D noise field in Fourier space by
        sampling a real-space Gaussian field and FFT-ing it.

        Parameters
        ----------
        key : jnp.ndarray
            A JAX PRNGKey used to draw real-space Gaussian noise.

        Returns
        -------
        noise_k : jnp.ndarray (complex)
            A 3D array of Hermitian-symmetric noise in Fourier space.
        """
        noise_real = random.normal(key, shape=self.shape, dtype=jnp.float64)
        noise_k = jnp.fft.fftn(noise_real) * self.norm_fft
        return noise_k.astype(jnp.complex128)
