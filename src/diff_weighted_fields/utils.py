# src/diff_weighted_fields/utils.py
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

def cic_paint_1d(positions: jnp.ndarray, grid) -> jnp.ndarray:
    """
    Standard (discontinuous) CIC deposit:
    - positions: 1D array of length M with values in [0, L).
    - grid with attributes shape, H, etc.

    Returns density array of shape (N,) on that grid.
    """

    N = grid.shape[0]
    dx = grid.H[0]
    density = jnp.zeros((N,))

    indices = positions / dx
    i_left = jnp.floor(indices).astype(int)      # left‐cell index
    delta = indices - i_left                     # fractional distance

    w_left = 1.0 - delta
    w_right = delta

    # Periodic wrap:
    density = density.at[i_left % N].add(w_left)
    density = density.at[(i_left + 1) % N].add(w_right)

    return density

def cic_paint_batch_1d(
    positions_batch: jnp.ndarray, 
    grid
) -> jnp.ndarray:
    """
    Vectorized CIC paint: 
      - positions_batch: shape (B, M) array of B separate particle‐lists
      - grid: the same Grid1D used for all B realizations
    Returns: a density array of shape (B, N), where N = grid.shape[0].
    
    Internally, this just does vmap(cic_paint_1d, in_axes=(0, None)).
    """
    # We map over axis 0 of positions_batch, keeping `grid` fixed.
    return jax.vmap(lambda pos: cic_paint_1d(pos, grid), in_axes=(0))(positions_batch)

def tsc_paint_1d(positions: jnp.ndarray, grid) -> jnp.ndarray:
    """
    Fully differentiable 1D TSC (triangular‐shaped–cloud) mass assignment.

    - positions:  1D array of length M, each x ∈ [0, L)
    - grid:       a Grid1D instance with attributes:
                    * shape = (N,),
                    * H[0] = Δx (grid spacing),
                    * kgrid, etc., unused here.

    Returns a real‐space density array of shape (N,) such that
    each particle deposits mass into its three nearest grid points
    with a C¹ (piecewise‐quadratic) kernel.

    The weight function w(|r|) is:
        w(r) = 3/4 - r²             for |r| ≤ 1/2
             = (1/2)*(1.5 - |r|)²    for 1/2 < |r| ≤ 3/2
             = 0                    otherwise
    """
    N = grid.shape[0]
    dx = grid.H[0]   # grid spacing (scalar)

    # Compute each particle's fractional index in [0, N):
    #   eta = x / dx, so that x = eta * dx.
    eta = positions / dx   # shape: (M,)

    # Find the integer index of the "center" cell for TSC:
    #   i_center = round(eta) = floor(eta + 0.5).
    i_center = jnp.floor(eta + 0.5).astype(int)  # shape: (M,)

    # Build the three offsets: center, left (=center-1), right (=center+1).
    i_left   = i_center - 1
    i_right  = i_center + 1

    # Distances r to each of those three cell-centers, in units of Δx.
    r_center = eta - i_center       # (M,)
    r_left   = r_center + 1.0       # (M,)
    r_right  = r_center - 1.0       # (M,)

    # Define a piecewise function w(r) that returns shape (M,) weights:
    def weight(r: jnp.ndarray) -> jnp.ndarray:
        # r can be positive or negative; use absolute value:
        ar = jnp.abs(r)

        # Case 1: |r| ≤ 0.5 ⇒ w = 3/4 - r²
        w1 = jnp.where(ar <= 0.5,
                       0.75 - r**2,
                       0.0)

        # Case 2:  0.5 < |r| ≤ 1.5 ⇒ w = 0.5 * (1.5 - |r|)²
        w2 = jnp.where((ar > 0.5) & (ar <= 1.5),
                       0.5 * (1.5 - ar)**2,
                       0.0)

        # Elsewhere: zero
        return w1 + w2  # shape: (M,)

    # Compute the three weight arrays for each particle:
    w_c = weight(r_center)  # weight for i_center
    w_l = weight(r_left)    # weight for i_left
    w_r = weight(r_right)   # weight for i_right

    # Now scatter these weights onto a length‐N array "density":
    # We start from zeros(N,) and add contributions with periodic wrapping.
    density = jnp.zeros((N,))

    # For JAX we must do everything in a vectorized (functional) style:
    #   (i_center % N, w_c), (i_left % N, w_l), (i_right % N, w_r)
    i_center_mod = i_center % N
    i_left_mod   = i_left % N
    i_right_mod  = i_right % N

    density = density.at[i_center_mod].add(w_c)
    density = density.at[i_left_mod].add(w_l)
    density = density.at[i_right_mod].add(w_r)

    return density

def PowerSpectrum(fieldA, fieldB, compensate = False):
    deltakA, WA = fieldA.delta_k, fieldA.W
    deltakB, WB = fieldB.delta_k, fieldB.W
    H = fieldA.grid.H
    Ndim = fieldA.grid.Ndim
    nbins = len(fieldA.grid.k_edges) - 1
    field_k_abs = deltakA * jnp.conjugate(deltakB)
    if compensate:
        eps = 1e-3
        safe_WA = jnp.where(jnp.abs(WA) < eps, 1.0, WA)
        safe_WB = jnp.where(jnp.abs(WB) < eps, 1.0, WB)
        field_k_abs = field_k_abs / safe_WA / safe_WB
        #field_k_abs = field_k_abs/WA/WB
    field_flat = field_k_abs.reshape(-1)
    k_mapping = fieldA.grid.k_mapping

    # Mask invalid bins by zeroing them
    valid = k_mapping >= 0
    k_mapping = jnp.where(valid, k_mapping, 0)
    field_flat = jnp.where(valid, field_flat, 0.0)

    # Count how many entries go into each bin
    counts = jnp.bincount(k_mapping, weights=valid.astype(field_flat.dtype), length=nbins)
    power = jnp.bincount(k_mapping, weights=field_flat, length=nbins)
    
    pk = jnp.real(jnp.where(counts > 0, power / counts, 0.0))
    pk = pk * H**Ndim
    return pk

def PowerSpectrum_batch(delta_k_batch: jnp.ndarray, W: jnp.ndarray, grid) -> jnp.ndarray:
    # Number of k‐bins
    nbins = len(grid.k_edges) - 1

    # Compute |δₖ|² for each batch entry
    field_k_abs = delta_k_batch * jnp.conjugate(delta_k_batch)  # shape (..., N)

    # Safe‐divide by W for compensation (avoid tiny denominators)
    eps = 1e-3
    safe_W = jnp.where(jnp.abs(W) < eps, 1.0, W)
    field_k_abs = field_k_abs / safe_W / safe_W  # still shape (..., N)

    # Flatten every leading batch dimension into a single axis of length B
    batch_shape = field_k_abs.shape[:-1]
    N = field_k_abs.shape[-1]
    flat_field = field_k_abs.reshape((-1, N))  # shape (B, N)

    # k_mapping: shape (N,) with bin indices or −1 for “ignore”
    k_mapping = grid.k_mapping
    valid = k_mapping >= 0
    kmap_safe = jnp.where(valid, k_mapping, 0)

    def ps_single(field_row):
        # Mask out invalid entries
        masked = jnp.where(valid, field_row, 0.0)
        counts = jnp.bincount(kmap_safe, weights=valid.astype(field_row.dtype), length=nbins)
        power  = jnp.bincount(kmap_safe, weights=masked,     length=nbins)
        pk     = jnp.real(jnp.where(counts > 0, power / counts, 0.0))
        return pk * (grid.H ** grid.Ndim)
    
    pk_flat = jax.vmap(ps_single)(flat_field)  # shape (B, nbins)
    return pk_flat.reshape(batch_shape + (nbins,))