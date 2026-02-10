"""Toy 3D example: scan C2/C3 polynomial coefficients and report a feature map."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_weighted_fields.generators import GaussianFieldGenerator3D
from diff_weighted_fields.grid import Grid3D


def toy_pk(k: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Simple toy power spectrum: A / (1 + (k / k0)**2)**2."""
    amplitude, k0 = theta
    return amplitude / (1.0 + (k / k0) ** 2) ** 2


def build_weighted_delta(delta: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    """Apply polynomial weights and return a normalized weighted overdensity."""
    basis = jnp.stack(
        [
            jnp.ones_like(delta),
            delta,
            delta**2,
            delta**3,
        ],
        axis=0,
    )
    mark = jnp.tensordot(coeffs, basis, axes=[0, 0])
    rho_weighted = mark * (1.0 + delta)
    delta_weighted = rho_weighted / jnp.mean(rho_weighted) - 1.0
    return delta_weighted


def feature_information(delta_weighted: jnp.ndarray) -> jnp.ndarray:
    """Scalar feature map proxy: variance of the weighted field."""
    return jnp.mean(delta_weighted**2)


def main() -> None:
    grid = Grid3D(shape=(16, 16, 16), L=200.0)
    gen = GaussianFieldGenerator3D(grid=grid, Pk_func=toy_pk)

    key = jax.random.PRNGKey(0)
    noise = grid.generate_hermitian_noise(key)
    theta = jnp.array([1.0, 0.3], dtype=jnp.float64)

    field = gen.make_realization_from_noise(theta, noise)
    field.compute_ifft()
    delta = field.delta

    c1 = 0.0
    c2_values = jnp.linspace(-1.0, 1.0, 5)
    c3_values = jnp.linspace(-1.0, 1.0, 5)
    c2_grid, c3_grid = jnp.meshgrid(c2_values, c3_values, indexing="ij")
    coeffs = jnp.stack(
        [
            2.0 * jnp.ones_like(c2_grid),
            c1 * jnp.ones_like(c2_grid),
            c2_grid,
            c3_grid,
        ],
        axis=-1,
    ).reshape(-1, 4)

    def info_for_coeffs(c: jnp.ndarray) -> jnp.ndarray:
        weighted_delta = build_weighted_delta(delta, c)
        return feature_information(weighted_delta)

    info_values = jax.vmap(info_for_coeffs)(coeffs)
    info_map = info_values.reshape(c2_values.shape[0], c3_values.shape[0])

    max_index = jnp.unravel_index(jnp.argmax(info_map), info_map.shape)
    best_c2 = c2_values[max_index[0]]
    best_c3 = c3_values[max_index[1]]

    print("Feature map (variance proxy) over C2/C3:")
    print(info_map)
    print(f"Best C2: {float(best_c2):.3f}")
    print(f"Best C3: {float(best_c3):.3f}")


if __name__ == "__main__":
    main()
