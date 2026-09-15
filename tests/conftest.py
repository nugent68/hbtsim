"""Shared test configuration: enable float64 in JAX once, before any
module is imported, so array dtypes do not depend on collection order
(the renderer and DFT kernels force float32 internally regardless)."""

import jax

jax.config.update("jax_enable_x64", True)
