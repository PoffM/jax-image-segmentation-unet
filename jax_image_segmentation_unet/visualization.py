import jax.numpy as jnp
from matplotlib import colors

to_rgb = jnp.array(
    [
        colors.to_rgb("orange"),
        colors.to_rgb("red"),
        colors.to_rgb("green"),
        colors.to_rgb("blue"),
        colors.to_rgb("yellow"),
        colors.to_rgb("magenta"),
        colors.to_rgb("purple"),
        colors.to_rgb("gray"),
        colors.to_rgb("white"),
        colors.to_rgb("brown"),
        # Number 11 will mean unclassified
        colors.to_rgb("black"),
    ]
)
