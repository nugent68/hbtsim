"""hbtsim: Hanbury Brown-Twiss intensity-interferometry simulation of
binary stars, starting with Beta Aurigae."""

from .params import (ALGOL, BETA_AUR, SYSTEMS, BinarySystem, GridConfig,
                     MovieConfig, Star)
from .spectral import spectral_vis2

__all__ = ["ALGOL", "BETA_AUR", "SYSTEMS", "BinarySystem", "GridConfig",
           "MovieConfig", "Star", "spectral_vis2"]
