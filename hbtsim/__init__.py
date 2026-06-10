"""hbtsim: Hanbury Brown-Twiss intensity-interferometry simulation of
binary stars, starting with Beta Aurigae."""

from .params import BETA_AUR, BinarySystem, GridConfig, MovieConfig, Star
from .spectral import spectral_vis2

__all__ = ["BETA_AUR", "BinarySystem", "GridConfig", "MovieConfig", "Star",
           "spectral_vis2"]
