"""hbtsim: Hanbury Brown-Twiss intensity-interferometry simulation of
binary and single stars.

Targets, telescopes, detectors, spectrographs, backends, sites, arrays and
campaigns are JSON definitions in hbtsim/configs, loaded through
hbtsim.catalog:

    from hbtsim import load_target, load_array
    spica = load_target("spica")
    vlt = load_array("vlt_ut")
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("hbtsim")
except PackageNotFoundError:          # a source tree that was never installed
    __version__ = "0+unknown"

from .catalog import (Catalog, load_array, load_backend, load_campaign, load_detector,
                      load_site, load_spectrograph, load_target, load_telescope, load_triangle)
from .params import BinarySystem, DiskTarget, GridConfig, MovieConfig, Star
from .spectral import spectral_vis2

__all__ = ["__version__", "Catalog", "load_array", "load_backend", "load_campaign", "load_detector",
           "load_site", "load_spectrograph", "load_target", "load_telescope", "load_triangle",
           "BinarySystem", "DiskTarget", "GridConfig", "MovieConfig", "Star", "spectral_vis2"]
