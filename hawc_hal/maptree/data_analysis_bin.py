from builtins import object, str
from typing import TypeAlias as T

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..healpix_handling import DenseHealpix, SparseHealpix

ndarri64: T = NDArray[np.int64]


class DataAnalysisBin(object):
    def __init__(
        self,
        name: str,
        observation_hpx_map: SparseHealpix | DenseHealpix,
        background_hpx_map: SparseHealpix | DenseHealpix,
        active_pixels_ids: list[str],
        n_transits: float,
        scheme: str = "RING",
    ):
        # Get nside
        self._nside: int = observation_hpx_map.nside

        nside_bkg = background_hpx_map.nside

        assert self._nside == nside_bkg, (
            "Observation and background maps have "
            "different nside (%i vs %i)" % (self._nside, nside_bkg)
        )

        self._npix = observation_hpx_map.npix

        # Store healpix maps
        self._observation_hpx_map: SparseHealpix | DenseHealpix = observation_hpx_map

        self._background_hpx_map: SparseHealpix | DenseHealpix = background_hpx_map

        # Store the active pixels (i.e., the pixels that are within the selected ROI)
        self._active_pixels_ids: list[str] = active_pixels_ids

        # Store name and scheme
        self._name: str = str(name)

        assert scheme in {"RING", "NEST"}, "Scheme must be either RING or NEST"

        self._scheme: str = scheme

        self._n_transits: float = n_transits

    def to_pandas(self) -> tuple[pd.DataFrame, dict[str, float | int]]:
        # Make a dataframe
        df = pd.DataFrame.from_dict(
            {
                "observation": self._observation_hpx_map.to_pandas(),
                "background": self._background_hpx_map.to_pandas(),
            }
        )

        if self._active_pixels_ids is not None:
            # We are saving only a subset
            df.set_index(self._active_pixels_ids, inplace=True)

        # Some metadata
        meta: dict[str, float | int] = {
            "scheme": 0 if self._scheme == "RING" else 1,
            "n_transits": self._n_transits,
            "nside": self._nside,
        }

        return df, meta

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_transits(self) -> float:
        return self._n_transits

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def npix(self) -> int:
        return self._npix

    @property
    def observation_map(self) -> SparseHealpix | DenseHealpix:
        return self._observation_hpx_map

    @property
    def background_map(self) -> SparseHealpix | DenseHealpix:
        return self._background_hpx_map
