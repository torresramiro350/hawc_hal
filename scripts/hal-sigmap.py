from __future__ import division

import argparse
import os
from builtins import range
from pathlib import Path

import astropy.units as u
import astropy.wcs as wcs
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates.angles import angular_separation
from astropy.io import fits as pyfits
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm.auto import tqdm

try:
    for env in ["OMP", "MKL", "NUMEXPR"]:
        os.environ[f"{env}_NUM_THREADS"] = "1"
    import threeML
    from astromodels.core.model_parser import clone_model
    from hawc_hal import HAL, HealpixConeROI
    from threeML.io.logging import setup_logger

except Exception as exc:
    raise Exception(f"Failed from {exc}")

log = setup_logger(__name__)
log.propagate = False

_worker_hal: HAL | None = None
_worker_like0: float | None = None
_worker_points: list[tuple[float, ...]] | None = None
_worker_model: threeML.Model | None = None
_worker_jl: threeML.JointLikelihood | None = None


def _source_model(norm: float, index: float, pivot: float):
    """Define simple model with a point source

    Parameters
    ----------
    norm : `float`
        Normalization (1/TeV cm2 s)
    index : `float`
        Spectral index
    pivot : `float`
        Pivot energy (TeV)

    Returns
    -------
    `threeML.Model`
        Likelihood model instance

    """
    crab_diff_flux_low = 1e-20 / (u.TeV * u.cm**2 * u.s)
    crab_diff_flux_hi = 1e-10 / (u.TeV * u.cm**2 * u.s)
    spectrum = threeML.Powerlaw()

    this_ra, this_dec = 0.0, 0.0

    this_source = threeML.PointSource(
        "TestSource", ra=this_ra, dec=this_dec, spectral_shape=spectrum
    )

    # Start from a flux 1/10 of the Crab
    spectrum.K = norm / (u.TeV * u.cm**2 * u.s)
    spectrum.K.bounds = (crab_diff_flux_low, crab_diff_flux_hi)

    spectrum.index = index
    spectrum.index.fix = True

    spectrum.piv = pivot * u.TeV
    spectrum.piv.fix = True

    model = threeML.Model(this_source)

    return model


def _init_worker(
    like0: float,
    points: list[tuple[float, ...]],
    model: threeML.Model,
) -> None:
    """Initialize global variable definitions to pass onto parallel processing pool

    Parameters
    ----------
    like0 : float
        Null log-likelihood
    points : list[tuple[float, ...]]
        List of coordinate positions over which to estimate the TS map
    model : threeML.Model
        Source model instance

    """
    global _worker_hal, _worker_like0, _worker_points
    global _worker_model, _worker_jl

    _like0 = like0
    _worker_points = points
    _worker_model = clone_model(model)

    data = threeML.DataList(_worker_hal)
    _worker_jl = threeML.JointLikelihood(_worker_model, data)


def _worker_func(interval_id: int) -> tuple[int, float]:
    """Perform fit for TS map

    Parameters
    ----------
    interval_id : int
        Current (RA, Dec) coordinate combination

    Returns
    -------
    tuple[int, float]
        Current interval id and corresponding minimized log-likelihood

    """
    global _worker_hal, _worker_like0, _worker_points
    global _worker_model, _worker_jl

    try:
        ra, dec = _worker_points[interval_id]
        source = _worker_model.TestSource
        source.position.ra = ra
        source.position.dec = dec
        # Fit
        _, like = _worker_jl.fit(quiet=True, compute_covariance=False)
        return interval_id, like["-log(likelihood)"]["HAWC"]
    except Exception:
        return interval_id, np.nan


class ParallelTSmap:
    def __init__(
        self,
        maptree: Path,
        response: Path,
        model: threeML.Model,
        ra_c: float,
        dec_c: float,
        xsize: int,
        ysize: int,
        pix_scale: float,
        bin_id_min: int | None = None,
        bin_id_max: int | None = None,
        bin_list: list[str] | None = None,
        projection: str = "AIT",
        roi_radius: float = 3.0,
    ):
        """Generate a TS map with HAL

        Parameters
        ----------
        maptree : `Path`
            Maptree in ROOT or HD5 format
        response : `Path`
            Detector response in ROOT or HD5 format
        model : `threeML.Model`
            Likelihood source model instance
        ra_c : `float`
            RA (J2000) position for ROI
        dec_c : `float`
            Dec (J2000) position for ROI
        xsize : `int`
            Width of map
        ysize : `int`
            Height of map
        pix_scale : `float`
            Pixel scale in degrees
        bin_id_min : `int | None`
            Minimum fHit bin id
        bin_id_max : `int | None`
            Maximum fHit bin id
        bin_list : `list[str] | None`
            Energy estimator bin list
        projection : `str`
            Astronomical projection
        roi_radius : `float`
            Radius in degrees for ROI

        """
        self._ra_c = ra_c
        self._dec_c = dec_c

        self._mtfile = maptree
        self._rsfile = response
        self._bin_id_min = bin_id_min
        self._bin_id_max = bin_id_max
        self._bin_list = bin_list
        self._model = model

        # Create a new WCS object so that we can compute the appropriare R.A. and Dec
        # where we need to compute the TS
        self._setup_wcs(xsize, ysize, pix_scale, projection)

        self._points: list[tuple[float, ...]] = []
        self._valid_mask = np.zeros((ysize, xsize), dtype=bool)

        # It is important that dec is the first one because the PSF for a Dec bin_name
        # is cached within one engine

        max_d = 0

        for idec in range(ysize):
            for ira in range(xsize):
                this_ra, this_dec = self._wcs.wcs_pix2world(ira, idec, 0)

                d = angular_separation(*np.deg2rad((this_ra, this_dec, ra_c, dec_c)))

                if np.rad2deg(d) <= roi_radius:
                    self._points.append((this_ra, this_dec))
                    self._valid_mask[idec, ira] = True

                if d > max_d:
                    max_d = d

        log.info(f"Maximum distance from center: {max_d:.3f} deg")

        # We keep track of how many ras we have so that when running in parallel all
        # the ras will run on the same engine with the same dec, maximizing the use
        # of the cache and minimizing the memory footprint
        self._n_ras = xsize
        self._n_decs = ysize
        self._ts_map = np.full((self._n_decs, self._n_ras), np.nan)

        self._roi_radius = float(roi_radius)

        roi = HealpixConeROI(
            self._roi_radius, self._roi_radius + 5.0, ra=ra_c, dec=dec_c
        )

        self._llh = HAL(
            "HAWC",
            self._mtfile,
            self._rsfile,
            roi,
            n_workers=4,
            flat_sky_pixels_size=0.1,
        )
        self._llh.set_active_measurements(
            bin_id_min=bin_id_min, bin_id_max=bin_id_max, bin_list=bin_list
        )
        global _worker_hal
        _worker_hal = self._llh
        self._like0 = self._compute_null_likelihood()
        # We will fill this with the maximum of the TS map

        self._max_ts: tuple[float, tuple[float, ...]] = None

    def _setup_wcs(
        self, xsize: int, ysize: int, pix_scale: float, projection: str
    ) -> None:
        """Generate a world cordinate system (WCS) for current source

        Parameters
        ----------
        xsize : `int`
            Number of pixels in x-direction
        ysize : `int`
            Number of pixels in y-direction
        pix_scale : `float`
            Step size in degrees for pixel
        projection : `str`
            Astrophysical projection

        """
        self._wcs = wcs.WCS(naxis=2)
        self._wcs.wcs.crpix = [xsize / 2.0, ysize / 2.0]
        self._wcs.wcs.cdelt = [-pix_scale, pix_scale]
        self._wcs.wcs.crval = [self._ra_c, self._dec_c]
        self._wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]

    def _compute_null_likelihood(self) -> float:
        """Compute likelihood under the null hypothesis (no source).

        Returns
        -------
        `float`:
            Log-likelihood value for null hypothesis.

        """
        # Make a fit with no source to get the likelihood for the null hypothesis
        model = clone_model(self._model)
        model.TestSource.spectrum.main.shape.K = (
            model.TestSource.spectrum.main.shape.K.min_value
        )

        self._llh.set_model(model)

        return self._llh.get_log_like()

    def get_data(self, interval_id: int) -> threeML.DataList:
        """Format HAL instance to a 3ML format.

        Parameters
        ----------
        interval_id : `int`
            Current inteval id for TS map.

        Returns
        -------
        `threeML.DataList`
            Container of current data sets.

        """
        datalist = threeML.DataList(self._llh)

        return datalist

    def go(self, nprocs: int = 6) -> NDArray[np.float64]:
        """Execute operations for TS map either in parallel or serial

        Parameters
        ----------
        nprocs : `int`
            Define the number of parallel processes over which to operate.

        Returns
        -------
        `NDArray[np.float64]`
            TS map image

        """
        from multiprocessing import Pool

        # nprocs = min(os.cpu_count(), 4)
        # if use_parallel:
        res = np.zeros(len(self._points))
        pool_args = list(range(len(self._points)))
        _init_args = (
            self._like0,
            self._points,
            self._model,
        )
        with Pool(processes=nprocs, initializer=_init_worker, initargs=_init_args) as p:
            with tqdm(total=len(self._points), desc="Parallel TS map") as pbar:
                for k, result in p.imap_unordered(
                    _worker_func, pool_args, chunksize=self._n_ras
                ):
                    res[k] = result
                    pbar.update(1)

        ts_values = 2 * (-res - self._like0)

        self._ts_map[self._valid_mask] = ts_values
        self._ts_map = np.nan_to_num(self._ts_map, nan=0.0)

        idx = ts_values.argmax()
        self._max_ts = (ts_values[idx], self._points[idx])

        log.info(
            f"Maximum TS is {self._max_ts[0]:.2f} at (R.A., Dec) = "
            + f"({self._max_ts[1][0]:.3f}, {self._max_ts[1][1]:.3f})"
        )

        return self._ts_map

    @property
    def maximum_of_map(self) -> float:
        return self._max_ts[0]

    def to_fits(self, filename: Path, overwrite: bool = True) -> None:
        """Write TS map to a FITS file for easy reading

        Parameters
        ----------
        filename : `Path`
            Path to file
        overwrite : `bool`
            Allows the overwriting of already existing file

        """
        primary_hdu = pyfits.PrimaryHDU(data=self._ts_map, header=self._wcs.to_header())

        primary_hdu.writeto(filename, overwrite=overwrite)

    def plot(
        self,
        cmap: LinearSegmentedColormap,
        amin: float = -5,
        amax: float = 25,
        contour_levels: list[float] = [3, 5, 7],
    ) -> Figure:
        """Draw the estimated significance (sqrt TS) map

        Parameters
        ----------
        cmap : LinearSegmentedColormap
            Colormap name
        amin : float
            Minimum significance boundary (sigmas)
        amax : float
            Maximum significance boundary (sigmas)
        contour_levels : list[float]
            Contour levels in significance units

        Returns
        -------
        Figure
            Significance map with astronomical projection

        """
        # Draw significance (sqrt TS) map
        fig, ax = plt.subplots(subplot_kw={"projection": self._wcs})

        # im = ax.imshow(self._ts_map, origin="lower", interpolation="none")
        sigmap = np.sign(self._ts_map) * np.sqrt(np.abs(self._ts_map))
        im = ax.imshow(
            sigmap,
            origin="lower",
            interpolation="bilinear",
            vmin=amin,
            vmax=amax,
            cmap=cmap,
        )

        # Draw colorbar
        cbar = fig.colorbar(im, format="%.1f")
        cbar.set_label(r"$\sqrt{TS}$")

        # Access the first WCS coordinate
        ra = ax.coords[0]
        dec = ax.coords[1]

        # Overlay the center
        ax.scatter(
            [self._ra_c], [self._dec_c], transform=ax.get_transform("world"), marker="."
        )

        # Overlay the maximum TS
        ra_max, dec_max = self._max_ts[1]
        ax.scatter([ra_max], [dec_max], transform=ax.get_transform("world"), marker="x")

        # Set the format of the tick labels
        ra.set_major_formatter("d.dd")
        dec.set_major_formatter("d.dd")
        cont = ax.contour(sigmap, levels=contour_levels, colors="black")

        ax.set_xlabel("R.A. (J2000)")
        ax.set_ylabel("Dec. (J2000)")
        ax.clabel(cont, inline=True, fmt="%d" + r"$\sigma$")
        ax.tick_params(which="major", direction="in", color="white")

        return fig


def set_linear_interp_cmap(
    amin: float, amax: float, threshold: float, cmap: Colormap, ncolors: int = 256
):
    modified_cmap = []

    threshMap = 0.2

    threshold2 = threshold + 0.4 * (amax - threshold)
    threshMap2 = 0.8

    for x in np.linspace(0, 1, ncolors):
        if x <= threshMap:
            y = (amin + (threshold - amin) * (x - 0) / (threshMap - 0) - amin) / (
                amax - amin
            )
        elif x <= threshMap2:
            y = (
                threshold
                + (threshold2 - threshold) * (x - threshMap) / (threshMap2 - threshMap)
                - amin
            ) / (amax - amin)
        else:
            y = (
                threshold2
                + (amax - threshold2) * (x - threshMap2) / (1 - threshMap2)
                - amin
            ) / (amax - amin)

        modified_cmap.append((y, cmap(x)))

    return modified_cmap


def setup_magma_colormap(
    amin: float,
    amax: float,
    threshold: float,
    reverse: bool = False,
    ncolors: int = 256,
) -> LinearSegmentedColormap:
    cmap_name = "magma_r" if reverse else "magma"
    magma = mpl.colormaps.get_cmap(cmap_name)

    threshMagma = set_linear_interp_cmap(amin, amax, threshold, magma, ncolors)

    newcm = mpl.colors.LinearSegmentedColormap.from_list(
        "threshMagma", threshMagma, ncolors
    )
    return newcm


def setup_ylgnbul_colormap(
    amin: float,
    amax: float,
    threshold: float,
    reverse: bool = False,
    ncolors: int = 256,
) -> LinearSegmentedColormap:
    cmap_name = "YlGnBu_r" if reverse else "YlGnBu"
    ylgnbul = mpl.colormaps.get_cmap(cmap_name)

    threshYlGnBu = set_linear_interp_cmap(amin, amax, threshold, ylgnbul, ncolors)

    newcm = mpl.colors.LinearSegmentedColormap.from_list(
        "threshYlGnBu", threshYlGnBu, ncolors
    )
    return newcm


def setup_spectral_colormap(
    amin: float,
    amax: float,
    threshold: float = 2,
    reverse: bool = False,
    ncolors: int = 256,
) -> LinearSegmentedColormap:
    cmap_name = "Spectral_r" if reverse else "Spectral"
    ylgnbul = mpl.colormaps.get_cmap(cmap_name)

    threshYlGnBu = set_linear_interp_cmap(amin, amax, threshold, ylgnbul, ncolors)

    newcm = mpl.colors.LinearSegmentedColormap.from_list(
        "threshSpectral", threshYlGnBu, ncolors
    )
    return newcm


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Generate a TS map with HAL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-m", "--maptree", dest="maptree", required=True, help="ROOT maptree file"
    )
    p.add_argument(
        "-d", "--det-res", dest="response", required=True, help="ROOT maptree file"
    )
    p.add_argument(
        "--roi-center",
        dest="roi_center",
        nargs=2,
        type=float,
        required=True,
        help="Center of ROI (RAJ2000, DecJ2000)",
    )
    p.add_argument(
        "--roi-radius",
        dest="roi_radius",
        type=float,
        default=3.0,
        help="Radius of region of interest (ROI)",
    )
    p.add_argument(
        "--bin-list",
        dest="bin_list",
        nargs="*",
        default=None,
        help="List of analysis bins",
    )
    p.add_argument(
        "--low-fhit-bin",
        dest="bin_min",
        type=int,
        default=None,
        help="Starting fHit bin",
    )
    p.add_argument(
        "--high-fhit-bin",
        dest="bin_max",
        type=int,
        default=None,
        help="Ending fHit bin",
    )
    p.add_argument(
        "-o",
        "--outname",
        dest="output",
        default="Crab",
        type=str,
        help="Output name for TS map files",
    )
    spectral_options = p.add_argument_group("Spectral Properties")
    spectral_options.add_argument(
        "--index", dest="spec_index", type=float, default=2.65, help="Spectral index"
    )
    spectral_options.add_argument(
        "--norm",
        dest="norm",
        type=float,
        default=1.4e-11,
        help="Normalization 1/(TeV cm2 s)",
    )
    spectral_options.add_argument(
        "--pivot", dest="pivot", type=float, default=7, help="Pivot energy [TeV]"
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    threeML.silence_warnings()
    threeML.silence_logs()
    maptree = Path(args.maptree)
    response = Path(args.response)
    roi_radius = args.roi_radius
    roi_center_ra, roi_center_dec = args.roi_center
    bin_list = args.bin_list
    src_model = _source_model(args.norm, -args.spec_index, args.pivot)
    cmap = setup_spectral_colormap(-5, 25, reverse=True)

    parallel_tsmap = ParallelTSmap(
        maptree,
        response,
        model=src_model,
        ra_c=roi_center_ra,
        dec_c=roi_center_dec,
        xsize=100,
        ysize=100,
        pix_scale=(hp.nside2resol(1024, arcmin=True) * u.arcmin).to(u.deg).value,
        bin_id_min=args.bin_min,
        bin_id_max=args.bin_max,
        bin_list=bin_list,
        roi_radius=roi_radius,
    )
    parallel_tsmap.go()
    fig = parallel_tsmap.plot(cmap, amin=-5, amax=25, contour_levels=[10, 15, 20, 25])
    fig.savefig(f"{args.output}_simple_ts_map.png", dpi=300, bbox_inches="tight")
    parallel_tsmap.to_fits(Path.cwd() / f"{args.output}_ts_map.fits", overwrite=True)


if __name__ == "__main__":
    args = options()
    main(args)
