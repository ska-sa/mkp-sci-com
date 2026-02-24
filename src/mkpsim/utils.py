import copy
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


def generate_stationspec(
    beam_path: str | Path, output_dir: str | Path, array: Literal["hetero", "homo"]
) -> Path:
    """Generate a stationspec JSON file from bundled templates.

    Parameters
    ----------
    beam_path:
        Path that will replace the literal string "beam-path" inside the template
        (e.g. `/abs/path/to/beams` or `relative/path/to/beams`).
    output_dir:
        Directory where the generated JSON will be written.
    arry:
        Array type, one of "hetero" (MeerKAT+ heterogeneous array) or
        "homo" (MeerKAT+ array with MeerKAT beam only).

    Returns
    -------
    str
        The path to the written stationspec JSON file.

    Notes
    -----
    The function looks for the template files shipped under `configs/`:
    - `stationspec_hetero.json`
    - `stationspec_homo.json`

    The literal substring "beam-path" in the template is replaced with
    the full path of `beam_path` paramter.
    """
    allowed = {
        "hetero": "stationspec_hetero.json",
        "homo": "stationspec_homo.json",
    }

    key = array.lower()
    if key not in allowed:
        raise ValueError(
            "array must be 'hetero' or 'homo')"
        )

    template_name = allowed[key]

    # try to read template from package resources, fall back to source-tree
    try:
        from importlib.resources import files

        tpl = (
            files("mkpsim")
            .joinpath("configs", template_name)
            .read_text(encoding="utf-8")
        )
    except Exception:
        tpl_path = Path(__file__).parent.joinpath("configs", template_name)
        tpl = tpl_path.read_text(encoding="utf-8")

    # replace placeholder
    beam_path = Path(beam_path).resolve()
    out_text = tpl.replace("beam-path", str(beam_path))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"stationspec_{key}.json"
    out_path.write_text(out_text, encoding="utf-8")
    return out_path


def generate_point_source_grid(
    flux: float,
    ra_cen: float,
    dec_cen: float,
    nrows: int,
    ncols: int,
    spacing: tuple[float, float] | None = None,
    extent: tuple[float, float] | None = None,
    filename: str | None = None,
) -> Path:
    """Generate a grid of point sources in Tigger-compatible ASCII format.

    Parameters
    ----------
    flux : float
        Flux of each point source (Jy).
    ra_cen : float
        Right ascension of grid center (decimal degrees).
    dec_cen : float
        Declination of grid center (decimal degrees).
    nrows : int
        Number of rows in the grid.
    ncols : int
        Number of columns in the grid.
    output_dir : str | Path
        Directory where the output ASCII file will be written.
    spacing : tuple[float, float], optional
        (RA, DEC) spacing between adjacent sources in decimal degrees.
        Mutually exclusive with extent.
    extent : tuple[float, float], optional
        (RA, DEC) extent of the entire grid in decimal degrees.
        Mutually exclusive with spacing.
    filename : str, optional
        Output filename, including path. If None, defaults to "./point_source_grid.txt".

    Returns
    -------
    Path
        Path to the written ASCII file.

    Notes
    -----
    Either spacing or extent must be provided, but not both.
    The output format is Tigger-compatible ASCII with columns:
    #format: ra_rad dec_rad i

    Examples
    --------
    Generate a 3x3 grid with 0.5 degree spacing centered at RA=04:00:00, DEC=-33:00:30:

    >>> generate_point_source_grid(
    ...     flux=1.0,
    ...     ra_cen=60.0,  # 04:00:00 in decimal degrees
    ...     dec_cen=-33.008333,  # -33:00:30
    ...     nrows=3,
    ...     ncols=3,
    ...     output_dir="sky_models",
    ...     spacing=(0.5, 0.5),
    ...     filename="grid_3x3.txt"
    ... )
    """
    if (spacing is None and extent is None) or (
        spacing is not None and extent is not None
    ):
        raise ValueError("Exactly one of 'spacing' or 'extent' must be provided")

    # Calculate spacing from extent if needed
    if extent is not None:
        extent_ra, extent_dec = extent
        spacing_ra = extent_ra / max(ncols - 1, 1)
        spacing_dec = extent_dec / max(nrows - 1, 1)
    else:
        spacing_ra, spacing_dec = spacing

    # Convert degrees to radians
    deg2rad = np.pi / 180.0

    # Generate grid of RA/DEC positions
    positions = []
    for row in range(nrows):
        for col in range(ncols):
            ra = ra_cen + (col - (ncols - 1) / 2) * spacing_ra
            dec = dec_cen + (row - (nrows - 1) / 2) * spacing_dec
            ra_rad = ra * deg2rad
            dec_rad = dec * deg2rad
            positions.append((ra_rad, dec_rad, flux))

    # Prepare grid description comment
    if extent is not None:
        grid_desc = (
            f"Grid: {nrows}x{ncols}, extent=({extent[0]:.4f}, {extent[1]:.4f}) deg"
        )
    else:
        grid_desc = (
            f"Grid: {nrows}x{ncols}, spacing=({spacing[0]:.4f}, {spacing[1]:.4f}) deg"
        )

    # Write to file
    if filename is None:
        out_path = Path("./point_source_grid.txt").resolve()
    else:
        out_path = Path(filename).resolve()
        outdir = out_path.parent
        outdir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("#format: ra_rad dec_rad i\n")
        f.write(
            f"# {grid_desc}, center=({ra_cen:.4f}, {dec_cen:.4f}) deg, flux={flux} Jy\n"
        )
        for ra_rad, dec_rad, fl in positions:
            f.write(f"{ra_rad:.10g} {dec_rad:.10g} {fl}\n")

    return out_path


def split_beam(
    beam_file: str | Path,
    outdir: str | Path | None = None,
    prefix: str | None = None,
    freq_range: tuple[float, float] | None = None,
):
    """
    Split a 5-axis beamfits file into 8 re/im voltage files.

    Assume the following:
    - Input beam has 5 axes: px---ssn, py---ssn, freq, stokes, complex (re,im)
    - Output beam has 3-axis: px, py, freq

    Parameters
    ----------
    beam_file: str | path
        Beam file in fits format
    outdir: str | path, optional
        Output directory to write the split beam files.
        Default to same directory as beam_file.
    prefix: str, optional
        Prefix of the output file. Use stem of beam_file if None
    freq_range: (float, float), optional
        Frequency range to split the beam in MHz.
        Default None will keep all frequencies.

    Returns
    -------
    out
        8 fits files written to <beam_file.parent>/<prefix>.<corr>_<reim>.fits
    """
    beam_file = Path(beam_file)
    outdir = Path(outdir) if outdir is not None else beam_file.parent
    prefix = beam_file.stem if prefix is None else prefix
    with fits.open(beam_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    # Update header
    # Change CTYPE value PX--SSN and PY--SSN to just PX an PY as expected by meqtree
    for key in header.keys():
        if header[key] == "PX---SSN":
            header[key] = "px"
        if header[key] == "PY---SSN":
            header[key] = "py"

    # Axis 4 and 5 are STOKES and COMPLEX
    # Get and delete keys associated with them
    keys_with_4_or_5 = [key for key in header.keys() if "4" in key or "5" in key]
    for key in keys_with_4_or_5:
        del header[key]
    header["NAXIS"] = 3

    # Frequency selection
    if freq_range is not None:
        # Get frequency information
        start = header["crval3"]
        step = header["cdelt3"]
        stop = start + header["naxis3"] * step + step / 2
        freqs = np.arange(start, stop, step)

        # Determine indices to select from
        freq_inds = np.where(
            np.logical_and(freqs > freq_range[0] * 1e6, freqs < freq_range[1] * 1e6)
        )[0]
        print(f"Selecting frequencies: {freqs[freq_inds]}")

        # Also update header
        header["NAXIS3"] = freq_inds.size
        header["CRVAL3"] = freqs[freq_inds[0]]
        header["CRPIX3"] = 1.0
    else:
        freq_inds = slice(None)

    # Loop over Stokes and real/imag and write out
    stokes = ["xx", "yy", "xy", "yx"]
    complex_part = ["re", "im"]
    for i, reim in enumerate(complex_part):
        for j, corr in enumerate(stokes):
            bm = data[i, j, freq_inds, :, :]
            hdr = copy.deepcopy(header)
            hdr["HISTORY"] = (
                f"{Time.now().isot} Split {reim} part of {corr} from {beam_file.name}"
            )
            hdu = fits.PrimaryHDU(data=bm, header=hdr)

            outfile = outdir / f"{prefix}.{corr}_{reim}{beam_file.suffix}"
            print(f"Splitting out {outfile}")
            hdu.writeto(outfile, overwrite=True)


def _translate_wcs_slices(slices: tuple) -> tuple:
    """Translate WCS slicing tuple to numpy slicing tuple."""
    new_slices = []
    for slc in slices[::-1]:
        if isinstance(slc, str):
            new_slices.append(slice(None))
        elif isinstance(slc, int):
            new_slices.append(slc)
        else:
            raise ValueError("slices value must be 'x', 'y' or integer")
    new_slices = tuple(new_slices)
    return new_slices


def plot_image(
    img1: str | Path,
    slices1: tuple = ("x", "y", 0, 0),
    img2: str | Path | None = None,
    slices2: tuple | None = None,
    quantity: Literal["diff", "fraction"] = "diff",
    figsize: tuple[int, int] = (6, 6),
    savepng: str | Path | None = None,
    imshow_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
):
    """Plot a FITS image with WCS projection.

    Parameters
    ----------
    img1 : str | Path
        Path to the FITS file to plot.
    slices1 : tuple, optional
        Data slices. Default is ("x", "y", 0, 0).
        See https://docs.astropy.org/en/stable/visualization/wcsaxes/slicing_datacubes.html
    img2 : str | Path, optional
        Path to a second FITS file to plot the difference.
    slices2 : tuple, optional
        Data slices for the second image. Use slices1 if None.
    quantity : "fraction" or "diff", optional
        Whether to plot img1/img2 ("fraction") or img1-img2 ("diff"). 
        Default "fraction". Only used if img2 is provided.
    figsize : tuple[int, int], optional
        Figure size (width, height) in inches, default (10, 10).
    savepng : str | Path, optional
        If provided, save the figure as a PNG file at this path.
    imshow_kwargs : dict, optional
        Additional keyword arguments to pass to ax.imshow().
    cbar_kwargs : dict, optional
        Additional keyword arguments to pass to fig.colorbar().
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """

    # Load and slice the image data to 2D array
    data1_slices = _translate_wcs_slices(slices1)
    data1 = fits.getdata(img1, 0)[data1_slices]
    if img2 is not None:
        if slices2 is None:
            data2_slices = data1_slices
        else:
            data2_slices = _translate_wcs_slices(slices2)
        data2 = fits.getdata(img2, 0)[data2_slices]
        if quantity == "fraction":
            data_plot = data1 / data2
        else:
            data_plot = data1 - data2
    else:
        data_plot = data1


    # Figure can use WCS from either images as long as the correct slices
    # are passed in.
    wcs = WCS(fits.getheader(img1))
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(projection=wcs, slices=slices1),
        constrained_layout=True,
    )
    im = ax.imshow(data_plot, origin="lower", **(imshow_kwargs or {}))
    ax.grid(color="white", ls="solid")
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    fig.colorbar(im, ax=ax, **cbar_kwargs or {})

    if savepng is not None:
        savepng = Path(savepng)
        savepng.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepng, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {savepng}")

    return fig, ax
