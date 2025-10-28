import copy
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from jsonargparse import auto_cli
from astropy.io import fits
from astropy.time import Time
from docstring_parser import DocstringStyle
from jsonargparse import set_parsing_settings

set_parsing_settings(docstring_parse_style=DocstringStyle.NUMPYDOC)


def split_beam(
    beam_file: str | Path,
    freq_range: Optional[Tuple[float, float]] = None,
    prefix: Optional[str] = None,
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
    freq_range: (float, float), optional
        Frequency range to split the beam in Hz. Default None will keep all frequencies.
    prefix: str, optional
        Prefix of the output file. Use stem of beam_file if None

    Returns
    -------
    out
        8 fits files written to <beam_file.parent>/<prefix>.<corr>_<reim>.fits
    """
    beam_file = Path(beam_file)
    prefix = beam_file.stem if prefix is None else prefix
    with fits.open(beam_file) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    # Update header
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
            np.logical_and(freqs > freq_range[0], freqs < freq_range[1])
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
            
            outfile = (
                beam_file.parent / f"{prefix}.{corr}_{reim}{beam_file.suffix}"
            )
            print(f"Splitting out {outfile}")
            hdu.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    auto_cli(split_beam)