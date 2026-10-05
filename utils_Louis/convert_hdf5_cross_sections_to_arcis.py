#!/usr/bin/env python3
"""Convert high-resolution HDF5 cross-sections to an ARCiS FITS k-table.

Expected input files are named ``cross_T<TEMPERATURE>.hdf5`` and contain:

* ``wave``: wavelength in m (linear), shape ``(n_wave,)``;
* ``P``: log10 pressure in Pa, shape ``(n_pressure,)``;
* ``T``: temperature in K, normally shape ``(1,)``;
* ``cross_sec``: log10 cross-section in m2/molecule, shape
  ``(n_wave, n_pressure, 1)``.

The output follows the layout read by ``OpacityFITS.f``:

* HDU 0: k coefficients in cm2/molecule, NumPy shape
  ``(n_pressure, n_temperature, n_g, n_wavelength)``;
* HDU 1: temperatures in K;
* HDU 2: pressures in bar;
* HDU 3: wavelength centres in cm;
* HDU 4: correlated-k quadrature weights on [0, 1].

The native cross-sections inside each output wavelength cell are sorted and
averaged over Gauss-Legendre probability intervals. This is the same kind of
wavelength-overlap and g-distribution rebinning used by ARCiS itself in
``OpacityFITS.f``.


Example for how to execute this code:


python /Users/louissiebenaler/ARCiS/src/utils_Louis/convert_hdf5_cross_sections_to_arcis.py \
    "/Volumes/L7aler_HD/PhD/Snellius/cross_sections/Li/combined" \
    "/Users/louissiebenaler/ARCiS/Data/Opacities/opacity_Li_Louis.fits" \
    --resolution 10000 \
    --ng 1 \
    --wavelength-min-micron 0.1 \
    --wavelength-max-micron 50 \
    --pressure-grid union \
    --pressure-max-bar 1000 \
    --pressure-extrapolation nearest

(ng is the number of Gauss-Legendre points)

"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


FILE_PATTERN = re.compile(
    r"^cross_T(?P<temperature>[0-9]+(?:\.[0-9]+)?)\.hdf5$"
)
PA_PER_BAR = 1.0e5
M_PER_MICRON = 1.0e-6
CM_PER_M = 1.0e2
M2_TO_CM2 = 1.0e4


@dataclass(frozen=True)
class InputTable:
    path: Path
    temperature: float
    log_pressure_pa: np.ndarray


def discover_tables(
    input_directory: Path,
    temperature_min: float | None,
    temperature_max: float | None,
) -> list[InputTable]:
    """Find, validate, and temperature-sort the real HDF5 input files."""
    tables: list[InputTable] = []

    for path in input_directory.iterdir():
        match = FILE_PATTERN.match(path.name)
        if match is None:
            # This also ignores macOS AppleDouble files named ._cross_T...
            continue

        filename_temperature = float(match.group("temperature"))
        if (
            temperature_min is not None
            and filename_temperature < temperature_min
        ):
            continue
        if (
            temperature_max is not None
            and filename_temperature > temperature_max
        ):
            continue

        with h5py.File(path, "r") as handle:
            missing = {
                name
                for name in ("wave", "P", "T", "cross_sec")
                if name not in handle
            }
            if missing:
                raise ValueError(
                    f"{path} is missing datasets: {', '.join(sorted(missing))}"
                )

            temperature_data = np.asarray(handle["T"][:], dtype=np.float64)
            if temperature_data.size != 1:
                raise ValueError(
                    f"{path}: expected one temperature, found "
                    f"{temperature_data.size}"
                )
            temperature = float(temperature_data[0])
            if not np.isclose(
                temperature, filename_temperature, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    f"{path}: filename says T={filename_temperature:g} K, "
                    f"but dataset T contains {temperature:g} K"
                )

            log_pressure_pa = np.asarray(handle["P"][:], dtype=np.float64)
            if (
                log_pressure_pa.ndim != 1
                or log_pressure_pa.size < 2
                or np.any(~np.isfinite(log_pressure_pa))
                or np.any(np.diff(log_pressure_pa) <= 0.0)
            ):
                raise ValueError(
                    f"{path}: P must be a finite, increasing 1-D grid"
                )

            cross_shape = handle["cross_sec"].shape
            expected_tail = (log_pressure_pa.size, 1)
            if len(cross_shape) != 3 or cross_shape[1:] != expected_tail:
                raise ValueError(
                    f"{path}: cross_sec has shape {cross_shape}; expected "
                    f"(n_wave, {log_pressure_pa.size}, 1)"
                )

        tables.append(
            InputTable(path, temperature, log_pressure_pa.copy())
        )

    tables.sort(key=lambda table: table.temperature)
    if len(tables) < 2:
        raise ValueError(
            "At least two temperature files are required for an ARCiS table"
        )

    temperatures = np.array(
        [table.temperature for table in tables], dtype=np.float64
    )
    if np.any(np.diff(temperatures) <= 0.0):
        raise ValueError("Input temperatures must be unique and increasing")
    return tables


def wavelength_edges(centres: np.ndarray) -> np.ndarray:
    """Return logarithmic cell edges, matching the convention in ARCiS."""
    centres = np.asarray(centres, dtype=np.float64)
    if (
        centres.ndim != 1
        or centres.size < 2
        or np.any(~np.isfinite(centres))
        or np.any(centres <= 0.0)
        or np.any(np.diff(centres) <= 0.0)
    ):
        raise ValueError("wave must be a finite, positive, increasing 1-D grid")

    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(centres[:-1] * centres[1:])
    edges[0] = centres[0] ** 2 / edges[1]
    edges[-1] = centres[-1] ** 2 / edges[-2]
    return edges


def constant_resolution_grid(
    lower: float,
    upper: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete constant-R cells between two wavelength edges."""
    if not (0.0 < lower < upper):
        raise ValueError("The requested wavelength limits are invalid")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be positive")

    # This is the convention used by the existing ARCiS downsampling utility.
    # It gives Delta(lambda)/lambda approximately equal to 1/R.
    ratio = 1.0 + 1.0 / resolution
    log_ratio = np.log(ratio)
    number = int(np.floor(np.log(upper / lower) / log_ratio + 1.0e-12))
    if number < 1:
        raise ValueError("The wavelength interval contains no output bin")

    edges = lower * np.exp(log_ratio * np.arange(number + 1))
    while edges[-1] > upper * (1.0 + 5.0e-14):
        edges = edges[:-1]
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres, edges


def gauss_legendre_weights(number: int) -> np.ndarray:
    """Return the weights generated by ARCiS gauleg on [0, 1]."""
    if number < 1:
        raise ValueError("ng must be at least one")
    _, weights = np.polynomial.legendre.leggauss(number)
    return np.asarray(weights / 2.0, dtype=np.float64)


def cumulative_integral_at(
    sorted_values: np.ndarray,
    sorted_weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Integrate piecewise-constant sorted distributions to quantiles."""
    cumulative_weight = np.cumsum(sorted_weights, axis=1)
    cumulative_weight[:, -1] = 1.0
    cumulative_value = np.cumsum(
        sorted_values * sorted_weights, axis=1
    )

    indices = np.sum(
        cumulative_weight[:, :, None] < quantiles[None, None, :], axis=1
    )
    indices = np.minimum(indices, sorted_values.shape[1] - 1)
    previous = np.maximum(indices - 1, 0)

    previous_weight = np.take_along_axis(
        cumulative_weight, previous, axis=1
    )
    previous_value = np.take_along_axis(
        cumulative_value, previous, axis=1
    )
    previous_weight = np.where(indices > 0, previous_weight, 0.0)
    previous_value = np.where(indices > 0, previous_value, 0.0)
    current_value = np.take_along_axis(sorted_values, indices, axis=1)

    return previous_value + current_value * (
        quantiles[None, :] - previous_weight
    )


def make_k_distribution(
    cross_sections: np.ndarray,
    spectral_weights: np.ndarray,
    target_g_weights: np.ndarray,
) -> np.ndarray:
    """Sort high-resolution cross-sections and average over target g cells.

    ``cross_sections`` has shape ``(n_pressure, n_spectral_samples)``.
    """
    if cross_sections.ndim != 2:
        raise ValueError("cross_sections must be a two-dimensional array")

    weights = np.asarray(spectral_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    order = np.argsort(cross_sections, axis=1)
    sorted_values = np.take_along_axis(cross_sections, order, axis=1)
    sorted_weights = np.take_along_axis(
        np.broadcast_to(weights, cross_sections.shape), order, axis=1
    )

    target_weights = np.asarray(target_g_weights, dtype=np.float64)
    target_weights = target_weights / np.sum(target_weights)
    quantile_edges = np.concatenate(([0.0], np.cumsum(target_weights)))
    quantile_edges[-1] = 1.0

    integral = cumulative_integral_at(
        sorted_values, sorted_weights, quantile_edges
    )
    return np.diff(integral, axis=1) / target_weights[None, :]


def common_pressure_grid(
    tables: list[InputTable],
    grid_mode: str,
    minimum_bar: float | None,
    maximum_bar: float | None,
) -> np.ndarray:
    """Construct a common log10(P/Pa) grid for the rectangular FITS cube."""
    rounded = [
        np.round(table.log_pressure_pa, decimals=12) for table in tables
    ]
    if grid_mode == "union":
        target = np.unique(np.concatenate(rounded))
    elif grid_mode == "intersection":
        shared = set(rounded[0])
        for pressure in rounded[1:]:
            shared.intersection_update(pressure)
        target = np.asarray(sorted(shared), dtype=np.float64)
    else:
        raise ValueError(f"Unknown pressure-grid mode {grid_mode!r}")

    pressure_bar = np.power(10.0, target) / PA_PER_BAR
    keep = np.ones(target.size, dtype=bool)
    if minimum_bar is not None:
        keep &= pressure_bar >= minimum_bar * (1.0 - 1.0e-12)
    if maximum_bar is not None:
        keep &= pressure_bar <= maximum_bar * (1.0 + 1.0e-12)
    target = target[keep]

    if target.size < 2:
        raise ValueError("The selected common pressure grid has fewer than two points")
    return target


def pressure_interpolation_indices(
    source: np.ndarray,
    target: np.ndarray,
    extrapolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return brackets for interpolation linear in log pressure and k."""
    outside = (target < source[0]) | (target > source[-1])
    if extrapolation == "error" and np.any(outside):
        missing = np.power(10.0, target[outside]) / PA_PER_BAR
        raise ValueError(
            "Target pressure grid exceeds this temperature file's range; "
            f"missing pressures in bar: {missing}"
        )

    upper = np.searchsorted(source, target, side="right")
    lower = np.clip(upper - 1, 0, source.size - 1)
    upper = np.clip(upper, 0, source.size - 1)

    # Exact and extrapolated locations use one source plane.
    exact_or_outside = outside | np.isclose(
        target, source[lower], rtol=0.0, atol=2.0e-12
    )
    upper[exact_or_outside] = lower[exact_or_outside]
    lower[target <= source[0]] = 0
    upper[target <= source[0]] = 0
    lower[target >= source[-1]] = source.size - 1
    upper[target >= source[-1]] = source.size - 1

    fraction = np.zeros(target.size, dtype=np.float64)
    interpolate = upper != lower
    fraction[interpolate] = (
        (target[interpolate] - source[lower[interpolate]])
        / (source[upper[interpolate]] - source[lower[interpolate]])
    )
    return lower, upper, fraction, int(np.count_nonzero(outside))


def source_bin_overlaps(
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Precompute source indices and wavelength overlaps for every bin."""
    overlaps: list[tuple[np.ndarray, np.ndarray]] = []
    source_count = source_edges.size - 1

    for lower, upper in zip(target_edges[:-1], target_edges[1:]):
        first = max(
            0,
            int(np.searchsorted(source_edges, lower, side="right")) - 1,
        )
        stop = min(
            source_count,
            int(np.searchsorted(source_edges, upper, side="left")),
        )
        if stop <= first:
            raise RuntimeError(
                f"No source wavelength cell overlaps [{lower}, {upper}] m"
            )

        cell_lower = source_edges[first:stop]
        cell_upper = source_edges[first + 1 : stop + 1]
        width = np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower)
        keep = width > 0.0
        indices = np.arange(first, stop, dtype=np.int64)[keep]
        width = width[keep]
        if indices.size == 0:
            raise RuntimeError(
                f"No positive overlap in wavelength cell [{lower}, {upper}] m"
            )
        overlaps.append((indices, width))
    return overlaps


def read_log_cross_sections(dataset: h5py.Dataset) -> np.ndarray:
    """Read one temperature file as float32 without a float64 copy."""
    output = np.empty(dataset.shape[:2], dtype=np.float32)
    try:
        dataset.read_direct(output, source_sel=np.s_[:, :, 0])
    except (TypeError, ValueError):
        # Compatibility fallback for older h5py versions.
        output[...] = np.asarray(dataset[:, :, 0], dtype=np.float32)
    return output


def validate_file_wavelengths(
    handle: h5py.File,
    reference: np.ndarray,
) -> None:
    """Check size and representative wavelength values without rereading 9 MB."""
    dataset = handle["wave"]
    if dataset.shape != reference.shape:
        raise ValueError(
            f"{handle.filename}: wave shape {dataset.shape} does not match "
            f"the reference shape {reference.shape}"
        )
    indices = np.unique(
        np.linspace(0, reference.size - 1, 33, dtype=np.int64)
    )
    values = np.asarray(dataset[indices], dtype=np.float64)
    if not np.allclose(values, reference[indices], rtol=2.0e-13, atol=0.0):
        raise ValueError(
            f"{handle.filename}: wavelength grid differs from the first file"
        )


def build_header(
    temperatures: np.ndarray,
    pressures_bar: np.ndarray,
    wavelengths_cm: np.ndarray,
    number_g: int,
    resolution: float,
    pressure_grid_mode: str,
    pressure_extrapolation: str,
) -> fits.Header:
    header = fits.Header()
    header["TMIN"] = (float(temperatures[0]), "[K]")
    header["TMAX"] = (float(temperatures[-1]), "[K]")
    header["PMIN"] = (float(pressures_bar[0]), "[bar]")
    header["PMAX"] = (float(pressures_bar[-1]), "[bar]")
    header["L_MIN"] = (float(wavelengths_cm[0]), "[cm]")
    header["L_MAX"] = (float(wavelengths_cm[-1]), "[cm]")
    header["NT"] = temperatures.size
    header["NP"] = pressures_bar.size
    header["NLAM"] = wavelengths_cm.size
    header["NG"] = number_g
    header["BUNIT"] = "cm2 molecule-1"
    header["RESOL"] = (float(resolution), "Approximate lambda/dlambda")
    header["PGRID"] = pressure_grid_mode
    header["PEXTRAP"] = pressure_extrapolation
    header.add_history("Input wavelength converted from m to cm")
    header.add_history("Input pressure converted from log10(Pa) to bar")
    header.add_history("Input log10(m2/molecule) converted to cm2/molecule")
    header.add_history("Native spectra rebinned into correlated-k distributions")
    return header


def convert(
    input_directory: Path,
    output_path: Path,
    resolution: float,
    number_g: int,
    wavelength_min_micron: float | None,
    wavelength_max_micron: float | None,
    temperature_min: float | None,
    temperature_max: float | None,
    pressure_grid_mode: str,
    pressure_min_bar: float | None,
    pressure_max_bar: float | None,
    pressure_extrapolation: str,
    output_dtype: str,
    temporary_directory: Path | None,
    overwrite: bool,
) -> None:
    """Perform the conversion."""
    input_directory = input_directory.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_directory.is_dir():
        raise NotADirectoryError(input_directory)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables = discover_tables(
        input_directory, temperature_min, temperature_max
    )
    temperatures = np.array(
        [table.temperature for table in tables], dtype=np.float64
    )
    target_log_pressure = common_pressure_grid(
        tables,
        pressure_grid_mode,
        pressure_min_bar,
        pressure_max_bar,
    )
    pressures_bar = np.power(10.0, target_log_pressure) / PA_PER_BAR

    with h5py.File(tables[0].path, "r") as first:
        source_wavelength = np.asarray(first["wave"][:], dtype=np.float64)
    source_edges = wavelength_edges(source_wavelength)

    lower = source_edges[0]
    upper = source_edges[-1]
    if wavelength_min_micron is not None:
        lower = max(lower, wavelength_min_micron * M_PER_MICRON)
    if wavelength_max_micron is not None:
        upper = min(upper, wavelength_max_micron * M_PER_MICRON)
    target_wavelength, target_edges = constant_resolution_grid(
        lower, upper, resolution
    )
    bin_overlaps = source_bin_overlaps(source_edges, target_edges)
    target_g_weights = gauss_legendre_weights(number_g)

    number_pressure = target_log_pressure.size
    number_temperature = temperatures.size
    number_wavelength = target_wavelength.size
    shape = (
        number_pressure,
        number_temperature,
        number_g,
        number_wavelength,
    )
    dtype = ">f4" if output_dtype == "float32" else ">f8"
    bytes_per_value = np.dtype(dtype).itemsize
    output_size_gib = np.prod(shape, dtype=np.int64) * bytes_per_value / 2**30

    print(
        f"Found {number_temperature} temperatures: "
        f"{temperatures[0]:g}--{temperatures[-1]:g} K"
    )
    print(
        f"Pressure grid: {number_pressure} points, "
        f"{pressures_bar[0]:.6g}--{pressures_bar[-1]:.6g} bar "
        f"({pressure_grid_mode})"
    )
    print(
        f"Source wavelength grid: {source_wavelength.size} points, "
        f"{source_wavelength[0] / M_PER_MICRON:.6g}--"
        f"{source_wavelength[-1] / M_PER_MICRON:.6g} micron"
    )
    print(
        f"Output wavelength grid: {number_wavelength} bins, "
        f"{target_wavelength[0] / M_PER_MICRON:.6g}--"
        f"{target_wavelength[-1] / M_PER_MICRON:.6g} micron, "
        f"R={resolution:g}, ng={number_g}"
    )
    print(
        f"FITS data cube: shape={shape}, dtype={output_dtype}, "
        f"approximately {output_size_gib:.2f} GiB"
    )
    print("A temporary cube of the same size is used while writing the FITS file.")

    temp_path: str | None = None
    start_time = time.monotonic()
    try:
        temp_parent = (
            temporary_directory.expanduser().resolve()
            if temporary_directory is not None
            else output_path.parent
        )
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="arcis_cross_sections_",
            suffix=".bin",
            dir=temp_parent,
            delete=False,
        ) as temporary:
            temp_path = temporary.name

        result = np.memmap(temp_path, mode="w+", dtype=dtype, shape=shape)

        for temperature_index, table in enumerate(tables):
            lower_pressure, upper_pressure, pressure_fraction, missing = (
                pressure_interpolation_indices(
                    table.log_pressure_pa,
                    target_log_pressure,
                    pressure_extrapolation,
                )
            )

            with h5py.File(table.path, "r") as handle:
                validate_file_wavelengths(handle, source_wavelength)
                dataset = handle["cross_sec"]
                expected_shape = (
                    source_wavelength.size,
                    table.log_pressure_pa.size,
                    1,
                )
                if dataset.shape != expected_shape:
                    raise ValueError(
                        f"{table.path}: cross_sec shape {dataset.shape}; "
                        f"expected {expected_shape}"
                    )
                log_cross_section = read_log_cross_sections(dataset)

            temperature_cube = np.empty(
                (number_pressure, number_g, number_wavelength),
                dtype=np.float64,
            )

            for wavelength_index, (indices, widths) in enumerate(bin_overlaps):
                log_values = np.asarray(
                    log_cross_section[indices, :].T,
                    dtype=np.float64,
                )
                if np.any(~np.isfinite(log_values)):
                    raise ValueError(
                        f"{table.path}: non-finite cross-section in output "
                        f"wavelength bin {wavelength_index}"
                    )

                # log10(m2/molecule) -> cm2/molecule.
                cross_sections = np.power(10.0, log_values) * M2_TO_CM2
                source_k = make_k_distribution(
                    cross_sections, widths, target_g_weights
                )

                # ARCiS interpolates k linearly with log10(P), so do the same
                # if a target pressure lies between source pressure planes.
                mapped = (
                    source_k[lower_pressure, :]
                    * (1.0 - pressure_fraction[:, None])
                    + source_k[upper_pressure, :]
                    * pressure_fraction[:, None]
                )
                temperature_cube[:, :, wavelength_index] = mapped

            result[:, temperature_index, :, :] = temperature_cube
            result.flush()
            del temperature_cube, log_cross_section

            elapsed_minutes = (time.monotonic() - start_time) / 60.0
            extrapolation_note = (
                f", {missing} pressure planes edge-extrapolated"
                if missing
                else ""
            )
            print(
                f"[{temperature_index + 1:2d}/{number_temperature}] "
                f"T={table.temperature:g} K complete"
                f"{extrapolation_note}; elapsed {elapsed_minutes:.1f} min",
                flush=True,
            )

        result.flush()
        wavelengths_cm = target_wavelength * CM_PER_M
        primary_header = build_header(
            temperatures,
            pressures_bar,
            wavelengths_cm,
            number_g,
            resolution,
            pressure_grid_mode,
            pressure_extrapolation,
        )

        hdus = fits.HDUList(
            [
                fits.PrimaryHDU(data=result, header=primary_header),
                fits.ImageHDU(
                    data=temperatures,
                    name="TEMPERATURE",
                    header=fits.Header({"BUNIT": "K"}),
                ),
                fits.ImageHDU(
                    data=pressures_bar,
                    name="PRESSURE",
                    header=fits.Header({"BUNIT": "bar"}),
                ),
                fits.ImageHDU(
                    data=wavelengths_cm,
                    name="WAVELENGTH",
                    header=fits.Header({"BUNIT": "cm"}),
                ),
                fits.ImageHDU(
                    data=target_g_weights,
                    name="GWEIGHTS",
                ),
            ]
        )
        hdus.writeto(output_path, overwrite=overwrite, checksum=True)
        hdus.close()
        del result

        with fits.open(
            output_path,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
            checksum=True,
        ) as check:
            if check[0].data.shape != shape:
                raise RuntimeError(
                    f"Written FITS shape is {check[0].data.shape}, expected {shape}"
                )
            if not np.isclose(np.sum(check[4].data), 1.0, atol=2.0e-14):
                raise RuntimeError("Written g weights do not sum to one")

        elapsed_minutes = (time.monotonic() - start_time) / 60.0
        print(f"Wrote and verified {output_path} in {elapsed_minutes:.1f} min")
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert temperature-separated high-resolution HDF5 "
            "cross-sections into an ARCiS correlated-k FITS file"
        )
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--resolution",
        "-R",
        type=float,
        default=1000.0,
        help="target resolving power lambda/dlambda (default: 1000)",
    )
    parser.add_argument(
        "--ng",
        type=int,
        default=25,
        help="number of correlated-k g points (default: 25)",
    )
    parser.add_argument(
        "--wavelength-min-micron",
        type=float,
        default=None,
        help="optional lower wavelength edge in micron",
    )
    parser.add_argument(
        "--wavelength-max-micron",
        type=float,
        default=None,
        help="optional upper wavelength edge in micron",
    )
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    parser.add_argument(
        "--pressure-grid",
        choices=("union", "intersection"),
        default="union",
        help=(
            "common pressure grid; union retains high-T pressure coverage, "
            "intersection avoids pressure extrapolation (default: union)"
        ),
    )
    parser.add_argument("--pressure-min-bar", type=float, default=None)
    parser.add_argument("--pressure-max-bar", type=float, default=None)
    parser.add_argument(
        "--pressure-extrapolation",
        choices=("nearest", "error"),
        default="nearest",
        help=(
            "handling of pressure planes absent from a temperature file "
            "(default: nearest)"
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "FITS opacity dtype (default: float64; float32 is smaller but "
            "underflows extremely weak cross-sections)"
        ),
    )
    parser.add_argument(
        "--temporary-directory",
        type=Path,
        default=None,
        help="directory for the temporary on-disk opacity cube",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert(
        input_directory=arguments.input_directory,
        output_path=arguments.output,
        resolution=arguments.resolution,
        number_g=arguments.ng,
        wavelength_min_micron=arguments.wavelength_min_micron,
        wavelength_max_micron=arguments.wavelength_max_micron,
        temperature_min=arguments.temperature_min,
        temperature_max=arguments.temperature_max,
        pressure_grid_mode=arguments.pressure_grid,
        pressure_min_bar=arguments.pressure_min_bar,
        pressure_max_bar=arguments.pressure_max_bar,
        pressure_extrapolation=arguments.pressure_extrapolation,
        output_dtype=arguments.output_dtype,
        temporary_directory=arguments.temporary_directory,
        overwrite=arguments.overwrite,
    )
