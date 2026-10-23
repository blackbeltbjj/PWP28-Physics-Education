# -*- coding: utf-8 -*-
"""
===============================================================================
PROJECT
    Pacific Warm Pool (PWP) Centroid and Area Analysis

SCRIPT
    011B_six_date_final_pwp_comparison_grayscale.py

PURPOSE
    Create one publication-quality grayscale figure comparing the canonical/full
    threshold-defined Pacific Warm Pool for exactly six daily NOAA OISST
    fields. Each panel reproduces the scientific and visual logic of Panel E
    in 011_methodological_domain_figure.py.

SCIENTIFIC DEFINITION
    Canonical PWP:
        All valid cells within the fixed Pacific Ocean mask satisfying
        SST >= SST_THRESHOLD_C. No largest-connected-component filter is used.

    Cell area:
        Exact spherical latitude-longitude cell area computed from cell bounds.

    Spherical centroid:
        Area-weighted three-dimensional unit-vector centroid converted back to
        latitude and longitude.

PROJECT ARCHITECTURE
    The source root is located from this script rather than from the current
    working directory. The script expects the existing project modules:

        src/config/config.py
        src/plotting_utils.py

    and the existing inputs used by Program 011:

        data/raw/**/*.nc
        data/processed/pacific_mask_oisst.npy
        data/processed/grid_lat.npy
        data/processed/grid_lon.npy

OUTPUT
    outputs/JPO/RBEF/figures/
        JPO_011B_six_date_canonical_pwp_grayscale_<dates>_<threshold>.png
        JPO_011B_six_date_canonical_pwp_grayscale_<dates>_<threshold>.pdf

    outputs/JPO/RBEF/tables/
        JPO_011B_six_date_canonical_pwp_grayscale_<dates>_<threshold>.csv

    outputs/JPO/RBEF/reports/
        JPO_011B_six_date_canonical_pwp_grayscale_<dates>_<threshold>.txt

NOTES
    - This is a visualization variation, not a new methodology.
    - The six panels use identical map limits, SST normalization, colormap,
      threshold definition, Pacific mask, reference grid, and geometry.
    - The visualization uses only grayscale tones for print-friendly output.
    - Existing outputs are never overwritten.

DEPENDENCIES
    numpy
    pandas
    matplotlib
    xarray

PYTHON
    Python 3.10+

AUTHOR
    Fabio Vieira Machado
===============================================================================
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# USER CONFIGURATION
# =============================================================================

TARGET_DATES = (
    "1995-09-01",  # Panel A
    "1983-04-30",  # Panel B (30 April; 31 April does not exist)
    "1989-09-01",  # Panel C
    "1998-04-01",  # Panel D
    "1999-09-01",  # Panel E
    "2016-04-01",  # Panel F
)

SST_THRESHOLD_C = 28.0  # °C; canonical PWP threshold for all six panels


# =============================================================================
# PROJECT ROOT AND IMPORTS
# =============================================================================

def locate_source_root() -> Path:
    """Locate the project's source root independently of the working directory."""

    script_directory = Path(__file__).resolve().parent

    for candidate in (script_directory, *script_directory.parents):
        if (candidate / "config" / "config.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project source root. Expected to find:\n"
        "  config/config.py\n"
        f"while searching upward from:\n  {script_directory}"
    )


SOURCE_ROOT = locate_source_root()
PROJECT_ROOT = SOURCE_ROOT.parent

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from config.config import GRID_ALPHA, GRID_LINESTYLE, GRID_LINEWIDTH  # noqa: E402
try:  # noqa: E402
    from plotting_utils import add_panel_label, save_publication_figure
except ModuleNotFoundError as error:  # noqa: E402
    if error.name != "plotting_utils":
        raise

    def add_panel_label(axis: plt.Axes, label: str) -> None:
        """Add a publication-style panel label when plotting_utils is absent."""

        axis.text(
            0.012,
            0.985,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 2.0,
            },
            zorder=100,
        )

    def save_publication_figure(
        figure: plt.Figure,
        output_file: Path,
        close: bool = True,
    ) -> None:
        """Save a publication-quality PNG when plotting_utils is absent."""

        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            transparent=False,
        )

        if close:
            plt.close(figure)


# =============================================================================
# PATHS AND FIXED SCIENTIFIC/PLOTTING CONSTANTS
# =============================================================================

MASK_FILE = PROCESSED_DIR / "pacific_mask_oisst.npy"
GRID_LAT_FILE = PROCESSED_DIR / "grid_lat.npy"
GRID_LON_FILE = PROCESSED_DIR / "grid_lon.npy"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "JPO" / "RBEF"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

EARTH_RADIUS_KM = 6371.0088

# Pacific-centred plotting convention copied from Program 011.
PLOT_LONGITUDE_MIN = 100.0
PLOT_LONGITUDE_MAX = 300.0
PLOT_LATITUDE_MIN = -45.0
PLOT_LATITUDE_MAX = 45.0

# SST plotting interval copied from Program 011.
SST_MIN_C = 18.0
SST_MAX_C = 32.0
SST_INTERVAL_C = 0.5

PANEL_LABELS = ("A)", "B)", "C)", "D)", "E)", "F)")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class PWPDiagnostics:
    """Scientific diagnostics for one canonical PWP daily field."""

    panel: str
    date: pd.Timestamp
    source_file: Path
    sst_variable: str
    centroid_longitude_deg_e: float
    centroid_latitude_deg: float
    area_km2: float
    mean_sst_c: float
    minimum_sst_c: float
    maximum_sst_c: float
    pwp_cell_count: int
    pacific_valid_cell_count: int
    pacific_mask_cell_count: int


@dataclass(frozen=True)
class DailyPWPField:
    """Aligned daily SST, canonical PWP mask, and diagnostics for one panel."""

    sst_c: np.ndarray
    pwp: np.ndarray
    diagnostics: PWPDiagnostics


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    """Print a standardized terminal header."""

    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    """Print a standardized terminal section."""

    print()
    print(title)
    print("-" * 78)


def validate_configuration() -> tuple[pd.Timestamp, ...]:
    """Validate and normalize the six user-configured dates and threshold."""

    if len(TARGET_DATES) != 6:
        raise ValueError(
            "TARGET_DATES must contain exactly six dates for Panels A through F. "
            f"Received {len(TARGET_DATES)} value(s)."
        )

    parsed_dates: list[pd.Timestamp] = []

    for panel, value in zip(PANEL_LABELS, TARGET_DATES, strict=True):
        try:
            parsed = pd.Timestamp(value).normalize()
        except Exception as error:
            raise ValueError(
                f"Invalid date for Panel {panel[0]}: {value!r}. "
                "Use a valid ISO calendar date such as '1998-03-31'."
            ) from error

        if pd.isna(parsed):
            raise ValueError(
                f"Invalid date for Panel {panel[0]}: {value!r}."
            )

        parsed_dates.append(parsed)

    if len(set(parsed_dates)) != 6:
        formatted = ", ".join(date.strftime("%Y-%m-%d") for date in parsed_dates)
        raise ValueError(
            "TARGET_DATES must contain six different dates. "
            f"Received: {formatted}"
        )

    if not np.isfinite(SST_THRESHOLD_C):
        raise ValueError("SST_THRESHOLD_C must be a finite numeric value.")

    return tuple(parsed_dates)

def threshold_token(threshold_c: float) -> str:
    """Return a filesystem-safe threshold token such as 28p5C."""

    text = f"{threshold_c:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") + "C"


def threshold_label(threshold_c: float) -> str:
    """Return a concise human-readable threshold label."""

    return f"{threshold_c:.3f}".rstrip("0").rstrip(".")


def build_output_paths(
    target_dates: tuple[pd.Timestamp, ...],
) -> tuple[Path, Path, Path, Path]:
    """Build unique output paths from all dates and the threshold."""

    dates_token = "_".join(date.strftime("%Y%m%d") for date in target_dates)
    stem = (
        "JPO_011B_six_date_canonical_pwp_grayscale_"
        f"{dates_token}_{threshold_token(SST_THRESHOLD_C)}"
    )

    return (
        FIGURE_DIR / f"{stem}.png",
        FIGURE_DIR / f"{stem}.pdf",
        TABLE_DIR / f"{stem}.csv",
        REPORT_DIR / f"{stem}.txt",
    )


def ensure_output_directories() -> None:
    """Create output directories without changing existing products."""

    for directory in (FIGURE_DIR, TABLE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def refuse_output_overwrite(output_paths: tuple[Path, ...]) -> None:
    """Abort clearly if any configured product already exists."""

    existing = [path for path in output_paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing scientific product(s):\n"
            + "\n".join(f"  {path}" for path in existing)
            + "\nMove, rename, or archive those products before running again."
        )


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 checksum."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# NETCDF DISCOVERY AND EXTRACTION
# =============================================================================

def discover_candidate_netcdf_files(target_date: pd.Timestamp) -> list[Path]:
    """
    Discover NetCDF files that could contain one target date.

    Files with 'anom' in the path or filename are excluded because the figure
    requires absolute SST, not SST anomalies. Files containing the target year
    in their filename are searched first, exactly as in Program 011.
    """

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found:\n{RAW_DIR}")

    all_files = sorted(
        path
        for path in RAW_DIR.rglob("*.nc")
        if "anom" not in str(path).lower()
    )

    year_text = str(target_date.year)
    preferred = [path for path in all_files if year_text in path.name]

    return preferred + [path for path in all_files if path not in preferred]


def identify_sst_variable(dataset: xr.Dataset) -> str:
    """Identify the absolute SST variable in one OISST dataset."""

    preferred_names = (
        "sst",
        "SST",
        "sea_surface_temperature",
        "analysed_sst",
    )

    for name in preferred_names:
        if name in dataset.data_vars:
            return name

    for name, variable in dataset.data_vars.items():
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        long_name = str(variable.attrs.get("long_name", "")).lower()
        units = str(variable.attrs.get("units", "")).lower()

        if (
            "sea_surface_temperature" in standard_name
            or "sea surface temperature" in long_name
            or name.lower() == "sst"
        ):
            if "anom" not in name.lower() and "anomal" not in long_name:
                return name

        if (
            "degree_c" in units
            or units in {"c", "degc", "degrees celsius"}
        ):
            if "anom" not in name.lower() and "anomal" not in long_name:
                return name

    raise ValueError("No absolute SST variable could be identified in the dataset.")


def identify_coordinate_name(
    dataset: xr.Dataset,
    candidates: tuple[str, ...],
) -> str:
    """Identify one coordinate name."""

    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.variables:
            return candidate

    raise ValueError(f"Coordinate not found. Candidates: {candidates}")


def convert_sst_to_celsius(values: np.ndarray, units: str) -> np.ndarray:
    """Convert SST to degrees Celsius when stored in kelvin."""

    values = np.asarray(values, dtype=np.float64)
    normalized_units = units.strip().lower()
    finite = values[np.isfinite(values)]
    median_value = float(np.median(finite)) if finite.size > 0 else np.nan

    units_indicate_kelvin = (
        normalized_units in {"k", "kelvin", "degrees kelvin"}
        or "kelvin" in normalized_units
    )
    magnitude_indicates_kelvin = (
        np.isfinite(median_value) and median_value > 100.0
    )

    if units_indicate_kelvin or magnitude_indicates_kelvin:
        return values - 273.15

    return values


def extract_target_sst(
    target_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, str]:
    """Locate and extract one target daily absolute-SST field."""

    candidate_files = discover_candidate_netcdf_files(target_date)

    if not candidate_files:
        raise FileNotFoundError(f"No NetCDF files found below:\n{RAW_DIR}")

    errors: list[str] = []

    for file_path in candidate_files:
        try:
            with xr.open_dataset(file_path, decode_times=True) as dataset:
                time_name = identify_coordinate_name(
                    dataset,
                    ("time", "Time", "TIME"),
                )
                times = pd.DatetimeIndex(
                    pd.to_datetime(dataset[time_name].values)
                ).normalize()
                matching = np.flatnonzero(times == target_date)

                if matching.size == 0:
                    continue

                latitude_name = identify_coordinate_name(
                    dataset,
                    ("lat", "latitude", "Latitude", "LAT"),
                )
                longitude_name = identify_coordinate_name(
                    dataset,
                    ("lon", "longitude", "Longitude", "LON"),
                )
                sst_name = identify_sst_variable(dataset)

                sst_data_array = dataset[sst_name].isel(
                    {time_name: int(matching[0])}
                ).squeeze(drop=True)

                # Preserve Program 011 handling of singleton non-spatial dims.
                for dimension in list(sst_data_array.dims):
                    if (
                        dimension not in {latitude_name, longitude_name}
                        and sst_data_array.sizes[dimension] == 1
                    ):
                        sst_data_array = sst_data_array.isel(
                            {dimension: 0},
                            drop=True,
                        )

                if set(sst_data_array.dims) != {
                    latitude_name,
                    longitude_name,
                }:
                    raise ValueError(
                        "SST field contains unexpected dimensions: "
                        f"{sst_data_array.dims}"
                    )

                sst_data_array = sst_data_array.transpose(
                    latitude_name,
                    longitude_name,
                )
                latitude = np.asarray(
                    dataset[latitude_name].values,
                    dtype=np.float64,
                )
                longitude = np.mod(
                    np.asarray(dataset[longitude_name].values, dtype=np.float64),
                    360.0,
                )
                sst_c = convert_sst_to_celsius(
                    sst_data_array.values,
                    units=str(sst_data_array.attrs.get("units", "")),
                )

                return sst_c, latitude, longitude, file_path, sst_name

        except Exception as error:
            errors.append(f"{file_path}: {error}")

    diagnostic = "\n".join(errors[:20])

    raise FileNotFoundError(
        "No absolute-SST NetCDF file containing "
        f"{target_date:%Y-%m-%d} was found below:\n"
        f"{RAW_DIR}\n\nDiagnostic errors:\n{diagnostic}"
    )


# =============================================================================
# GRID AND MASK ALIGNMENT
# =============================================================================

def load_reference_grid_and_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the processed OISST grid and fixed Pacific mask."""

    missing = [
        path
        for path in (MASK_FILE, GRID_LAT_FILE, GRID_LON_FILE)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required processed file(s) not found:\n"
            + "\n".join(str(path) for path in missing)
        )

    mask = np.load(MASK_FILE).astype(bool)
    latitude = np.load(GRID_LAT_FILE).astype(np.float64)
    longitude = np.mod(np.load(GRID_LON_FILE).astype(np.float64), 360.0)

    if mask.shape != (latitude.size, longitude.size):
        raise ValueError(
            "Pacific mask and processed grid are inconsistent:\n"
            f"Mask shape : {mask.shape}\n"
            f"Grid shape : {(latitude.size, longitude.size)}"
        )

    return mask, latitude, longitude


def coordinate_reindex(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Return indices that reorder source coordinates to target coordinates."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.size != target.size:
        raise ValueError("Coordinate lengths differ.")

    order = np.argsort(source)
    sorted_source = source[order]
    insertion = np.searchsorted(sorted_source, target)
    insertion = np.clip(insertion, 0, sorted_source.size - 1)
    candidate = order[insertion]
    difference = np.abs(source[candidate] - target)

    if np.any(difference > tolerance):
        raise ValueError(
            "NetCDF coordinates cannot be aligned with the processed grid. "
            f"Maximum difference: {difference.max():.8g}"
        )

    return candidate


def align_sst_to_reference_grid(
    sst_c: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Align one SST field to the processed mask grid."""

    if sst_c.shape != (source_latitude.size, source_longitude.size):
        raise ValueError("SST field shape is inconsistent with its coordinates.")

    latitude_indices = coordinate_reindex(
        source=source_latitude,
        target=target_latitude,
    )
    longitude_indices = coordinate_reindex(
        source=np.mod(source_longitude, 360.0),
        target=np.mod(target_longitude, 360.0),
    )

    return sst_c[latitude_indices, :][:, longitude_indices]


# =============================================================================
# SPHERICAL GEOMETRY — UNCHANGED FROM PROGRAM 011
# =============================================================================

def coordinate_edges(coordinate: np.ndarray, cyclic: bool = False) -> np.ndarray:
    """Calculate cell edges from one-dimensional cell centres."""

    coordinate = np.asarray(coordinate, dtype=np.float64)

    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError(
            "Coordinate must be one-dimensional with at least two values."
        )

    differences = np.diff(coordinate)

    if not np.allclose(
        differences,
        differences[0],
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("This script expects a regular grid.")

    spacing = float(differences[0])
    edges = np.empty(coordinate.size + 1, dtype=np.float64)
    edges[1:-1] = (coordinate[:-1] + coordinate[1:]) / 2.0
    edges[0] = coordinate[0] - spacing / 2.0
    edges[-1] = coordinate[-1] + spacing / 2.0

    if cyclic:
        return edges

    return np.clip(edges, -90.0, 90.0)


def calculate_cell_areas_km2(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Calculate exact spherical grid-cell areas."""

    latitude_edges = coordinate_edges(latitude, cyclic=False)
    longitude_edges = coordinate_edges(longitude, cyclic=True)
    latitude_edges_rad = np.deg2rad(latitude_edges)
    longitude_edges_rad = np.deg2rad(longitude_edges)
    latitude_factor = np.abs(
        np.sin(latitude_edges_rad[1:]) - np.sin(latitude_edges_rad[:-1])
    )
    longitude_width = np.abs(np.diff(longitude_edges_rad))

    return (
        EARTH_RADIUS_KM ** 2
        * latitude_factor[:, None]
        * longitude_width[None, :]
    )


def calculate_spherical_centroid(
    selected: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cell_area_km2: np.ndarray,
) -> tuple[float, float]:
    """Calculate area-weighted spherical centroid."""

    if not np.any(selected):
        return np.nan, np.nan

    latitude_2d, longitude_2d = np.meshgrid(
        latitude,
        longitude,
        indexing="ij",
    )
    latitude_rad = np.deg2rad(latitude_2d)
    longitude_rad = np.deg2rad(longitude_2d)
    weights = np.where(selected, cell_area_km2, 0.0)

    x = np.sum(weights * np.cos(latitude_rad) * np.cos(longitude_rad))
    y = np.sum(weights * np.cos(latitude_rad) * np.sin(longitude_rad))
    z = np.sum(weights * np.sin(latitude_rad))

    longitude_centroid = np.mod(
        np.rad2deg(np.arctan2(y, x)),
        360.0,
    )
    horizontal = np.hypot(x, y)
    latitude_centroid = np.rad2deg(np.arctan2(z, horizontal))

    return float(longitude_centroid), float(latitude_centroid)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def calculate_diagnostics(
    panel: str,
    target_date: pd.Timestamp,
    sst_c: np.ndarray,
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cell_area_km2: np.ndarray,
    source_file: Path,
    sst_variable: str,
) -> tuple[PWPDiagnostics, np.ndarray]:
    """Calculate canonical PWP mask, centroid, area, and diagnostics."""

    finite = np.isfinite(sst_c)
    pacific_valid = mask & finite
    pwp = pacific_valid & (sst_c >= SST_THRESHOLD_C)

    if not np.any(pwp):
        raise ValueError(
            f"No canonical PWP cells satisfy SST >= {SST_THRESHOLD_C:g} °C "
            f"on {target_date:%Y-%m-%d}."
        )

    centroid_longitude, centroid_latitude = calculate_spherical_centroid(
        selected=pwp,
        latitude=latitude,
        longitude=longitude,
        cell_area_km2=cell_area_km2,
    )
    selected_sst = sst_c[pwp]

    diagnostics = PWPDiagnostics(
        panel=panel,
        date=target_date,
        source_file=source_file,
        sst_variable=sst_variable,
        centroid_longitude_deg_e=centroid_longitude,
        centroid_latitude_deg=centroid_latitude,
        area_km2=float(np.sum(cell_area_km2[pwp])),
        mean_sst_c=float(np.mean(selected_sst)),
        minimum_sst_c=float(np.min(selected_sst)),
        maximum_sst_c=float(np.max(selected_sst)),
        pwp_cell_count=int(np.count_nonzero(pwp)),
        pacific_valid_cell_count=int(np.count_nonzero(pacific_valid)),
        pacific_mask_cell_count=int(np.count_nonzero(mask)),
    )

    return diagnostics, pwp


# =============================================================================
# FIGURE PREPARATION
# =============================================================================

def select_plot_domain(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the Pacific-centred plotting domain."""

    latitude_selection = (
        (latitude >= PLOT_LATITUDE_MIN) & (latitude <= PLOT_LATITUDE_MAX)
    )
    longitude_selection = (
        (longitude >= PLOT_LONGITUDE_MIN)
        & (longitude <= PLOT_LONGITUDE_MAX)
    )

    return (
        field[latitude_selection, :][:, longitude_selection],
        latitude[latitude_selection],
        longitude[longitude_selection],
    )


def configure_geographic_axis(axis: plt.Axes) -> None:
    """Configure longitude-latitude axes exactly as in Program 011."""

    axis.set_xlim(PLOT_LONGITUDE_MIN, PLOT_LONGITUDE_MAX)
    axis.set_ylim(PLOT_LATITUDE_MIN, PLOT_LATITUDE_MAX)
    axis.set_xticks(np.arange(100.0, 301.0, 20.0))
    axis.set_yticks(np.arange(-40.0, 41.0, 20.0))
    axis.set_xticklabels(
        [
            f"{int(value)}°E"
            if value <= 180.0
            else f"{int(360.0 - value)}°W"
            for value in axis.get_xticks()
        ]
    )
    axis.set_yticklabels(
        [
            "0°"
            if value == 0.0
            else (
                f"{abs(int(value))}°N"
                if value > 0.0
                else f"{abs(int(value))}°S"
            )
            for value in axis.get_yticks()
        ]
    )
    axis.grid(
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )
    axis.set_aspect("auto")


def create_comparison_figure(
    fields: tuple[DailyPWPField, ...],
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Create a 3-by-2 comparison of six canonical-PWP daily maps."""

    if len(fields) != 6:
        raise ValueError(
            f"Expected six fields for Panels A through F; received {len(fields)}."
        )

    mask_plot, plot_latitude, plot_longitude = select_plot_domain(
        field=mask.astype(np.float64),
        latitude=latitude,
        longitude=longitude,
    )
    longitude_2d, latitude_2d = np.meshgrid(
        plot_longitude,
        plot_latitude,
    )
    sst_levels = np.arange(
        SST_MIN_C,
        SST_MAX_C + SST_INTERVAL_C,
        SST_INTERVAL_C,
    )
    # Print-friendly sequential grayscale: warmer SST values are darker.
    sst_colormap = plt.get_cmap("Greys")
    sst_norm = BoundaryNorm(
        sst_levels,
        ncolors=sst_colormap.N,
        clip=True,
    )

    figure, axes_2d = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16.0, 13.2),
        constrained_layout=False,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.91,
        bottom=0.065,
        top=0.925,
        wspace=0.13,
        hspace=0.24,
    )
    axes = axes_2d.ravel()

    first_mesh = None

    for axis, field in zip(axes, fields, strict=True):
        sst_plot, _, _ = select_plot_domain(
            field=field.sst_c,
            latitude=latitude,
            longitude=longitude,
        )
        pwp_plot, _, _ = select_plot_domain(
            field=field.pwp.astype(np.float64),
            latitude=latitude,
            longitude=longitude,
        )
        masked_sst_plot = np.where(mask_plot > 0.5, sst_plot, np.nan)

        mesh = axis.pcolormesh(
            longitude_2d,
            latitude_2d,
            masked_sst_plot,
            cmap=sst_colormap,
            norm=sst_norm,
            shading="auto",
            rasterized=True,
        )

        if first_mesh is None:
            first_mesh = mesh

        axis.contourf(
            longitude_2d,
            latitude_2d,
            pwp_plot,
            levels=[0.5, 1.5],
            colors=["0.30"],
            alpha=0.20,
        )
        axis.contour(
            longitude_2d,
            latitude_2d,
            pwp_plot,
            levels=[0.5],
            colors=["black"],
            linewidths=1.3,
        )

        diagnostics = field.diagnostics

        axis.scatter(
            diagnostics.centroid_longitude_deg_e,
            diagnostics.centroid_latitude_deg,
            s=100,
            marker="X",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=15,
            label="Area-weighted spherical centroid",
        )
        axis.annotate(
            (
                f"Centroid: {diagnostics.centroid_longitude_deg_e:.2f}°E, "
                f"{diagnostics.centroid_latitude_deg:.2f}°\n"
                f"Area: {diagnostics.area_km2 / 1.0e6:.2f} × 10⁶ km²"
            ),
            xy=(
                diagnostics.centroid_longitude_deg_e,
                diagnostics.centroid_latitude_deg,
            ),
            xytext=(16, 16),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.25",
                "alpha": 0.88,
                "pad": 3.5,
            },
            arrowprops={"arrowstyle": "-", "linewidth": 0.8},
            zorder=16,
        )
        axis.set_title(
            "Canonical PWP domain — "
            f"{diagnostics.date:%d %B %Y}"
        )
        axis.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="X",
                    linestyle="None",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=9,
                    label="Spherical centroid",
                ),
                Patch(
                    facecolor="0.30",
                    alpha=0.20,
                    edgecolor="black",
                    label=f"SST ≥ {threshold_label(SST_THRESHOLD_C)} °C",
                ),
            ],
            frameon=False,
            loc="lower left",
            fontsize=8.5,
        )
        add_panel_label(axis, diagnostics.panel)
        configure_geographic_axis(axis)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")

    if first_mesh is None:
        raise RuntimeError("No SST mesh was created for the shared colorbar.")

    colorbar_axis = figure.add_axes([0.93, 0.14, 0.016, 0.70])
    colorbar = figure.colorbar(
        first_mesh,
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label("Sea surface temperature (°C)")

    figure.suptitle(
        "Six-date grayscale comparison of the canonical Pacific Warm Pool — "
        f"SST ≥ {threshold_label(SST_THRESHOLD_C)} °C",
        y=0.975,
    )

    save_publication_figure(
        figure=figure,
        output_file=output_png,
        close=False,
    )
    figure.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=False,
    )
    plt.close(figure)


# =============================================================================
# OUTPUT TABLE AND REPORT
# =============================================================================

def export_summary(
    diagnostics: tuple[PWPDiagnostics, ...],
    output_csv: Path,
) -> None:
    """Export the six-panel scientific summary."""

    rows = []

    for item in diagnostics:
        rows.append(
            {
                "panel": item.panel[0],
                "date": item.date.strftime("%Y-%m-%d"),
                "threshold_c": SST_THRESHOLD_C,
                "centroid_longitude_deg_e": item.centroid_longitude_deg_e,
                "centroid_latitude_deg": item.centroid_latitude_deg,
                "area_km2": item.area_km2,
                "area_million_km2": item.area_km2 / 1.0e6,
                "pwp_cell_count": item.pwp_cell_count,
                "mean_pwp_sst_c": item.mean_sst_c,
                "minimum_pwp_sst_c": item.minimum_sst_c,
                "maximum_pwp_sst_c": item.maximum_sst_c,
                "pacific_valid_cell_count": item.pacific_valid_cell_count,
                "pacific_mask_cell_count": item.pacific_mask_cell_count,
                "sst_variable": item.sst_variable,
                "source_file": str(item.source_file),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
        float_format="%.10f",
        lineterminator="\n",
    )


def write_report(
    diagnostics: tuple[PWPDiagnostics, ...],
    output_paths: tuple[Path, Path, Path, Path],
    source_checksums: dict[Path, str],
) -> None:
    """Write a reproducible comparison-figure report."""

    output_png, output_pdf, output_csv, output_report = output_paths
    generated_utc = datetime.now(timezone.utc).isoformat()

    lines = [
        "SIX-DATE CANONICAL PACIFIC WARM POOL COMPARISON REPORT",
        "=" * 78,
        "",
        "1. PROCESSING INFORMATION",
        "-" * 78,
        f"Generated UTC              : {generated_utc}",
        f"Script                     : {Path(__file__).name}",
        f"SST threshold              : {SST_THRESHOLD_C:.3f} °C",
        "PWP definition             : Full canonical threshold-defined domain",
        "Visual style               : Grayscale publication version",
        "Connectivity filter        : None (not LCC)",
        "",
        "2. PANEL DIAGNOSTICS",
        "-" * 78,
    ]

    for item in diagnostics:
        lines.extend(
            [
                f"Panel {item.panel[0]}",
                f"Date                       : {item.date:%Y-%m-%d}",
                f"Source NetCDF              : {item.source_file}",
                f"Source SHA-256             : {source_checksums[item.source_file]}",
                f"SST variable               : {item.sst_variable}",
                (
                    "Centroid longitude         : "
                    f"{item.centroid_longitude_deg_e:.6f} °E"
                ),
                (
                    "Centroid latitude          : "
                    f"{item.centroid_latitude_deg:.6f} °"
                ),
                f"Total PWP area             : {item.area_km2:,.3f} km²",
                (
                    "Total PWP area             : "
                    f"{item.area_km2 / 1.0e6:.6f} × 10⁶ km²"
                ),
                f"Mean PWP SST               : {item.mean_sst_c:.6f} °C",
                f"Minimum PWP SST            : {item.minimum_sst_c:.6f} °C",
                f"Maximum PWP SST            : {item.maximum_sst_c:.6f} °C",
                f"PWP cells                  : {item.pwp_cell_count:,}",
                (
                    "Valid Pacific cells        : "
                    f"{item.pacific_valid_cell_count:,}"
                ),
                (
                    "Pacific-mask cells         : "
                    f"{item.pacific_mask_cell_count:,}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "3. OUTPUT FILES",
            "-" * 78,
            str(output_png),
            str(output_pdf),
            str(output_csv),
            str(output_report),
            "",
            "=" * 78,
            "END OF REPORT",
            "=" * 78,
        ]
    )

    output_report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Execute the complete six-date canonical-PWP comparison workflow."""

    print_header("SIX-DATE CANONICAL PACIFIC WARM POOL COMPARISON")

    target_dates = validate_configuration()
    output_paths = build_output_paths(target_dates)

    ensure_output_directories()
    refuse_output_overwrite(output_paths)

    print(f"Project root               : {PROJECT_ROOT}")
    print(f"Source root                : {SOURCE_ROOT}")
    print(f"SST threshold              : {SST_THRESHOLD_C:.3f} °C")
    print(
        "Target dates               : "
        + ", ".join(date.strftime("%Y-%m-%d") for date in target_dates)
    )

    pacific_mask, reference_latitude, reference_longitude = (
        load_reference_grid_and_mask()
    )
    cell_area_km2 = calculate_cell_areas_km2(
        latitude=reference_latitude,
        longitude=reference_longitude,
    )

    fields: list[DailyPWPField] = []

    for panel, target_date in zip(PANEL_LABELS, target_dates, strict=True):
        print_section(
            f"PANEL {panel[0]} — LOADING {target_date:%Y-%m-%d}"
        )

        (
            source_sst,
            source_latitude,
            source_longitude,
            source_file,
            sst_variable,
        ) = extract_target_sst(target_date)

        sst_c = align_sst_to_reference_grid(
            sst_c=source_sst,
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            target_latitude=reference_latitude,
            target_longitude=reference_longitude,
        )
        diagnostics, pwp = calculate_diagnostics(
            panel=panel,
            target_date=target_date,
            sst_c=sst_c,
            mask=pacific_mask,
            latitude=reference_latitude,
            longitude=reference_longitude,
            cell_area_km2=cell_area_km2,
            source_file=source_file,
            sst_variable=sst_variable,
        )
        fields.append(
            DailyPWPField(
                sst_c=sst_c,
                pwp=pwp,
                diagnostics=diagnostics,
            )
        )

        print(f"Source file                : {source_file}")
        print(f"SST variable               : {sst_variable}")
        print(f"Source grid shape          : {source_sst.shape}")
        print(f"PWP cells                  : {diagnostics.pwp_cell_count:,}")
        print(
            "Centroid longitude         : "
            f"{diagnostics.centroid_longitude_deg_e:.4f} °E"
        )
        print(
            "Centroid latitude          : "
            f"{diagnostics.centroid_latitude_deg:.4f} °"
        )
        print(
            "Total area                 : "
            f"{diagnostics.area_km2 / 1.0e6:.4f} × 10⁶ km²"
        )
        print(f"Mean PWP SST               : {diagnostics.mean_sst_c:.4f} °C")

    if len(fields) != 6:
        raise RuntimeError(
            f"Internal error: expected six processed fields, received {len(fields)}."
        )

    fields_tuple = tuple(fields)
    diagnostics_typed = tuple(
        field.diagnostics for field in fields_tuple
    )

    output_png, output_pdf, output_csv, _ = output_paths

    print_section("CREATING SIX-PANEL COMPARISON FIGURE")
    create_comparison_figure(
        fields=fields_tuple,
        mask=pacific_mask,
        latitude=reference_latitude,
        longitude=reference_longitude,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    export_summary(
        diagnostics=diagnostics_typed,
        output_csv=output_csv,
    )

    source_checksums = {
        item.source_file: calculate_sha256(item.source_file)
        for item in diagnostics_typed
    }
    write_report(
        diagnostics=diagnostics_typed,
        output_paths=output_paths,
        source_checksums=source_checksums,
    )

    print_section("FILES CREATED")
    for output_path in output_paths:
        print(output_path)

    print()
    print("=" * 78)
    print("Finished successfully.")
    print("=" * 78)


if __name__ == "__main__":
    main()
