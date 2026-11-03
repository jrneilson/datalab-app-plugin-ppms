"""Parsing, unit conversion and entropy integration for PPMS heat capacity data."""

import io
from pathlib import Path

import numpy as np
import pandas as pd

TEMPERATURE_CANDIDATES = ("Sample Temp (Kelvin)", "Sample Temp (K)", "Temperature (K)", "T")
HC_CANDIDATES = ("Samp HC (µJ/K)", "Samp HC (uJ/K)", "µJ/K", "uJ/K")
ERROR_CANDIDATES = {
    "Samp HC (µJ/K)": "Samp HC Err (µJ/K)",
    "Samp HC (uJ/K)": "Samp HC Err (uJ/K)",
}
FIELD_CANDIDATES = ("Field (Oersted)", "Field (Oe)", "Magnetic Field (Oe)")

SOURCE_UNITS = {
    # unit of the heat capacity column: (scale to J, basis)
    "µJ/K": (1e-6, "sample"),
    "J/K": (1.0, "sample"),
    "µJ/(g·K)": (1e-6, "g"),
    "J/(g·K)": (1.0, "g"),
    "µJ/(mol·K)": (1e-6, "mol"),
    "J/(mol·K)": (1.0, "mol"),
}
"""Units that the heat capacity column may be in."""

UNITS = ("J/(mol·K)", "J/(g·K)", "J/K")
"""Units that heat capacity can be converted to: per mole of formula units, per gram, per sample."""

FLAVORS = {"Cp": 0, "Cp/T": 1, "Cp/T²": 2, "Cp/T³": 3}
"""Plotted quantities and the power of T that Cp is divided by."""


def parse_hc_file(location: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse a heat capacity file into a DataFrame and a dictionary of header `INFO` entries.

    Accepts Quantum Design `.dat` files (with `[Header]`/`[Data]` sections) and plain
    comma- or tab-delimited tables with a single header row. Handles CR, LF or CRLF line
    endings and UTF-8 or Latin-1 encoding. Columns that contain no data are dropped.
    """
    raw = Path(location).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    header: dict[str, str] = {}
    data_start = next((i for i, line in enumerate(lines) if line.strip() == "[Data]"), None)
    if data_start is not None:
        for line in lines[:data_start]:
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "INFO" and len(parts) >= 3:
                header[parts[-1]] = ",".join(parts[1:-1])
        lines = lines[data_start + 1 :]

    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis="columns", how="all")
    return df, header


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns that contain numeric data."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]


def detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column that is present with numeric data."""
    available = numeric_columns(df)
    return next((c for c in candidates if c in available), None)


def guess_source_unit(column: str) -> str:
    """Guess the unit of a heat capacity column from its name, defaulting to µJ/K."""
    name = column.replace(" ", "").lower()
    micro = "µj" in name or "uj" in name
    if "mol" in name:
        return "µJ/(mol·K)" if micro else "J/(mol·K)"
    if "/g" in name or "kg" in name or "g)" in name:
        return "µJ/(g·K)" if micro else "J/(g·K)"
    return "J/K" if ("j/k" in name and not micro) else "µJ/K"


def available_units(
    source_unit: str, mass_mg: float | None, molecular_weight: float | None
) -> list[str]:
    """The output units that can be computed from the source unit and sample parameters."""
    basis = SOURCE_UNITS[source_unit][1]
    if basis == "sample":
        units = ["J/K"]
        if mass_mg:
            units.append("J/(g·K)")
            if molecular_weight:
                units.append("J/(mol·K)")
    elif molecular_weight:
        units = ["J/(mol·K)", "J/(g·K)"]
    else:
        units = ["J/(g·K)" if basis == "g" else "J/(mol·K)"]
    return [u for u in UNITS if u in units]


def convert_heat_capacity(
    hc: np.ndarray,
    hc_err: np.ndarray | None,
    source_unit: str,
    unit: str,
    mass_mg: float | None = None,
    mass_err_mg: float | None = None,
    molecular_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert heat capacity and its uncertainty from `source_unit` to `unit`.

    Molecular weight is treated as exact; the mass uncertainty applies only when converting
    a per-sample heat capacity to a per-gram or per-mole one.

    Returns:
        The converted heat capacity, its point-wise (statistical) uncertainty, and the
        relative systematic uncertainty from the sample mass, which is common to all points.

    """
    if unit not in available_units(source_unit, mass_mg, molecular_weight):
        raise ValueError(f"Cannot convert {source_unit} to {unit} with the given sample parameters")

    scale, basis = SOURCE_UNITS[source_unit]
    factor = scale
    mass_rel_err = 0.0
    if basis == "sample" and unit != "J/K":
        assert mass_mg  # guaranteed by available_units
        factor /= mass_mg * 1e-3
        mass_rel_err = (mass_err_mg or 0) / mass_mg
    if unit == "J/(mol·K)" and basis != "mol":
        assert molecular_weight  # guaranteed by available_units
        factor *= molecular_weight
    elif unit == "J/(g·K)" and basis == "mol":
        assert molecular_weight
        factor /= molecular_weight

    hc = np.asarray(hc, dtype=float)
    err = np.zeros_like(hc) if hc_err is None else np.abs(np.asarray(hc_err, dtype=float))
    return hc * factor, err * factor, mass_rel_err


def integrate_entropy(
    temperature: np.ndarray,
    hc: np.ndarray,
    hc_err: np.ndarray,
    mass_rel_err: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> dict:
    """Integrate ΔS = ∫ Cp/T dT by the trapezoidal rule over the points within [t_start, t_end].

    The statistical uncertainty propagates the point-wise uncertainties through the (linear)
    trapezoidal sum; the mass uncertainty is fully correlated across points and so scales ΔS
    directly. The two are combined in quadrature.

    Returns:
        A dictionary with `dS`, `dS_err`, `dS_err_stat`, `dS_err_mass`, the actual
        integration limits `t_min` and `t_max`, and the number of points `n_points`.

    """
    temperature = np.asarray(temperature, dtype=float)
    mask = np.isfinite(temperature) & np.isfinite(hc) & (temperature > 0)
    if t_start is not None:
        mask &= temperature >= t_start
    if t_end is not None:
        mask &= temperature <= t_end
    if mask.sum() < 2:
        raise ValueError("At least two points are needed in the integration range")

    order = np.argsort(temperature[mask])
    t = temperature[mask][order]
    y = (np.asarray(hc)[mask] / temperature[mask])[order]
    y_err = (np.nan_to_num(np.asarray(hc_err)[mask]) / temperature[mask])[order]

    # trapezoidal weights: ΔS = Σ w_i y_i
    dt = np.diff(t)
    weights = np.zeros_like(t)
    weights[:-1] += dt / 2
    weights[1:] += dt / 2

    dS = float(np.sum(weights * y))
    err_stat = float(np.sqrt(np.sum((weights * y_err) ** 2)))
    err_mass = abs(dS) * mass_rel_err
    return {
        "dS": dS,
        "dS_err": float(np.hypot(err_stat, err_mass)),
        "dS_err_stat": err_stat,
        "dS_err_mass": err_mass,
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
        "n_points": int(len(t)),
    }
