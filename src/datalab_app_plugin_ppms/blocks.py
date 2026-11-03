"""A datalab block for Quantum Design PPMS heat capacity data."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydatalab.blocks.base import DataBlock, event

from datalab_app_plugin_ppms._version import __version__
from datalab_app_plugin_ppms.utils import (
    ERROR_CANDIDATES,
    FIELD_CANDIDATES,
    FLAVORS,
    HC_CANDIDATES,
    SOURCE_UNITS,
    TEMPERATURE_CANDIDATES,
    UNITS,
    available_units,
    convert_heat_capacity,
    detect_column,
    guess_source_unit,
    integrate_entropy,
    numeric_columns,
    parse_hc_file,
)

TEMPERATURE = "T (K)"
FIELD = "Field (Oe)"
NONE = "None"
WIDGET_WIDTH = 240

NUMERIC_SETTINGS = (
    "mass_mg",
    "mass_err_mg",
    "molecular_weight",
    "filter_field_min",
    "filter_field_max",
    "entropy_t_start",
    "entropy_t_end",
)
POSITIVE_SETTINGS = ("mass_mg", "molecular_weight")
CHOICE_SETTINGS = {
    "flavor": tuple(FLAVORS),
    "unit": UNITS,
    "source_unit": tuple(SOURCE_UNITS),
    # column choices are validated against the file at plot time
    "temperature_column": None,
    "hc_column": None,
    "error_column": None,
    "field_column": None,
}

# The JS callbacks and widget style below are shared with the MPMS plugin.
EXPORT_CSV_CALLBACK = """
const columns = Object.keys(source.data);
const rows = [columns.map((c) => '"' + c + '"').join(",")];
for (let i = 0; i < source.data[columns[0]].length; i++) {
    rows.push(columns.map((c) => {
        const v = source.data[c][i];
        return (v === null || Number.isNaN(v)) ? "" : v;
    }).join(","));
}
const blob = new Blob([rows.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"});
const link = document.createElement("a");
link.href = URL.createObjectURL(blob);
link.download = filename;
document.body.appendChild(link);
link.click();
document.body.removeChild(link);
"""

AXIS_MENU_CALLBACK = """
// Open a menu of plot choices when the x or y axis (outside the plot frame) is clicked.
const [x0, x1] = [Math.min(xr.start, xr.end), Math.max(xr.start, xr.end)];
const [y0, y1] = [Math.min(yr.start, yr.end), Math.max(yr.start, yr.end)];
const in_x = cb_obj.x >= x0 && cb_obj.x <= x1;
const in_y = cb_obj.y >= y0 && cb_obj.y <= y1;
let options = null;
if (in_x && cb_obj.y < y0) options = menus.x;
else if (in_y && cb_obj.x < x0) options = menus.y;
if (!options || options.length === 0) return;

document.getElementById("ppms-axis-menu")?.remove();
// tap coordinates are relative to the plot canvas
const canvas = document.querySelector(".ppms-" + block_id + " .bk-canvas-events");
const rect = canvas ? canvas.getBoundingClientRect() : {left: 0, top: 0};
const menu = document.createElement("div");
menu.id = "ppms-axis-menu";
Object.assign(menu.style, {
    position: "absolute", zIndex: 10000, background: "#fff", border: "1px solid #ccc",
    borderRadius: "4px", boxShadow: "0 2px 8px rgba(0,0,0,0.15)", padding: "4px 0",
    fontSize: "14px", whiteSpace: "nowrap",
    left: (rect.left + window.scrollX + cb_obj.sx) + "px",
    top: (rect.top + window.scrollY + cb_obj.sy) + "px",
});
for (const option of options) {
    const item = document.createElement("div");
    item.textContent = (option.current ? "✓ " : "\\u2003") + option.label;
    Object.assign(item.style, {padding: "4px 12px", cursor: "pointer"});
    item.onmouseenter = () => { item.style.background = "#e9ecef"; };
    item.onmouseleave = () => { item.style.background = ""; };
    item.onclick = (e) => {
        e.stopPropagation();
        menu.remove();
        const detail = Object.assign({block_id: block_id, event_name: "update_settings"}, option.settings);
        document.dispatchEvent(new CustomEvent("block-event", {detail: detail, bubbles: true}));
    };
    menu.appendChild(item);
}
document.body.appendChild(menu);
setTimeout(() => document.addEventListener("pointerdown", function close(e) {
    if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener("pointerdown", close);
    }
}), 0);
"""

# datalab restyles Bokeh widgets with Bootstrap classes after rendering, which changes their
# size after Bokeh has laid them out; restore Bokeh's geometry within this block.
WIDGET_STYLE = """<style>
.ppms-block .input-group.form-inline.col-sm-6 {
    flex: none; max-width: none; width: 100%; height: 100%; padding: 0;
    display: inline-flex; flex-direction: column; flex-wrap: nowrap;
    align-items: stretch; white-space: nowrap;
}
.ppms-block .input-group > label { justify-content: flex-start; }
.ppms-block .input-group > select.form-control {
    margin-left: 0 !important; width: 100%; height: auto; min-height: 31px;
    padding: 0 12px; font-size: inherit;
}
</style>"""

SUPERSCRIPTS = {1: "", 2: "²", 3: "³", 4: "⁴"}


def _flavor_unit(flavor: str, unit: str) -> str:
    """The unit of Cp/Tⁿ given the unit of Cp, e.g. J/(mol·K) → J/(mol·K²) for Cp/T."""
    return unit.replace("K", "K" + SUPERSCRIPTS[1 + FLAVORS[flavor]])


def _y_columns(flavor: str, unit: str) -> tuple[str, str]:
    """The output column names for a plotted quantity and its uncertainty."""
    label = _flavor_unit(flavor, unit)
    return f"{flavor} ({label})", f"σ{flavor} ({label})"


def _js_update_setting(key: str, block_id: str) -> str:
    """A Bokeh JS callback that sends a widget's value to the block's `update_settings` event."""
    detail = {"block_id": block_id, "event_name": "update_settings"}
    return (
        f"const detail = {json.dumps(detail)};\n"
        f"detail[{json.dumps(key)}] = (cb_obj.value ?? cb_obj.text);\n"
        "document.dispatchEvent(new CustomEvent('block-event', {detail: detail, bubbles: true}));"
    )


def _parse_float(key: str, value) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {key}: {value!r}. Must be a number.")
    if key in POSITIVE_SETTINGS and value <= 0:
        raise ValueError(f"{key} must be positive")
    if key == "mass_err_mg" and value < 0:
        raise ValueError(f"{key} must not be negative")
    return value


class PPMSHeatCapacityBlock(DataBlock):
    version = __version__
    blocktype: str = "ppms-hc"
    name: str = "PPMS heat capacity"
    description: str = (
        "Plot Quantum Design PPMS heat capacity as Cp, Cp/T, Cp/T² or Cp/T³ in J/(mol·K), "
        "J/(g·K) or J/K with propagated uncertainties, filter by field, integrate the "
        "entropy over a chosen temperature range, and export the converted data."
    )
    accepted_file_extensions = (".dat", ".txt", ".csv")

    defaults = {"flavor": "Cp", "unit": "J/(mol·K)"}

    @property
    def plot_functions(self):
        return (self.generate_plot,)

    @event()
    def update_settings(self, **kwargs):
        """Update one or more block settings from the plot widgets."""
        for key, value in kwargs.items():
            if key in NUMERIC_SETTINGS:
                self.data[key] = _parse_float(key, value)
            elif key in CHOICE_SETTINGS:
                allowed = CHOICE_SETTINGS[key]
                if allowed is not None and value not in allowed:
                    raise ValueError(f"Invalid value for {key}: {value!r}")
                if key == "hc_column" and value != self.data.get("hc_column"):
                    # re-guess the unit of the newly chosen column
                    self.data.pop("source_unit", None)
                self.data[key] = value
            else:
                raise ValueError(f"Unknown setting {key!r}")

    def _prefill_from_header(self, header: dict[str, str]):
        """Fill sample mass and molecular weight from the file header the first time it is read."""
        for key, header_key in (
            ("mass_mg", "SAMPLE_MASS"),
            ("molecular_weight", "SAMPLE_MOLECULAR_WEIGHT"),
        ):
            if key not in self.data:
                try:
                    self.data[key] = _parse_float(key, header.get(header_key))
                except ValueError:
                    self.data[key] = None

    def _resolve_columns(self, df: pd.DataFrame):
        """Choose the temperature, heat capacity, error and field columns, keeping valid choices."""
        available = numeric_columns(df)
        if not available:
            raise ValueError("No numeric columns found in the file")
        if self.data.get("temperature_column") not in available:
            self.data["temperature_column"] = (
                detect_column(df, TEMPERATURE_CANDIDATES) or available[0]
            )
        if self.data.get("hc_column") not in available:
            others = [c for c in available if c != self.data["temperature_column"]]
            self.data["hc_column"] = detect_column(df, HC_CANDIDATES) or (others or available)[0]
            self.data.pop("source_unit", None)
        if self.data.get("source_unit") not in SOURCE_UNITS:
            self.data["source_unit"] = guess_source_unit(self.data["hc_column"])
        if self.data.get("error_column") not in [*available, NONE]:
            error_column = ERROR_CANDIDATES.get(self.data["hc_column"])
            self.data["error_column"] = error_column if error_column in available else NONE
        if self.data.get("field_column") not in [*available, NONE]:
            self.data["field_column"] = detect_column(df, FIELD_CANDIDATES) or NONE

    def compute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, dict]]:
        """Convert and filter the raw data according to the block settings, and integrate
        the entropy in each available unit.

        Returns:
            A DataFrame with temperature, field (if any), the raw heat capacity and every
            flavor in every unit that can be computed from the given sample parameters;
            the unit actually used for plotting; and the entropy results keyed by unit
            (or an `error` message if the integration was not possible).

        """
        self._resolve_columns(df)

        field_column = self.data["field_column"]
        mask = np.ones(len(df), dtype=bool)
        if field_column != NONE:
            if (lower := self.data.get("filter_field_min")) is not None:
                mask &= df[field_column].to_numpy() >= lower
            if (upper := self.data.get("filter_field_max")) is not None:
                mask &= df[field_column].to_numpy() <= upper
        df = df[mask].reset_index(drop=True)

        out = pd.DataFrame()
        out[TEMPERATURE] = df[self.data["temperature_column"]]
        if field_column != NONE:
            out[FIELD] = df[field_column]
        hc = df[self.data["hc_column"]].to_numpy(dtype=float)
        error_column = self.data["error_column"]
        hc_err = None if error_column == NONE else df[error_column].to_numpy(dtype=float)
        source_unit = self.data["source_unit"]
        out[f"HC ({source_unit})"] = hc
        if hc_err is not None:
            out[f"σHC ({source_unit})"] = hc_err

        sample = {
            "mass_mg": self.data.get("mass_mg"),
            "mass_err_mg": self.data.get("mass_err_mg"),
            "molecular_weight": self.data.get("molecular_weight"),
        }
        units = available_units(source_unit, sample["mass_mg"], sample["molecular_weight"])
        temperature = out[TEMPERATURE].to_numpy(dtype=float)
        entropy: dict[str, dict] = {}
        for unit in units:
            cp, cp_err, mass_rel_err = convert_heat_capacity(
                hc, hc_err, source_unit, unit, **sample
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                for flavor, power in FLAVORS.items():
                    y_label, y_err_label = _y_columns(flavor, unit)
                    out[y_label] = cp / temperature**power
                    # the mass uncertainty is common to all points and included here too
                    out[y_err_label] = np.hypot(cp_err, cp * mass_rel_err) / temperature**power
            try:
                entropy[unit] = integrate_entropy(
                    temperature,
                    cp,
                    cp_err,
                    mass_rel_err,
                    self.data.get("entropy_t_start"),
                    self.data.get("entropy_t_end"),
                )
            except ValueError as exc:
                entropy[unit] = {"error": str(exc)}

        unit = self.data.get("unit", "J/(mol·K)")
        if unit not in units:
            missing = {"J/(mol·K)": "mass and molecular weight", "J/(g·K)": "mass"}
            fallback = units[0]
            warnings.warn(
                f"Enter the sample {missing.get(unit, 'parameters')} to plot in {unit}; "
                f"showing {fallback} instead."
            )
            unit = fallback

        self.data["computed"] = {
            "entropy": {u: r for u, r in entropy.items() if "error" not in r},
        }
        return out, unit, entropy

    def generate_plot(self, location: str | Path | None = None):
        """Read the attached file (or `location`, if given) and store the Bokeh plot in the block."""
        import bokeh.embed
        from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

        if location is None:
            if "file_id" not in self.data:
                return
            from pydatalab.file_utils import get_file_info_by_id

            location = get_file_info_by_id(self.data["file_id"], update_if_live=True)["location"]

        df, header = parse_hc_file(location)
        self._prefill_from_header(header)
        out, unit, entropy = self.compute(df)

        layout = self._make_layout(
            df, out, unit, entropy, filename=f"{Path(location).stem}_converted.csv"
        )
        self.data["bokeh_plot_data"] = bokeh.embed.json_item(layout, theme=DATALAB_BOKEH_THEME)

    def _make_layout(
        self,
        df: pd.DataFrame,
        out: pd.DataFrame,
        unit: str,
        entropy: dict[str, dict],
        filename: str,
    ):
        from bokeh.layouts import column, row
        from bokeh.models import (
            BoxAnnotation,
            Button,
            ColumnDataSource,
            CustomJS,
            Div,
            HoverTool,
            Select,
            TextInput,
            Whisker,
        )
        from bokeh.plotting import figure

        def text_input(key: str, title: str) -> TextInput:
            value = self.data.get(key)
            widget = TextInput(
                title=title, value="" if value is None else f"{value:g}", width=WIDGET_WIDTH
            )
            widget.js_on_change("value", CustomJS(code=_js_update_setting(key, self.block_id)))
            return widget

        def select(key: str, title: str, options) -> Select:
            widget = Select(
                title=title, value=self.data[key], options=list(options), width=WIDGET_WIDTH
            )
            widget.js_on_change("value", CustomJS(code=_js_update_setting(key, self.block_id)))
            return widget

        flavor = self.data["flavor"]
        y_label, y_err_label = _y_columns(flavor, unit)
        has_field = FIELD in out

        y = out[y_label].to_numpy()
        y_err = out[y_err_label].to_numpy()
        data = {
            "x": out[TEMPERATURE].to_numpy(),
            "y": y,
            "upper": y + y_err,
            "lower": y - y_err,
            "err": y_err,
        }
        tooltips = [(TEMPERATURE, "@x{0.000}")]
        if has_field:
            data["field"] = out[FIELD].to_numpy()
            tooltips.append((FIELD, "@field{0.00}"))
        tooltips.append((y_label, "@y{0.0000e} ± @err{0.0e}"))
        source = ColumnDataSource(data)

        p = figure(
            x_axis_label=TEMPERATURE,
            y_axis_label=f"{y_label} ▾",
            sizing_mode="scale_width",
            aspect_ratio=1.5,
        )
        p.circle("x", "y", source=source, size=4)
        p.add_layout(Whisker(source=source, base="x", upper="upper", lower="lower", line_alpha=0.5))
        p.add_tools(HoverTool(tooltips=tooltips))

        result = entropy.get(unit, {"error": "No data"})
        if "error" in result:
            entropy_text = f"<b>Integrated entropy:</b> {result['error']}"
        else:
            p.add_layout(
                BoxAnnotation(
                    left=result["t_min"], right=result["t_max"], fill_alpha=0.08, fill_color="gray"
                )
            )
            entropy_text = (
                f"<b>Integrated entropy</b> ΔS = ∫ Cp/T dT from {result['t_min']:g} to "
                f"{result['t_max']:g} K ({result['n_points']} points): "
                f"<b>{result['dS']:.5g} ± {result['dS_err']:.2g} {unit}</b> "
                f"(statistical ± {result['dS_err_stat']:.2g}, mass ± {result['dS_err_mass']:.2g})"
            )

        menus = {
            "x": [],
            "y": [
                {
                    "label": _y_columns(f, u)[0],
                    "settings": {"flavor": f, "unit": u},
                    "current": (f, u) == (flavor, unit),
                }
                for f in FLAVORS
                for u in UNITS
                if _y_columns(f, u)[0] in out
            ],
        }
        p.js_on_event(
            "tap",
            CustomJS(
                args=dict(xr=p.x_range, yr=p.y_range, menus=menus, block_id=self.block_id),
                code=AXIS_MENU_CALLBACK,
            ),
        )

        columns = numeric_columns(df)
        export = Button(
            label="Export converted data (.csv)", button_type="primary", width_policy="min"
        )
        export.js_on_click(
            CustomJS(
                args=dict(source=ColumnDataSource(out), filename=filename), code=EXPORT_CSV_CALLBACK
            )
        )

        if has_field:
            field_filter = [
                row(
                    text_input("filter_field_min", f"{FIELD} ≥"),
                    text_input("filter_field_max", f"{FIELD} ≤"),
                )
            ]
        else:
            field_filter = [Div(text="No field column selected.")]

        return column(
            Div(text=WIDGET_STYLE),
            Div(text="<b>Sample</b>"),
            row(text_input("mass_mg", "Mass (mg)"), text_input("mass_err_mg", "± mass (mg)")),
            row(text_input("molecular_weight", "Molecular weight (g/mol f.u.)")),
            Div(text="<b>Data columns</b>"),
            row(
                select("hc_column", "Heat capacity column", columns),
                select("source_unit", "Heat capacity column unit", SOURCE_UNITS),
            ),
            row(
                select("error_column", "Heat capacity error column", [NONE, *columns]),
                select("temperature_column", "Temperature column", columns),
            ),
            row(select("field_column", "Field column", [NONE, *columns])),
            Div(text="<b>Field filter</b> (leave blank for no limit)"),
            *field_filter,
            Div(text="<b>Entropy integration range</b> (leave blank for all data)"),
            row(
                text_input("entropy_t_start", "T start (K)"),
                text_input("entropy_t_end", "T end (K)"),
            ),
            Div(text=entropy_text),
            export,
            Div(text="Click the y axis to change what is plotted."),
            p,
            sizing_mode="scale_width",
            css_classes=["ppms-block", f"ppms-{self.block_id}"],
        )
