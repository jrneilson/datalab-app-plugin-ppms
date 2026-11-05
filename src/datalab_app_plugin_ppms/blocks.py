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
    field_groups,
    guess_source_unit,
    integrate_entropy,
    numeric_columns,
    parse_hc_file,
)

TEMPERATURE = "T (K)"
FIELD = "Field (Oe)"
FIELD_SETPOINT = "Field setpoint (Oe)"
ALL_FIELDS = "all"
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
X_AXES = {
    TEMPERATURE: lambda t: t,
    "T² (K²)": lambda t: t**2,
    "T³ (K³)": lambda t: t**3,
    "ln(T/K)": np.log,
}
"""Plotted x axes and their functions of temperature."""

CHOICE_SETTINGS = {
    "x_axis": tuple(X_AXES),
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


def _field_label(setpoint_oe: float) -> str:
    return f"{setpoint_oe / 1e4:g} T"


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


CLEARED = ""
"""Stored for a cleared numeric setting. datalab drops `None` values when it sends the block
to the browser, which then keeps (and later sends back) the old value; "" replaces it."""


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

    defaults = {"x_axis": TEMPERATURE, "flavor": "Cp", "unit": "J/(mol·K)"}

    @property
    def plot_functions(self):
        return (self.generate_plot,)

    def _number(self, key: str) -> float | None:
        """A numeric setting, or None if it is unset or cleared."""
        value = self.data.get(key)
        return None if value in (None, CLEARED) else float(value)

    @event()
    def update_settings(self, **kwargs):
        """Update one or more block settings from the plot widgets."""
        for key, value in kwargs.items():
            if key in NUMERIC_SETTINGS:
                parsed = _parse_float(key, value)
                self.data[key] = CLEARED if parsed is None else parsed
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
        """Fill sample mass, its uncertainty and molecular weight from the file header the
        first time it is read. Quantum Design header keys look like `MASS:Sample Mass (mg)`.
        """
        codes = {k.split(":")[0].strip(): v for k, v in header.items()}
        for key, code in (
            ("mass_mg", "MASS"),
            ("mass_err_mg", "MASSERR"),
            ("molecular_weight", "MOLWGHT"),
        ):
            if key not in self.data:
                try:
                    parsed = _parse_float(key, codes.get(code))
                    self.data[key] = CLEARED if parsed is None else parsed
                except ValueError:
                    self.data[key] = CLEARED

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
        the entropy separately for each field setpoint in each available unit.

        Returns:
            A DataFrame with temperature, field and field setpoint (if any), the raw heat
            capacity and every flavor in every unit that can be computed from the given
            sample parameters; the unit actually used for plotting; and the entropy results
            keyed by field setpoint label (`ALL_FIELDS` without a field column) and then by
            unit (or an `error` message if the integration was not possible).

        """
        self._resolve_columns(df)

        field_column = self.data["field_column"]
        # e.g. comment rows in Quantum Design files
        mask = df[self.data["hc_column"]].notna().to_numpy()
        if field_column != NONE:
            if (lower := self._number("filter_field_min")) is not None:
                mask &= df[field_column].to_numpy() >= lower
            if (upper := self._number("filter_field_max")) is not None:
                mask &= df[field_column].to_numpy() <= upper
        df = df[mask].reset_index(drop=True)

        out = pd.DataFrame()
        out[TEMPERATURE] = df[self.data["temperature_column"]]
        if field_column != NONE:
            out[FIELD] = df[field_column]
            out[FIELD_SETPOINT] = field_groups(out[FIELD])
        hc = df[self.data["hc_column"]].to_numpy(dtype=float)
        error_column = self.data["error_column"]
        hc_err = None if error_column == NONE else df[error_column].to_numpy(dtype=float)
        source_unit = self.data["source_unit"]
        out[f"HC ({source_unit})"] = hc
        if hc_err is not None:
            out[f"σHC ({source_unit})"] = hc_err

        sample = {
            "mass_mg": self._number("mass_mg"),
            "mass_err_mg": self._number("mass_err_mg"),
            "molecular_weight": self._number("molecular_weight"),
        }
        units = available_units(source_unit, sample["mass_mg"], sample["molecular_weight"])
        temperature = out[TEMPERATURE].to_numpy(dtype=float)
        if field_column != NONE:
            groups = {
                _field_label(h): (out[FIELD_SETPOINT] == h).to_numpy()
                for h in sorted(out[FIELD_SETPOINT].dropna().unique())
            }
        else:
            groups = {ALL_FIELDS: np.ones(len(out), dtype=bool)}
        entropy: dict[str, dict] = {label: {} for label in groups}
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
            for label, in_group in groups.items():
                try:
                    entropy[label][unit] = integrate_entropy(
                        temperature[in_group],
                        cp[in_group],
                        cp_err[in_group],
                        mass_rel_err,
                        self._number("entropy_t_start"),
                        self._number("entropy_t_end"),
                    )
                except ValueError as exc:
                    entropy[label][unit] = {"error": str(exc)}

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
            "entropy": {
                label: {u: r for u, r in results.items() if "error" not in r}
                for label, results in entropy.items()
            },
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
            Legend,
            Select,
            TeeHead,
            TextInput,
            Whisker,
        )
        from bokeh.palettes import Category10_10
        from bokeh.plotting import figure

        def text_input(key: str, title: str) -> TextInput:
            value = self._number(key)
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

        x_axis = self.data.get("x_axis", TEMPERATURE)
        x_of_t = X_AXES[x_axis]
        tooltips = [(TEMPERATURE, "@t{0.000}")]
        if has_field:
            tooltips.append((FIELD, "@field{0.00}"))
        tooltips.append((y_label, "@y{0.0000e} ± @err{0.0e}"))

        p = figure(
            x_axis_label=f"{x_axis} ▾",
            y_axis_label=f"{y_label} ▾",
            sizing_mode="scale_width",
            aspect_ratio=1.5,
        )
        legend_items = []
        if has_field:
            groups = [
                (out[out[FIELD_SETPOINT] == h], _field_label(h))
                for h in sorted(out[FIELD_SETPOINT].dropna().unique())
            ]
        else:
            groups = [(out, ALL_FIELDS)]
        for i, (group, label) in enumerate(groups):
            y = group[y_label].to_numpy()
            y_err = group[y_err_label].to_numpy()
            data = {
                "x": x_of_t(group[TEMPERATURE].to_numpy()),
                "t": group[TEMPERATURE].to_numpy(),
                "y": y,
                "upper": y + y_err,
                "lower": y - y_err,
                "err": y_err,
            }
            if has_field:
                data["field"] = group[FIELD].to_numpy()
            source = ColumnDataSource(data)
            color = Category10_10[i % len(Category10_10)]
            renderer = p.circle("x", "y", source=source, size=4, color=color)
            p.add_layout(
                Whisker(
                    source=source,
                    base="x",
                    upper="upper",
                    lower="lower",
                    line_color=color,
                    line_alpha=0.5,
                    upper_head=TeeHead(line_color=color, line_alpha=0.5, size=6),
                    lower_head=TeeHead(line_color=color, line_alpha=0.5, size=6),
                )
            )
            legend_items.append((label, [renderer]))
        p.add_tools(HoverTool(tooltips=tooltips))
        if has_field:
            p.add_layout(Legend(items=legend_items, click_policy="hide"), "right")

        lines = []
        for label, results in entropy.items():
            result = results.get(unit, {"error": "No data"})
            prefix = "" if label == ALL_FIELDS else f"{label}: "
            if "error" in result:
                lines.append(f"{prefix}{result['error']}")
            else:
                lines.append(
                    f"{prefix}<b>{result['dS']:.5g} ± {result['dS_err']:.2g} {unit}</b> "
                    f"from {result['t_min']:g} to {result['t_max']:g} K "
                    f"({result['n_points']} points; statistical ± {result['dS_err_stat']:.2g}, "
                    f"mass ± {result['dS_err_mass']:.2g})"
                )
        t_start, t_end = self._number("entropy_t_start"), self._number("entropy_t_end")
        with np.errstate(divide="ignore", invalid="ignore"):
            left, right = (
                float(x) if t is not None and np.isfinite(x := x_of_t(t)) else None
                for t in (t_start, t_end)
            )
        if left is not None or right is not None:
            p.add_layout(
                BoxAnnotation(
                    left=left,
                    right=right,
                    fill_alpha=0.08,
                    fill_color="gray",
                )
            )
        entropy_text = "<b>Integrated entropy</b> ΔS = ∫ Cp/T dT<br>" + "<br>".join(lines)

        menus = {
            "x": [{"label": x, "settings": {"x_axis": x}, "current": x == x_axis} for x in X_AXES],
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
            Div(text="Click the x or y axis to change what is plotted."),
            p,
            sizing_mode="scale_width",
            css_classes=["ppms-block", f"ppms-{self.block_id}"],
        )
