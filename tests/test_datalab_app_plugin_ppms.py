from pathlib import Path

import numpy as np
import pytest

from datalab_app_plugin_ppms import __version__
from datalab_app_plugin_ppms.blocks import PPMSHeatCapacityBlock
from datalab_app_plugin_ppms.utils import (
    convert_heat_capacity,
    field_groups,
    guess_source_unit,
    integrate_entropy,
    parse_hc_file,
)

DATA = Path(__file__).parent / "data"
# Quantum Design heat capacity file at 0, 1, 2, 5 and 9 T, with sample HC in J/mole-K
MULTI_FIELD = DATA / "LiFeBO3-1p93mg_20130425.dat"
# Quantum Design heat capacity file with sample HC in J/g-K and no molecular weight
PER_GRAM = DATA / "EMM_pellet_100to250.dat"
# Long-pulse analysis export with no header and no errors
LONG_PULSE = DATA / "LiFeBO3-1p93mg_20130425_9TDS.dat"


def test_version():
    assert __version__
    assert PPMSHeatCapacityBlock.version == __version__


def test_parse_qd_file():
    df, header = parse_hc_file(MULTI_FIELD)
    assert header["MASS:Sample Mass (mg)"] == "1.93"
    assert header["MOLWGHT:Formula Weight (g/mole)"] == "121.60"
    df = df.dropna(subset=["Samp HC (J/mole-K)"])
    assert len(df) == 1173

    # the derived per-sample HC reproduces the software's per-mole HC with the header mass
    cp, _, _ = convert_heat_capacity(
        df["Samp HC (µJ/K)"], None, "µJ/K", "J/(mol·K)", mass_mg=1.93, molecular_weight=121.6
    )
    np.testing.assert_allclose(cp, df["Samp HC (J/mole-K)"], rtol=1e-6)

    assert guess_source_unit("Samp HC (J/mole-K)") == "J/(mol·K)"
    assert guess_source_unit("Samp HC (J/g-K)") == "J/(g·K)"
    assert guess_source_unit("HCSample (µJ/K)") == "µJ/K"


def test_conversion():
    cp, err, mass_rel = convert_heat_capacity(
        [100.0], [1.0], "µJ/K", "J/(g·K)", mass_mg=10.0, mass_err_mg=0.1
    )
    assert cp[0] == pytest.approx(1e-2)
    assert err[0] == pytest.approx(1e-4)
    assert mass_rel == pytest.approx(0.01)

    cp, _, mass_rel = convert_heat_capacity(
        [2.0], None, "J/(g·K)", "J/(mol·K)", molecular_weight=50
    )
    assert cp[0] == pytest.approx(100)
    assert mass_rel == 0

    with pytest.raises(ValueError):
        convert_heat_capacity([1.0], None, "µJ/K", "J/(mol·K)", mass_mg=10.0)


def test_field_groups():
    field = np.array([0.061, -0.0, 9999.0, 10000.4, 50001.0, 49999.0, 89998.469, np.nan])
    labels = field_groups(field)
    np.testing.assert_array_equal(labels[:-1], [0, 0, 10000, 10000, 50000, 50000, 90000])
    assert np.isnan(labels[-1])
    assert not np.signbit(labels[0])


def test_entropy_integration():
    # Cp = a T³ gives ΔS = a (T2³ - T1³) / 3
    t = np.linspace(2, 10, 2001)
    a = 1e-3
    result = integrate_entropy(t, a * t**3, np.full_like(t, 1e-4), 0.02, t_start=4, t_end=8)
    assert result["t_min"] == pytest.approx(4)
    assert result["t_max"] == pytest.approx(8)
    assert result["dS"] == pytest.approx(a * (8**3 - 4**3) / 3, rel=1e-5)
    assert result["dS_err_mass"] == pytest.approx(0.02 * result["dS"])

    # statistical error for two points: ΔS = (y0 + y1) Δt / 2
    two = integrate_entropy(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0)
    assert two["dS"] == pytest.approx(1.0)
    assert two["dS_err_stat"] == pytest.approx(np.hypot(0.1 / 2, 0.1 / 2))
    assert two["dS_err"] == pytest.approx(two["dS_err_stat"])

    with pytest.raises(ValueError):
        integrate_entropy(t, t, t, 0, t_start=20)


def test_block_multi_field():
    block = PPMSHeatCapacityBlock(item_id="test")
    block.generate_plot(location=MULTI_FIELD)
    assert block.data["bokeh_plot_data"]
    assert (block.data["mass_mg"], block.data["mass_err_mg"]) == (1.93, 0.01)
    assert block.data["molecular_weight"] == 121.6
    assert block.data["hc_column"] == "Samp HC (µJ/K)"
    assert block.data["error_column"] == "Samp HC Err (µJ/K)"
    assert block.data["field_column"] == "Field (Oersted)"

    entropy = block.data["computed"]["entropy"]
    assert list(entropy) == ["0 T", "1 T", "2 T", "5 T", "9 T"]

    # per-field entropy matches integrating the software's own per-mole HC
    df, _ = parse_hc_file(MULTI_FIELD)
    df = df.dropna(subset=["Samp HC (J/mole-K)"])
    in_1T = field_groups(df["Field (Oersted)"]) == 10000
    expected = integrate_entropy(
        df["Sample Temp (Kelvin)"][in_1T], df["Samp HC (J/mole-K)"][in_1T], np.zeros(201), 0
    )
    result = entropy["1 T"]["J/(mol·K)"]
    assert result["dS"] == pytest.approx(expected["dS"], rel=1e-6)
    assert result["dS_err_mass"] == pytest.approx(result["dS"] * 0.01 / 1.93)

    block.process_events(
        [
            {"event_name": "update_settings", "filter_field_min": "40000"},
            {"event_name": "update_settings", "flavor": "Cp/T³"},
            {"event_name": "update_settings", "entropy_t_start": "2", "entropy_t_end": "20"},
        ]
    )
    assert "errors" not in block.data
    block.generate_plot(location=MULTI_FIELD)
    assert list(block.data["computed"]["entropy"]) == ["5 T", "9 T"]
    assert block.data["computed"]["entropy"]["9 T"]["J/(mol·K)"]["t_max"] <= 20

    # choosing a different column re-guesses its unit
    block.process_events({"event_name": "update_settings", "hc_column": "Samp HC (J/mole-K)"})
    block.compute(df)
    assert block.data["source_unit"] == "J/(mol·K)"


def test_block_other_formats():
    block = PPMSHeatCapacityBlock(item_id="test")
    with pytest.warns(UserWarning, match="mass and molecular weight"):
        block.generate_plot(location=PER_GRAM)
    assert block.data["molecular_weight"] is None
    assert list(block.data["computed"]["entropy"]["0 T"]) == ["J/(g·K)", "J/K"]

    block = PPMSHeatCapacityBlock(item_id="test")
    with pytest.warns(UserWarning):
        block.generate_plot(location=LONG_PULSE)
    assert block.data["hc_column"] == "HCSample (µJ/K)"
    assert block.data["error_column"] == "None"
    assert list(block.data["computed"]["entropy"]) == ["9 T"]
