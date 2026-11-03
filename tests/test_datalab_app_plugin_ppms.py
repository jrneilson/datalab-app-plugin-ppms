from pathlib import Path

import numpy as np
import pytest

from datalab_app_plugin_ppms import __version__
from datalab_app_plugin_ppms.blocks import PPMSHeatCapacityBlock
from datalab_app_plugin_ppms.utils import (
    convert_heat_capacity,
    guess_source_unit,
    integrate_entropy,
    parse_hc_file,
)

# Processed table (tab-delimited, CR line endings, Latin-1) of T and HC in several units
HC_TABLE = Path(__file__).parent / "data" / "CSU_JN_CH(NH2)2PbBr3_ 20160719.dat"


@pytest.fixture
def table():
    df, _ = parse_hc_file(HC_TABLE)
    # sample mass and molecular weight implied by the columns of the file
    mass_mg = 1e3 * df["µJ/K"][0] / df["µJ/Kg"][0]
    mw = df["µJ/Kmol"][0] / df["µJ/Kg"][0]
    return df, mass_mg, mw


def test_version():
    assert __version__
    assert PPMSHeatCapacityBlock.version == __version__


def test_parse_table(table):
    df, mass_mg, mw = table
    assert list(df.columns) == ["T", "µJ/K", "µJ/Kg", "µJ/Kmol", "J/mol K"]
    assert len(df) == 237
    assert mass_mg == pytest.approx(145.0, rel=1e-3)
    assert mw == pytest.approx(491.9, rel=1e-3)
    assert [guess_source_unit(c) for c in df.columns[1:]] == [
        "µJ/K",
        "µJ/(g·K)",
        "µJ/(mol·K)",
        "J/(mol·K)",
    ]


def test_conversion_matches_file(table):
    df, mass_mg, mw = table
    for column, source_unit in (("µJ/K", "µJ/K"), ("µJ/Kg", "µJ/(g·K)"), ("µJ/Kmol", "µJ/(mol·K)")):
        cp, _, _ = convert_heat_capacity(
            df[column], None, source_unit, "J/(mol·K)", mass_mg=mass_mg, molecular_weight=mw
        )
        np.testing.assert_allclose(cp, df["J/mol K"], rtol=1e-6)

    cp, err, mass_rel = convert_heat_capacity(
        [100.0], [1.0], "µJ/K", "J/(g·K)", mass_mg=10.0, mass_err_mg=0.1
    )
    assert cp[0] == pytest.approx(1e-2)
    assert err[0] == pytest.approx(1e-4)
    assert mass_rel == pytest.approx(0.01)

    with pytest.raises(ValueError):
        convert_heat_capacity([1.0], None, "µJ/K", "J/(mol·K)", mass_mg=10.0)


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


def test_block(table):
    df, mass_mg, mw = table
    block = PPMSHeatCapacityBlock(item_id="test")
    with pytest.warns(UserWarning, match="mass and molecular weight"):
        block.generate_plot(location=HC_TABLE)
    assert block.data["bokeh_plot_data"]
    assert block.data["temperature_column"] == "T"
    assert block.data["hc_column"] == "µJ/K"
    assert block.data["source_unit"] == "µJ/K"
    assert block.data["field_column"] == "None"

    block.process_events(
        [
            {"event_name": "update_settings", "mass_mg": str(mass_mg), "block_id": block.block_id},
            {"event_name": "update_settings", "mass_err_mg": "0.1"},
            {"event_name": "update_settings", "molecular_weight": str(mw)},
            {"event_name": "update_settings", "flavor": "Cp/T³"},
            {"event_name": "update_settings", "entropy_t_start": "2", "entropy_t_end": "50"},
        ]
    )
    assert "errors" not in block.data
    out, unit, entropy = block.compute(df)
    assert unit == "J/(mol·K)"
    assert "Cp/T³ (J/(mol·K⁴))" in out
    result = entropy["J/(mol·K)"]
    in_range = df[(df["T"] >= 2) & (df["T"] <= 50)]
    expected = integrate_entropy(in_range["T"], in_range["J/mol K"], np.zeros(len(in_range)), 0)
    assert result["dS"] == pytest.approx(expected["dS"], rel=1e-6)
    assert result["dS_err_mass"] == pytest.approx(result["dS"] * 0.1 / mass_mg)
    assert block.data["computed"]["entropy"]["J/(mol·K)"]["dS"] == result["dS"]

    # choosing a different column re-guesses its unit
    block.process_events({"event_name": "update_settings", "hc_column": "J/mol K"})
    block.compute(df)
    assert block.data["source_unit"] == "J/(mol·K)"
