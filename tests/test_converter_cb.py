import csv

import pytest

from converter_cb import convert


def write_scores(path, rows, *, columns=None):
    columns = columns or [
        "site_name",
        "site_url",
        "fit_score",
        "core__specific_counterpart",
        "core__individualized_data_link",
        "core__simulation_prediction",
        "core__repeated_synchronization",
        "core__self_claim",
        "aux__enabling_technology",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_mode_two_ignores_non_gate_scores_and_keeps_inclusive_boundary(tmp_path):
    source = tmp_path / "twins.csv"
    rows = [
        {
            "site_name": "candidate",
            "site_url": "https://example.test",
            "fit_score": 20,
            "core__specific_counterpart": 30,
            "core__individualized_data_link": 45,
            "core__simulation_prediction": 50,
            "core__repeated_synchronization": 2,
            "core__self_claim": 2,
            "aux__enabling_technology": 2,
        },
        {
            "site_name": "generic",
            "site_url": "https://generic.test",
            "fit_score": 90,
            "core__specific_counterpart": 29,
            "core__individualized_data_link": 45,
            "core__simulation_prediction": 50,
            "core__repeated_synchronization": 95,
            "core__self_claim": 95,
            "aux__enabling_technology": 95,
        },
    ]
    write_scores(source, rows)
    convert(str(tmp_path / "twins"), 2, 70, 30)
    with (tmp_path / "twins_simples.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["Company", "URL"],
            ["candidate", "https://example.test"],
        ]


def test_mode_one_keeps_strict_fit_threshold(tmp_path):
    write_scores(
        tmp_path / "twins.csv",
        [
            {"site_name": "equal", "site_url": "https://equal.test", "fit_score": 70},
            {"site_name": "above", "site_url": "https://above.test", "fit_score": 71},
        ],
    )
    convert(str(tmp_path / "twins"), 1, 70, 30)
    with (tmp_path / "twins_simples.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["Company", "URL"],
            ["above", "https://above.test"],
        ]


def test_mode_two_reports_missing_gate_column_before_overwriting_output(tmp_path):
    source = tmp_path / "twins.csv"
    write_scores(source, [], columns=["site_name", "site_url"])
    target = tmp_path / "twins_simples.csv"
    target.write_text("existing")
    with pytest.raises(ValueError, match="core__specific_counterpart"):
        convert(str(tmp_path / "twins"), 2, 70, 30)
    assert target.read_text() == "existing"
