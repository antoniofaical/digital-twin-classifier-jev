from __future__ import annotations

import io
import json
from typing import Any

import pytest

import fit


class FakeResponse:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "answers": {
                name: {"type": "noul", "noul": probability}
                for name, probability in self.probabilities.items()
            }
        }


def test_text_chunks_preserve_all_input() -> None:
    text = "abcdefghij"
    assert list(fit.text_chunks(text, 4)) == ["abcd", "efgh", "ij"]


def test_live_entrypoint_scores_arbitrary_file_and_saves_replay(
    tmp_path, monkeypatch
) -> None:
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        json.dumps(
            {
                "profile_schema_version": 1,
                "id": "job_fit",
                "version": "1.0.0",
                "name": "Job fit",
                "instruction": "Evaluate the candidate against the role.",
                "criteria": [
                    {
                        "id": "skills",
                        "role": "core",
                        "weight": 1.0,
                        "instructions": "Does the candidate have the required skills?",
                    },
                    {
                        "id": "experience",
                        "role": "core",
                        "weight": 1.0,
                        "instructions": "Does the candidate have relevant experience?",
                    },
                ],
                "evidence_aggregation": {"top_weights": [1.0]},
                "score_aggregation": {"method": "weighted_mean"},
            }
        ),
        encoding="utf-8",
    )
    input_file = tmp_path / "candidate.txt"
    input_file.write_text("candidate evidence", encoding="utf-8")
    output_file = tmp_path / "result.json"
    responses_file = tmp_path / "responses.jsonl"
    calls: list[dict[str, Any]] = []

    def fake_post(*args: object, **kwargs: Any) -> FakeResponse:
        del args
        calls.append(kwargs)
        return FakeResponse({"skills": 0.8, "experience": 0.6})

    monkeypatch.setenv(fit.DEFAULT_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(fit.requests, "post", fake_post)

    fit.main(
        [
            "--profile",
            str(profile_file),
            "--input",
            str(input_file),
            "--name",
            "Candidate A",
            "--output",
            str(output_file),
            "--responses-output",
            str(responses_file),
            "--yes",
        ]
    )

    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert result["profile"]["id"] == "job_fit"
    assert result["subject"] == "Candidate A"
    assert result["fit_score"] == 70.0
    assert set(calls[0]["json"]["questions"]) == {"skills", "experience"}
    assert calls[0]["json"]["state"]["input"] == "candidate evidence"
    assert "test-secret" not in output_file.read_text(encoding="utf-8")
    assert responses_file.is_file()


def test_offline_rescore_needs_no_key_or_network(tmp_path, monkeypatch, capsys) -> None:
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        json.dumps(
            {
                "profile_schema_version": 1,
                "id": "simple",
                "version": "1",
                "name": "Simple fit",
                "instruction": "Evaluate fit.",
                "criteria": [
                    {
                        "id": "fit",
                        "role": "core",
                        "instructions": "Does it fit?",
                    }
                ],
                "score_aggregation": {"method": "weighted_mean"},
            }
        ),
        encoding="utf-8",
    )
    responses_file = tmp_path / "responses.jsonl"
    responses_file.write_text(
        json.dumps(
            {
                "request": 1,
                "evidence_unit": "input_chunk_1",
                "probabilities": {"fit": 0.75},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_post(*args: object, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("offline rescore must not call Jev")

    monkeypatch.delenv(fit.DEFAULT_API_KEY_ENV, raising=False)
    monkeypatch.setattr(fit.requests, "post", fail_post)

    fit.main(
        [
            "--profile",
            str(profile_file),
            "--responses",
            str(responses_file),
            "--name",
            "saved input",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["fit_score"] == 75.0
    assert result["offline_rescore"]
    assert "no API calls" in captured.err


def test_stdin_input_requires_noninteractive_approval() -> None:
    with pytest.raises(SystemExit):
        fit.parse_args(["--profile", "profile.json", "--input", "-"])

    args = fit.parse_args(["--profile", "profile.json", "--input", "-", "--yes"])
    assert args.input == "-"


def test_read_input_from_stdin() -> None:
    text, name = fit.read_input("-", stdin=io.StringIO("arbitrary evidence"))
    assert text == "arbitrary evidence"
    assert name == "stdin"
