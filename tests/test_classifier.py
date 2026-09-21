from __future__ import annotations

import json
from typing import Any

import classifier


class FakeJevResponse:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "answers": {
                name: {"type": "noul", "noul": probability}
                for name, probability in self._probabilities.items()
            }
        }


def test_evidence_chunks_preserve_all_text(tmp_path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    records = [
        {"url": "https://a.test/", "text": "A" * 12},
        {"url": "https://a.test/about", "text": "B" * 8},
    ]
    evidence.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    chunks = list(classifier.evidence_chunks(evidence, chunk_chars=10))
    recovered = "".join(
        page["text"] for chunk in chunks for page in chunk["pages"]
    )

    assert recovered == "A" * 12 + "B" * 8


def test_decide_requires_coherent_evidence_for_verified_label() -> None:
    probabilities = {name: 0.95 for name in classifier.QUESTIONS}
    assert classifier.decide(probabilities, True, 0.70, 0.30) == (
        "verified_digital_twin",
        True,
        False,
    )

    probabilities["self_claim"] = 0.10
    assert classifier.decide(probabilities, False, 0.70, 0.30) == (
        "manual_review_cross_chunk_evidence",
        None,
        True,
    )


def test_decide_distinguishes_biosimulation() -> None:
    probabilities = {name: 0.10 for name in classifier.QUESTIONS}
    probabilities["simulation_prediction"] = 0.90
    probabilities["enabling_technology"] = 0.90
    assert classifier.decide(probabilities, False, 0.70, 0.30) == (
        "virtual_model_or_biosimulation",
        False,
        True,
    )


def test_classify_site_uses_passed_key_and_never_calls_live_api(
    tmp_path, monkeypatch
) -> None:
    probabilities = {
        "specific_counterpart": 0.95,
        "individualized_data_link": 0.92,
        "repeated_synchronization": 0.91,
        "simulation_prediction": 0.96,
        "self_claim": 0.80,
        "enabling_technology": 0.99,
    }
    calls: list[dict[str, Any]] = []

    def fake_post(*args: object, **kwargs: Any) -> FakeJevResponse:
        del args
        calls.append(kwargs)
        return FakeJevResponse(probabilities)

    monkeypatch.setattr(classifier.requests, "post", fake_post)
    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="test-secret",
    )

    assert result["classification"] == "verified_digital_twin"
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer test-secret"
    assert "test-secret" not in (site_dir / "classification.json").read_text()
