from __future__ import annotations

import json
from typing import Any

import pytest

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
    recovered = "".join(page["text"] for chunk in chunks for page in chunk["pages"])

    assert recovered == "A" * 12 + "B" * 8


def test_evenly_spaced_indices_span_the_corpus() -> None:
    assert classifier.evenly_spaced_indices(4, 2) == [0, 3]
    assert classifier.evenly_spaced_indices(5, 3) == [0, 2, 4]


def test_aggregation_uses_weighted_top_three_distinct_evidence_units() -> None:
    records = [
        {
            "request": index,
            "urls": [f"https://example.com/{index}"],
            "probabilities": {name: probability for name in classifier.QUESTIONS},
        }
        for index, probability in enumerate((0.90, 0.60, 0.30, 0.10), start=1)
    ]

    scores, evidence, unit_count = classifier.aggregate_criterion_scores(records)

    assert scores["specific_counterpart"] == pytest.approx(0.735)
    assert [item["score"] for item in evidence["specific_counterpart"]] == [
        90.0,
        60.0,
        30.0,
    ]
    assert unit_count == 4


def test_repeated_chunks_from_same_url_are_one_evidence_unit() -> None:
    records = [
        {
            "request": index,
            "urls": ["https://example.com/product"],
            "probabilities": {name: probability for name in classifier.QUESTIONS},
        }
        for index, probability in enumerate((0.90, 0.60, 0.30), start=1)
    ]

    scores, evidence, unit_count = classifier.aggregate_criterion_scores(records)

    assert scores["specific_counterpart"] == pytest.approx(0.90)
    assert len(evidence["specific_counterpart"]) == 1
    assert unit_count == 1


def test_self_claim_does_not_increase_core_adherence() -> None:
    core = {
        "specific_counterpart": 0.90,
        "individualized_data_link": 0.80,
        "repeated_synchronization": 0.40,
        "simulation_prediction": 0.70,
        "enabling_technology": 0.95,
    }

    def result(self_claim: float) -> dict[str, Any]:
        return classifier.build_adherence_result(
            site_name="site1",
            records=[
                {
                    "request": 1,
                    "urls": ["https://example.com/product"],
                    "probabilities": {**core, "self_claim": self_claim},
                }
            ],
        )

    without_claim = result(0.0)
    with_claim = result(1.0)

    assert (
        without_claim["digital_twin_adherence_score"]
        == with_claim["digital_twin_adherence_score"]
    )
    assert not without_claim["self_claim"]
    assert with_claim["self_claim"]
    assert with_claim["main_gap"] == "repeated_synchronization"


def _mock_jev(monkeypatch, probabilities, calls) -> None:
    def fake_post(*args: object, **kwargs: Any) -> FakeJevResponse:
        del args
        calls.append(kwargs)
        return FakeJevResponse(probabilities)

    monkeypatch.setattr(classifier.requests, "post", fake_post)


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
    progress_events: list[str] = []
    _mock_jev(monkeypatch, probabilities, calls)

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
        progress_callback=progress_events.append,
    )

    expected = classifier.adherence_score(
        {name: probabilities[name] for name in classifier.CORE_CRITERIA}
    )
    assert result["digital_twin_adherence_score"] == pytest.approx(
        round(100 * expected, 2)
    )
    assert result["classification_schema_version"] == 2
    assert "classification" not in result
    assert not result["evidence_is_partial"]
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer test-secret"
    assert progress_events == ["jev"]
    assert "test-secret" not in (site_dir / "classification.json").read_text()


def test_crawl_budget_limit_forces_partial_classification(
    tmp_path, monkeypatch
) -> None:
    probabilities = {name: 0.95 for name in classifier.QUESTIONS}
    calls: list[dict[str, Any]] = []
    _mock_jev(monkeypatch, probabilities, calls)
    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
        encoding="utf-8",
    )
    (site_dir / "manifest.json").write_text(
        json.dumps(
            {
                "crawl_limited": True,
                "crawl_limit_reasons": ["request_limit"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="test-secret",
    )

    assert result["digital_twin_adherence_score"] == 95.0
    assert result["crawl_was_limited"]
    assert result["crawl_limit_reasons"] == ["request_limit"]
    assert result["evidence_is_partial"]


def test_classify_site_balances_single_request_across_saved_pages(
    tmp_path, monkeypatch
) -> None:
    probabilities = {name: 0.95 for name in classifier.QUESTIONS}
    calls: list[dict[str, Any]] = []
    _mock_jev(monkeypatch, probabilities, calls)

    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    records = [
        {"url": "https://example.com/privacy", "text": "A" * 12},
        {"url": "https://example.com/product", "text": "B" * 8},
    ]
    (site_dir / "evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="test-secret",
        chunk_chars=10,
        max_chunks=1,
    )

    submitted_pages = calls[0]["json"]["state"]["pages"]
    assert len(calls) == 1
    assert {page["url"] for page in submitted_pages} == {
        "https://example.com/privacy",
        "https://example.com/product",
    }
    assert sum(len(page["text"]) for page in submitted_pages) == 10
    assert result["digital_twin_adherence_score"] == 95.0
    assert result["evidence_is_partial"]
    assert result["sampling_strategy"] == "balanced_across_pages"
    assert result["evidence_chunks_available"] == 2
    assert result["evidence_chunks_sent"] == 1


def test_percentage_selection_is_evenly_spaced(tmp_path, monkeypatch) -> None:
    probabilities = {name: 0.95 for name in classifier.QUESTIONS}
    calls: list[dict[str, Any]] = []
    _mock_jev(monkeypatch, probabilities, calls)

    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    text = "A" * 10 + "B" * 10 + "C" * 10 + "D" * 10
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": text}) + "\n",
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="test-secret",
        chunk_chars=10,
        evidence_percentage=50,
    )

    submitted_text = [call["json"]["state"]["pages"][0]["text"] for call in calls]
    assert submitted_text == ["A" * 10, "D" * 10]
    assert result["evidence_percentage_requested"] == 50
    assert result["sampling_strategy"] == "evenly_spaced_chunks"
    assert result["source_chunk_numbers"] == [1, 4]
    assert result["evidence_chunks_available"] == 4
    assert result["evidence_chunks_sent"] == 2


def test_auto_translation_sends_non_english_text_to_deepl(
    tmp_path, monkeypatch
) -> None:
    probabilities = {name: 0.10 for name in classifier.QUESTIONS}
    calls: list[tuple[str, dict[str, Any]]] = []
    progress_events: list[str] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self.payload

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append((url, kwargs))
        if url == classifier.DEEPL_FREE_ENDPOINT:
            return FakeResponse(
                {
                    "translations": [
                        {
                            "detected_source_language": "ES",
                            "text": "Digital twin evidence",
                            "billed_characters": 24,
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "answers": {
                    name: {"type": "noul", "noul": probability}
                    for name, probability in probabilities.items()
                }
            }
        )

    monkeypatch.setattr(classifier.requests, "post", fake_post)
    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps(
            {
                "url": "https://example.com/",
                "language_hint": "es",
                "text": "Evidencia de gemelo digital",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="jev-secret",
        translation_mode="auto",
        translation_target="EN",
        deepl_api_key="deepl-secret:fx",
        progress_callback=progress_events.append,
    )

    assert [url for url, _ in calls] == [
        classifier.DEEPL_FREE_ENDPOINT,
        classifier.JEV_ENDPOINT,
    ]
    jev_pages = calls[1][1]["json"]["state"]["pages"]
    assert jev_pages[0]["text"] == "Digital twin evidence"
    assert result["translation"]["deepl_requests"] == 1
    assert result["translation"]["billed_characters"] == 24
    assert progress_events == ["deepl", "jev"]
    translation = json.loads(
        (site_dir / "translations.jsonl").read_text().splitlines()[0]
    )
    assert translation["url"] == "https://example.com/"
    assert translation["detected_source_language"] == "ES"
    assert translation["translated_text"] == "Digital twin evidence"


def test_auto_translation_skips_explicit_english_hint(tmp_path, monkeypatch) -> None:
    probabilities = {name: 0.10 for name in classifier.QUESTIONS}
    calls: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> FakeJevResponse:
        del kwargs
        calls.append(url)
        if url != classifier.JEV_ENDPOINT:
            raise AssertionError("English evidence must not call DeepL")
        return FakeJevResponse(probabilities)

    monkeypatch.setattr(classifier.requests, "post", fake_post)
    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps(
            {
                "url": "https://example.com/",
                "language_hint": "en-US",
                "text": "English evidence",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = classifier.classify_site(
        site_name="site1",
        evidence_dir=site_dir,
        api_key="jev-secret",
        translation_mode="auto",
        deepl_api_key="deepl-secret:fx",
    )

    assert calls == [classifier.JEV_ENDPOINT]
    assert result["translation"]["deepl_requests"] == 0
    assert result["translation"]["skipped_english_page_fragments"] == 1


def test_rescore_site_replaces_categorical_result_without_api_calls(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    site_dir.mkdir(parents=True)
    previous = {
        "site_name": "site1",
        "classification": "claimed_digital_twin_unverified",
        "is_digital_twin": None,
        "requires_human_review": True,
        "evidence_is_partial": True,
        "evidence_chunks_available": 4,
        "evidence_chunks_sent": 1,
        "sampling_strategy": "balanced_across_pages",
        "model": "jev-latest",
    }
    (site_dir / "classification.json").write_text(
        json.dumps(previous) + "\n",
        encoding="utf-8",
    )
    probabilities = {
        "specific_counterpart": 0.90,
        "individualized_data_link": 0.80,
        "repeated_synchronization": 0.50,
        "simulation_prediction": 0.70,
        "self_claim": 0.95,
        "enabling_technology": 0.90,
    }
    (site_dir / "jev_chunks.jsonl").write_text(
        json.dumps(
            {
                "request": 1,
                "urls": ["https://example.com/product"],
                "probabilities": probabilities,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_post(*args, **kwargs):
        del args, kwargs
        raise AssertionError("offline rescoring must not call an API")

    monkeypatch.setattr(classifier.requests, "post", fail_post)
    result = classifier.rescore_site(site_name="site1", evidence_dir=site_dir)
    persisted = json.loads((site_dir / "classification.json").read_text())

    assert persisted == result
    assert result["classification_schema_version"] == 2
    assert result["evidence_is_partial"]
    assert result["evidence_chunks_available"] == 4
    assert result["self_claim"]
    assert result["main_gap"] == "repeated_synchronization"
    assert "classification" not in result
    assert "is_digital_twin" not in result
    assert "requires_human_review" not in result
