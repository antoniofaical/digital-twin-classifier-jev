from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


try:
    import requests  # noqa: F401
except ImportError:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Response = object
    requests_stub.Session = object
    requests_stub.post = None
    sys.modules["requests"] = requests_stub


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


class ClassifierUnitTests(unittest.TestCase):
    def test_evidence_chunks_preserve_all_text(self) -> None:
        with TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "evidence.jsonl"
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
            self.assertEqual(recovered, "A" * 12 + "B" * 8)

    def test_decide_requires_coherent_evidence_for_verified_label(self) -> None:
        probabilities = {name: 0.95 for name in classifier.QUESTIONS}
        verified = classifier.decide(probabilities, True, 0.70, 0.30)
        split = classifier.decide(probabilities, False, 0.70, 0.30)
        self.assertEqual(verified, ("verified_digital_twin", True, False))
        self.assertEqual(split, ("claimed_digital_twin_unverified", None, True))

    def test_decide_distinguishes_biosimulation(self) -> None:
        probabilities = {name: 0.10 for name in classifier.QUESTIONS}
        probabilities["simulation_prediction"] = 0.90
        probabilities["enabling_technology"] = 0.90
        result = classifier.decide(probabilities, False, 0.70, 0.30)
        self.assertEqual(result, ("virtual_model_or_biosimulation", False, True))

    def test_classify_site_uses_passed_key_and_never_calls_live_api(self) -> None:
        probabilities = {
            "specific_counterpart": 0.95,
            "individualized_data_link": 0.92,
            "repeated_synchronization": 0.91,
            "simulation_prediction": 0.96,
            "self_claim": 0.80,
            "enabling_technology": 0.99,
        }
        calls: list[dict[str, object]] = []

        def fake_post(*args: object, **kwargs: object) -> FakeJevResponse:
            calls.append(kwargs)
            return FakeJevResponse(probabilities)

        with TemporaryDirectory() as tmp:
            site_dir = Path(tmp) / "evidence" / "site1"
            site_dir.mkdir(parents=True)
            (site_dir / "evidence.jsonl").write_text(
                json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
                encoding="utf-8",
            )

            with patch.object(classifier.requests, "post", side_effect=fake_post):
                result = classifier.classify_site(
                    site_name="site1",
                    evidence_dir=site_dir,
                    api_key="test-secret",
                )

            self.assertEqual(result["classification"], "verified_digital_twin")
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0]["headers"]["Authorization"],  # type: ignore[index]
                "Bearer test-secret",
            )
            saved = (site_dir / "classification.json").read_text()
            self.assertNotIn("test-secret", saved)


if __name__ == "__main__":
    unittest.main()
