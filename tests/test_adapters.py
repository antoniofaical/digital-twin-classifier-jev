from pathlib import Path

import pytest

from startup_adherence.adapters.jev import JevClient
from startup_adherence.domain.profile import criterion_ids, load_profile

PROFILE = Path(__file__).parents[1] / "profiles" / "digital_twin.json"


@pytest.mark.parametrize("style", ["pages", "text"])
def test_jev_adapter_keeps_website_and_text_contracts_separate(monkeypatch, style):
    p = load_profile(PROFILE)
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"answers": {key: {"noul": 0.3} for key in criterion_ids(p)}}

    def post(endpoint, *, headers, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr("startup_adherence.adapters.jev.requests.post", post)
    values, _ = JevClient("key", input_style=style).evaluate(
        subject="A",
        pages=[{"url": "https://a.test/", "text": "hello"}],
        profile=p,
        model="fixture",
    )
    assert values[criterion_ids(p)[0]] == 0.3
    assert calls[0]["state"]["instruction"] == p["instruction"]
    if style == "pages":
        assert calls[0]["state"]["company"] == "A"
        assert calls[0]["state"]["pages"][0]["text"] == "hello"
    else:
        assert calls[0]["state"]["subject"] == "A"
        assert calls[0]["state"]["input"] == "hello"
