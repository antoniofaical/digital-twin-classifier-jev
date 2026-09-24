"""Jev HTTP boundary. The domain never imports this module."""

from __future__ import annotations

from typing import Any

import requests

from ..domain.profile import profile_questions
from ..domain.scoring import normalize_records

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class JevClient:
    def __init__(self, api_key: str, *, timeout: int = 60, input_style: str = "pages"):
        if not api_key:
            raise ValueError("Jev API key required")
        if input_style not in {"pages", "text"}:
            raise ValueError("Invalid Jev input style")
        self.api_key = api_key
        self.timeout = timeout
        self.input_style = input_style

    def evaluate(
        self,
        *,
        subject: str,
        pages: list[dict[str, Any]],
        profile: dict[str, Any],
        model: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        questions = profile_questions(profile)
        state = (
            {
                "subject": subject,
                "instruction": profile["instruction"],
                "input": "".join(page["text"] for page in pages),
            }
            if self.input_style == "text"
            else {
                "company": subject,
                "instruction": profile["instruction"],
                "pages": pages,
            }
        )
        response = requests.post(
            JEV_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "state": state,
                "questions": questions,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Jev response must be an object")
        try:
            probabilities = {
                name: payload["answers"][name]["noul"] for name in questions
            }
        except (KeyError, TypeError) as exc:
            raise ValueError("Jev returned incomplete answers") from exc
        normalized = normalize_records(
            [{"probabilities": probabilities}], tuple(questions)
        )[0]["probabilities"]
        return normalized, payload
