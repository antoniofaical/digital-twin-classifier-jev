"""Optional text translation adapter."""

from __future__ import annotations

import hashlib
from typing import Any

import requests

DEEPL_FREE_ENDPOINT = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_ENDPOINT = "https://api.deepl.com/v2/translate"


def deepl_endpoint(api_key: str) -> str:
    return DEEPL_FREE_ENDPOINT if api_key.endswith(":fx") else DEEPL_PRO_ENDPOINT


def is_english_hint(language_hint: str) -> bool:
    normalized = language_hint.lower().replace("_", "-")
    return normalized == "en" or normalized.startswith("en-")


def translate_chunk(
    chunk: dict[str, Any],
    *,
    api_key: str,
    mode: str,
    target_language: str,
    request_timeout: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Translate selected page fragments while preserving URL provenance."""
    pages = [dict(page) for page in chunk["pages"]]
    candidate_indexes = [
        index
        for index, page in enumerate(pages)
        if mode == "deepl" or not is_english_hint(str(page.get("language_hint", "")))
    ]
    records: list[dict[str, Any]] = []

    for index, page in enumerate(pages):
        if index not in candidate_indexes:
            records.append(
                {
                    "page_index": index,
                    "url": page["url"],
                    "status": "skipped_english_hint",
                    "language_hint": page.get("language_hint", ""),
                    "source_characters": len(page["text"]),
                    "original_sha256": hashlib.sha256(
                        page["text"].encode("utf-8")
                    ).hexdigest(),
                }
            )

    if not candidate_indexes:
        return {"pages": pages}, records, False

    response = requests.post(
        deepl_endpoint(api_key),
        headers={
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "text": [pages[index]["text"] for index in candidate_indexes],
            "target_lang": target_language.upper(),
            "show_billed_characters": True,
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    translations = response.json().get("translations", [])
    if len(translations) != len(candidate_indexes):
        raise ValueError("DeepL returned an unexpected number of translations")

    for index, translation in zip(candidate_indexes, translations, strict=True):
        original_text = pages[index]["text"]
        translated_text = str(translation["text"])
        detected_language = str(translation.get("detected_source_language", ""))
        pages[index]["text"] = translated_text
        pages[index]["detected_source_language"] = detected_language
        pages[index]["translation_target_language"] = target_language.upper()
        records.append(
            {
                "page_index": index,
                "url": pages[index]["url"],
                "status": "translated",
                "language_hint": pages[index].get("language_hint", ""),
                "detected_source_language": detected_language,
                "target_language": target_language.upper(),
                "source_characters": len(original_text),
                "billed_characters": int(
                    translation.get("billed_characters", len(original_text))
                ),
                "original_sha256": hashlib.sha256(
                    original_text.encode("utf-8")
                ).hexdigest(),
                "translated_text": translated_text,
            }
        )
    records.sort(key=lambda record: record["page_index"])
    return {"pages": pages}, records, True
