"""CLI for scoring arbitrary text-based input against a versioned fit profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import requests

from fit_engine import (
    build_fit_result,
    criterion_ids,
    load_profile,
    normalize_records,
    profile_questions,
    question_set_sha256,
)

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_API_KEY_ENV = "TYPESAFE_PSN_DIG_TWIN_CLASS"
DEFAULT_MODEL = "jev-latest"
DEFAULT_CHUNK_CHARS = 20_000


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score arbitrary text-based input against a versioned fit profile."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="JSON profile defining the target, criteria, weights, and aggregation",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=str,
        help="text-based input file, or '-' to read the input from stdin",
    )
    source.add_argument(
        "--responses",
        type=Path,
        help="score saved Jev response records from JSONL without API calls",
    )
    source.add_argument(
        "--validate-profile",
        action="store_true",
        help="validate the profile and exit without reading input or calling APIs",
    )
    parser.add_argument(
        "--name",
        help="subject name stored in the result; defaults to the input filename",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the result as JSON; otherwise print it to stdout",
    )
    parser.add_argument(
        "--responses-output",
        type=Path,
        help="save raw Jev response records as JSONL for offline rescoring",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"environment variable containing the Jev key, default: {DEFAULT_API_KEY_ENV}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chunk-chars",
        type=positive_integer,
        default=DEFAULT_CHUNK_CHARS,
        help=f"maximum characters sent per Jev request, default: {DEFAULT_CHUNK_CHARS}",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_integer,
        default=60,
        help="HTTP timeout in seconds, default: 60",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="approve the displayed Jev request plan without prompting",
    )
    args = parser.parse_args(argv)
    if args.responses is not None and args.responses_output is not None:
        parser.error("--responses-output cannot be used with --responses")
    if args.validate_profile and args.responses_output is not None:
        parser.error("--responses-output cannot be used with --validate-profile")
    if args.input == "-" and not args.yes:
        parser.error("--input - requires --yes because stdin contains the scored input")
    return args


def read_input(source: str, stdin: TextIO | None = None) -> tuple[str, str]:
    """Read arbitrary UTF-8 text from a path or stdin."""
    if source == "-":
        text = (stdin or sys.stdin).read()
        default_name = "stdin"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        text = path.read_text(encoding="utf-8")
        default_name = path.stem
    if not text.strip():
        raise ValueError("Input contains no text")
    return text, default_name


def text_chunks(text: str, chunk_chars: int) -> Iterator[str]:
    """Yield bounded chunks while preserving every input character."""
    for start in range(0, len(text), chunk_chars):
        yield text[start : start + chunk_chars]


def load_response_records(
    path: Path,
    criterion_names: tuple[str, ...],
    expected_question_set_sha256: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            records.append(record)
    recorded_hashes = {
        str(record["question_set_sha256"])
        for record in records
        if isinstance(record, dict) and record.get("question_set_sha256")
    }
    records_with_hash = sum(
        isinstance(record, dict) and bool(record.get("question_set_sha256"))
        for record in records
    )
    if records_with_hash not in {0, len(records)}:
        raise ValueError("Saved responses mix versioned and legacy question sets")
    if len(recorded_hashes) > 1:
        raise ValueError("Saved responses contain multiple question sets")
    if (
        expected_question_set_sha256 is not None
        and recorded_hashes
        and recorded_hashes != {expected_question_set_sha256}
    ):
        raise ValueError(
            "Saved responses were created with different profile questions. "
            "Run live fit scoring again."
        )
    return normalize_records(records, criterion_names)


def query_jev(
    *,
    text: str,
    subject: str,
    profile: dict[str, Any],
    api_key: str,
    model: str,
    chunk_chars: int,
    request_timeout: int,
    progress_stream: TextIO | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all input chunks and retain replayable probability records."""
    if not api_key:
        raise ValueError("A Jev API key is required")
    chunks = list(text_chunks(text, chunk_chars))
    questions = profile_questions(profile)
    questions_sha256 = question_set_sha256(profile)
    records: list[dict[str, Any]] = []
    stream = progress_stream or sys.stderr
    for number, chunk in enumerate(chunks, start=1):
        response = requests.post(
            JEV_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "state": {
                    "subject": subject,
                    "instruction": profile["instruction"],
                    "input": chunk,
                },
                "questions": questions,
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            probabilities = {
                name: float(payload["answers"][name]["noul"]) for name in questions
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Jev returned an invalid answers object") from exc
        records.append(
            {
                "request": number,
                "evidence_unit": f"input_chunk_{number}",
                "profile_id": profile["id"],
                "profile_version": profile["version"],
                "question_set_sha256": questions_sha256,
                "probabilities": probabilities,
            }
        )
        print(f"Jev {number}/{len(chunks)}", file=stream, flush=True)
    return normalize_records(records, tuple(questions))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def prompt_confirmation(message: str) -> bool:
    print(f"{message} Continue? [y/N]", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().casefold() in {"y", "yes"}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = load_profile(args.profile)
    all_criteria = criterion_ids(profile)
    questions_sha256 = question_set_sha256(profile)

    if args.validate_profile:
        print(
            f"PROFILE VALID: {profile['id']}@{profile['version']} "
            f"criteria={len(all_criteria)} question_set_sha256={questions_sha256}"
        )
        return

    if args.responses is not None:
        subject = args.name or args.responses.stem
        print(
            "FIT RESCORE: saved Jev responses only; no API calls will be made.",
            file=sys.stderr,
        )
        records = load_response_records(
            args.responses,
            all_criteria,
            expected_question_set_sha256=questions_sha256,
        )
        metadata = {
            "model": args.model,
            "input_chunks_sent": len(records),
            "source": str(args.responses),
            "offline_rescore": True,
        }
    else:
        text, default_name = read_input(args.input)
        subject = args.name or default_name
        chunk_count = (len(text) + args.chunk_chars - 1) // args.chunk_chars
        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{args.api_key_env} is not set. Set it before running live fit scoring."
            )
        print("FIT PLAN", file=sys.stderr)
        print(f"Profile: {profile['id']}@{profile['version']}", file=sys.stderr)
        print(f"Subject: {subject}", file=sys.stderr)
        print(f"Characters: {len(text)}", file=sys.stderr)
        print(f"Jev requests: {chunk_count}", file=sys.stderr)
        if not args.yes and not prompt_confirmation(
            f"The run will send {len(text)} characters in {chunk_count} Jev request(s)."
        ):
            print("Fit scoring cancelled; no API calls were made.", file=sys.stderr)
            return
        records = query_jev(
            text=text,
            subject=subject,
            profile=profile,
            api_key=api_key,
            model=args.model,
            chunk_chars=args.chunk_chars,
            request_timeout=args.request_timeout,
        )
        if args.responses_output is not None:
            write_jsonl(args.responses_output, records)
        metadata = {
            "model": args.model,
            "input_characters": len(text),
            "input_chunks_sent": len(records),
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source": args.input,
            "offline_rescore": False,
        }

    result = build_fit_result(
        subject=subject,
        records=records,
        profile=profile,
        metadata=metadata,
    )
    if args.output is not None:
        write_json(args.output, result)
        print(f"FIT COMPLETE: score={result['fit_score']:.2f}/100", file=sys.stderr)
        print(f"Output: {args.output}", file=sys.stderr)
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
