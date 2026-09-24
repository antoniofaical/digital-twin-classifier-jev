"""Generic text CLI using the same profile, Jev adapter and score engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from .adapters.jev import JevClient
from .application.classification import classify_saved, plan_classification
from .domain.profile import criterion_ids, load_profile, question_set_sha256
from .domain.scoring import build_fit_result, normalize_records
from .domain.urls import evidence_directory_name
from .storage import RunStore, read_jsonl, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score arbitrary UTF-8 text with a theme profile"
    )
    parser.add_argument("--profile", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="UTF-8 file or '-' for stdin")
    inputs.add_argument("--responses", type=Path)
    inputs.add_argument("--validate-profile", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--responses-output", type=Path)
    parser.add_argument("--api-key-env", default="TYPESAFE_PSN_DIG_TWIN_CLASS")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--chunk-chars", type=int, default=20_000)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=Path("fit_runs"))
    args = parser.parse_args(argv)
    if args.chunk_chars <= 0 or args.request_timeout <= 0:
        parser.error("Chunk size and timeout must be positive")
    if args.responses and args.responses_output:
        parser.error("--responses-output is for live runs")
    if args.input == "-" and not args.yes:
        parser.error("--input - requires --yes because stdin contains input data")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = load_profile(args.profile)
    if args.validate_profile:
        print(
            f"PROFILE VALID: {profile['id']}@{profile['version']} question_set_sha256={question_set_sha256(profile)}"
        )
        return 0
    if args.responses:
        records = read_jsonl(args.responses)
        expected = question_set_sha256(profile)
        if not records or any(
            record.get("question_set_sha256") != expected for record in records
        ):
            raise ValueError("Saved response questions do not match the profile")
        if len({record.get("model") for record in records}) != 1:
            raise ValueError("Saved responses contain multiple models")
        result = build_fit_result(
            subject=args.name or args.responses.stem,
            records=normalize_records(records, criterion_ids(profile)),
            profile=profile,
            metadata={
                "offline_rescore": True,
                "model": records[0]["model"],
                "source": str(args.responses),
            },
        )
    else:
        source = args.input
        content = (
            sys.stdin.read()
            if source == "-"
            else Path(source).read_text(encoding="utf-8")
        )
        if not content.strip():
            raise ValueError("Input contains no text")
        subject = args.name or ("stdin" if source == "-" else Path(source).stem)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
        site_dir = (
            args.runs_root / evidence_directory_name(subject) / source_digest / digest
        )
        evidence = site_dir / "evidence.jsonl"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        if not evidence.exists():
            evidence.write_text(
                json.dumps({"url": str(source), "text": content}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        plan = plan_classification(
            evidence, chunk_chars=args.chunk_chars, preserve_whitespace=True
        )
        print(
            f"FIT PLAN: profile={profile['id']}@{profile['version']}; characters={len(content)}; Jev requests={plan.jev_requests}",
            file=sys.stderr,
        )
        if not args.yes:
            print(
                "Paid API calls are planned. Continue? [y/N] ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if sys.stdin.readline().strip().casefold() not in {"y", "yes"}:
                print("Cancelled; no API calls were made.", file=sys.stderr)
                return 0
        key = os.getenv(args.api_key_env, "")
        if not key:
            raise ValueError(f"Missing Jev API key in {args.api_key_env}")
        result = classify_saved(
            site_name=subject,
            site_dir=site_dir,
            profile=profile,
            plan=plan,
            jev_client=JevClient(key, timeout=args.request_timeout, input_style="text"),
            model=args.model,
            extra_metadata={"input_sha256": digest, "source": source},
            evidence_unit_prefix="input_chunk",
        )
        if args.responses_output:
            directory, _, _ = RunStore(site_dir, profile).current()
            args.responses_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / "responses.jsonl", args.responses_output)
        print(f"Saved run: {site_dir}", file=sys.stderr)
    if args.output:
        write_json(args.output, result)
        print(
            f"FIT COMPLETE: score={result['fit_score']:.2f}/100; output={args.output}",
            file=sys.stderr,
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
