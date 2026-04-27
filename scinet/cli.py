#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .core.common import (
    DEFAULT_ENV_PATH,
    DEFAULT_RUN_ROOT,
    normalize_whitespace,
    read_json,
    resolve_run_dir,
    write_json,
    write_text,
)
from .core.schemas import (
    SUPPORTED_TASK_TYPES,
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
    SciNetRequest,
)
from .renderers.markdown import render_response_markdown
from .tasks.dispatcher import execute_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SciNet workflows and emit JSON + Markdown results.")
    parser.add_argument("--task-type", choices=SUPPORTED_TASK_TYPES, default=None, help="Task type to run.")
    parser.add_argument(
        "--idea-text",
        default=None,
        help=f"Idea text input for {TASK_GROUNDED_REVIEW} or {TASK_RELATED_AUTHORS}.",
    )
    parser.add_argument(
        "--pdf-path",
        default=None,
        help=f"PDF input for {TASK_GROUNDED_REVIEW} or {TASK_RELATED_AUTHORS}.",
    )
    parser.add_argument(
        "--topic-text",
        default=None,
        help=f"Topic text input for {TASK_TOPIC_TREND_REVIEW} or {TASK_IDEA_GENERATION}.",
    )
    parser.add_argument("--author-name", default=None, help=f"Author name input for {TASK_AUTHOR_PROFILE}.")
    parser.add_argument("--params-file", default=None, help="Path to a JSON file with task params overrides.")
    parser.add_argument("--params-json", default=None, help="Inline JSON object for task params overrides.")
    parser.add_argument("--api-timeout-default", type=float, default=None, help="Default SciNet API read timeout in seconds.")
    parser.add_argument("--api-timeout-search", type=float, default=None, help="Read timeout in seconds for /v1/search.")
    parser.add_argument(
        "--api-timeout-authors-related",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/related.",
    )
    parser.add_argument(
        "--api-timeout-authors-papers",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/papers.",
    )
    parser.add_argument(
        "--api-timeout-support-papers",
        type=float,
        default=None,
        help="Read timeout in seconds for /v1/authors/support-papers.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_RUN_ROOT), help="Root folder for SciNet runs.")
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    parser.add_argument("--env", default=str(DEFAULT_ENV_PATH), help="Path to the SciNet .env file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final JSON to stdout.")
    return parser


def _load_optional_json(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    payload = read_json(Path(path_value).expanduser().resolve())
    return payload


def _parse_inline_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--params-json must decode to a JSON object")
    return payload


def _build_request_from_args(args: argparse.Namespace) -> SciNetRequest:
    if not args.task_type:
        raise ValueError("--task-type is required")

    input_payload: dict[str, Any] = {}
    if args.task_type in {TASK_GROUNDED_REVIEW, TASK_RELATED_AUTHORS}:
        if bool(args.idea_text) == bool(args.pdf_path):
            raise ValueError(f"{TASK_GROUNDED_REVIEW} and {TASK_RELATED_AUTHORS} require exactly one of --idea-text or --pdf-path")
        if args.idea_text:
            input_payload["idea_text"] = args.idea_text
        else:
            input_payload["pdf_path"] = str(Path(args.pdf_path).expanduser().resolve())
    elif args.task_type == TASK_TOPIC_TREND_REVIEW:
        if not args.topic_text:
            raise ValueError(f"{TASK_TOPIC_TREND_REVIEW} requires --topic-text")
        input_payload["topic_text"] = args.topic_text
    elif args.task_type == TASK_AUTHOR_PROFILE:
        if not args.author_name:
            raise ValueError(f"{TASK_AUTHOR_PROFILE} requires --author-name")
        input_payload["author_name"] = args.author_name
    elif args.task_type == TASK_IDEA_GENERATION:
        if not args.topic_text:
            raise ValueError(f"{TASK_IDEA_GENERATION} requires --topic-text")
        input_payload["topic_text"] = args.topic_text

    params = {}
    params.update(_load_optional_json(args.params_file))
    params.update(_parse_inline_json(args.params_json))
    cli_timeout_params = {
        "api_timeout_default": "scinet_api_timeout_default",
        "api_timeout_search": "scinet_api_timeout_search",
        "api_timeout_authors_related": "scinet_api_timeout_authors_related",
        "api_timeout_authors_papers": "scinet_api_timeout_authors_papers",
        "api_timeout_support_papers": "scinet_api_timeout_support_papers",
    }
    for arg_name, param_name in cli_timeout_params.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            params[param_name] = value
    return SciNetRequest(
        task_type=args.task_type,
        input_payload=input_payload,
        params=params,
        output_root=Path(args.output_root).expanduser().resolve(),
        env_path=Path(args.env).expanduser().resolve(),
        run_id=args.run_id,
    )


def _build_input_summary_for_run_id(request: SciNetRequest) -> str:
    for key in ("idea_text", "topic_text", "author_name", "pdf_path"):
        value = normalize_whitespace(request.input_payload.get(key))
        if value:
            return value
    return request.task_type


def main() -> int:
    args = build_parser().parse_args()
    request = _build_request_from_args(args)
    run_dir = resolve_run_dir(request.output_root, request.task_type, request.run_id, _build_input_summary_for_run_id(request))
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    markdown_path = run_dir / "result.md"

    request_payload = {
        "task_type": request.task_type,
        "input": request.input_payload,
        "params": request.params,
        "output_root": str(request.output_root),
        "env": str(request.env_path),
        "run_id": request.run_id,
    }
    write_json(request_path, request_payload)

    try:
        response = execute_request(request, run_dir)
        response.setdefault("artifacts", {})
        response["artifacts"]["json_path"] = str(result_path.resolve())
        response["artifacts"]["markdown_path"] = str(markdown_path.resolve())
        markdown = render_response_markdown(response)
        write_json(result_path, response)
        write_text(markdown_path, markdown)
    except Exception as exc:
        error_payload = {
            "status": "error",
            "task_type": request.task_type,
            "input_summary": request.input_payload,
            "params_effective": request.params,
            "artifacts": {
                "request_path": str(request_path.resolve()),
                "json_path": str(result_path.resolve()),
                "markdown_path": str(markdown_path.resolve()),
            },
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir.resolve()),
        }
        write_json(result_path, error_payload)
        write_text(markdown_path, f"# Error\n\n`{exc.__class__.__name__}`: {exc}\n")
        response = error_payload

    if args.pretty:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
