from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal end-to-end paper search interface without stage logs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--idea-text", help="English idea text used for retrieval.")
    input_group.add_argument("--pdf-path", help="PDF path used for retrieval.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--Target-Field",
        dest="target_field",
        default=None,
        help="Filter final ranked papers by Field before applying top-k.",
    )
    parser.add_argument("--after", default=None, help="Lower date bound in YYYY-MM-DD format.")
    parser.add_argument("--before", default=None, help="Upper date bound in YYYY-MM-DD format.")
    parser.add_argument(
        "--unable-title-ft",
        action="store_true",
        help="Disable title fulltext fallback and use exact title matching only.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def _build_search_config(args: argparse.Namespace):
    from .config import SearchConfig, parse_date_arg

    config = SearchConfig()
    config.final_top_k = args.top_k
    config.target_field = args.target_field
    config.uniform_importance = True
    if args.unable_title_ft:
        config.enable_title_ft = False

    after_date = None
    before_date = None
    if args.after is not None:
        after_date, config.after_year = parse_date_arg(args.after, "--after")
    if args.before is not None:
        before_date, config.before_year = parse_date_arg(args.before, "--before")
    if after_date is not None and before_date is not None and after_date > before_date:
        raise ValueError(f"--after must be <= --before, got after={after_date}, before={before_date}")
    return config


def run_search_with_authors(args: argparse.Namespace) -> dict[str, object]:
    from .neo4j_repository import Neo4jSearchRepository
    from .pipeline import PaperSearchPipeline

    config = _build_search_config(args)

    with redirect_stderr(io.StringIO()):
        pipeline = PaperSearchPipeline(config)
        if args.pdf_path:
            result = pipeline.search_pdf(args.pdf_path)
        else:
            result = pipeline.search(args.idea_text)

    paper_ids = [item.paper_id for item in result.results if item.paper_id]
    with Neo4jSearchRepository(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password or "",
        database=config.neo4j_database,
    ) as repository:
        papers = repository.fetch_full_paper_records(paper_ids)

    return {
        "papers": papers,
        "authors": [
            {
                "author_id": item.author_id,
                "name": item.name,
                "score": item.score,
            }
            for item in result.authors
        ],
    }


def run_search(args: argparse.Namespace) -> list[dict]:
    payload = run_search_with_authors(args)
    papers = payload.get("papers")
    if not isinstance(papers, list):
        raise RuntimeError("run_search_with_authors() returned an invalid papers payload")
    return papers


def main() -> int:
    args = build_parser().parse_args()
    payload = run_search_with_authors(args)
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
