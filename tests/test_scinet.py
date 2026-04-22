from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scinet.core.api_client import SciNetApiClient, load_scinet_api_settings
from scinet.core.common import truncate_text
from scinet.core.schemas import (
    TASK_AUTHOR_PROFILE,
    TASK_GROUNDED_REVIEW,
    TASK_IDEA_GENERATION,
    TASK_RELATED_AUTHORS,
    TASK_TOPIC_TREND_REVIEW,
    SciNetRequest,
    merge_task_params,
)
from scinet.renderers.markdown import render_response_markdown
from run_scinet import build_parser


class SciNetRendererTests(unittest.TestCase):
    def test_merge_task_params_overrides_defaults(self) -> None:
        params = merge_task_params(TASK_TOPIC_TREND_REVIEW, {"final_paper_count_for_summary": 12})
        self.assertEqual(params["final_paper_count_for_summary"], 12)
        self.assertEqual(params["search_api_top_k"], 40)

    def test_truncate_text_shortens_long_text(self) -> None:
        text = "a" * 80
        truncated = truncate_text(text, max_chars=20)
        self.assertTrue(truncated.endswith("..."))
        self.assertEqual(len(truncated), 20)

    def test_render_grounding_markdown_contains_sections(self) -> None:
        response = {
            "task_type": TASK_GROUNDED_REVIEW,
            "input_summary": {"input_mode": "idea_text", "idea_text": "idea"},
            "params_effective": {"kg_top_k": 20},
            "result": {
                "summary": "summary",
                "retrieved_papers": [{"title": "Paper A", "year": 2024, "source": "kg"}],
                "top_matches": [{"paper_title": "Paper A", "query_sentence": "query", "grounded_passage": "match"}],
                "per_paper": {
                    "Paper A": {
                        "matches": [{"query_sentence": "query", "grounded_passage": "match"}],
                        "similar_points": ["shared point"],
                        "different_points": ["different point"],
                    }
                },
            },
        }
        markdown = render_response_markdown(response)
        self.assertIn("# Task 1", markdown)
        self.assertIn("Retrieved Papers", markdown)
        self.assertIn("Similar Points", markdown)

    def test_render_trend_review_markdown_contains_sections(self) -> None:
        response = {
            "task_type": TASK_TOPIC_TREND_REVIEW,
            "input_summary": {"topic_text": "topic"},
            "params_effective": {"kg_top_k": 40},
            "result": {
                "trend_summary": {
                    "one_sentence_summary": "summary",
                    "stage_summary": [{"period": "2019-2021", "theme": "theme"}],
                    "methodological_shifts": ["shift"],
                    "emerging_topics": ["topic"],
                    "open_gaps": ["gap"],
                },
                "representative_papers": [{"title": "Paper A", "year": 2024}],
                "papers_by_year": [{"title": "Paper A", "year": 2024}],
            },
        }
        markdown = render_response_markdown(response)
        self.assertIn("# Task 2", markdown)
        self.assertIn("Stage Summary", markdown)
        self.assertIn("Chronological Paper List", markdown)

    def test_render_related_authors_markdown_contains_author_table(self) -> None:
        response = {
            "task_type": TASK_RELATED_AUTHORS,
            "input_summary": {"input_mode": "idea_text", "idea_text": "idea"},
            "params_effective": {"author_top_k": 10},
            "result": {
                "summary": "summary",
                "authors": [{"name": "Alice", "score": 0.9, "support_papers": [{"title": "Paper A"}]}],
                "supporting_papers": [{"title": "Paper A", "year": 2024}],
            },
        }
        markdown = render_response_markdown(response)
        self.assertIn("# Task 3", markdown)
        self.assertIn("Related Authors", markdown)
        self.assertIn("Supporting Papers", markdown)

    def test_render_author_profile_markdown_contains_trajectory(self) -> None:
        response = {
            "task_type": TASK_AUTHOR_PROFILE,
            "input_summary": {"author_name": "Alice"},
            "params_effective": {"author_paper_sample_size": 40},
            "result": {
                "overall_academic_profile": "profile",
                "main_research_directions": [{"theme": "Theme A", "active_years": "2020-2024"}],
                "technical_arsenal": ["method"],
                "representative_papers": [{"title": "Paper A", "year": 2024}],
            },
        }
        markdown = render_response_markdown(response)
        self.assertIn("# Task 4", markdown)
        self.assertIn("Research Trajectory", markdown)
        self.assertIn("Technical Arsenal", markdown)

    def test_render_idea_generation_markdown_contains_generated_section(self) -> None:
        response = {
            "task_type": TASK_IDEA_GENERATION,
            "input_summary": {"topic_text": "topic"},
            "params_effective": {"idea_count": 5},
            "result": {
                "ideas": [
                    {
                        "title": "Idea A",
                        "description": "desc",
                        "novelty": "novel",
                        "significance": "important",
                        "related_papers": [{"title": "Paper A", "year": 2024}],
                    }
                ]
            },
        }
        markdown = render_response_markdown(response)
        self.assertIn("# Task 5", markdown)
        self.assertIn("Generated Ideas", markdown)
        self.assertIn("Idea A", markdown)


class GroundingPipelineTests(unittest.TestCase):
    def test_build_fallback_queries_uses_basic_idea_when_motivation_and_method_missing(self) -> None:
        from scinet.evidence.grounding import (
            StructuredExtraction,
            build_fallback_queries_from_extraction,
        )

        queries, reason = build_fallback_queries_from_extraction(
            StructuredExtraction(
                basic_idea=["Use literature-grounded evidence to evaluate research ideas."],
                motivation=[],
                method=[],
                experimental_focus=[],
            ),
            max_queries=3,
        )

        self.assertEqual(reason, "basic_idea_fallback")
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].section, "method")
        self.assertEqual(queries[0].query_text, "Use literature-grounded evidence to evaluate research ideas.")

    def test_build_fallback_queries_uses_idea_text_when_extraction_is_empty(self) -> None:
        from scinet.evidence.grounding import (
            StructuredExtraction,
            build_fallback_queries_from_extraction,
        )

        queries, reason = build_fallback_queries_from_extraction(
            StructuredExtraction(
                basic_idea=[],
                motivation=[],
                method=[],
                experimental_focus=[],
            ),
            max_queries=2,
            idea_text="Knowledge-grounded evaluation of scientific research ideas",
        )

        self.assertEqual(reason, "idea_text_fallback")
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].section, "method")
        self.assertEqual(queries[0].query_text, "Knowledge-grounded evaluation of scientific research ideas")


class ConfigAndCliTests(unittest.TestCase):
    def test_load_scinet_api_settings_reads_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "SCINET_API_BASE_URL=http://127.0.0.1:8000\n"
                "SCINET_API_KEY=test-key\n"
                "SCINET_API_TIMEOUT=42\n",
                encoding="utf-8",
            )
            settings = load_scinet_api_settings(env_path)
        self.assertEqual(settings.base_url, "http://127.0.0.1:8000")
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.timeout, 42.0)

    def test_build_parser_parses_topic_task(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--task-type", "topic_trend_review", "--topic-text", "topic"])
        self.assertEqual(args.task_type, "topic_trend_review")
        self.assertEqual(args.topic_text, "topic")

    def test_scinet_request_defaults(self) -> None:
        request = SciNetRequest(task_type=TASK_RELATED_AUTHORS, input_payload={"idea_text": "idea"})
        self.assertEqual(request.task_type, TASK_RELATED_AUTHORS)
        self.assertEqual(request.input_payload["idea_text"], "idea")

    def test_authors_support_papers_strips_extra_author_fields(self) -> None:
        client = SciNetApiClient.__new__(SciNetApiClient)
        captured: dict[str, object] = {}

        def fake_request(path: str, payload: dict[str, object]) -> dict[str, object]:
            captured["path"] = path
            captured["payload"] = payload
            return {"status": "ok"}

        client._request = fake_request  # type: ignore[method-assign]

        client.authors_support_papers(
            query_text="knowledge-grounded evaluation",
            authors=[
                {"author_id": " A1 ", "name": " Alice ", "score": 0.9, "rank": 1},
                {"name": "Bob", "extra": "ignored"},
            ],
            options={"top_k_per_author": 2},
        )

        self.assertEqual(captured["path"], "/v1/authors/support-papers")
        self.assertEqual(
            captured["payload"],
            {
                "query_text": "knowledge-grounded evaluation",
                "authors": [
                    {"author_id": "A1", "name": "Alice"},
                    {"name": "Bob"},
                ],
                "options": {"top_k_per_author": 2},
            },
        )


if __name__ == "__main__":
    unittest.main()
