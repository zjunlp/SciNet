from __future__ import annotations

import argparse
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from innoeval_search.combined import merge_search


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        idea_text="test idea",
        pdf_path=None,
        kg_top_k=5,
        s2_top_k=7,
        s2_mode=None,
        s2_search_top_k=None,
        s2_recommend_top_k=None,
        s2_per_keyword_limit=3,
        include_s2_per_keyword_results=False,
        target_field=None,
        after=None,
        before=None,
        unable_title_ft=False,
        cache_path="/tmp/nonexistent-cache.json",
        disable_cache_reuse=True,
        reuse_cached_s2=False,
        env="/tmp/test.env",
        keyword_model="test-model",
        keyword_api_url="http://example.test",
        keyword_timeout=10,
        search_timeout=11,
        search_retries=1,
        recommendation_limit=9,
        grobid_base_url="http://127.0.0.1:8070",
        grobid_start_page=None,
        use_env_proxy=False,
        disable_llm_ranking=True,
        llm_model="test-llm",
        llm_timeout=10,
        llm_max_tokens=400,
        llm_temperature=0.1,
        llm_batch_size=4,
        llm_paper_coverage=2,
        llm_max_parallel=4,
        final_top_k=None,
        result_tag="test_merge_search",
        disable_result_log=True,
        pretty=False,
    )


class MergeSearchTests(unittest.TestCase):
    def test_run_combined_search_merges_two_successful_sources(self) -> None:
        args = make_args()
        kg_payload = [{"paper_id": "kg-1", "title": "KG Paper"}]
        s2_payload = {
            "input_type": "idea_text",
            "keywords": ["graph learning"],
            "papers": [{"paperId": "s2-1", "title": "S2 Paper"}],
        }

        with patch.object(merge_search, "run_kg_search", return_value=kg_payload), patch.object(
            merge_search.s2_search, "run_pipeline", return_value=s2_payload
        ):
            result = merge_search.run_combined_search(args)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["successful_source_count"], 2)
        self.assertEqual(result["failed_source_count"], 0)
        self.assertEqual(result["sources"]["kg"]["paper_count"], 1)
        self.assertEqual(result["sources"]["s2"]["paper_count"], 1)
        self.assertEqual(result["combined"]["paper_count"], 2)
        self.assertEqual(result["combined"]["papers"][0]["source"], "kg")
        self.assertEqual(result["combined"]["papers"][1]["source"], "s2")
        self.assertEqual(result["ranking"]["status"], "skipped")
        self.assertEqual(result["enabled_sources"], ["kg", "s2"])

    def test_run_combined_search_keeps_partial_result_when_one_source_fails(self) -> None:
        args = make_args()
        s2_payload = {
            "input_type": "idea_text",
            "keywords": ["retrieval"],
            "papers": [{"paperId": "s2-1", "title": "Only S2"}],
        }

        with patch.object(merge_search, "run_kg_search", side_effect=RuntimeError("neo4j down")), patch.object(
            merge_search.s2_search, "run_pipeline", return_value=s2_payload
        ):
            result = merge_search.run_combined_search(args)

        self.assertEqual(result["status"], "partial_error")
        self.assertEqual(result["successful_source_count"], 1)
        self.assertEqual(result["failed_source_count"], 1)
        self.assertEqual(result["sources"]["kg"]["status"], "error")
        self.assertEqual(result["sources"]["s2"]["status"], "ok")
        self.assertEqual(result["combined"]["paper_count"], 1)
        self.assertEqual(result["combined"]["papers"][0]["source"], "s2")

    def test_run_combined_search_supports_explicit_s2_only_mode(self) -> None:
        args = make_args()
        args.disable_kg = True
        s2_payload = {
            "input_type": "idea_text",
            "keywords": ["retrieval"],
            "papers": [{"paperId": "s2-1", "title": "Only S2"}],
        }

        with patch.object(merge_search.s2_search, "run_pipeline", return_value=s2_payload):
            result = merge_search.run_combined_search(args)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["enabled_sources"], ["s2"])
        self.assertEqual(result["sources"]["kg"]["status"], "disabled")
        self.assertEqual(result["sources"]["s2"]["status"], "ok")
        self.assertEqual(result["successful_source_count"], 1)
        self.assertEqual(result["disabled_source_count"], 1)

    def test_filter_and_group_papers_deduplicates_and_prefers_pdf_available_variants(self) -> None:
        combined = [
            {
                "source": "kg",
                "source_rank": 1,
                "paper": {
                    "id": "https://openalex.org/W1",
                    "title": "ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models",
                    "abstract": "Short KG abstract.",
                    "pdf_url": "https://kg.example/researchagent.pdf",
                    "doi": "https://doi.org/10.18653/v1/2025.naacl-long.342",
                    "cited_by_count": 3,
                    "publication_year": 2025,
                },
            },
            {
                "source": "s2",
                "source_rank": 1,
                "paper": {
                    "paperId": "s2-r1",
                    "title": "ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models",
                    "abstract": "Longer S2 abstract for the same paper.",
                    "openAccessPdf": {"url": "https://s2.example/researchagent.pdf"},
                    "externalIds": {"DOI": "10.18653/v1/2025.naacl-long.342"},
                    "citationCount": 9,
                    "year": 2025,
                },
            },
            {
                "source": "kg",
                "source_rank": 2,
                "paper": {
                    "id": "https://openalex.org/W2",
                    "title": "Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination",
                    "abstract": "KG version without pdf.",
                    "pdf_url": "",
                    "cited_by_count": 2,
                    "publication_year": 2024,
                },
            },
            {
                "source": "s2",
                "source_rank": 2,
                "paper": {
                    "paperId": "s2-s1",
                    "title": "Scideator: Human-LLM Scientific Idea Generation and Novelty Evaluation Grounded in Research-Paper Facet Recombination",
                    "abstract": "S2 version with pdf.",
                    "openAccessPdf": {"url": "https://s2.example/scideator.pdf"},
                    "citationCount": 4,
                    "year": 2024,
                },
            },
            {
                "source": "kg",
                "source_rank": 3,
                "paper": {
                    "id": "https://openalex.org/W3",
                    "title": "Completely Unusable Paper",
                    "abstract": "No pdf available.",
                    "pdf_url": "",
                    "cited_by_count": 0,
                    "publication_year": 2024,
                },
            },
        ]

        payload = merge_search.filter_and_group_papers(combined)

        self.assertEqual(payload["input_candidate_count"], 5)
        self.assertEqual(payload["duplicate_group_count"], 3)
        self.assertEqual(payload["unique_paper_count"], 2)
        self.assertEqual(payload["dropped_group_count_without_pdf"], 1)
        self.assertEqual(payload["dropped_member_count_without_pdf"], 2)

        unique_by_title = {item["title"]: item for item in payload["unique_papers"]}
        research_agent = unique_by_title[
            "ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models"
        ]
        self.assertEqual(research_agent["source"], "s2")
        self.assertEqual(research_agent["variant_count"], 2)

        scideator_title = (
            "Scideator: Human-LLM Scientific Idea Generation and Novelty Evaluation "
            "Grounded in Research-Paper Facet Recombination"
        )
        scideator = unique_by_title[scideator_title]
        self.assertEqual(scideator["source"], "s2")
        self.assertEqual(scideator["removed_variant_count_without_pdf"], 1)

    def test_build_scoring_batches_respects_requested_paper_coverage(self) -> None:
        batches = merge_search.build_scoring_batches(paper_count=10, batch_size=4, paper_coverage=2, seed=123)

        self.assertEqual(len(batches), 6)
        counts = Counter()
        for batch in batches:
            self.assertLessEqual(len(batch["paper_indices"]), 4)
            for paper_index in batch["paper_indices"]:
                counts[paper_index] += 1

        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(set(counts.keys()), set(range(10)))

    def test_run_combined_search_reuses_matching_cache_for_pdf_and_kg(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "cached.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            cache_path = tmp_path / "target.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "pdf_path": str(pdf_path),
                        "pdf": {
                            "title": "Cached Title",
                            "abstract": "Cached abstract.",
                            "body": "Cached body.",
                        },
                        "kg": {
                            "papers": [
                                {
                                    "id": "https://openalex.org/W9",
                                    "title": "Cached KG Paper",
                                    "abstract": "Cached KG abstract.",
                                    "pdf_url": "https://kg.example/cached.pdf",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = make_args()
            args.idea_text = None
            args.pdf_path = str(pdf_path)
            args.cache_path = str(cache_path)
            args.disable_cache_reuse = False

            captured: dict[str, object] = {}

            def fake_s2_run(s2_args: argparse.Namespace) -> dict[str, object]:
                captured["pre_extracted_pdf"] = s2_args.pre_extracted_pdf
                return {
                    "input_type": "pdf",
                    "pdf": s2_args.pre_extracted_pdf,
                    "papers": [
                        {
                            "paperId": "s2-1",
                            "title": "S2 Paper",
                            "abstract": "S2 abstract.",
                            "openAccessPdf": {"url": "https://s2.example/paper.pdf"},
                        }
                    ],
                }

            with patch.object(merge_search, "run_kg_search", side_effect=AssertionError("kg should not run")), patch.object(
                merge_search.s2_search, "run_pipeline", side_effect=fake_s2_run
            ):
                result = merge_search.run_combined_search(args)

            self.assertEqual(result["sources"]["kg"]["paper_count"], 1)
            self.assertEqual(result["sources"]["kg"]["papers"][0]["title"], "Cached KG Paper")
            self.assertIn("kg", result["cache"]["reused_sources"])
            self.assertTrue(result["cache"]["reused_pdf"])
            self.assertEqual(captured["pre_extracted_pdf"], {"title": "Cached Title", "abstract": "Cached abstract.", "body": "Cached body."})


if __name__ == "__main__":
    unittest.main()
