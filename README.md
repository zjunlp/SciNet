<div align="center">
  <h1>SciNet: A Large-Scale Knowledge Graph for Automated Scientific Research</h1>
</div>

<p align="center">
  🌐 <strong>English</strong> · <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
<<<<<<< HEAD
  <a href="docs/api/SCINET_API_DOC.html">📚 API Docs Website</a>
=======
  <a href="https://huadongjian.github.io/SciNet/api/SCINET_API_DOC.html">📚 API Docs Website</a>
>>>>>>> 3d1a104 (Update docs and polish downstream frontend outputs)
</p>

<p align="center">
  A pip-installable client and CLI for literature-grounded scientific research workflows on top of the hosted SciNet API.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.14367">📄 arXiv</a>
  ·
  <a href="http://scinet.openkg.cn/register">🔑 Get API Token</a>
  ·
  <a href="http://scinet.openkg.cn/healthz">🩺 API Health</a>
</p>

<p align="center">
  <a href="https://github.com/zjunlp/SciNet">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://github.com/zjunlp/SciNet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/github/last-commit/zjunlp/SciNet?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>

---

## ✨ Overview

SciNet is a research map you can use from the command line. Give it a topic, an idea, an author, or a paper trail, and it helps you look up literature, gather graph-backed evidence, and turn the result into readable reports and reusable JSON artifacts.

Behind that simple workflow is a large scientific knowledge graph. SciNet connects papers, authors, institutions, venues, keywords, citations, and a four-level research taxonomy from domains down to topics. That means a search is not limited to matching words: it can follow how research areas, people, concepts, and papers relate to one another.

This repository packages that capability as a lightweight **SciNet client**. New users can install it with `pip`, register an API token, and start running literature-grounded research tasks without setting up Neo4j, maintaining graph data, or touching backend infrastructure.

<p align="center">
  <img src="imgs/field_distribution_pie.png" alt="SciNet field distribution across research areas" width="92%">
</p>

<p align="center">
  <em>SciNet spans a broad research landscape, from medicine and social sciences to engineering, computer science, materials science, mathematics, and more.</em>
</p>

<p align="center">
  <img src="imgs/schema.png" alt="SciNet knowledge graph schema" width="92%">
</p>

<p align="center">
  <em>The graph links papers with authors, institutions, sources, keywords, citations, related work, and the domain-field-subfield-topic hierarchy.</em>
</p>

With the client, you can:

- search for papers with keyword, semantic, title, reference, and graph-aware retrieval;
- run research workflows such as literature review, idea grounding, idea evaluation, idea generation, trend analysis, related-author retrieval, and researcher profiling;
- save reproducible outputs such as `request.json`, `response.json`, `summary.txt`, and `report.md`;
- customize downstream workflows through editable CLI **skills**.

---

## 📑 Table of Contents

- [✨ Overview](#-overview)
<<<<<<< HEAD
- [🚀 Quick Start](#-quick-start)
- [🔑 API Token](#-api-token)
- [🧠 What SciNet Does](#-what-scinet-does)
- [🧩 Supported Tasks](#-supported-tasks)
- [🛠️ CLI-First Workflow](#-cli-first-workflow)
- [🧰 Editable Skills](#-editable-skills)
- [🐍 Python SDK](#-python-sdk)
- [⚙️ Configuration](#-configuration)
- [🧪 Examples](#-examples)
- [📦 Outputs and Artifacts](#-outputs-and-artifacts)
- [🛠️ GROBID for PDF Workflows](#-grobid-for-pdf-workflows)
- [📂 Repository Layout](#-repository-layout)
- [🧯 Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap](#-roadmap)
- [✍️ Citation](#-citation)
=======
- [� Table of Contents](#-table-of-contents)
- [🚀 Quick Start](#-quick-start)
  - [1. Install](#1-install)
  - [2. Register an API Token](#2-register-an-api-token)
  - [3. Configure](#3-configure)
  - [4. Test](#4-test)
  - [5. Run a Paper Search](#5-run-a-paper-search)
- [🔑 API Token](#-api-token)
  - [Browser Registration](#browser-registration)
  - [Check Token Status](#check-token-status)
  - [Check Usage](#check-usage)
- [🧠 What SciNet Does](#-what-scinet-does)
- [🧩 Supported Tasks](#-supported-tasks)
- [🛠️ CLI-First Workflow](#️-cli-first-workflow)
  - [Help](#help)
  - [Basic Retrieval](#basic-retrieval)
  - [Retrieval Modes](#retrieval-modes)
  - [Expert Anchors](#expert-anchors)
  - [Graph Bias Parameters](#graph-bias-parameters)
- [🧰 Editable Skills](#-editable-skills)
- [🐍 Python SDK](#-python-sdk)
- [⚙️ Configuration](#️-configuration)
- [🧪 Examples](#-examples)
  - [Literature Review](#literature-review)
  - [Idea Evaluation](#idea-evaluation)
  - [Idea Generation](#idea-generation)
  - [Trend Report](#trend-report)
  - [Researcher Review](#researcher-review)
- [📦 Outputs and Artifacts](#-outputs-and-artifacts)
- [🛠️ GROBID for PDF Workflows](#️-grobid-for-pdf-workflows)
- [📂 Repository Layout](#-repository-layout)
- [🧯 Troubleshooting](#-troubleshooting)
  - [`scinet health` works but `search-papers` returns 401](#scinet-health-works-but-search-papers-returns-401)
  - [No email verification code](#no-email-verification-code)
  - [Retrieval is slow or times out](#retrieval-is-slow-or-times-out)
  - [`scinet` command is not found on Windows](#scinet-command-is-not-found-on-windows)
- [📝 TODO](#-todo)
- [✍️ Citation](#️-citation)
>>>>>>> 3d1a104 (Update docs and polish downstream frontend outputs)
- [📄 License](#-license)

---

## 🚀 Quick Start

### 1. Install

Install directly from GitHub:

```bash
pip install "git+https://github.com/zjunlp/SciNet.git#subdirectory=scinet"
```

For isolated CLI usage:

```bash
pipx install "git+https://github.com/zjunlp/SciNet.git#subdirectory=scinet"
```

After installation:

```bash
scinet -h
```

### 2. Register an API Token

Open:

```text
http://scinet.openkg.cn/register
```

Complete email verification and copy your personal token.

### 3. Configure

Linux / macOS:

```bash
export SCINET_API_BASE_URL="http://scinet.openkg.cn"
export SCINET_API_KEY="your-personal-scinet-token"
```

Windows CMD:

```bat
set SCINET_API_BASE_URL=http://scinet.openkg.cn
set SCINET_API_KEY=your-personal-scinet-token
```

### 4. Test

```bash
scinet health
scinet config
```

### 5. Run a Paper Search

```bash
scinet search-papers \
  --query "open world agent" \
  --keyword "high:open world agent" \
  --top-k 3
```

---

## 🔑 API Token

SciNet uses personal API tokens for public access.

### Browser Registration

Visit:

```text
http://scinet.openkg.cn/register
```

Steps:

1. enter your name, email, organization, and use case;
2. click **Send code**;
3. check your inbox for the verification code;
4. enter the code and create a token;
5. copy the returned `scinet_xxx` token.

The token is shown only once.

### Check Token Status

```bash
curl -H "Authorization: Bearer $SCINET_API_KEY" \
  http://scinet.openkg.cn/v1/auth/token/status
```

### Check Usage

```bash
curl -H "Authorization: Bearer $SCINET_API_KEY" \
  "http://scinet.openkg.cn/v1/auth/usage?days=7"
```

---

## 🧠 What SciNet Does

SciNet is built for research workflows rather than plain keyword search.

1. **Search + KG Retrieval**: retrieve papers using keywords, semantic matching, title anchors, references, and graph propagation.
2. **Research Workflow Automation**: run literature review, idea grounding, idea evaluation, idea generation, trend analysis, related-author retrieval, and researcher profiling.
3. **Agent-Friendly Outputs**: every run keeps machine-readable JSON artifacts and a human-readable Markdown report.
4. **Editable Skills**: downstream workflows can be represented as JSON skills, inspected, copied, modified, and invoked through CLI.

---

## 🧩 Supported Tasks

| Command | Scenario | Main Output |
|---|---|---|
| `scinet search-papers` | Paper search | Related papers and Markdown report |
| `scinet related-authors` | Related-author discovery | Candidate authors and scores |
| `scinet author-papers` | Author paper lookup | Papers by a specified author |
| `scinet support-papers` | Support-paper retrieval | Evidence papers for candidate authors |
| `scinet paper-search` | Lightweight low-level paper search | Fast paper candidates |
| `scinet literature-review` | Literature review | Core paper pool, timeline, writing hints |
| `scinet idea-grounding` | Idea grounding | Similar works and differentiation evidence |
| `scinet idea-evaluate` | Idea evaluation | Evidence for novelty, feasibility, and soundness |
| `scinet idea-generate` | Idea generation | Topic combinations and idea seeds |
| `scinet trend-report` | Trend analysis | Evolution evidence and representative works |
| `scinet researcher-review` | Researcher background review | Research trajectory and representative works |
| `scinet skill` | Editable skill registry | Reusable workflow presets |

---

## 🛠️ CLI-First Workflow

SciNet is CLI-first. The CLI is the primary interface for both users and AI agents.

### Help

```bash
scinet -h
scinet search-papers -h
scinet literature-review -h
scinet skill -h
```

### Basic Retrieval

```bash
scinet search-papers \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --top-k 3 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0
```

### Retrieval Modes

| Mode | Meaning | Best For |
|---|---|---|
| `keyword` | Keyword-driven KG retrieval | Clear terminology |
| `semantic` | Semantic retrieval | Broad semantic matching |
| `title` | Title-anchor retrieval | Known paper titles |
| `hybrid` | Keyword + semantic + title + graph walk | Default and recommended |

If `--retrieval-mode` is omitted, SciNet uses `hybrid`.

### Expert Anchors

```bash
--keyword "high:open world agent"
--title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models"
--reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models"
```

### Graph Bias Parameters

| Parameter | Meaning |
|---|---|
| `--bias-keyword` | Keyword association strength |
| `--bias-non-seed-keyword` | Non-seed keyword expansion |
| `--bias-citation` | Citation edge strength |
| `--bias-related` | Paper relatedness strength |
| `--bias-authorship` | Author-paper relation strength |
| `--bias-coauthorship` | Coauthor network strength |
| `--bias-cooccurrence` | Keyword co-occurrence strength |
| `--bias-exploration` | Graph exploration level |
| `--ranking-profile` | Ranking preference: `precision`, `balanced`, `discovery`, `impact` |

Recommended safe defaults:

```bash
--top-k 3
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

---

## 🧰 Editable Skills

SciNet skills are JSON presets for downstream research workflows. They make complex workflows easier to inspect, reuse, and customize.

```bash
scinet skill list
scinet skill show literature-review
scinet skill run literature-review --query "open world agent" --keyword "high:open world agent"
scinet skill run --dry-run literature-review --query "open world agent" --keyword "high:open world agent"
```

Create a custom skill:

```bash
scinet skill init my-review --from literature-review
```

This creates:

```text
./skills/my-review.json
```

Edit it, then run:

```bash
scinet skill run my-review --query "your topic"
```

User-defined skills are loaded from:

1. `./skills/*.json`
2. `~/.scinet/skills/*.json`
3. directories specified by `SCINET_SKILLS_DIR`

User-defined skills can override built-in skills with the same name.

---

## 🐍 Python SDK

SciNet also provides a lightweight Python client.

```python
from scinet import SciNetClient

client = SciNetClient()

print(client.health())

result = client.search_papers(
    query="open world agent",
    keywords=[{"text": "open world agent", "score": 10}],
    top_k=3,
)

print(result)
```

You can also pass credentials directly:

```python
from scinet import SciNetClient

client = SciNetClient(
    base_url="http://scinet.openkg.cn",
    api_key="your-personal-scinet-token",
)

print(client.token_status())
```

---

## ⚙️ Configuration

```env
SCINET_API_BASE_URL=http://scinet.openkg.cn
SCINET_API_KEY=your-personal-scinet-token
SCINET_TIMEOUT=900
SCINET_RUNS_DIR=./runs
```

Optional compatibility variables:

```env
KG2API_BASE_URL=http://scinet.openkg.cn
KG2API_API_KEY=your-personal-scinet-token
```

For new setups, prefer `SCINET_*`.

---

## 🧪 Examples

### Literature Review

```bash
scinet literature-review \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 5
```

### Idea Evaluation

```bash
scinet idea-evaluate \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:idea evaluation" \
  --keyword "middle:LLM as a judge" \
  --top-k 3
```

### Idea Generation

```bash
scinet idea-generate \
  --query "knowledge editing for large language models" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:knowledge editing" \
  --keyword "middle:large language models" \
  --keyword "low:continual learning" \
  --top-k 5
```

### Trend Report

```bash
scinet trend-report \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --keyword "middle:knowledge graph" \
  --top-k 5
```

### Researcher Review

```bash
scinet researcher-review \
  --author "Yoshua Bengio" \
  --limit 10 \
  --no-abstract
```

---

## 📦 Outputs and Artifacts

Terminal output is concise and table-based. Full outputs are saved under:

```text
runs/<run_id>/
```

Common artifacts:

| File | Description |
|---|---|
| `plan.json` | Structured search plan |
| `request.json` | Full request sent to SciNet API |
| `response.json` | Raw backend response |
| `summary.txt` | Short summary |
| `report.md` | User-facing Markdown report |
| `metadata.json` | Runtime metadata |

---

## 🛠️ GROBID for PDF Workflows

GROBID extracts structured metadata from scientific PDFs, including titles, authors, abstracts, and references. It is only needed for PDF-based workflows.

```bash
docker pull lfoppiano/grobid:latest
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:latest
curl http://127.0.0.1:8070/api/isalive
```

Then configure:

```env
GROBID_BASE_URL=http://127.0.0.1:8070
```

---

## 📂 Repository Layout

```text
SciNet/
  pyproject.toml
  README.md
  README_zh.md
  README_skills.md
  .env.example
  src/
    scinet/
      __init__.py
      cli.py
      client.py
      config.py
      skills.py
      builtin_skills.json
  examples/
    search_papers.sh
    literature_review.sh
    idea_evaluate.sh
  tests/
    test_import.py
  references/
    search/
```

---

## 🧯 Troubleshooting

### `scinet health` works but `search-papers` returns 401

Your token is missing or invalid.

```bash
echo $SCINET_API_KEY
export SCINET_API_KEY="your-personal-scinet-token"
```

Windows CMD:

```bat
set SCINET_API_KEY=your-personal-scinet-token
```

### No email verification code

Check the email address, spam folder, and resend interval.

### Retrieval is slow or times out

Use lightweight settings:

```bash
--top-k 3
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

### `scinet` command is not found on Windows

Use the virtual environment executable directly:

```bat
.venv\Scripts\scinet.exe -h
```

or reinstall:

```bat
.venv\Scripts\python.exe -m pip install -e .
```

---

<<<<<<< HEAD
## 🗺️ Roadmap
=======
## 📝 TODO
>>>>>>> 3d1a104 (Update docs and polish downstream frontend outputs)

- [ ] **CLI Tools.** Add more user-facing CLI capabilities so downstream users and AI agents can invoke retrieval workflows without touching database internals.
- [ ] **Skills.** Package reusable agent skills for common scientific discovery workflows and expose best practices as easier-to-load components.
- [ ] **More Knowledge.** Integrate more knowledge forms beyond paper-centric entities, such as datasets, code, standards, theorems, and experimental experience.
- [ ] **Benchmark and Evaluation.** Build dedicated benchmarks and evaluation protocols for downstream scientific research tasks supported by SciNet.
- [ ] **Dynamic Update**Improve dynamic knowledge updates toward a more systematic and frequent refresh mechanism.
- [ ] **Dynamic Update.** Improve dynamic knowledge updates toward a more systematic and frequent refresh mechanism.

---

## ✍️ Citation

If you find SciNet helpful, please cite:

```

```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
