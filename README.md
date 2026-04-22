# SciNet

<div align="center">
  <h1>SciNet: Literature-Grounded Research Workflows</h1>
</div>

<p align="center">
  Open-source client for running literature-grounded scientific research tasks on top of a hosted <code>SciNet API</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Tasks-5-success" alt="5 Tasks">
  <img src="https://img.shields.io/badge/Runtime-CLI-orange" alt="CLI">
</p>

---

## ✨ Overview

This repository provides a runnable client for several scientific research workflows, including idea evaluation, topic review, author discovery, author profiling, and idea generation.

The local client is responsible for:

- building a structured request
- calling a hosted `SciNet API`
- running client-side post-processing such as reranking, PDF parsing, grounding, and Markdown report generation

Users do **not** need to connect to Neo4j or other database components directly.

## 🔍 Scope

This repository is intended to be a lightweight, runnable demo client.

- `run_scinet.py` is the main entrypoint.
- `scinet/` contains the runnable workflow code.
- `references/search/` is a reference implementation for the standalone search stack and is **not** part of the main demo runtime.

## 🧩 Supported Tasks

| Task Type | Required Input | Main Output |
| --- | --- | --- |
| `grounded_review` | `--idea-text` or `--pdf-path` | grounded evidence, paragraph matches, and idea-level analysis |
| `topic_trend_review` | `--topic-text` | topic evolution summary and representative papers |
| `related_authors` | `--idea-text` or `--pdf-path` | related authors and supporting papers |
| `author_profile` | `--author-name` | research trajectory and representative works |
| `idea_generation` | `--topic-text` | generated ideas grounded in retrieved literature |

## 🏗️ Workflow

```text
Input -> Local Planning -> SciNet API Retrieval -> Local Post-processing -> JSON + Markdown Reports
```

Typical post-processing includes reranking, PDF extraction, evidence grounding, and response rendering.

## 📦 Installation

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Create the environment file

```bash
cp .env.example .env
```

### 2. Fill in the required variables

```env
SCINET_API_BASE_URL=https://your-scinet-api.example.com
SCINET_API_KEY=replace-me
SCINET_API_TIMEOUT=120

OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your-model-name

GROBID_BASE_URL=http://127.0.0.1:8070
OA_API_KEY=
OPENALEX_MAILTO=
```

### 3. Know what is required

| Variable | Required For | Notes |
| --- | --- | --- |
| `SCINET_API_BASE_URL` | all tasks | hosted `SciNet API` base URL |
| `SCINET_API_KEY` | all tasks | sent as `X-API-Key` |
| `OPENAI_API_KEY` | all tasks | used for planning and LLM summarization |
| `OPENAI_BASE_URL` | all tasks | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | all tasks | chat model name |
| `GROBID_BASE_URL` | PDF tasks | needed for `--pdf-path` flows |
| `OA_API_KEY` | optional | OpenAlex fallback support |
| `OPENALEX_MAILTO` | optional | OpenAlex contact email |

The code still accepts legacy `SCIMAP_*` and `KG2API_*` variables for compatibility, but new setups should use `SCINET_API_*`.

## 🚀 Quick Start

If you only want the shortest path to a working run:

### 1. Make sure the following services are ready

- a hosted `SciNet API`
- an OpenAI-compatible LLM endpoint
- GROBID if you want to use `--pdf-path`

### 2. Run a task

```bash
python3 run_scinet.py \
  --task-type topic_trend_review \
  --topic-text "research idea evaluation with large language models" \
  --pretty
```

### 3. Check the output

Each run creates a directory under `runs/` containing:

- `request.json`
- `result.json`
- `result.md`

## 🧪 Run Tasks

### `grounded_review`

```bash
python3 run_scinet.py \
  --task-type grounded_review \
  --idea-text "Use literature-grounded evidence to evaluate research ideas." \
  --pretty
```

With PDF input:

```bash
python3 run_scinet.py \
  --task-type grounded_review \
  --pdf-path /absolute/path/to/paper.pdf \
  --params-file examples/grounded_review_params.example.json \
  --pretty
```

### `topic_trend_review`

```bash
python3 run_scinet.py \
  --task-type topic_trend_review \
  --topic-text "research idea evaluation with large language models" \
  --pretty
```

### `related_authors`

```bash
python3 run_scinet.py \
  --task-type related_authors \
  --idea-text "knowledge-grounded evaluation of scientific research ideas" \
  --pretty
```

### `author_profile`

```bash
python3 run_scinet.py \
  --task-type author_profile \
  --author-name "Geoffrey Hinton" \
  --pretty
```

### `idea_generation`

```bash
python3 run_scinet.py \
  --task-type idea_generation \
  --topic-text "scientific idea generation with retrieval-augmented large language models" \
  --pretty
```

## 📁 Request Files

You can also run tasks from JSON request files in `examples/`:

```bash
python3 run_scinet.py --request-file examples/grounded_review_request.json --pretty
python3 run_scinet.py --request-file examples/topic_trend_review_request.json --pretty
python3 run_scinet.py --request-file examples/related_authors_request.json --pretty
python3 run_scinet.py --request-file examples/author_profile_request.json --pretty
python3 run_scinet.py --request-file examples/idea_generation_request.json --pretty
```

For `grounded_review`, you can also override model-related parameters with:

- `examples/grounded_review_params.example.json`
- `examples/grounded_review_params.cpu.example.json`

By default, `grounded_review` uses:

- embedding model: `BAAI/bge-large-en-v1.5`
- reranker model: `BAAI/bge-reranker-large`

The first run may download these models into the local Hugging Face cache.

## 🛠️ GROBID

GROBID is needed for:

- `grounded_review`
- `related_authors` when using `--pdf-path`

Example startup with Docker:

```bash
docker pull lfoppiano/grobid:latest
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:latest
curl http://127.0.0.1:8070/api/isalive
```

## 📂 Repository Layout

```text
.
├── run_scinet.py
├── scinet/
│   ├── cli.py
│   ├── core/
│   ├── llm/
│   ├── search/
│   ├── tasks/
│   ├── evidence/
│   └── renderers/
├── examples/
├── tests/
└── references/
    └── search/
```

Key directories:

- `scinet/core/`: shared config, schemas, and API client code
- `scinet/tasks/`: task dispatch and task-specific logic
- `scinet/evidence/`: PDF manifest building and evidence grounding
- `scinet/renderers/`: Markdown rendering
- `examples/`: runnable request examples
- `references/search/`: standalone search reference code

## ✅ Testing

```bash
python3 -m unittest discover -s tests
```

## 📝 TODO

- [ ] Add more user-facing CLI capabilities so downstream users and AI agents can invoke retrieval workflows without touching database internals.
- [ ] Package reusable agent skills for common scientific discovery workflows and expose best practices as easier-to-load components.
- [ ] Integrate more knowledge forms beyond paper-centric entities, such as datasets, code, standards, theorems, and experimental experience.
- [ ] Build dedicated benchmarks and evaluation protocols for downstream scientific research tasks supported by SciNet.
- [ ] Improve dynamic knowledge updates toward a more systematic and frequent refresh mechanism.
