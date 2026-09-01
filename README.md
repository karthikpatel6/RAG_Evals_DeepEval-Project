# RAG Eval Project

A focused RAG evaluation project built to measure how well a course-assistant system answers questions grounded in transcript data, while also checking for scope adherence, leakage, toxicity, and retrieval quality.

This repository was designed as a practical evaluation sandbox for a Retrieval-Augmented Generation (RAG) pipeline over course content extracted from VTT transcripts. The goal is not just to build a working pipeline, but to evaluate it systematically using DeepEval and a set of golden datasets.

---

## Current evaluation results

The current benchmark results for the course-assistant RAG pipeline are:

- Recall: 98%
- Precision: 85%
- Faithfulness: 93%
- Answer Relevancy: 90%
- Contextual Relevancy: 83%
- Correctness: 83%
- Completeness: 75%
- Style: 74%
- Toxicity: 0
- PII Leakage: 96% success
- Scope Adherence: 99%
- Latency (P95): 5.2 seconds
- Cost per query: 2 paise

These results indicate strong retrieval quality, high faithfulness, and excellent scope and safety performance, while highlighting opportunities to improve completeness, style quality, and contextual relevance in the generated answers.

---

## What this project does

The project creates a full RAG workflow:

- loads course transcript data from the `data/` folder
- chunks and stores it in a local Chroma vector database
- retrieves the most relevant chunks for a query
- reranks the retrieved results with a cross-encoder
- generates a grounded answer from the retrieved context
- evaluates the system across multiple dimensions using DeepEval

The evaluation suite checks both component-level and end-to-end behavior:

- **retrieval quality** — measures whether the retriever finds the right supporting chunks for the query; it is used when validating search and ranking pipelines before trusting downstream answers.
- **grounding and faithfulness** — checks whether each claim in the answer is supported by the retrieved context; it is used whenever the system must provide evidence-backed responses and avoid hallucinations.
- **answer relevance** — measures how directly the response addresses the user’s question; it is used to catch generic, off-topic, or partially answered outputs.
- **correctness** — checks whether the answer matches the expected factual content; it is used when comparing generated output to an ideal answer or ground-truth fact set.
- **completeness** — evaluates whether the answer covers all key points expected from the ideal response; it is used when user satisfaction depends on coverage, not just correctness.
- **scope adherence** — verifies the assistant stays within its allowed role and does not perform out-of-scope tasks; it is used for safety, policy compliance, and role-controlled assistant behavior.
- **prompt and content leakage** — tests whether hidden system prompts, internal instructions, or protected course content are exposed; it is used when protecting proprietary instructions or educational material.
- **PII leakage** — checks whether personal or sensitive information is revealed in the output; it is used anytime the system may handle user data, credentials, or private identifiers.
- **toxicity** — evaluates the tone and safety of the generated answer; it is used to detect abusive, hateful, threatening, or otherwise unsafe output before deployment.

---

## Project architecture

### RAG pipeline

The main system lives in:

- `src/retriever.py` — loads transcript documents, builds a Chroma vector store, creates a retriever
- `src/reranker.py` — over-retrieves candidate chunks and reranks them with a cross-encoder
- `src/generator.py` — generates the final answer from the retrieved context
- `src/rag_pipeline.py` — orchestrates retrieve + rerank + generate into one usable pipeline

### Evaluation modules

Each `evals/*.py` file focuses on a specific metric or evaluation goal:

- `evals/eval_retriever.py` — retrieval metrics using contextual precision and recall
- `evals/eval_generator.py` — generator metrics for faithfulness and answer relevancy
- `evals/eval_rag_pipeline.py` — end-to-end RAG metrics for contextual relevance, faithfulness, and answer relevancy
- `evals/eval_application.py` — correctness and completeness for the full application behavior
- `evals/eval_scope.py` — scope adherence to ensure the assistant stays inside its role
- `evals/eval_leakage.py` — prompt leakage, course-content leakage, and PII leakage checks
- `evals/eval_toxicity.py` — toxicity evaluation for generated answers
 - `evals/eval_latency.py` — measures response latency and pipeline timing characteristics
 - `evals/eval_cost.py` — estimates API/model cost for evaluation runs and per-query cost
 - `evals/eval_reliability.py` — checks stability and repeatability of answers across runs

---

## Data and golden sets

The benchmark data is stored in the following folders:

- `data/` — raw VTT transcripts
- `goldens/` — evaluation datasets used for scoring the model

Examples include:

- `goldens/retriever_goldens.json`
- `goldens/faithfulness_dataset.json`
- `goldens/correctness_goldens.json`
- `goldens/scope_goldens.json`
- `goldens/leakage_goldens.json`
- `goldens/toxicity_goldens.json`

These goldens are used to simulate realistic prompts and assess how the assistant behaves under both normal and adversarial situations.

---

## Retrieval and generation stack

### Retrieval

The retriever uses:

- Chroma as the vector store
- Hugging Face embeddings (`BAAI/bge-large-en-v1.5`)
- transcript splitting via `RecursiveCharacterTextSplitter`

### Reranking

The reranker uses:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- over-retrieval followed by reranking for higher precision

### Generation

The generator uses:

- LangChain + Groq
- a course-teaching assistant prompt
- retrieval-grounded generation rules that prefer evidence from the provided context

This means the model is prompted to answer only from the retrieved context and abstain when the contents do not support an answer.

---

## Evaluation dimensions covered

### 1. Retriever quality

Measured in `evals/eval_retriever.py` using:

- `ContextualRecallMetric`
- `ContextualPrecisionMetric`

These assess whether the retriever actually brings back relevant context and whether the ranking is precise.

### 2. Generator faithfulness and answer relevance

Measured in `evals/eval_generator.py` and `evals/eval_rag_pipeline.py` using:

- `FaithfulnessMetric`
- `AnswerRelevancyMetric`
- `ContextualRelevancyMetric`

This checks whether the answer is grounded in the provided context and whether it addresses the user’s request.

### 3. Correctness and completeness

Handled in `evals/eval_application.py` using G-Eval style rubrics:

- correctness: factual alignment with the expected output
- completeness: coverage of the expected key points

This is a strong judge for whether the final answer is not just plausible, but actually correct and comprehensive.

### 4. Scope adherence

Handled in `evals/eval_scope.py`.

The project evaluates whether the assistant stays in its role as a teaching assistant and does not accept or perform unrelated general-purpose tasks. This covers:

- answer cases
- decline cases
- partial response cases
- jailbreak / roleplay attempt handling

### 5. Leakage

Handled in `evals/eval_leakage.py`.

This checks for:

- prompt leakage
- course-content leakage
- PII leakage

This is especially important for agentic or educational systems that must protect internal instructions and proprietary course material.

### 6. Toxicity

Handled in `evals/eval_toxicity.py`.

This checks whether the generated answer contains abusive, hateful, threatening, or otherwise toxic language.

---

### 7. Latency

Handled in `evals/eval_latency.py`.

This measures end-to-end response latency and timing breakdowns across the pipeline (retrieval, rerank, generation). It's useful for spotting performance regressions and understanding per-query time costs.

### 8. Cost

Handled in `evals/eval_cost.py`.

This estimates model/API cost for evaluation runs and provides per-query cost approximations to help balance quality against budget.

### 9. Reliability

Handled in `evals/eval_reliability.py`.

This checks the stability and repeatability of generated answers across multiple runs and random seeds, helping identify brittle or nondeterministic behavior.


## Why this project matters

The key idea is to move beyond “does the model answer something vaguely plausible?” and instead ask:

- Is the answer grounded in the retrieved evidence?
- Is it factually correct?
- Does it stay within scope?
- Does it leak hidden instructions or course material?
- Is it safe and respectful?
- Is the retriever actually finding the right information?

That is the core of a serious RAG evaluation workflow.

---

## Repository structure

```text
RAG_Eval_Project/
├── data/                             # VTT transcripts
├── goldens/                          # evaluation goldens and synthetic datasets
├── chroma_store/                     # persisted local vector database
├── evals/                            # all evaluation scripts
│   ├── eval_application.py
│   ├── eval_generator.py
│   ├── eval_leakage.py
│   ├── eval_rag_pipeline.py
│   ├── eval_retriever.py
│   ├── eval_scope.py
│   ├── eval_toxicity.py
│   └── __init__.py
├── src/                              # RAG components
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   ├── retriever.py
│   └── __init__.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── .env.example (if used locally)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or with the project config:

```bash
pip install -e .
```

### 3. Configure environment variables

The project uses environment variables for model access and runtime configuration. Typical keys include:

```bash
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key  # if used by a judge model
HF_TOKEN=your_hf_token  # optional for faster model downloads
```

You can place these in a local `.env` file at the repository root.

---

## Running the evaluations

Each evaluation is designed to be run as a module, for example:

```bash
python -m evals.eval_retriever
python -m evals.eval_generator
python -m evals.eval_rag_pipeline
python -m evals.eval_application
python -m evals.eval_scope
python -m evals.eval_leakage
python -m evals.eval_toxicity
python -m evals.eval_latency
python -m evals.eval_cost
python -m evals.eval_reliability
```

These scripts use `DeepEval` to score the model answers against the golden datasets and evaluation metrics.

---

## Notes on usage

This project is intentionally a research and evaluation playground rather than a production API service. It is useful for:

- testing RAG quality on a real dataset
- comparing different retrieval strategies
- auditing model safety and compliance behavior
- measuring answer quality with deterministic evals
- exploring LLM judge behavior in a practical setting

---

## Tech stack

- Python
- LangChain
- ChromaDB
- Hugging Face embeddings
- SentenceTransformers cross-encoder
- Groq models
- DeepEval
- dotenv

---

## Future improvements

Possible next steps for this repository include:

- adding automated benchmark reporting summaries
- saving evaluation results to CSV/JSON dashboards
- comparing several retrievers and rerankers against the same golden dataset
- implementing a small CLI wrapper for running benchmark suites
- adding CI checks for reproducible evaluation runs

---

## Summary

This project is a complete RAG evaluation workflow for a course-assistant application, designed to answer the question: “How good is the system, really?”

It does not just test whether the app returns a response — it checks whether the answer is:

- retrieved correctly
- grounded in evidence
- relevant
- correct
- complete
- in-scope
- non-leaky
- safe
- not toxic

That makes it a strong example of practical, production-minded LLM evaluation work in a real repository.
