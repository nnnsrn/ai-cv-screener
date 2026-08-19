# AI CV Screener Benchmark — Hybrid Architecture

This repository contains a portable Machine Learning experimentation and benchmarking framework for an **AI CV Screening pipeline**. It provides a side-by-side benchmark comparing:

1. **Legacy Pipeline**: Semantic Embedding Similarity (`BAAI/bge-m3`) + Document Similarity (`spaCy en_core_web_md`).
2. **Proposed Hybrid Pipeline**: Semantic + Similarity + **Structured Quantitative** (rule-based GPA, degree hierarchy, experience duration) + **Structured Qualitative** (LLM-based experience evaluation via Ollama `qwen2.5:7b-instruct`).

---

## 🏗️ Architecture Overview

The repository consists of two execution pathways:

* **Legacy Path**:
  * `src/legacy_scoring.py`: Computes semantic score using `BAAI/bge-m3` cosine similarity with scaled baseline transformation, alongside document similarity using `spaCy en_core_web_md`.
* **Proposed Hybrid Path**:
  * `src/ingestion.py`: PyMuPDF text extraction with PyTesseract OCR fallback, language detection (`langdetect`), and automatic English translation (`deep-translator`).
  * `src/parsers.py`: Structured JSON extraction via Ollama (`qwen2.5:7b-instruct`) validated with Pydantic schemas (`educations`, `experiences`, `certifications`).
  * `src/proposed_scoring.py`: Deterministic rule-based quantitative scoring (GPA, degree hierarchy, experience years) and qualitative LLM evidence evaluation against role requirements.
  * `src/aggregator.py`: Weighted score fusion combining Semantic, Similarity, and Structurized sub-scores into a normalized final candidate score (0.0 to 1.0).
  * `src/metrics.py`: Ranking correlation evaluation metrics using Kendall's Tau ($\tau$) and Spearman Rank Correlation ($\rho$) against recruiter ground-truth rankings.

---

## ⚙️ Environment Separation

This project explicitly separates the **Development Environment** from the **Experiment Environment**:

```text
ANTIGRAVITY (Development Workspace)
    │
    │ Scaffolding, implementation, module syntax & JSON notebook validation
    │ Git Commit & Push
    ▼
PRIVATE GITHUB REPOSITORY
    │
    │ Git Clone / Git Pull
    ▼
REMOTE GPU JUPYTER SERVER (Experiment Environment)
    │
    │ Install requirements.txt & run setup.sh
    │ Execute full ML benchmark notebook
    ▼
ACTUAL BENCHMARK EVALUATION (Kendall Tau, Spearman Rho, Latency)
```

* **Development (Antigravity)**: Used for scaffolding repository structure, implementing Python modules, writing the Jupyter notebook, syntax validation, and version control. Light-weight environment; full ML inference is not required to run locally here.
* **Experiment (Remote GPU Jupyter Server)**: High-performance GPU environment where Python dependencies, spaCy models, and Ollama `qwen2.5:7b-instruct` execute the full benchmarking pipeline to generate experimental results.

---

## 🚀 Installation & Setup on GPU Experiment Server

### 1. Repository Authentication & Cloning

Authenticate on your GPU server using GitHub CLI, SSH, or Git Credential Manager. **Do not embed Personal Access Tokens (PATs) directly into Git clone URLs.**

Using HTTPS via Git Credential Manager / GitHub CLI:
```bash
git clone https://github.com/nnnsrn/ai-cv-screener.git
cd ai-cv-screener
```

Or using SSH:
```bash
git clone git@github.com:nnnsrn/ai-cv-screener.git
cd ai-cv-screener
```

### 2. Environment Activation & Dependencies

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Model Download & Setup

Run the automated setup script to download the spaCy English model and pull the Ollama Qwen model:

```bash
bash setup.sh
```

Ensure the Ollama service is running in your GPU server background (`ollama serve`).

---

## 📊 Running the Benchmark

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open `notebooks/01_pipeline_benchmark.ipynb` and execute all cells sequentially.

The notebook will:
1. Detect compute device (`cuda` GPU or `cpu`).
2. Initialize models once (`BAAI/bge-m3`, `en_core_web_md`, Ollama client).
3. Execute the Legacy pipeline and record scores, ranks, and per-CV latency.
4. Execute the Proposed Hybrid pipeline and record structured scores, ranks, and per-CV latency.
5. Compute Kendall's Tau and Spearman Rho rank correlation against recruiter ground-truth rankings.
6. Display comparative summary DataFrames and answer key research efficiency questions.

---

## 🔬 Research Objective

The benchmark evaluates whether adding **Structured Quantitative Rules** and **Qualitative LLM Experience Scoring** improves agreement with recruiter ground-truth rankings sufficiently to justify the additional computational latency per CV over the Legacy vector-similarity approach.

---

## 🛡️ Security & Privacy

* **Outputs**: Generated benchmark results are stored locally in `outputs/` and excluded from Git tracking by `.gitignore`.
* **CV Data**: Real candidate CVs or personal information must never be committed to the repository. Synthetic/mock data is provided for benchmarking.
* **Secrets**: API tokens, credentials, and `.env` files are excluded from Git.

---

## 📁 Directory Structure

```text
ai-cv-screener/
├── src/
│   ├── __init__.py           # Package initializer
│   ├── ingestion.py          # PyMuPDF, PyTesseract OCR, langdetect & translation
│   ├── parsers.py            # Legacy summary & Ollama structured JSON parsing with Pydantic
│   ├── legacy_scoring.py     # BGE-M3 semantic scoring & SpaCy document similarity
│   ├── proposed_scoring.py   # Quantitative rules (GPA, degree, exp) & Qualitative LLM scoring
│   ├── aggregator.py         # Configurable weighted score aggregation
│   └── metrics.py            # Kendall's Tau & Spearman correlation helpers
├── notebooks/
│   └── 01_pipeline_benchmark.ipynb  # End-to-end ML benchmark notebook
├── data/
│   └── .gitkeep              # Placeholder for local CV files (ignored)
├── outputs/                  # Generated benchmark outputs (ignored)
├── requirements.txt          # Target environment Python packages
├── setup.sh                  # Shell script for spaCy & Ollama model setup
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation
```
