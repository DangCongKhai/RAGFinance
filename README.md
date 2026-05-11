## FinanceRAG (ACM-ICAIF'24) – Retrieval Experiments

This repository is a personal experimentation sandbox for the **ACM-ICAIF'24 FinanceRAG Challenge (Task 1: Retrieval)**. It implements and compares multiple retrieval pipelines for financial QA benchmarks (e.g., FinQA, TATQA, FinanceBench, ConvFinQA, FinQABench, MultiHiertt, FinDER), focusing on improving **nDCG@10** on the competition leaderboard.

- **Competition**: [ACM-ICAIF '24 FinanceRAG Challenge (Kaggle)](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge)

### What’s inside

- **Retrievers**
  - Dense retrieval (FAISS + `GoogleGenerativeAIEmbeddings`)
  - BM25 (`rank_bm25`)
  - Dense on **split/chunked** documents
- **Reranking**
  - Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- **Graph-based retrieval**
  - HippoRAG variants (`financerag/hipporag/` and `financerag/hipporag_v2/`)
- **Notebook**
  - `finance.ipynb` contains most experiment orchestration and submission file generation.

### Setup

#### Requirements

- Python 3.10+ recommended (tested with Python 3.11)

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
GOOGLE_API_KEY=<your_google_api_key_here>
PINECONE_API_KEY=<your_pinecone_api_key_here>
```

### Experiments tried

The main ideas explored so far:

- **Dense retrieval baseline**: embed query + corpus, retrieve top-\(k\) via vector similarity.
- **Dense retrieval on split documents**: chunk long documents to improve recall.
- **BM25 baseline**: lexical retrieval only.
- **BM25 + reranker**: rerank BM25 candidates using a cross-encoder.
- **Dense + reranker**: rerank dense candidates using a cross-encoder.
- **BM25 + corpus summarization + reranker**: summarize corpus text then rerank (performed poorly in this run).
- **HippoRAG**: knowledge-graph-assisted retrieval (work-in-progress in this repo).

### Results (nDCG@10)

Leaderboard scores from your runs:

| Approach | Public nDCG@10 | Private nDCG@10 |
|---|---:|---:|
| Dense Retrieval only | 0.32004 | 0.28345 |
| Dense Retrieval on split documents | **0.35434** | **0.32395** |
| BM25 only | 0.31839 | 0.29124 |
| BM25 + Reranker | 0.32322 | 0.29414 |
| BM25 + Corpus_Summarization + Reranker | 0.16984 | 0.15283 |
| Dense Retrieval + Reranker | 0.34384 | 0.31264 |
| HippoRAG | _in progress_ | _in progress_ |

### How to run (high level)

- Open and run `finance.ipynb` to:
  - load datasets (expects a dataset folder like `finance_dataset/` or `finance_subset_dataset/`)
  - build retrieval indices
  - run retrieval / reranking
  - export a submission CSV (see notebook cell that writes `submission_<method_name>.csv`)


