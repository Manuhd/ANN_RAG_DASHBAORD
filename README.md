

## ANN-Parameter-Controlled RAG System

### Hallucination Reduction & Cost-Aware Retrieval

This project implements a **Retrieval-Augmented Generation (RAG)** system where
**Approximate Nearest Neighbor (ANN) parameters** are explicitly controlled to
balance **retrieval accuracy, latency, hallucination risk, and LLM cost**.
A Streamlit dashboard provides visibility into ANN tuning effects, token usage,
and response quality.

---

## 🎯 Why ANN Parameter Control Matters

In large-scale RAG systems:

* Aggressive ANN settings → fast but may retrieve wrong context (hallucinations)
* Conservative ANN settings → accurate but slower and more expensive

This system exposes ANN parameters so retrieval behavior is **measurable,
tunable, and explainable** instead of being a black box.

---

## 📂 Project Structure

```
rag-system/
├── data/
│   └── loan_faq.csv
├── ingest.py                  # Offline embedding & index build
├── retriever.py               # ANN-based vector search
├── reranker.py                # Re-ranking for grounding
├── generator.py               # LLM answer generation
├── self_corrector.py          # Hallucination mitigation
├── metrics.py                 # Retrieval & quality metrics
├── ingestion_metrics.py       # Ingestion-time cost metrics
├── token_utils.py             # Token & cost estimation
├── logger.py                  # Structured logs
├── main.py                    # CLI-based RAG execution
├── app.py               # ANN & cost control dashboard
├── requirements.txt
└── README.md
```

---

## 🧠 Core Design Principle

> **Control retrieval quality before paying for generation.**

ANN parameters are tuned **before** LLM calls to:

* Reduce irrelevant context
* Minimize hallucinations
* Lower token usage
* Keep latency predictable

---

## 🔍 ANN Parameters Controlled

Typical ANN parameters exposed in this system (varies by index type):

| Parameter             | Purpose                        |
| --------------------- | ------------------------------ |
| `top_k`               | Number of candidates retrieved |
| `nprobe` / `efSearch` | Search depth vs recall         |
| `index_type`          | IVF / HNSW / Flat              |
| `distance_metric`     | Cosine / L2                    |
| `score_threshold`     | Drop low-confidence matches    |

These parameters directly impact:

* Recall quality
* Hallucination risk
* Token consumption
* End-to-end latency

---

## ⚙️ Setup

```bash
git clone https://github.com/Manuhd/RAG.git

cd RAG/ANN_RAG_DASHBAORD


```

---

## Step 1: Offline Ingestion (ANN Index Build)

Run ingestion **only when data or embedding strategy changes**:

```bash
python ingest.py
```

---

## Step 2: Runtime Query Execution

```bash
python main.py
```

Query-time flow:

1. Embed query
2. ANN retrieval using configured parameters
3. Re-ranking for grounding
4. Answer generation
5. Self-correction
6. Metric logging

---

## 📊 Step 3: ANN & Cost Control Dashboard

```bash
py streamlit run app.py
```

Dashboard enables:

* Live tuning of ANN parameters
* Monitoring recall vs latency
* Token usage per query
* Estimated LLM cost
* Detection of low-confidence retrievals

Access:

```
http://localhost:8501
```

---

## 🧪 Hallucination Control Strategy

| Layer                | Role                         |
| -------------------- | ---------------------------- |
| ANN parameter tuning | Controls candidate quality   |
| Re-ranking           | Improves relevance grounding |
| Threshold filtering  | Drops weak context           |
| Self-correction      | Refines uncertain answers    |
| Metrics              | Detects degradation          |

---

## 🚀 Use Cases

* Enterprise document search
* Financial & compliance Q&A
* Large knowledge bases
* Cost-sensitive LLM deployments



---

## Author

**Manu**
GenAI 
Just tell me 🚀
