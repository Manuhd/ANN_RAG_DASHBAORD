import streamlit as st
import os
import pandas as pd
import time

from retriever import load_data, build_index, retrieve
from reranker import rerank
from generator import generate_answer
from self_corrector import self_correct
from metrics import hallucination_risk
from logger import save_to_csv
from ingestion_metrics import ingestion_token_stats_from_df
from token_utils import query_token_stats


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG Hallucination Dashboard",
    layout="wide"
)

st.title("🧠 Gemini RAG – Hallucination Control Dashboard")


# ---------------- VECTOR INDEX CONFIG ----------------
st.subheader("⚙️ Vector Index Configuration")

index_type = st.selectbox(
    "FAISS Index Type",
    ["flat", "hnsw", "ivf"],
    help="Choose index type for vector search"
)


# ---------------- LOAD DATA & BUILD INDEX ----------------
@st.cache_resource
def setup(index_type):
    df = load_data()
    index, embeddings = build_index(df, index_type=index_type)
    return df, index

df, index = setup(index_type)

# Runtime tuning
if index_type == "hnsw":
    index.hnsw.efSearch = st.slider(
        "HNSW efSearch",
        min_value=10,
        max_value=200,
        value=50
    )

if index_type == "ivf":
    index.nprobe = st.slider(
        "IVF nprobe",
        min_value=1,
        max_value=10,
        value=3
    )


# ---------------- INGESTION METRICS ----------------
st.subheader("📦 Ingestion Token Metrics (One-Time Cost)")

ingestion_stats = ingestion_token_stats_from_df(df)

m1, m2, m3 = st.columns(3)
m1.metric("Documents (Rows)", ingestion_stats["docs"])
m2.metric("Chunks Created", ingestion_stats["chunks"])
m3.metric("Total Chunk Tokens", f"{ingestion_stats['total_chunk_tokens']:,}")

st.caption("🔹 Embedding cost is paid once during ingestion.")


# ---------------- USER INPUT ----------------
query = st.text_input("Ask a question:")
ask = st.button("Ask")


# ---------------- RECALL@K FUNCTION ----------------
def recall_at_k(retrieved_df, ground_truth_answer):
    if ground_truth_answer is None:
        return None
    return int(
        ground_truth_answer.strip()
        in retrieved_df["answer"].values
    )


# ---------------- MAIN PIPELINE ----------------
if ask and query:
    start_time = time.time()

    # ---------- RETRIEVAL ----------
    docs = retrieve(query, df, index, top_k=3)

    # ---------- RERANKING ----------
    reranked_docs = rerank(query, docs)
    top_context = reranked_docs.head(1)

    retrieved_context_text = "\n".join(top_context["answer"].tolist())

    # ---------- GENERATION ----------
    answer = generate_answer(retrieved_context_text, query)

    # ---------- SELF-CORRECTION ----------
    final_answer, corrected, faithfulness = self_correct(
        answer, top_context
    )

    latency = round(time.time() - start_time, 2)
    risk = hallucination_risk(faithfulness)

    # ---------- GROUND TRUTH ----------
    gt_row = df[df["question"] == query]
    gt_answer = gt_row["answer"].values[0] if len(gt_row) > 0 else None

    recall_before = recall_at_k(docs, gt_answer)
    recall_after = recall_at_k(reranked_docs, gt_answer)

    # ---------- UI OUTPUT ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model Answer")
        st.success(final_answer)

        st.subheader("📊 Evaluation Metrics")
        st.metric("Faithfulness", round(faithfulness, 2))
        st.metric("Hallucination Risk", risk)
        st.metric("Latency (seconds)", latency)

        if recall_before is not None:
            st.metric("Recall@3 (Before Rerank)", recall_before)
            st.metric("Recall@1 (After Rerank)", recall_after)

    with col2:
        st.subheader("📚 Retrieved Context")
        for _, row in reranked_docs.iterrows():
            st.write(f"**Q:** {row['question']}")
            st.write(f"**A:** {row['answer']}")
            st.write("---")

    # ---------- SAVE LOG ----------
    save_to_csv(
        question=query,
        answer=final_answer,
        faithfulness=faithfulness,
        hallucination_risk=risk,
        corrected=corrected
    )

    # ---------- TOKEN USAGE ----------
    system_prompt = "You are a factual assistant answering only from context."

    token_stats = query_token_stats(
        system_prompt=system_prompt,
        question=query,
        retrieved_context=retrieved_context_text,
        output=final_answer
    )

    st.subheader("🧮 Token Usage (Per Query)")

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("System", token_stats["system_tokens"])
    t2.metric("Question", token_stats["question_tokens"])
    t3.metric("Retrieved", token_stats["retrieved_tokens"])
    t4.metric("Output", token_stats["output_tokens"])
    t5.metric("Total", token_stats["total_tokens"])


# ---------------- LOG VIEWER ----------------
st.subheader("📁 Question Answer Logs")

if os.path.exists("logs/qa_logs.csv"):
    logs_df = pd.read_csv("logs/qa_logs.csv")
    st.dataframe(logs_df, width="stretch", hide_index=True)
else:
    st.info("No logs found yet.")
