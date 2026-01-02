# ingestion_metrics.py
from token_utils import estimate_tokens

def ingestion_token_stats_from_df(df):
    """
    Dynamically calculate ingestion tokens from actual data.
    1 row = 1 chunk (FAQ-style data)
    """
    total_tokens = 0
    total_chunks = 0

    for _, row in df.iterrows():
        text = f"{row['question']} {row['answer']}"
        tokens = estimate_tokens(text)
        total_tokens += tokens
        total_chunks += 1

    return {
        "docs": len(df),
        "chunks": total_chunks,
        "total_chunk_tokens": total_tokens
    }
