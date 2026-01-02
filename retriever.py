import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    df = pd.read_csv("data/loan_faq.csv")
    return df

def build_index(df, index_type="flat", nlist=10):
    embeddings = model.encode(df["question"].tolist())
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    if index_type == "ivf":
        # Auto-adjust nlist for small datasets
        nlist = min(nlist, max(1, num_vectors // 2))

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            faiss.METRIC_L2
        )
        index.train(embeddings)
        index.add(embeddings)

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.add(embeddings)

    else:
        # Flat index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    return index, embeddings

def retrieve(query, df, index, top_k=3):
    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), top_k)
    return df.iloc[indices[0]]
