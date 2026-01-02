from retriever import load_data, build_index, retrieve
from reranker import rerank
from generator import generate_answer
from self_corrector import self_correct
from metrics import hallucination_risk

df = load_data()
index, _ = build_index(df)

query = input("Enter your question: ")

docs = retrieve(query, df, index)
reranked_docs = rerank(query, docs)
top_context = reranked_docs.head(1)

answer = generate_answer(top_context, query)


final_answer, corrected, faithfulness = self_correct(answer, top_context)

print("\nAnswer:", final_answer)
print("Faithfulness Score:", round(faithfulness, 2))
print("Hallucination Risk:", hallucination_risk(faithfulness))
