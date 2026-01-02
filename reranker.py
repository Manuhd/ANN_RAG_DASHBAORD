from difflib import SequenceMatcher

def rerank(query, docs):
    docs = docs.copy()  # ✅ FIX: avoid SettingWithCopyWarning

    def score(text):
        return SequenceMatcher(None, query.lower(), text.lower()).ratio()

    docs.loc[:, "rerank_score"] = docs["question"].apply(score)
    return docs.sort_values("rerank_score", ascending=False)
