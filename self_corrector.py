def self_correct(answer, context):
    context_text = " ".join(context["answer"].tolist()).lower()

    # Simple grounding check
    overlap = sum(1 for word in answer.lower().split() if word in context_text)

    faithfulness = overlap / max(len(answer.split()), 1)

    if faithfulness < 0.5:
        return "I don't know", True, faithfulness

    return answer, False, faithfulness
