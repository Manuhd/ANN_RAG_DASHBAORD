def estimate_tokens(text: str) -> int:
    """
    Industry standard rough estimation:
    1 token ≈ 4 characters (English text)
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def query_token_stats(system_prompt, question, retrieved_context, output):
    system_tokens = estimate_tokens(system_prompt)
    question_tokens = estimate_tokens(question)
    retrieved_tokens = estimate_tokens(retrieved_context)
    output_tokens = estimate_tokens(output)

    total_tokens = (
        system_tokens
        + question_tokens
        + retrieved_tokens
        + output_tokens
    )

    return {
        "system_tokens": system_tokens,
        "question_tokens": question_tokens,
        "retrieved_tokens": retrieved_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }
