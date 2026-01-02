def recall_at_k(retrieved_df, ground_truth_answer):
    """
    retrieved_df: DataFrame returned by retrieve()
    ground_truth_answer: correct answer from dataset
    """
    return int(
        ground_truth_answer.strip()
        in retrieved_df["answer"].values
    )
