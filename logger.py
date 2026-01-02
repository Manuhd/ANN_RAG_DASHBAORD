import csv
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, "qa_logs.csv")

def save_to_csv(question, answer, faithfulness, hallucination_risk, corrected):
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "answer",
                "faithfulness",
                "hallucination_risk",
                "corrected"
            ])

        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
            question,
            answer,
            round(faithfulness, 2),
            hallucination_risk,
            "Yes" if corrected else "No"
        ])
