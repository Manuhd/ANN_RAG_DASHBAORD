def hallucination_risk(faithfulness):
    risk = round((1 - faithfulness) * 100, 2)
    return f"{risk}%"
