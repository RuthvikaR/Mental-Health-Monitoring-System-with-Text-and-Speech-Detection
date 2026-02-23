def compute_risk(sentiment, dep_score, anx_score):
    risk = 0

    # sentiment weight
    if sentiment["compound"] < -0.5:
        risk += 2

    # keyword weights
    risk += dep_score * 1.5
    risk += anx_score * 1.2

    return round(risk, 2)


def risk_label(score):
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Moderate"
    else:
        return "High"
