def compute_risk(sentiment, dep_score, anx_score):
    raw_risk = 0

    if not sentiment or "vader" not in sentiment:
        return 0

    compound = sentiment["vader"]["compound"]

    # 🔴 Detect negativity earlier (more sensitive)
    if compound < -0.3:
        raw_risk += 2.5
    elif compound < -0.1:
        raw_risk += 1.5

    # 🔴 Stronger emotional intensity contribution
    raw_risk += abs(compound) * 2

    # 🔴 Cap keyword scores (important for stability)
    dep_score = min(dep_score, 5)
    anx_score = min(anx_score, 5)

    # 🔴 Slightly increased keyword weights
    raw_risk += dep_score * 1.8
    raw_risk += anx_score * 1.5

    # 🔴 Nonlinear boost (makes small signals noticeable)
    raw_risk = raw_risk ** 1.1

    # Normalize to 0–10
    max_score = 20  # increased because we boosted weights
    normalized = (raw_risk / max_score) * 10

    return round(min(normalized, 10), 2)


def risk_label(score):
    if score <= 2.5:
        return "Low"
    elif score <= 5:
        return "Moderate"
    elif score <= 7.5:
        return "High"
    else:
        return "Critical"

