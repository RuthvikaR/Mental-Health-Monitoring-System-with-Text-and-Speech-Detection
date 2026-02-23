depression_keywords = [
    "tired", "hopeless", "worthless", "low",
    "unmotivated", "empty", "sad", "alone"
]

anxiety_keywords = [
    "worried", "overthinking", "stress",
    "fear", "panic", "nervous"
]

def keyword_scores(text):
    text = text.lower()

    dep_score = sum(word in text for word in depression_keywords)
    anx_score = sum(word in text for word in anxiety_keywords)

    return dep_score, anx_score
