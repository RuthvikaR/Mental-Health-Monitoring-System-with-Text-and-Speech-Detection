depression_keywords = [
    # Core emotional states
    "sad", "very sad", "deeply sad", "unhappy", "miserable", "down", "low",
    "empty", "numb", "hollow", "void", "lifeless",

    # Hopelessness / worthlessness
    "hopeless", "helpless", "worthless", "useless", "failure", "burden",
    "no purpose", "pointless", "meaningless", "nothing matters",

    # Low energy / fatigue
    "tired", "exhausted", "drained", "fatigued", "no energy", "burnt out",
    "can’t get out of bed", "sleeping too much",

    # Loss of interest (anhedonia)
    "unmotivated", "no motivation", "lost interest", "don’t care anymore",
    "nothing excites me", "bored with life",

    # Isolation / loneliness
    "alone", "lonely", "isolated", "no one cares", "no friends",
    "feel disconnected", "withdrawn",

    # Self-worth / guilt
    "guilty", "ashamed", "self-hate", "hate myself", "not good enough",
    "inferior", "regret everything",

    # Cognitive patterns
    "overthinking life", "negative thoughts", "dark thoughts",
    "can't focus", "brain fog", "confused",

    # Sleep & appetite
    "insomnia", "can't sleep", "sleeping all day", "no appetite",
    "overeating",

    # Severe indicators (important for risk systems)
    "want to disappear", "no reason to live", "give up", "end it all",
    "life is pointless"
]

anxiety_keywords = [
    # Core anxiety emotions
    "worried", "anxious", "nervous", "tense", "uneasy",
    "restless", "on edge", "panic", "fear", "scared",

    # Overthinking patterns
    "overthinking", "thinking too much", "racing thoughts",
    "can't stop thinking", "looping thoughts", "what if",

    # Stress indicators
    "stress", "stressed", "overwhelmed", "pressure",
    "too much to handle", "burnout",

    # Panic / physical symptoms
    "panic attack", "heart racing", "shortness of breath",
    "sweating", "shaking", "dizzy", "chest tightness",

    # Control / uncertainty
    "losing control", "can't handle this", "uncertain",
    "fear of future", "fear of failure",

    # Social anxiety
    "social anxiety", "afraid to talk", "fear of judgment",
    "embarrassed", "awkward",

    # Anticipatory anxiety
    "something bad will happen", "expecting worst",
    "doom", "impending danger",

    # Sleep issues
    "can't relax", "can't calm down", "mind won't stop",
    "trouble sleeping",

    # Avoidance behavior
    "avoiding people", "avoiding situations",
    "don't want to go out"
]


def keyword_scores(text):
    text = text.lower()
    words = text.split()

    dep_score = sum(word in text for word in depression_keywords)
    anx_score = sum(word in text for word in anxiety_keywords)

    total_keywords = dep_score + anx_score

    keyword_density = total_keywords / len(words) if words else 0

    return dep_score, anx_score, round(keyword_density, 3)
