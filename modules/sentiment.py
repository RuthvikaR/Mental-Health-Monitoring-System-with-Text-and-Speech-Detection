from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    vader = analyzer.polarity_scores(text)

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Label logic
    if vader["compound"] > 0.05:
        label = "Positive"
    elif vader["compound"] < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader": vader,
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "label": label
    }
