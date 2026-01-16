import re
import spacy
from typing import Tuple, Dict
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is available
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Load spaCy model (small model)
# If not installed, run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def mask_with_spacy(text: str, labels_to_mask=("PERSON", "ORG", "GPE", "LOC")) -> Tuple[str, Dict]:
    """
    Uses spaCy NER to find entities and replace them with [REDACTED].
    Returns masked text and a dict of detected entities (for display / debug).
    """
    doc = nlp(text)
    masked = text
    entities = []
    # We will replace longer entity spans first to avoid partial overlap issues
    # Collect entities
    ents = [(ent.start_char, ent.end_char, ent.text, ent.label_) for ent in doc.ents if ent.label_ in labels_to_mask]
    # Sort in reverse order by start_char so replace doesn't shift following indices
    for start, end, ent_text, ent_label in sorted(ents, key=lambda x: x[0], reverse=True):
        masked = masked[:start] + "[REDACTED]" + masked[end:]
        entities.append({"text": ent_text, "label": ent_label})
    return masked, {"spacy_entities": entities}


def mask_regex_pii(text: str) -> Tuple[str, Dict]:
    """
    Apply regex-based masks for email, phone, Aadhaar, PAN, SSN, credit-card-like numbers.
    Returns masked text and a dict of which regexes matched.
    """
    masked = text
    matches = {}

    # Emails
    email_pat = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    masked, email_count = re.subn(email_pat, "[EMAIL]", masked)
    matches["email_count"] = email_count

    # Phone numbers (various formats)
    phone_pat = r"(\+?\d{1,3}[-\s]?)?(?:\d{10}|\d{5}[-\s]\d{5}|\d{3}[-\s]\d{3}[-\s]\d{4})"
    masked, phone_count = re.subn(phone_pat, "[PHONE]", masked)
    matches["phone_count"] = phone_count

    # Aadhaar: XXXX XXXX XXXX (allow digits w/ spaces)
    aadhaar_pat = r"\b\d{4}\s\d{4}\s\d{4}\b"
    masked, aadhaar_count = re.subn(aadhaar_pat, "[AADHAAR]", masked)
    matches["aadhaar_count"] = aadhaar_count

    # PAN: 5 letters, 4 digits, 1 letter (case-insensitive)
    pan_pat = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    masked, pan_count = re.subn(pan_pat, "[PAN]", masked, flags=re.IGNORECASE)
    matches["pan_count"] = pan_count

    # US SSN: 123-45-6789
    ssn_pat = r"\b\d{3}-\d{2}-\d{4}\b"
    masked, ssn_count = re.subn(ssn_pat, "[SSN]", masked)
    matches["ssn_count"] = ssn_count

    # Generic long numeric sequences (credit-card-like) -> optional
    cc_pat = r"\b(?:\d[ -]*?){13,19}\b"
    masked, cc_count = re.subn(cc_pat, "[LONG_NUM]", masked)
    matches["long_number_count"] = cc_count

    return masked, matches


def mask_all(text: str) -> Dict:
    """
    Master masking function: applies spaCy masking then regex masking.
    Returns dict containing:
      - original
      - masked_text
      - details (counts and found entities)
      - sentiment scores
    """
    original = text if text else ""
    # First use spaCy to mask persons/orgs/locations
    masked_spacy, spacy_info = mask_with_spacy(original)
    # Then apply regex-based masks (email/phone/Aadhaar/PAN/SSN/long numbers)
    masked_final, regex_info = mask_regex_pii(masked_spacy)

    # Sentiment using VADER
    sentiment_scores = sia.polarity_scores(original)
    compound = sentiment_scores.get("compound", 0.0)
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "original": original,
        "masked_text": masked_final,
        "details": {**spacy_info, **regex_info},
        "sentiment": {"label": sentiment, "scores": sentiment_scores}
    }
