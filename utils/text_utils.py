import spacy
from rapidfuzz import fuzz
import logging


logger = logging.getLogger("text_utils")
# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

# --------------------------------------
# Replaces null/empty values with fallback
# --------------------------------------
def replace_null(value):
    """Return value if not empty/null; else return 'Not Available'."""
    return value if value not in [None, "None", ""] else "Not Available"


# --------------------------------------
# Preprocess query using spaCy tokenizer
# --------------------------------------
def clean_query_text(query: str) -> str:
    """Removes punctuation and lowercases the input query."""
    doc = nlp(query)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return " ".join(tokens)


# ----------------------------------------------------------------
# Check if tokens match target keywords using string similarity
# ----------------------------------------------------------------
def fuzzy_match_keywords(query_tokens: list[str], keywords: set[str], threshold: int = 90) -> bool:
    """Returns True if any token matches keywords directly or via fuzzy match."""
    filtered_tokens = [token for token in query_tokens if len(token) > 2 and not nlp(token)[0].is_stop]

    # Exact keyword match
    for token in filtered_tokens:
        if token in keywords:
            return True

    # Fuzzy match comparison
    for token in filtered_tokens:
        for keyword in keywords:
            score = fuzz.partial_ratio(token, keyword)
            logger.debug(f"Matching token '{token}' with keyword '{keyword}': Score = {score}")
            if score >= threshold:
                return True

    return False
