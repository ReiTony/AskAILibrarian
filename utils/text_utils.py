import spacy
from rapidfuzz import fuzz
from rapidfuzz.fuzz import token_sort_ratio
import logging


logger = logging.getLogger("text_utils")
# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

# --------------------------------------
# Replaces null/empty values with fallback
# --------------------------------------
def replace_null(value):
    """Return cleaned string if valid; else return 'Not Available'."""
    if value is None:
        return "Not Available"
    
    str_val = str(value).strip()
    if str_val.lower() in {"", "none", "null", "not available"}:
        return "Not Available"
    
    return str_val

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


def fuzzy_match_text_to_targets(query: str, *targets: str, threshold: int = 75) -> bool:
    """
    Fuzzy match a query string against one or more target strings using token_sort_ratio.
    This is stricter and accounts for word order/duplicates.
    """
    query_clean = clean_query_text(query)

    for target in targets:
        if not target:
            continue
        target_clean = clean_query_text(target)

        score = token_sort_ratio(query_clean, target_clean)
        logger.debug(f"[FuzzyMatch] '{query_clean}' vs '{target_clean}' → Score: {score}")
        if score >= threshold:
            return True

    return False