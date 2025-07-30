import spacy
from rapidfuzz import fuzz
from rapidfuzz.fuzz import token_sort_ratio
import logging

logger = logging.getLogger("text_utils")

# Load spaCy English model once to tokenize and process user queries
nlp = spacy.load("en_core_web_sm")

# --------------------------------------
# Utility: Replace null or invalid values
# --------------------------------------
def replace_null(value):
    """
    Cleans a given value by checking for null/empty/invalid cases.
    Returns a standardized placeholder string if invalid.
    """
    if value is None:
        return "Not Available"
    
    str_val = str(value).strip()
    if str_val.lower() in {"", "none", "null", "not available"}:
        return "Not Available"
    
    return str_val

# --------------------------------------
# Utility: Clean query string for processing
# --------------------------------------
def clean_query_text(query: str) -> str:
    """
    Uses spaCy to tokenize and remove punctuation.
    Lowercases all tokens and returns a clean string.
    """
    doc = nlp(query)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return " ".join(tokens)

# ----------------------------------------------------------------
# Keyword Detection: Fuzzy match individual tokens with a set
# ----------------------------------------------------------------
def fuzzy_match_keywords(query_tokens: list[str], keywords: set[str], threshold: int = 90) -> bool:
    """
    Matches tokens from the user query with target keywords.
    Uses both exact match and fuzzy partial ratio scoring.

    Parameters:
        query_tokens: list of individual tokens from the query
        keywords: set of valid trigger keywords
        threshold: fuzzy score threshold to qualify as match

    Returns:
        True if a match is found, else False
    """
    # Remove short or stopword tokens
    filtered_tokens = [token for token in query_tokens if len(token) > 2 and not nlp(token)[0].is_stop]

    # Check for exact match
    for token in filtered_tokens:
        if token in keywords:
            return True

    # Apply fuzzy partial ratio match
    for token in filtered_tokens:
        for keyword in keywords:
            score = fuzz.partial_ratio(token, keyword)
            logger.debug(f"Matching token '{token}' with keyword '{keyword}': Score = {score}")
            if score >= threshold:
                return True

    return False

# ----------------------------------------------------------------
# Fuzzy match full query to target phrases (stricter)
# ----------------------------------------------------------------
def fuzzy_match_text_to_targets(query: str, *targets: str, threshold: int = 75) -> bool:
    """
    Performs a stricter fuzzy match between the full cleaned query
    and one or more target phrases, using token_sort_ratio.

    Parameters:
        query: the user query string
        targets: one or more string targets to compare with
        threshold: minimum score to count as match

    Returns:
        True if match found, False otherwise
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

# ----------------------------------------------------------------
# Intent Classification for Book Queries
# ----------------------------------------------------------------
def classify_book_intent(query: str) -> str:
    """
    Classifies the user's query into one of the following intents:
    - 'isbn_lookup'
    - 'recommend'
    - 'search' (default fallback)

    Returns:
        One of: 'isbn_lookup', 'recommend', 'search'
    """
    q = query.lower()

    if "isbn" in q:
        return "isbn_lookup"

    if any(kw in q for kw in ["recommend", "suggest", "can you recommend"]):
        return "recommend"

    return "search"
