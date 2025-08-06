import re
import logging
import spacy
from rapidfuzz import fuzz
from rapidfuzz.fuzz import token_sort_ratio
from typing import Optional

# Setup
logger = logging.getLogger("text_utils")
nlp = spacy.load("en_core_web_sm")

# -----------------------
# Constants
# -----------------------
SPECIFIC_BOOK_PHRASES = [
    "find book", "look for book", "lookup title", "specific book",
    "find this book", "find the book titled", "lookup the book",
    "find title", "find the book", "lookup"
]

RECOMMEND_PHRASES = [
    "recommend", "suggest", "can you recommend", "book suggestion"
]

# -----------------------
# ISBN Extraction
# -----------------------
def extract_isbn(text: str) -> Optional[str]:
    """
    Extracts an ISBN from text with hyphens preserved and appends a period if not present.
    """
    pattern = r'\b(?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]\b'
    match = re.search(pattern, text)
    if match:
        isbn = match.group(0).strip()
        isbn = re.sub(r'\s+', '-', isbn)  # Normalize spaces to hyphens
        return isbn if isbn.endswith('.') else f"{isbn}."
    return None

# -----------------------
# Null Replacement
# -----------------------
def replace_null(value) -> str:
    """
    Standardizes missing or null-like values to 'Not Available'.
    """
    if value is None:
        return "Not Available"
    
    str_val = str(value).strip().lower()
    return "Not Available" if str_val in {"", "none", "null", "not available"} else str(value).strip()

# -----------------------
# Query Cleaning
# -----------------------
def clean_query_text(query: str) -> str:
    """
    Tokenizes and lowercases a query string, removing punctuation.
    """
    return " ".join(token.text.lower() for token in nlp(query) if not token.is_punct)

# -----------------------
# Token-Level Fuzzy Match
# -----------------------
def fuzzy_match_keywords(query_tokens: list[str], keywords: set[str], threshold: int = 90) -> bool:
    """
    Checks for both exact and fuzzy token matches in a query against a keyword set.
    """
    filtered = [t for t in query_tokens if len(t) > 2 and not nlp(t)[0].is_stop]

    for token in filtered:
        if token in keywords:
            return True

    for token in filtered:
        for keyword in keywords:
            score = fuzz.partial_ratio(token, keyword)
            logger.debug(f"Fuzzy match: '{token}' vs '{keyword}' = {score}")
            if score >= threshold:
                return True

    return False

# -----------------------
# Full Query Fuzzy Match
# -----------------------
def fuzzy_match_text_to_targets(query: str, *targets: str, threshold: int = 75) -> bool:
    """
    Compares a query against multiple target phrases using fuzzy token_sort_ratio.
    """
    query_clean = clean_query_text(query)

    for target in targets:
        if not target:
            continue
        score = token_sort_ratio(query_clean, clean_query_text(target))
        logger.debug(f"[FuzzyMatch] '{query_clean}' vs '{target}': Score = {score}")
        if score >= threshold:
            return True

    return False

# -----------------------
# Book Intent Classifier
# -----------------------
def classify_book_intent(query: str) -> str:
    """
    Classifies a book-related query into one of:
    - 'specific_book_search'
    - 'recommend'
    - 'search'
    """
    query_lower = query.lower()

    if "isbn" in query_lower or fuzzy_match_text_to_targets(query_lower, *SPECIFIC_BOOK_PHRASES):
        return "specific_book_search"

    if any(phrase in query_lower for phrase in RECOMMEND_PHRASES):
        return "recommend"

    if query_lower.startswith(("find", "lookup")):
        noun_chunks = list(nlp(query).noun_chunks)
        if noun_chunks and len(noun_chunks[0].text.split()) >= 2:
            return "specific_book_search"

    return "search"