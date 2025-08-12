import re
import logging
import spacy
from rapidfuzz import fuzz
from rapidfuzz.fuzz import token_sort_ratio
from typing import Optional, Dict, List

# Setup
logger = logging.getLogger("text_utils")
nlp = spacy.load("en_core_web_sm")


def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _digits_x(s: str) -> str:
    return re.sub(r'[^0-9Xx]', '', s)

# ---------- Validators ----------
def is_valid_isbn10(s: str) -> bool:
    s = _digits_x(s).upper()
    if len(s) != 10: return False
    try:
        total = 0
        for i, ch in enumerate(s):
            val = 10 if (ch == 'X' and i == 9) else int(ch)
            total += (10 - i) * val
        return total % 11 == 0
    except ValueError:
        return False

def is_valid_isbn13(s: str) -> bool:
    s = re.sub(r'\D', '', s)
    if len(s) != 13: return False
    try:
        total = sum((int(d) * (3 if i % 2 else 1)) for i, d in enumerate(s[:12]))
        check = (10 - (total % 10)) % 10
        return check == int(s[-1])
    except ValueError:
        return False

def is_valid_issn(s: str) -> bool:
    s = _digits_x(s).upper()
    if len(s) != 8: return False
    try:
        total = 0
        for i, ch in enumerate(s):
            val = 10 if (ch == 'X' and i == 7) else int(ch)
            total += (8 - i) * val
        return total % 11 == 0
    except ValueError:
        return False

# -----------------------
# Candidate patterns
# -----------------------
# Broad ISBN candidates; validated after matching
_ISBN_CAND_RE = re.compile(r'\b[0-9Xx][0-9Xx\-\s]{8,16}[0-9Xx]\b')
_ISSN_CAND_RE = re.compile(r'\b\d{4}[-\s]?\d{3}[\dXx]\b')
# Pure 9-digit SBN (pre-ISBN, no hyphens)
_SBN_CAND_RE  = re.compile(r'\b\d{9}\b')

_CALLNO_RE = re.compile(
    r'\b(?:[A-Z]{1,3}\s*\d{1,4}(?:\.\d+)?)'           
    r'(?:\s*[.\s]?[A-Z]{1,3}\d{0,4}[A-Z]{0,3})*'      
    r'(?:\s*\d{4})?\b'                                 
)

def expand_sbn_to_isbn10(compact_digits: str) -> Optional[str]:
    """
    Convert 9-digit SBN to 10-digit ISBN by prefixing '0' and validating.
    Returns the 10-digit compact string (digits only) if valid, else None.
    """
    cand = "0" + compact_digits
    return cand if is_valid_isbn10(cand) else None

# -----------------------
# Identifier Extraction
# -----------------------
def extract_identifiers(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"isbn": [], "issn": [], "call_numbers": []}

    # --- ISBN & SBN-in-ISBN pass ---
    isbn_found: List[str] = []
    for m in _ISBN_CAND_RE.finditer(text):
        raw = _norm(m.group(0))
        if m.end() < len(text) and text[m.end()] == '.':
            raw = raw + '.'
        compact = _digits_x(raw)

        if len(compact) == 10 and is_valid_isbn10(compact):
            isbn_found.append(raw)
            continue
        if len(compact) == 13 and is_valid_isbn13(compact):
            isbn_found.append(raw)
            continue
        if len(compact) == 9:
            padded = expand_sbn_to_isbn10(compact)
            if padded:
                trailing_dot = raw.endswith('.')
                base = raw[:-1] if trailing_dot else raw
                display = ('0-' + base) if '-' in base else ('0' + base)
                if trailing_dot:
                    display += '.'
                isbn_found.append(display)

    # Extra pass: pure 9-digit SBN tokens (no hyphens) like "870228412"
    for m in _SBN_CAND_RE.finditer(text):
        compact9 = m.group(0)
        trailing_dot = (m.end() < len(text) and text[m.end()] == '.')
        padded = expand_sbn_to_isbn10(compact9)
        if padded:
            display = '0' + compact9  
            if trailing_dot:
                display += '.'
            isbn_found.append(display)

    # --- ISSN ---
    issn_found: List[str] = []
    for m in _ISSN_CAND_RE.finditer(text):
        raw = _norm(m.group(0))
        if m.end() < len(text) and text[m.end()] == '.':
            raw = raw + '.'
        compact = _digits_x(raw)
        if is_valid_issn(compact):
            # normalize to ####-#### but preserve trailing '.'
            if len(compact) == 8:
                core = f"{compact[:4]}-{compact[4:]}"
                raw = core + ('.' if raw.endswith('.') else '')
            issn_found.append(raw)

    # --- Call numbers ---
    lower = text.lower()
    want_call = (
        "call no" in lower or "call num" in lower or "call number" in lower or "call code" in lower
        or (not isbn_found and not issn_found)
    )
    callnos: List[str] = []
    if want_call:
        for m in _CALLNO_RE.finditer(text):
            callnos.append(_norm(m.group(0)))

    # De-dup preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    ids = {
        "isbn": dedup(isbn_found),
        "issn": dedup(issn_found),
        "call_numbers": dedup(callnos),
    }
    logger.info(f"[IDs] Extracted: {ids}")
    return ids

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


