import base64
import json
import logging
import requests
from typing import Any, Literal, Union, List, Dict
from decouple import config
# from cachetools import TTLCache
from collections import defaultdict
import re

# -------------------------------
# Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
TIMEOUT = 10  # seconds

logger = logging.getLogger("koha_client")

# A set to store phrases that have already been searched in this session.
# _search_cache = TTLCache(maxsize=1000, ttl=300)

# -------------------------------
# Authentication Header
# -------------------------------
def get_auth_headers() -> dict[str, str]:
    """Returns HTTP headers with Basic Auth for Koha API."""
    auth = f"{USERNAME}:{PASSWORD}"
    token = base64.b64encode(auth.encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json"
    }


# -------------------------------
# Helpers for Koha query building
# -------------------------------
def _q(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _format_list(resp_json: Any) -> list[dict[str, Any]]:
    if not resp_json:
        return []
    return [format_book_data(b) for b in resp_json]

def _get(url: str, headers: dict) -> Any:
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# -------------------------------
# Search Books (General)
# -------------------------------
def search_books(query: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Search books in Koha by title.
    This function is stateless and does not perform any caching.
    It attempts a search with the full query, and if that yields no results,
    it tries a fallback search using just the first word of the query.
    """
    headers = get_auth_headers()
    
    # List of phrases to attempt to search for
    phrases = [query]
    words = query.split()
    if words and words[0].lower() != query.lower():
        phrases.append(words[0])

    for phrase in phrases:
        params = {"title": {"-like": f"%{phrase}%"}}
        url = f"{API_URL}?q={json.dumps(params)}"

        try:
            logger.info(f"[Koha Search] Searching title with: {phrase}")
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            books = response.json()

            if books and isinstance(books, list):
                return [format_book_data(book) for book in books]

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"[Koha Search] Error during API call for phrase '{phrase}': {e}")
            return {"error": f"Koha API could not be reached: {e}"}

    # If the loop completes without returning, it means no books were found for any phrase.
    return {"error": "No books found."}

def fetch_items_for_multiple_biblios(biblio_ids: List[Union[str, int]]) -> Dict[int, List[Dict]]:
    """
    Fetches all items for a given list of biblio_ids in a single API call.

    Args:
        biblio_ids: A list of biblio_id integers or strings.

    Returns:
        A dictionary mapping each biblio_id to a list of its item records.
        Example: {101: [{item1_data}, {item2_data}], 204: [{item3_data}]}
    """
    if not biblio_ids:
        return {}

    # Ensure all IDs are integers for consistency, as biblio_id is a number.
    clean_biblio_ids = [int(bid) for bid in biblio_ids]
    
    headers = get_auth_headers()
    params = {"biblio_id": clean_biblio_ids}
    url = f"{API_URL}/items?q={json.dumps(params)}"

    try:
        logger.info(f"[Koha Items] Fetching items for {len(clean_biblio_ids)} biblio records.")
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        
        items = response.json()
        
        # Group the flat list of items by their biblio_id
        items_by_biblio = defaultdict(list)
        if isinstance(items, list):
            for item in items:
                # Koha API returns biblio_id as an integer
                items_by_biblio[item.get("biblio_id")].append(item)
        
        return items_by_biblio

    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"[Koha Items] Error fetching items for multiple biblios: {e}")
        return {}

def fetch_quantity_from_biblio_id(biblio_id: str) -> int:
    """Fetch number of items available for a given biblio_id."""
    try:
        url = f"{API_URL}/{biblio_id}/items"
        headers = get_auth_headers()
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        items = response.json()
        return len(items) if isinstance(items, list) else 0
    except Exception as e:
        logger.error(f"[Koha] Error fetching quantity for biblio_id {biblio_id}: {e}")
        return 0
    
# -------------------------------
# Search by any identifier (ISBN, ISSN, Call No.)
# -------------------------------
FIELD_ISBN = "isbn"        

def _with_period_variants(val: str) -> list[str]:
    return list(dict.fromkeys([val, val.rstrip('.')] if val.endswith('.') else [val, f"{val}."]))

def _like_contains(value: str) -> dict:
    return {"-like": f"%{value}%"}

def _perform_identifier_search(headers, field, value):
    # Tries an exact match, then a 'like' match for a given field and value
    # Returns the result list if found, otherwise None
    
    # 1. Try exact match
    try:
        url = f"{API_URL}?q={_q({field: value})}"
        data = _get(url, headers)
        if data:
            return _format_list(data)
    except Exception:
        pass # Log warning

    # 2. Try contains match
    try:
        url = f"{API_URL}?q={_q({field: _like_contains(value)})}"
        data = _get(url, headers)
        if data:
            return _format_list(data)
    except Exception:
        pass # Log warning
        
    return None

def _perform_identifier_search(headers: dict, field: str, value: str) -> list[dict[str, Any]] | None:
    """
    Tries to find a match for a given field and value, first by exact match, then by 'contains'.
    If a match is found, it formats the result and adds debugging info.

    Returns:
        A list of formatted book dictionaries if found, otherwise None.
    """
    # 1. Try for an exact match
    try:
        query_obj = {field: value}
        url = f"{API_URL}?q={_q(query_obj)}"
        logger.info(f"[Koha Lookup] Trying {field} (exact): {value!r}")
        data = _get(url, headers)
        if data:
            out = _format_list(data)
            for b in out:
                b["matched_on"] = {"field": f"{field} (exact)", "value": value}
            return out
    except Exception as e:
        logger.warning(f"[Koha Lookup] Error on {field} (exact) for {value!r}: {e}")

    # 2. If no exact match, try for a 'contains' match
    try:
        query_obj = {field: _like_contains(value)}
        url = f"{API_URL}?q={_q(query_obj)}"
        logger.info(f"[Koha Lookup] Trying {field} (contains): {value!r}")
        data = _get(url, headers)
        if data:
            out = _format_list(data)
            for b in out:
                b["matched_on"] = {"field": f"{field} (contains)", "value": value}
            return out
    except Exception as e:
        logger.warning(f"[Koha Lookup] Error on {field} (contains) for {value!r}: {e}")

    return None

def search_by_identifiers(identifiers: Dict[str, List[str]]) -> Union[list[dict[str, Any]], dict[str, str]]:
    """
    Searches for books by ISBN, ISSN, or Call Number.
    It handles a special case where Call Numbers are indexed in the 'isbn' field.
    """
    headers = get_auth_headers()
    
    # --- Part 1: Search for standard identifiers first ---
    # First, we search for ISBN and ISSN in their correct fields.
    standard_search_map = {
        "isbn": identifiers.get("isbn", []),
        "issn": identifiers.get("issn", [])
    }

    for field, values in standard_search_map.items():
        for value in values:
            variants = _with_period_variants(value)
            for variant in variants:
                result = _perform_identifier_search(headers, field, variant)
                if result:
                    return result
    # --- Part 2: Search for Call Numbers ---
    call_numbers = identifiers.get("call_numbers", [])
    if call_numbers:
        logger.info("[Koha Lookup] No match on standard fields. Now trying Call Number search via the 'isbn' field.")
        for cn in call_numbers:
            # Generate the same variants as the original code
            variants = _with_period_variants(cn)
            variants.append(re.sub(r"\s+", " ", cn).strip())
            variants.append(cn.replace(".", " ").strip())
            # De-duplicate while preserving order
            seen = set()
            variants = [v for v in variants if not (v in seen or seen.add(v))]

            for variant in variants:
                # IMPORTANT: We are calling the helper with the field hardcoded to "isbn"
                result = _perform_identifier_search(headers, "isbn", variant)
                if result:
                    # Add a note to the debugging info to make this clear
                    for book in result:
                        book["matched_on"]["field"] = f"isbn (callno-search)"
                    return result

    return {"error": "No matching records found."}


# -------------------------------
# Helpers
# -------------------------------
def format_book_data(book: dict[str, Any]) -> dict[str, Any]:
    """Formats book dictionary from Koha API into standard response schema."""
    return {
        "title": book.get("title", "N/A"),
        "publisher": book.get("publisher", "N/A"),
        "isbn": book.get("isbn", "N/A"),
        "quantity": book.get("quantity", "N/A"),
        "author": book.get("author", "N/A"),
        "year": book.get("copyright_date", "N/A"),
        "biblio_id": book.get("biblio_id", "N/A"),
    }