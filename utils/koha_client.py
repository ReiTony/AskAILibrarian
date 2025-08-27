import base64
import json
import logging
import requests
from typing import Any, Union, List, Dict
from decouple import config
from collections import defaultdict
from cachetools import TTLCache
import re

# -------------------------------
# Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
TIMEOUT = 10  # seconds

logger = logging.getLogger("koha_client")

# -------------------------------
# Cache
# -------------------------------
_search_cache = TTLCache(maxsize=1000, ttl=300)

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
def search_books(query: str, session_id: str = "global") -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Search books in Koha by title.
    Tries full query and fallback on first word.
    Uses caching to avoid duplicate searches.
    """
    headers = get_auth_headers()
    phrases = [query]

    if (words := query.split()) and words[0].lower() != query.lower():
        phrases.append(words[0])

    for phrase in phrases:
        key = (session_id, "title", phrase.lower())
        if key in _search_cache:
            logger.info(f"[Koha Search] Cache hit for: {phrase}")
            return _search_cache[key]

        params = {"title": {"-like": f"%{phrase}%"}}
        url = f"{API_URL}?q={json.dumps(params)}"

        try:
            logger.info(f"[Koha Search] Searching title with: {phrase}")
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            books = response.json()

            result = [format_book_data(book) for book in books] if books else {"error": "No books found."}
            _search_cache[key] = result
            if books:
                return result
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"[Koha Search] Error during API call for phrase '{phrase}': {e}")
            return {"error": f"Koha API could not be reached: {e}"}

    return {"error": "No books found."}

# -------------------------------
# Fetch Items
# -------------------------------
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
# Identifier Search
# -------------------------------
FIELD_ISBN = "isbn"        

def _with_period_variants(val: str) -> list[str]:
    return list(dict.fromkeys([val, val.rstrip('.')] if val.endswith('.') else [val, f"{val}."]))

def _like_contains(value: str) -> dict:
    return {"-like": f"%{value}%"}

def _perform_identifier_search(headers: dict, field: str, value: str) -> list[dict[str, Any]] | None:
    """Exact match then 'contains' match for a given field/value."""
    try:
        url = f"{API_URL}?q={_q({field: value})}"
        logger.info(f"[Koha Lookup] Trying {field} (exact): {value!r}")
        data = _get(url, headers)
        if data:
            out = _format_list(data)
            for b in out:
                b["matched_on"] = {"field": f"{field} (exact)", "value": value}
            return out
    except Exception as e:
        logger.warning(f"[Koha Lookup] Error on {field} (exact) for {value!r}: {e}")

    try:
        url = f"{API_URL}?q={_q({field: _like_contains(value)})}"
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
    """
    headers = get_auth_headers()

    # ISBN / ISSN
    for field in ["isbn", "issn"]:
        for value in identifiers.get(field, []):
            variants = _with_period_variants(value)
            for variant in variants:
                result = _perform_identifier_search(headers, field, variant)
                if result:
                    return result

    # Call numbers (using ISBN field in Koha)
    call_numbers = identifiers.get("call_numbers", [])
    if call_numbers:
        logger.info("[Koha Lookup] No match on ISBN/ISSN. Trying Call Number search via 'isbn' field.")
        for cn in call_numbers:
            variants = _with_period_variants(cn)
            variants.append(re.sub(r"\s+", " ", cn).strip())
            variants.append(cn.replace(".", " ").strip())
            seen = set()
            variants = [v for v in variants if not (v in seen or seen.add(v))]

            for variant in variants:
                result = _perform_identifier_search(headers, "isbn", variant)
                if result:
                    for book in result:
                        book["matched_on"]["field"] = "isbn (callno-search)"
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
