import base64
import json
import logging
import re
import requests
from typing import Any, Union, List, Dict
from decouple import config

# -------------------------------
# Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
TIMEOUT = 8# seconds

logger = logging.getLogger("koha_client")

# -------------------------------
# Authentication Header
# -------------------------------
def get_auth_headers() -> dict[str, str]:
    """Returns HTTP headers with Basic Auth for Koha API."""
    auth = f"{USERNAME}:{PASSWORD}"
    token = base64.b64encode(auth.encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }

# -------------------------------
# Helpers
# -------------------------------
def _q(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _format_list(resp_json: Any) -> list[dict[str, Any]]:
    return [format_book_data(b) for b in resp_json] if resp_json else []

def _safe_request(url: str, headers: dict) -> Any:
    """Perform GET request with unified error handling + logging."""
    try:
        logger.debug(f"[Koha Request] GET {url}")
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.Timeout:
        logger.error(f"[Koha Request] Timeout after {TIMEOUT}s for URL: {url}")
    except requests.RequestException as e:
        logger.error(f"[Koha Request] Request error for URL {url}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"[Koha Request] JSON decode error for URL {url}: {e}")
    except Exception as e:
        logger.error(f"[Koha Request] Unexpected error for URL {url}: {e}")
    return None

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

# -------------------------------
# Search Books (General)
# -------------------------------
def search_books(query: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Search books in Koha by title.
    Tries full query and fallback on first word.
    """
    headers = get_auth_headers()
    phrases = [query]

    if (words := query.split()) and words[0].lower() != query.lower():
        phrases.append(words[0])

    for phrase in phrases:
        params = {"title": {"-like": f"%{phrase}%"}}
        url = f"{API_URL}?q={json.dumps(params)}"
        logger.info(f"[Koha Search] Searching title with phrase: {phrase!r}")

        data = _safe_request(url, headers)
        if data:
            return [format_book_data(book) for book in data]

    logger.warning(f"[Koha Search] No results found for query: {query!r}")
    return {"error": "No books found."}

# -------------------------------
# Fetch Items
# -------------------------------
def fetch_quantity_from_biblio_id(biblio_id: str) -> int:
    """Fetch number of items available for a given biblio_id."""
    url = f"{API_URL}/{biblio_id}/items"
    headers = get_auth_headers()

    data = _safe_request(url, headers)
    if isinstance(data, list):
        return len(data)

    logger.warning(f"[Koha Quantity] No items returned for biblio_id={biblio_id}")
    return 0

# -------------------------------
# Identifier Search
# -------------------------------
FIELD_ISBN = "isbn"        

def _with_period_variants(val: str) -> list[str]:
    return [val, val.rstrip(".")] if val.endswith(".") else [val, f"{val}."]

def _like_contains(value: str) -> dict:
    return {"-like": f"%{value}%"}

def _perform_identifier_search(headers: dict, field: str, value: str) -> list[dict[str, Any]] | None:
    """Exact match then 'contains' match for a given field/value."""
    # Exact
    url = f"{API_URL}?q={_q({field: value})}"
    logger.info(f"[Koha Lookup] Trying exact {field}: {value!r}")
    data = _safe_request(url, headers)
    if data:
        out = _format_list(data)
        for b in out:
            b["matched_on"] = {"field": f"{field} (exact)", "value": value}
        return out

    # Contains
    url = f"{API_URL}?q={_q({field: _like_contains(value)})}"
    logger.info(f"[Koha Lookup] Trying contains {field}: {value!r}")
    data = _safe_request(url, headers)
    if data:
        out = _format_list(data)
        for b in out:
            b["matched_on"] = {"field": f"{field} (contains)", "value": value}
        return out

    logger.debug(f"[Koha Lookup] No match for {field}: {value!r}")
    return None

def search_by_identifiers(identifiers: Dict[str, List[str]]) -> Union[list[dict[str, Any]], dict[str, str]]:
    """
    Searches for books by ISBN, ISSN, or Call Number.
    """
    headers = get_auth_headers()

    # ISBN / ISSN
    for field in ["isbn", "issn"]:
        for value in identifiers.get(field, []):
            for variant in _with_period_variants(value):
                logger.debug(f"[Koha Lookup] Searching {field} variant: {variant!r}")
                result = _perform_identifier_search(headers, field, variant)
                if result:
                    return result

    # Call numbers (fallback via ISBN field)
    for cn in identifiers.get("call_numbers", []):
        variants = _with_period_variants(cn)
        variants += [
            re.sub(r"\s+", " ", cn).strip(),
            cn.replace(".", " ").strip(),
        ]
        seen = set()
        variants = [v for v in variants if not (v in seen or seen.add(v))]

        for variant in variants:
            logger.debug(f"[Koha Lookup] Searching call number variant: {variant!r}")
            result = _perform_identifier_search(headers, "isbn", variant)
            if result:
                for book in result:
                    book["matched_on"]["field"] = "isbn (callno-search)"
                return result

    logger.warning("[Koha Lookup] No matches found for identifiers.")
    return {"error": "No matching records found."}
