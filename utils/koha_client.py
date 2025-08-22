import base64
import json
import logging
import requests
from typing import Any, Literal, Union, List, Dict
from decouple import config
from cachetools import TTLCache

# -------------------------------
# Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
TIMEOUT = 10  # seconds

logger = logging.getLogger("koha_client")

# A set to store phrases that have already been searched in this session.
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
def search_books(query: str, session_id: str = "global") -> Union[list[dict[str, Any]], dict[str, str]]:
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
            logger.error(f"[Koha Search] Error: {e}")
            return {"error": f"Koha API unreachable: {e}"}

    return {"error": "No books found."}

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

def search_by_identifiers(identifiers: Dict[str, List[str]]) -> Union[list[dict[str, Any]], dict[str, str]]:
    headers = get_auth_headers()

    # 1) ISBN
    for isbn in identifiers.get("isbn", []):
        for v in _with_period_variants(isbn):
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: v})}"
                logger.info(f"[Koha Lookup] ISBN exact via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (isbn-exact)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] ISBN exact error for {v!r}: {e}")

        for v in _with_period_variants(isbn):
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: _like_contains(v)})}"
                logger.info(f"[Koha Lookup] ISBN contains via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (isbn-contains)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] ISBN contains error for {v!r}: {e}")

    # 2) ISSN 
    for issn in identifiers.get("issn", []):
        for v in _with_period_variants(issn):
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: v})}"
                logger.info(f"[Koha Lookup] ISSN exact via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (issn-exact)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] ISSN exact error for {v!r}: {e}")

        for v in _with_period_variants(issn):
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: _like_contains(v)})}"
                logger.info(f"[Koha Lookup] ISSN contains via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (issn-contains)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] ISSN contains error for {v!r}: {e}")

    # 3) Call Numbers 
    import re as _re
    callnos = identifiers.get("call_numbers", [])
    for cn in callnos:
        variants = _with_period_variants(cn)
        variants += [
            _re.sub(r"\s+", " ", cn).strip(),     
            cn.replace(".", " ").strip(),     
        ]
        # de-dup while preserving order
        seen = set(); variants = [x for x in variants if not (x in seen or seen.add(x))]

        # exact
        for v in variants:
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: v})}"
                logger.info(f"[Koha Lookup] CALLNO exact via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (callno-exact)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] CALLNO exact error for {v!r}: {e}")

        # contains
        for v in variants:
            try:
                url = f"{API_URL}?q={_q({FIELD_ISBN: _like_contains(v)})}"
                logger.info(f"[Koha Lookup] CALLNO contains via {FIELD_ISBN}: {v!r}")
                data = _get(url, headers)
                if data:
                    out = _format_list(data)
                    for b in out:
                        b["matched_on"] = {"field": f"{FIELD_ISBN} (callno-contains)", "value": v}
                    return out
            except Exception as e:
                logger.warning(f"[Koha Lookup] CALLNO contains error for {v!r}: {e}")

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