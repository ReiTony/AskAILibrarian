import base64
import json
import logging
import requests
from typing import Any, Literal, Union
from decouple import config

# -------------------------------
# Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
TIMEOUT = 10  # seconds

logger = logging.getLogger("koha_client")

# NEW: A set to store phrases that have already been searched in this session.
_searched_phrases = set()

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
# Search Books (General)
# -------------------------------
def search_books(
    search_type: Literal["title", "author", "publisher"],
    query: str
) -> Union[list[dict[str, Any]], dict[str, str]]:
    """
    Searches Koha using the specified field (title, author, publisher).
    Attempts both the full query and a fallback on the first word,
    avoiding duplicate searches within the same session.
    """
    headers = get_auth_headers()
    phrases = [query]
    
    if (words := query.split()) and words[0].lower() != query.lower():
        phrases.append(words[0])

    for phrase in phrases:
        # MODIFIED: Check if we have already searched for this phrase
        if phrase.lower() in _searched_phrases:
            logger.info(f"[Koha Search] Skipping duplicate search for: {phrase}")
            continue

        params = {search_type: {"-like": f"%{phrase}%"}}
        try:
            url = f"{API_URL}?q={json.dumps(params)}"
            logger.info(f"[Koha Search] Searching {search_type} with: {phrase}")
            
            # MODIFIED: Add the phrase to our set of searched terms *before* the request
            _searched_phrases.add(phrase.lower())

            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()

            books = response.json()
            if books:
                # We found results, so we can stop searching and return them.
                return [format_book_data(book) for book in books]

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"[Koha Search] Error: {e}")
            return {"error": f"Koha API unreachable: {e}"}

    return {"error": "No books found."}


def fetch_quantity_from_biblio_id(biblio_id: str) -> int:
    """Fetch number of items available for a given biblio_id."""
    try:
        url = f"http://192.168.1.68:8080/api/v1/biblios/{biblio_id}/items"
        headers = get_auth_headers()
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        items = response.json()
        return len(items) if isinstance(items, list) else 0
    except Exception as e:
        print(f"[Koha] Error fetching quantity for biblio_id {biblio_id}: {e}")
        return 0
    
# -------------------------------
# Search Specific Book (e.g., ISBN)
# -------------------------------
def search_specific_book(
    field: Literal["isbn", "title", "author"],
    value: str
) -> Union[list[dict[str, Any]], dict[str, str]]:
    """
    Searches Koha using an exact match on the given field.
    """
    headers = get_auth_headers()
    value = value.strip()

    query = json.dumps({field: value})
    url = f"{API_URL}?q={query}"
    logger.info(f"[Koha Specific Search] Final URL: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        books = response.json()

        return [format_book_data(book) for book in books] if books else []

    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"[Koha Specific Search] Error: {e}")
        return {"error": f"Failed to fetch from Koha: {e}"}

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