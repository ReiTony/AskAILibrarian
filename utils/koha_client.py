import requests
import json
import base64
import logging
from decouple import config

# -------------------------------
# Koha API Configuration
# -------------------------------
API_URL = config("KOHA_API")
USERNAME = config("KOHA_USERNAME")
PASSWORD = config("KOHA_PASSWORD")
KOHA_REQUEST_TIMEOUT = 6  

# -------------------------------
# Create Authorization Headers
# -------------------------------
def get_auth_headers():
    """Returns HTTP headers with Basic Auth for Koha API."""
    auth_string = f"{USERNAME}:{PASSWORD}"
    base64_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    return {
        'Authorization': f'Basic {base64_auth_string}',
        'Accept': 'application/json'
    }
    
# -------------------------------
# Search Books by Type and Query
# -------------------------------
def search_books(search_type: str, query: str):
    """
    Searches books in Koha. Tries the full query and (at most) one backup phrase.
    Returns list of books or an error message.
    """
    search_phrases = [query]
    words = query.split()
    if words and words[0] != query:
        search_phrases.append(words[0])

    headers = get_auth_headers()
    last_error = None
    logged_error = False

    for idx, phrase in enumerate(search_phrases):
        search_params = {
            search_type: {'-like': f"%{phrase}%"}
        }
        encoded_query = json.dumps(search_params)
        url = f'{API_URL}?q={encoded_query}'

        logging.info(f"Trying search with phrase: {phrase}")

        try:
            response = requests.get(url, headers=headers, timeout=KOHA_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if data:
                return [
                    {
                        'title': book.get('title', 'N/A'),
                        'publisher': book.get('publisher', 'N/A'),
                        'isbn': book.get('isbn', 'N/A'),
                        'quantity': book.get('quantity', 'N/A'),
                        'author': book.get('author', 'N/A'),
                        'year': book.get('copyright_date', 'N/A'),
                    }
                    for book in data
                ]
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            last_error = e
            if not logged_error:
                logging.error(f"Koha API error: {e}")
                logged_error = True
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                break
            continue

    if last_error:
        return {"error": f"Koha API unavailable or unreachable. Last error: {last_error}"}
    else:
        return {"error": "No books found."}
