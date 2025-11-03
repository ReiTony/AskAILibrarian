import logging
import asyncio
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Import Koha client helpers
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.koha_client import search_books, format_book_data, fetch_items_for_multiple_biblios
from utils.text_utils import clean_query_text
from utils.llm_client import generate_response

logger = logging.getLogger(__name__)


# ---------- Query Expansion ----------
async def expand_query(user_query: str) -> list[str]:
    """Expand user query into multiple useful keywords using LLM."""
    prompt = (
        "You are helping to search a library catalog. Expand the user's topic into 5 concise search terms.\n"
        f"User topic: {user_query!r}\n\n"
        "Rules:\n- Return ONLY a comma-separated list (no bullets, no numbering).\n"
        "- Prefer concrete subject/keyword terms.\n"
        "- Avoid made-up phrases.\n"
        "- Do NOT return generic terms like 'books', 'novels', 'stories'.\n"
        "- Do not repeat keywords.\n"
        "- Don't return specific book titles.\n"
        "- Avoid returning incoherent phrases."
    )
    try:
        raw = await generate_response(prompt)
        keywords = [kw.strip().lower() for kw in raw.split(",") if kw.strip()]
        if not keywords:
            raise ValueError("Empty keyword list")
        return keywords
    except Exception as e:
        logger.warning(f"[expand_query] LLM failed: {e}, using fallback.")
        return [user_query]


# ---------- Multi Search ----------
async def koha_multi_search(keywords: list[str]) -> list[dict]:
    """Run multiple Koha searches in parallel and return deduped results."""
    sem = asyncio.Semaphore(3)

    async def safe_search(term: str):
        async with sem:
            try:
                return await asyncio.to_thread(search_books, term)
            except Exception as e:
                logger.error(f"[Koha] error for {term!r}: {e}")
                return []

    tasks = [safe_search(kw) for kw in keywords[:8]]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    books = []
    for r in results:
        if isinstance(r, list):
            books.extend(r)

    # Deduplicate by biblio_id
    unique_books = {b.get("biblio_id"): b for b in books if b.get("biblio_id")}
    return list(unique_books.values())[:100]


# ---------- Quantity Enrichment ----------
async def fetch_and_add_quantities(books: list[dict]) -> list[dict]:
    """Batch fetch quantities and merge into book results."""
    biblio_ids = [
        b.get("biblio_id") for b in books
        if b.get("biblio_id") and b.get("biblio_id") != "N/A"
    ]
    if not biblio_ids:
        for b in books:
            b["quantity_available"] = 0
        return books

    try:
        items_by_biblio = await asyncio.to_thread(fetch_items_for_multiple_biblios, biblio_ids)
        for book in books:
            book["quantity_available"] = 0
            try:
                bid = int(book.get("biblio_id"))
                if bid in items_by_biblio:
                    book["quantity_available"] = len(items_by_biblio[bid])
            except (ValueError, TypeError):
                pass
    except Exception as e:
        logger.error(f"[Quantity Fetch] Batch fetch error: {e}")
        for b in books:
            b["quantity_available"] = 0

    return books


# ---------- Rasa Action ----------
class ActionSearchBook(Action):
    def name(self):
        return "action_search_book"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: dict):

        query_raw = (
            next(tracker.get_latest_entity_values("isbn"), None)
            or next(tracker.get_latest_entity_values("title"), None)
            or next(tracker.get_latest_entity_values("author"), None)
        )

        if not query_raw:
            dispatcher.utter_message(text="Please provide a title, author, or ISBN to search.")
            return []

        query_clean = clean_query_text(query_raw)
        logger.info(f"[Koha Action] Searching for: {query_clean!r}")

        try:
            keywords = await expand_query(query_clean)
            results = await koha_multi_search(keywords)

            if not results:
                dispatcher.utter_message(text=f"Sorry, I couldn't find any books for '{query_clean}'.")
                return []

            # Add quantities in batch
            results = await fetch_and_add_quantities(results)

            dispatcher.utter_message(text="Here are some books I found:")

            for book in results:
                book_fmt = format_book_data(book)
                dispatcher.utter_message(json_message={
                    "type": "book",
                    "title": book_fmt.get("title", "Unknown Title"),
                    "author": book_fmt.get("author", "Unknown Author"),
                    "isbn": book_fmt.get("isbn", "N/A"),
                    "publisher": book_fmt.get("publisher", "N/A"),
                    "year": book_fmt.get("year", "N/A"),
                    "quantity_available": book.get("quantity_available", 0),
                    "biblio_id": book_fmt.get("biblio_id", "N/A"),
                })

        except Exception as e:
            logger.error(f"[Koha Action] Fatal error: {e}")
            dispatcher.utter_message(text="There was a problem searching the catalog.")

        return []

class ActionLookupISBN(Action):
    def name(self):
        return "action_lookup_isbn"

    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: dict):

        user_query = tracker.latest_message.get("text", "").strip()
        if not user_query:
            dispatcher.utter_message(text="Please provide an ISBN to search.")
            return []

        logger.info(f"[Koha Action] ISBN lookup for: {user_query!r}")

        try:
            # Extract identifiers (ISBN/ISSN/Call Number)
            from utils.text_utils import extract_identifiers
            ids = extract_identifiers(user_query)

            if not any(ids.values()):
                dispatcher.utter_message(text="Sorry, I couldn’t detect a valid ISBN/ISSN/Call Number.")
                return []

            # Call Koha search
            from utils.koha_client import search_by_identifiers
            results = await asyncio.to_thread(search_by_identifiers, ids)

            if not results or (isinstance(results, dict) and "error" in results):
                dispatcher.utter_message(text="Sorry, no records found for that ISBN/ISSN/Call Number.")
                return []

            # Enrich with quantities
            results = await fetch_and_add_quantities(results)

            # Return results
            for book in results[:5]:  
                dispatcher.utter_message(json_message={
                    "type": "book",
                    "title": book.get("title", "Unknown Title"),
                    "author": book.get("author", "Unknown Author"),
                    "isbn": book.get("isbn", "N/A"),
                    "publisher": book.get("publisher", "N/A"),
                    "year": book.get("year", "N/A"),
                    "quantity_available": book.get("quantity_available", 0),
                    "biblio_id": book.get("biblio_id", "N/A"),
                })

        except Exception as e:
            logger.error(f"[Koha Action] ISBN lookup failed: {e}")
            dispatcher.utter_message(text="There was a problem performing the ISBN lookup.")
            return []

        return []
    
class ActionGeneralQuery(Action):
    def name(self):
        return "action_general_query"

    async def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text", "").strip()

        if not user_query:
            dispatcher.utter_message(custom={
                "answer": "Could you repeat that?"
            })
            return []

        try:
            response = await generate_response(f"Answer the user query clearly: {user_query}")
            dispatcher.utter_message(custom={
                "answer": response
            })
        except Exception as e:
            logger.error(f"[General Query] LLM failed: {e}")
            dispatcher.utter_message(custom={
                "answer": "Sorry, I couldn’t answer that right now."
            })

        return []
