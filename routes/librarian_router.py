import logging
import re
import asyncio
import spacy
from threading import RLock
from cachetools import TTLCache
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from utils.llm_client import generate_response
from utils.koha_client import search_books, search_specific_book
from utils.sessions import get_session_and_user_data
from utils.text_utils import (
    replace_null,
    clean_query_text,
    fuzzy_match_keywords,
    classify_book_intent,
    extract_isbn,
)
from utils.prompt_templates import (
    search_books_prompt,
    specific_book_found_prompt,
    specific_book_not_found_prompt,
    recommend_books_prompt,
)

from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.chat_retention import get_retained_history, save_conversation_turn # <-- NEW
from db.connection import get_db
from utils.koha_client import search_books, search_specific_book, fetch_quantity_from_biblio_id

router = APIRouter()
logger = logging.getLogger("search_books_api")
nlp = spacy.load("en_core_web_sm")

# Caches
EXPANSION_CACHE = TTLCache(maxsize=1000, ttl=86400)  # 24h
KOHA_CACHE = TTLCache(maxsize=5000, ttl=600)         # 10m
AGG_CACHE = TTLCache(maxsize=2000, ttl=600)          # 10m

EXPANSION_LOCK = RLock()
KOHA_LOCK = RLock()
AGG_LOCK = RLock()

SEARCH_BOOKS_KEYWORDS = {"search", "quantity", "title", "find", "lookup", "copies", "available", "book", "books"}
KOHA_TIMEOUT_SECONDS = 8


# ---------- Utility Functions ----------
def extract_search_terms(text: str) -> list[str]:
    return [t.text for t in nlp(text) if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop]

def parse_llm_keyword_list(s: str, max_terms: int = 12) -> list[str]:
    parts = re.split(r"[,|\n]+", s)
    seen, out = set(), []
    for p in parts:
        kw = p.strip().strip("\"“”'").lower()
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
            if len(out) >= max_terms:
                break
    return out

def _cache_key(field: str, term: str) -> str:
    return f"{field}::{term.strip().lower()}"

def _agg_key(keywords: list[str]) -> str:
    return f"agg::{'|'.join(sorted(set(map(str.lower, keywords))))}"


# ---------- Cached Operations ----------
async def expand_query(user_query: str) -> list[str]:
    qnorm = clean_query_text(user_query).lower()
    with EXPANSION_LOCK:
        if qnorm in EXPANSION_CACHE:
            return EXPANSION_CACHE[qnorm]

    prompt = (
        "You are helping to search a library catalog. Expand the user's topic into 8–15 concise search terms.\n"
        f"User topic: {user_query!r}\n\n"
        "Rules:\n- Return ONLY a comma-separated list (no bullets, no numbering).\n"
        "- Prefer concrete book title terms.\n"
        "- Avoid made-up phrases."
    )
    try:
        raw = await generate_response(prompt)
        keywords = parse_llm_keyword_list(raw)
        if not keywords:
            raise ValueError("LLM returned empty keywords")
    except Exception as e:
        logger.error(f"[LLM expand] fallback triggered: {e}")
        keywords = extract_search_terms(user_query) or [user_query]

    with EXPANSION_LOCK:
        EXPANSION_CACHE[qnorm] = keywords
    return keywords

def cached_koha_search(field: str, term: str):
    key = _cache_key(field, term)
    with KOHA_LOCK:
        if key in KOHA_CACHE:
            return KOHA_CACHE[key]

    result = search_books(field, term)
    with KOHA_LOCK:
        KOHA_CACHE[key] = result
    return result

async def koha_multi_search(keywords: list[str]) -> list[dict]:
    keywords = keywords[:8]
    tasks = [asyncio.to_thread(cached_koha_search, "title", kw) for kw in keywords]

    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=KOHA_TIMEOUT_SECONDS + 4)
    except asyncio.TimeoutError:
        logger.error("[Koha] all search tasks timed out")
        return [{"answer": "Sorry, our book database is taking too long. Please try again."}]

    books, errors = [], 0
    for i, res in enumerate(results):
        if isinstance(res, Exception) or (isinstance(res, dict) and "error" in res):
            logger.error(f"[Koha] error for '{keywords[i]}': {res}")
            errors += 1
            continue
        if not res:
            continue
        books.extend(res)

    if not books and errors == len(keywords):
        return [{"answer": "Library database is currently unavailable or returned no results."}]
    return books

async def fetch_and_add_quantities(books: list[dict]) -> list[dict]:
    """
    Fetches quantities for a list of books in parallel and adds the 'quantity_available' key.
    """
    if not books:
        return []

    # Create a list of tasks to run concurrently
    tasks = [
        asyncio.to_thread(fetch_quantity_from_biblio_id, book.get("biblio_id"))
        for book in books if book.get("biblio_id")
    ]
    
    # Run all tasks in parallel
    quantities = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Map the fetched quantities back to the books
    for i, book in enumerate(books):
        biblio_id = book.get("biblio_id")
        if biblio_id:
            quantity_result = quantities[i]
            if isinstance(quantity_result, Exception):
                logger.error(f"Failed to fetch quantity for biblio_id {biblio_id}: {quantity_result}")
                book["quantity_available"] = "Error"
            else:
                book["quantity_available"] = quantity_result
        else:
            book["quantity_available"] = "N/A" # Or 0, or some other default

    return books


# ---------- Main API Route ----------
@router.post("/search_books")
async def search_books_api(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)#Huwag alisin. For CardNumber compatibility
        if not user_query:
            return JSONResponse(content={"error": "Query parameter is required"}, status_code=400)

        query_clean = clean_query_text(user_query)
        intent = classify_book_intent(query_clean)
        logger.info(f"[Intent] Detected: {intent}")

        try:
            # Get long-term history from our new utility
            retained_history = await get_retained_history(db, cardnumber)
            # Get short-term (current session) history
            session_history = await chat_session.get_history()
            # Combine them for full context. Retained history comes first.
            full_history = retained_history + session_history
        except Exception as e:
            logger.warning(f"Failed to fetch or combine chat history: {e}")
            full_history = []
        
        # Use the combined history for the LLM prompt
        history_text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in full_history)

        # ---------- INTENT: Specific Book ----------
        if intent == "specific_book_search":
            logger.info("Executing path: specific_book_search")
            isbn = extract_isbn(user_query)
            result = await asyncio.to_thread(search_specific_book, "isbn", isbn)

            if not result or (isinstance(result, dict) and "error" in result):
                bot_response = specific_book_not_found_prompt(isbn)
            else:
                formatted_books = [{"title": replace_null(b.get("title")).strip(" ,;:"), "author": replace_null(b.get("author")).strip(" ,;:"), "isbn": replace_null(b.get("isbn")), "publisher": replace_null(b.get("publisher")).strip(" ,;:"), "year": replace_null(b.get("year")), "biblio_id": replace_null(b.get("biblio_id"))} for b in result[:5]]
                books_with_quantities = await fetch_and_add_quantities(formatted_books)
            
                bot_response = specific_book_found_prompt(books_with_quantities[0]["title"], books_with_quantities[0]["isbn"])
            await save_conversation_turn(db, cardnumber, user_query, bot_response)
            return JSONResponse(content={"response": [{"type": "specific_book_search", "answer": bot_response, "books": books_with_quantities}]}, status_code=200)

        # ---------- INTENT: Recommendations ----------
        elif intent == "recommend":
            logger.info("Executing path: recommend")
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)

            if not raw_results or ("answer" in raw_results[0] and len(raw_results) == 1):
                bot_response = raw_results[0].get("answer", f"I couldn't find any books to recommend for '{user_query}'.")
                await save_conversation_turn(db, cardnumber, user_query, bot_response)
                return JSONResponse(content={"answer": bot_response}, status_code=200)
            
            # Process results...
            aggregated = {}
            for book in raw_results:
                key = f"{replace_null(book.get('title'))}|{replace_null(book.get('author'))}"
                if key not in aggregated and "Not Available" not in key:
                    aggregated[key] = {"title": replace_null(book.get("title")), "author": replace_null(book.get("author")), "isbn": replace_null(book.get("isbn")), "publisher": replace_null(book.get("publisher")), "biblio_id": replace_null(book.get("biblio_id")), "year": replace_null(book.get("year"))}
                    if len(aggregated) >= 15: # Limit the number of books before fetching quantities
                        break
            
            books_without_quantities = list(aggregated.values())
            
            # Now fetch quantities for the aggregated list in parallel
            books_with_quantities = await fetch_and_add_quantities(books_without_quantities)
            
            # DEFINE PROMPT and call LLM
            prompt = recommend_books_prompt(query_clean, history_text, user_query)
            bot_response = await generate_response(prompt)

            logger.info(f"=======Recieved Prompt========: \n{prompt}\n\n")

            await save_conversation_turn(db, cardnumber, user_query, bot_response)
            return JSONResponse(content={"response": [{"type": "recommendation", "answer": bot_response, "books": books_with_quantities}]}, status_code=200)

        # ---------- INTENT: Search ----------
        elif intent == "search":
            logger.info("Executing path: search")
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)

            if not raw_results or ("answer" in raw_results[0] and len(raw_results) == 1):
                bot_response = raw_results[0].get("answer", f"I couldn't find any books for '{user_query}'.")
                await save_conversation_turn(db, cardnumber, user_query, bot_response)
                return JSONResponse(content={"answer": bot_response}, status_code=200)

            # Process results...
            aggregated = {}
            for book in raw_results:
                key = f"{replace_null(book.get('title'))}|{replace_null(book.get('author'))}"
                if key not in aggregated and "Not Available" not in key:
                    aggregated[key] = {"title": replace_null(book.get("title")), "author": replace_null(book.get("author")), "isbn": replace_null(book.get("isbn")), "publisher": replace_null(book.get("publisher")), "biblio_id": replace_null(book.get("biblio_id")), "year": replace_null(book.get("year"))}
                    if len(aggregated) >= 15: # Limit the number of books before fetching quantities
                        break
            
            books_without_quantities = list(aggregated.values())
            
            # Now fetch quantities for the aggregated list in parallel
            books_with_quantities = await fetch_and_add_quantities(books_without_quantities)

            # DEFINE PROMPT and call LLM
            prompt = search_books_prompt(query_clean, history_text, user_query)
            bot_response = await generate_response(prompt)

            logger.info(f"=======Recieved Prompt========: \n{prompt}\n\n")
            
            await save_conversation_turn(db, cardnumber, user_query, bot_response)
            return JSONResponse(content={"response": [{"type": "booksearch", "answer": bot_response, "books": books_with_quantities}]}, status_code=200)
        
        else: 
            logger.warning(f"Query did not match any specific route. Intent: '{intent}'. Falling back.")
            # You can either return an error or route to the general library_info
            return JSONResponse(content={"error": "I'm not sure how to handle that request. Please try rephrasing, for example 'find books about...'"}, status_code=400)
    
    except Exception as e:
        logger.error(f"[Search Books] Unhandled error: {e}")
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)