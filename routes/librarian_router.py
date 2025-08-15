import logging
import re
import asyncio
import spacy
from threading import RLock
from cachetools import TTLCache
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.llm_client import generate_response
from utils.koha_client import search_books, search_specific_book, fetch_quantity_from_biblio_id
from utils.sessions import get_session_and_user_data
from utils.chat_retention import get_retained_history, save_conversation_turn
from utils.text_utils import (
    replace_null, clean_query_text,  extract_isbn
)
from utils.prompt_templates import (
    search_books_prompt, specific_book_found_prompt,
    specific_book_not_found_prompt, recommend_books_prompt,
    contextual_search_topic_prompt
)
from db.connection import get_db

router = APIRouter()
logger = logging.getLogger("search_books_api")
nlp = spacy.load("en_core_web_sm")

# Caches
EXPANSION_CACHE = TTLCache(maxsize=1000, ttl=86400)  # 24h
KOHA_CACHE = TTLCache(maxsize=5000, ttl=600)         # 10m

EXPANSION_LOCK = RLock()
KOHA_LOCK = RLock()

KOHA_TIMEOUT_SECONDS = 8


# ---------- Utility ----------
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

async def resolve_search_topic(user_query: str, history_text: str) -> str:
    """
    Uses an LLM to determine the true search topic based on conversation context.
    Returns the resolved topic as a string.
    """
    # A simple heuristic: if the query is very short or contains follow-up words,
    follow_up_words = {'more', 'another', 'else', 'other', 'others', 'again'}
    query_words = set(user_query.lower().split())

    # If the query is long or doesn't seem like a follow-up, just use it as is
    # to save an LLM call. You can adjust this logic.
    if len(query_words) > 4 and not query_words.intersection(follow_up_words):
        logger.info("[Context Resolver] Query is specific, skipping LLM resolution.")
        return user_query

    logger.info(f"[Context Resolver] Query '{user_query}' might be a follow-up. Asking LLM for context.")
    try:
        # Ask the LLM to figure out the real topic
        prompt = contextual_search_topic_prompt(history_text, user_query)
        resolved_topic = await generate_response(prompt)
        
        # Clean up the LLM response
        resolved_topic = resolved_topic.strip().strip('"')
        
        if not resolved_topic:
             logger.warning("[Context Resolver] LLM returned empty topic, falling back to original query.")
             return user_query

        logger.info(f"[Context Resolver] Resolved topic: '{user_query}' -> '{resolved_topic}'")
        return resolved_topic

    except Exception as e:
        logger.error(f"[Context Resolver] Error resolving topic: {e}. Falling back to original query.")
        return user_query

# ---------- Cached + Parallel Ops ----------
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
    tasks = [asyncio.to_thread(cached_koha_search, "title", kw) for kw in keywords[:8]]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=KOHA_TIMEOUT_SECONDS + 4)
    except asyncio.TimeoutError:
        logger.error("[Koha] search timeout")
        return [{"answer": "Sorry, our book database is taking too long."}]
    
    books, errors = [], 0
    for i, res in enumerate(results):
        if isinstance(res, Exception) or (isinstance(res, dict) and "error" in res):
            errors += 1
            continue
        if res:
            books.extend(res)

    if not books and errors == len(keywords):
        return [{"answer": "Library database is unavailable or empty."}]
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


# ---------- Main Route ----------
@router.post("/search_books")
async def search_books_api(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db),
    intent: str = None   # <-- Passed by the main router!
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)
        if not user_query:
            return JSONResponse(content={"error": "Query is required."}, status_code=400)

        logger.info(f"[search_books_api] Received intent: {intent}")
        # Load chat history for context
        try:
            full_history = await get_retained_history(db, cardnumber) + await chat_session.get_history()
        except Exception as e:
            logger.warning(f"History load error: {e}")
            full_history = []
        history_text = "\n".join(f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in full_history[-6:])

        if intent in ["book_search", "book_recommend"]:
            contextual_query = await resolve_search_topic(user_query, history_text)
        else:
            contextual_query = user_query
        
        query_clean = clean_query_text(contextual_query)

        # ----- ISBN Lookup -----
        if intent == "book_lookup_isbn":
            isbn = extract_isbn(user_query)
            books = await asyncio.to_thread(search_specific_book, "isbn", isbn)
            if not books or (isinstance(books, dict) and "error" in books):
                bot_reply = specific_book_not_found_prompt(isbn)
                await save_conversation_turn(db, cardnumber, user_query, bot_reply)
                return JSONResponse(content={"response": [{"type": "specific_book_search", "answer": bot_reply, "books": []}]}, status_code=200)

            formatted = [{
                "title": replace_null(b.get("title")).strip(" ,;:"),
                "author": replace_null(b.get("author")).strip(" ,;:"),
                "isbn": replace_null(b.get("isbn")),
                "publisher": replace_null(b.get("publisher")),
                "year": replace_null(b.get("year")),
                "biblio_id": replace_null(b.get("biblio_id"))
            } for b in books[:5]]

            books = await fetch_and_add_quantities(formatted)
            bot_reply = specific_book_found_prompt(books[0]["title"], books[0]["isbn"])
            await save_conversation_turn(db, cardnumber, user_query, bot_reply)
            return JSONResponse(content={"response": [{"type": "specific_book_search", "answer": bot_reply, "books": books}]}, status_code=200)

        # ----- Book Recommendation -----
        elif intent == "book_recommend":
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)
            if not raw_results or ("answer" in raw_results[0] and len(raw_results) == 1):
                reply = raw_results[0]["answer"]
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(content={"answer": reply}, status_code=200)

            books = {}
            for book in raw_results:
                key = f"{replace_null(book.get('title'))}|{replace_null(book.get('author'))}"
                if key not in books:
                    books[key] = {
                        "title": replace_null(book.get("title")),
                        "author": replace_null(book.get("author")),
                        "isbn": replace_null(book.get("isbn")),
                        "publisher": replace_null(book.get("publisher")),
                        "biblio_id": replace_null(book.get("biblio_id")),
                        "year": replace_null(book.get("year")),
                    }
                    if len(books) >= 10:
                        break

            books = await fetch_and_add_quantities(list(books.values()))
            prompt = recommend_books_prompt(query_clean, history_text, user_query)
            reply = await generate_response(prompt)
            await save_conversation_turn(db, cardnumber, user_query, reply)
            return JSONResponse(content={"response": [{"type": "recommendation", "answer": reply, "books": books}]}, status_code=200)

        # ----- Book Search -----
        elif intent == "book_search":
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)
            if not raw_results or ("answer" in raw_results[0] and len(raw_results) == 1):
                reply = raw_results[0]["answer"]
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(content={"answer": reply}, status_code=200)

            books = {}
            for book in raw_results:
                key = f"{replace_null(book.get('title'))}|{replace_null(book.get('author'))}"
                if key not in books:
                    books[key] = {
                        "title": replace_null(book.get("title")),
                        "author": replace_null(book.get("author")),
                        "isbn": replace_null(book.get("isbn")),
                        "publisher": replace_null(book.get("publisher")),
                        "biblio_id": replace_null(book.get("biblio_id")),
                        "year": replace_null(book.get("year")),
                    }
                    if len(books) >= 50:
                        break

            books = await fetch_and_add_quantities(list(books.values()))
            prompt = search_books_prompt(query_clean, history_text, user_query)
            reply = await generate_response(prompt)
            await save_conversation_turn(db, cardnumber, user_query, reply)
            return JSONResponse(content={"response": [{"type": "booksearch", "answer": reply, "books": books}]}, status_code=200)

        else:
            return JSONResponse(content={"error": "Unrecognized query intent."}, status_code=400)

    except Exception as e:
        logger.error(f"[Unhandled Error] {e}")
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)