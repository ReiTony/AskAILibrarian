import logging
import re
import asyncio
import spacy
from threading import RLock
from cachetools import TTLCache
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from utils.llm_client import generate_response
from utils.koha_client import search_books, search_specific_book, fetch_quantity_from_biblio_id
from utils.sessions import ChatSession, get_chat_session
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


# ---------- Main API Route ----------
@router.post("/search_books")
async def search_books_api(request: Request, chat_session: ChatSession = Depends(get_chat_session)):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return JSONResponse(content={"error": "Query parameter is required"}, status_code=400)

        query_clean = clean_query_text(user_query)
        intent = classify_book_intent(query_clean)
        logger.info(f"[Intent] Detected: {intent}")

        try:
            history = await chat_session.get_history()
        except Exception as e:
            logger.warning(f"Failed to fetch chat history: {e}")
            history = []

        history_text = "\n".join(f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in history)

        # ---------- INTENT: Specific Book ----------
        if intent == "specific_book_search":
            isbn = extract_isbn(user_query)
            logger.info(f"[Specific Book Search] ISBN: {isbn}")
            result = await asyncio.to_thread(search_specific_book, "isbn", isbn)

            if not result or (isinstance(result, dict) and "error" in result):
                return JSONResponse(content={"response": [{
                    "type": "specific_book_search",
                    "answer": specific_book_not_found_prompt(isbn),
                    "books": [],
                }]}, status_code=200)

            formatted = [{
                "title": replace_null(b.get("title")).strip(" ,;:"),
                "author": replace_null(b.get("author")).strip(" ,;:"),
                "isbn": replace_null(b.get("isbn")),
                "publisher": replace_null(b.get("publisher")).strip(" ,;:"),
                "year": replace_null(b.get("year")),
                "biblio_id": replace_null(b.get("biblio_id")),
                "quantity_available": await asyncio.to_thread(fetch_quantity_from_biblio_id, replace_null(b.get("biblio_id"))),
            } for b in result[:5]]

            prompt = specific_book_found_prompt(formatted[0]["title"], formatted[0]["isbn"])
            bot_response = await generate_response(prompt)
            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", bot_response)

            return JSONResponse(content={"response": [{
                "type": "specific_book_search",
                "answer": bot_response,
                "books": formatted
            }]}, status_code=200)

        # ---------- INTENT: Recommendations ----------
        elif intent == "recommend":
            keywords = await expand_query(query_clean)
            logger.info(f"[Recommend] Expanded: {keywords}")
            raw_results = await koha_multi_search(keywords)

            if not raw_results or "answer" in raw_results[0]:
                return JSONResponse(content={"answer": raw_results[0]["answer"]}, status_code=200)

            aggregated = {}
            for book in raw_results:
                key = f"{replace_null(book.get('title'))}|{replace_null(book.get('author'))}"
                if key not in aggregated and "Not Available" not in key:
                    aggregated[key] = {
                        "title": replace_null(book.get("title")),
                        "author": replace_null(book.get("author")),
                        "isbn": replace_null(book.get("isbn")),
                        "publisher": replace_null(book.get("publisher")),
                        "biblio_id": replace_null(book.get("biblio_id")),
                        "quantity_available": await asyncio.to_thread(fetch_quantity_from_biblio_id, replace_null(book.get("biblio_id"))),
                        "year": replace_null(book.get("year")),
                    }
                    if len(aggregated) >= 10:
                        break

            books = list(aggregated.values())
            prompt = recommend_books_prompt(query_clean, history_text)
            try:
                bot_response = await generate_response(prompt)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                bot_response = "Here are some recommended books based on your topic."

            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", bot_response)

            return JSONResponse(content={"response": [{
                "type": "recommendation",
                "answer": bot_response,
                "books": books
            }]}, status_code=200)

        # ---------- INTENT: General Search ----------
        if not fuzzy_match_keywords(query_clean.split(), SEARCH_BOOKS_KEYWORDS, threshold=75):
            return JSONResponse(content={"error": "Query does not contain valid search keywords."}, status_code=400)

        keywords = await expand_query(user_query)
        logger.info(f"[Search] Expanded: {keywords}")
        cache_key = _agg_key(keywords)

        with AGG_LOCK:
            cached_books = AGG_CACHE.get(cache_key)
        if cached_books:
            logger.info(f"[Search] Served from cache: {len(cached_books)}")
            books = cached_books
        else:
            raw_results = await koha_multi_search(keywords)
            if not raw_results or "answer" in raw_results[0]:
                return JSONResponse(content={"answer": raw_results[0]["answer"]}, status_code=200)

            books, seen = {}, set()
            for book in raw_results:
                title = replace_null(book.get("title"))
                if title in seen or title == "Not Available":
                    continue
                key = replace_null(book.get("isbn")) or f"{title}|{replace_null(book.get('author'))}"
                books[key] = {
                    "title": title.strip(" ,;:"),
                    "author": replace_null(book.get("author")).strip(" ,;:"),
                    "isbn": replace_null(book.get("isbn")),
                    "publisher": replace_null(book.get("publisher")).strip(" ,;:"),
                    "quantity_available": await asyncio.to_thread(fetch_quantity_from_biblio_id, replace_null(book.get("biblio_id"))),
                    "year": replace_null(book.get("year")),
                    "biblio_id": replace_null(book.get("biblio_id")),
                }
                seen.add(title)

            formatted_books = list(books.values())
            with AGG_LOCK:
                AGG_CACHE[cache_key] = formatted_books

        prompt = search_books_prompt(query_clean, history_text)
        try:
            bot_response = await generate_response(prompt)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            bot_response = "Here are the books I found that match your search."

        await chat_session.add_message("user", user_query)
        await chat_session.add_message("assistant", bot_response)

        return JSONResponse(content={"response": [{
            "type": "booksearch",
            "answer": bot_response,
            "books": formatted_books,
        }]}, status_code=200)

    except Exception as e:
        logger.error(f"[Search Books] Unhandled error: {e}")
        return JSONResponse(content={"error": "Internal server error."}, status_code=500)
