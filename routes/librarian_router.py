import logging
import re
import asyncio
import spacy
from threading import RLock
from cachetools import TTLCache
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from motor.motor_asyncio import AsyncIOMotorDatabase
from utils.llm_client import generate_response
from utils.koha_client import (
    search_books,
    # fetch_quantity_from_biblio_id,
    # fetch_and_add_quantities,
    fetch_items_for_multiple_biblios,
    search_by_identifiers,
)

from utils.sessions import get_session_and_user_data
from utils.chat_retention import get_retained_history, save_conversation_turn

from utils.text_utils import replace_null, clean_query_text, extract_identifiers
from utils.prompt_templates import (
    search_books_prompt,
    specific_book_found_prompt,
    specific_book_not_found_prompt,
    recommend_books_prompt,
    contextual_search_topic_prompt
)

from db.connection import get_db

router = APIRouter()
logger = logging.getLogger("search_books_api")
nlp = spacy.load("en_core_web_sm")

# Caches
EXPANSION_CACHE = TTLCache(maxsize=1000, ttl=86400)  # 24h
KOHA_CACHE = TTLCache(maxsize=5000, ttl=600)  # 10m

EXPANSION_LOCK = RLock()
KOHA_LOCK = RLock()

KOHA_TIMEOUT_SECONDS = 8


# ---------- Utility ----------
def extract_search_terms(text: str) -> list[str]:
    return [
        t.text
        for t in nlp(text)
        if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop
    ]


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
    follow_up_words = {'more', 'another', 'else', 'other', 'others', 'again', 'some more', 'show me more'}
    query_words = set(clean_query_text(user_query).lower().split())

    # --- REVISED HEURISTIC ---
    # We will perform a contextual search if the query is very short (<= 2 words)
    # OR if it clearly contains a follow-up phrase.
    is_follow_up = bool(query_words.intersection(follow_up_words))
    is_short_query = len(query_words) <= 2

    # If the query is NOT a follow-up AND is long enough, we can skip the LLM call.
    if not is_follow_up and not is_short_query:
        logger.info("[Context Resolver] Query seems specific, skipping LLM resolution.")
        return user_query

    logger.info(f"[Context Resolver] Query '{user_query}' is short or a follow-up. Asking LLM to resolve topic from history.")
    try:
        # Ask the LLM to figure out the real topic
        prompt = contextual_search_topic_prompt(history_text, user_query)
        logger.info(f"[Context Resolver] History context:\n{history_text}\n\n")

        resolved_topic = await generate_response(prompt)
        
        # Clean up the LLM response
        resolved_topic = resolved_topic.strip().strip('"').strip()
        
        # Fallback if the LLM returns an empty string or a useless phrase
        if not resolved_topic or resolved_topic.lower() in ["similar books", "more books"]:
             logger.warning(f"[Context Resolver] LLM returned a weak topic ('{resolved_topic}'). Falling back to original query.")
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
        "You are helping to search a library catalog. Expand the user's topic into 7 concise search terms.\n"
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


def cached_koha_search(term: str):
    key = _cache_key("title", term)
    with KOHA_LOCK:
        if key in KOHA_CACHE:
            return KOHA_CACHE[key]
    # The call to search_books no longer passes the invalid argument
    result = search_books(query=term)
    with KOHA_LOCK:
        KOHA_CACHE[key] = result
    return result



async def koha_multi_search(keywords: list[str], session_id: str = "global") -> list[dict]:
    tasks = [asyncio.to_thread(cached_koha_search, kw) for kw in keywords[:8]]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=KOHA_TIMEOUT_SECONDS + 4,
        )
    except asyncio.TimeoutError:
        logger.error("[Koha] search timeout")
        return [{"answer": "Sorry, our book database is taking too long."}]

    books, errors = [], 0
    for res in results:
        if isinstance(res, Exception) or (isinstance(res, dict) and "error" in res):
            errors += 1
            continue
        if res:
            books.extend(res)

    if not books and errors == len(keywords):
        return [{"answer": "Library database is unavailable or empty."}]
    return books

async def fetch_and_add_quantities(books: list[dict]) -> list[dict]:
    # 1. Collect all the unique biblio_ids from the list of books
    biblio_ids = list(set(
        book.get("biblio_id") for book in books if book.get("biblio_id")
    ))
    
    if not biblio_ids:
        return books # Return early if no IDs to process

    # 2. Make ONE single, efficient API call to get all items
    items_by_biblio_id = await asyncio.to_thread(
        fetch_items_for_multiple_biblios, biblio_ids
    )

    # 3. Add the quantity to each book by checking the count of items
    for book in books:
        book_id = book.get("biblio_id")
        if book_id in items_by_biblio_id:
            book["quantity_available"] = len(items_by_biblio_id[book_id])
        else:
            book["quantity_available"] = 0
            
    return books

# ---------- Main Route ----------
@router.post("/search_books")
async def search_books_api(
    session_data: tuple = Depends(get_session_and_user_data),
    db: AsyncIOMotorDatabase = Depends(get_db),
    intent: str = None,  # <-- Passed by the main router!
):
    try:
        chat_session, cardnumber, data = session_data
        user_query = data.get("query", "").strip()
        cardnumber = data.get("cardNumber") or getattr(chat_session, 'cardNumber', None)
        if not user_query:
            return JSONResponse(
                content={"error": "Query is required."}, status_code=400
            )

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

        # ----- Identifier Lookup (ISBN / ISSN / Call Number) -----
        if intent == "book_lookup_isbn":
            ids = extract_identifiers(user_query)
            if not any(ids.values()):
                bot_reply = specific_book_not_found_prompt("ISBN/ISSN/Call Number")
                await save_conversation_turn(db, cardnumber, user_query, bot_reply)
                return JSONResponse(
                    content={
                        "response": [
                            {
                                "type": "specific_book_search",
                                "answer": bot_reply,
                                "books": [],
                            }
                        ]
                    },
                    status_code=200,
                )

            books = await asyncio.to_thread(search_by_identifiers, ids)

            if not books or (isinstance(books, dict) and "error" in books):
                reason = (
                    books.get("error")
                    if isinstance(books, dict)
                    else "No matching records"
                )
                bot_reply = specific_book_not_found_prompt(reason)
                await save_conversation_turn(db, cardnumber, user_query, bot_reply)
                return JSONResponse(
                    content={
                        "response": [
                            {
                                "type": "specific_book_search",
                                "answer": bot_reply,
                                "books": [],
                            }
                        ]
                    },
                    status_code=200,
                )

            formatted = []
            for b in books[:5]:
                formatted.append(
                    {
                        "title": replace_null(b.get("title")).strip(" ,;:"),
                        "author": replace_null(b.get("author")).strip(" ,;:"),
                        "isbn": replace_null(b.get("isbn")),
                        "publisher": replace_null(b.get("publisher")),
                        "year": replace_null(b.get("year")),
                        "biblio_id": replace_null(b.get("biblio_id")),
                    }
                )

            formatted = await fetch_and_add_quantities(formatted)
            lead = formatted[0]
            bot_reply = (
                specific_book_found_prompt(lead["title"], lead["isbn"]))
            await save_conversation_turn(db, cardnumber, user_query, bot_reply)
            return JSONResponse(
                content={
                    "response": [
                        {
                            "type": "specific_book_search",
                            "answer": bot_reply,
                            "books": formatted,
                        }
                    ]
                },
                status_code=200,
            )
        # ----- Book Recommendation -----
        elif intent == "book_recommend":
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)
            if raw_results and isinstance(raw_results[0], dict) and "answer" in raw_results[0]:
                reply = raw_results[0]["answer"]
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(content={"answer": reply}, status_code=200)

            if not raw_results:
                reply = f"I'm sorry, I couldn't find any books matching '{query_clean}'. Please try another search term."
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(
                    content={"response": [{"type": "booksearch", "answer": reply, "books": []}]},
                    status_code=200,
                )
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
            return JSONResponse(
                content={
                    "response": [
                        {"type": "recommendation", "answer": reply, "books": books}
                    ]
                },
                status_code=200,
            )

        # ----- Book Search -----
        elif intent == "book_search":
            keywords = await expand_query(query_clean)
            raw_results = await koha_multi_search(keywords)
            if raw_results and isinstance(raw_results[0], dict) and "answer" in raw_results[0]:
                reply = raw_results[0]["answer"]
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(content={"answer": reply}, status_code=200)

            # Case 2: No books were found at all (empty list)
            if not raw_results:
                reply = f"I'm sorry, I couldn't find any books matching '{query_clean}'. Please try another search term."
                await save_conversation_turn(db, cardnumber, user_query, reply)
                return JSONResponse(
                    content={"response": [{"type": "booksearch", "answer": reply, "books": []}]},
                    status_code=200,
                )
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
            return JSONResponse(
                content={
                    "response": [
                        {"type": "booksearch", "answer": reply, "books": books}
                    ]
                },
                status_code=200,
            )

        else:
            return JSONResponse(
                content={"error": "Unrecognized query intent."}, status_code=400
            )

    except Exception as e:
        logger.error(f"[Unhandled Error] {e}")
        return JSONResponse(
            content={"error": "Internal server error."}, status_code=500
        )
