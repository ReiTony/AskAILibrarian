import logging
import re
import asyncio
import spacy
from threading import RLock
from cachetools import TTLCache

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from utils.llm_client import generate_response
from utils.koha_client import search_books
from utils.sessions import ChatSession, get_chat_session
from utils.text_utils import (
    replace_null,
    clean_query_text,
    fuzzy_match_keywords,
    classify_book_intent,
)
from utils.prompt_templates import (
    search_books_prompt,
    lookup_isbn_prompt,
    lookup_isbn_not_found_prompt,
    recommend_books_prompt,
)

router = APIRouter()
logger = logging.getLogger("search_books_api")

# -------- NLP --------
nlp = spacy.load("en_core_web_sm")

# -------- Caching (per-process) --------
# NOTE: For multi-worker deployments use Redis for shared caching across processes.
EXPANSION_CACHE = TTLCache(maxsize=1000, ttl=60 * 60 * 24)  # 24h
KOHA_CACHE = TTLCache(maxsize=5000, ttl=60 * 10)  # 10m
AGG_CACHE = TTLCache(maxsize=2000, ttl=60 * 10)  # 10m

EXPANSION_LOCK = RLock()
KOHA_LOCK = RLock()
AGG_LOCK = RLock()

SEARCH_BOOKS_KEYWORDS = {
    "search",
    "quantity",
    "title",
    "find",
    "lookup",
    "copies",
    "available",
    "book",
    "books",
}
KOHA_TIMEOUT_SECONDS = 8

# ---------- Helpers ----------


def extract_search_terms(text: str) -> list[str]:
    """Fallback extraction: nouns, proper nouns, adjectives (no stopwords/punct)."""
    doc = nlp(text)
    return [t.text for t in doc if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop]


def parse_llm_keyword_list(s: str, max_terms: int = 12) -> list[str]:
    parts = re.split(r"[,|\n]+", s)
    out, seen = [], set()
    for p in parts:
        kw = p.strip().strip("\"“”'").lower()
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
            if len(out) >= max_terms:
                break
    return out


def _agg_key_from_keywords(keywords: list[str]) -> str:
    base = "|".join(sorted({kw.strip().lower() for kw in keywords if kw.strip()}))
    return f"agg::{base}"


def _koha_key(field: str, term: str) -> str:
    return f"{field}::{term.strip().lower()}"


# -------- Cached primitives --------


async def expand_query_with_llm(user_query: str) -> list[str]:
    """LLM expansion with 24h cache + spaCy fallback."""
    qnorm = clean_query_text(user_query).strip().lower()
    with EXPANSION_LOCK:
        cached = EXPANSION_CACHE.get(qnorm)
    if cached:
        return cached

    prompt = (
        "You are helping to search a library catalog. "
        "Expand the user's topic into 8–15 concise, distinct search keywords and subtopics.\n"
        f"User topic: {user_query!r}\n\n"
        "Rules:\n"
        "- Return ONLY a comma-separated list (no bullets, no numbering, no extra text).\n"
        "- Prefer concrete terms that appear in book titles (e.g., algebra, calculus, geometry, statistics, trigonometry, number theory, probability, discrete mathematics, linear algebra, analysis).\n"
        "- Do not invent phrases; keep to single or short multiword terms."
    )
    try:
        raw = await generate_response(prompt)
        keywords = parse_llm_keyword_list(raw, max_terms=12)
        if not keywords:
            raise ValueError("Empty expansion from LLM")
    except Exception as e:
        logger.error(f"[LLM expand] error: {e}")
        keywords = extract_search_terms(user_query) or [user_query]

    with EXPANSION_LOCK:
        EXPANSION_CACHE[qnorm] = keywords
    return keywords


def cached_koha_search(field: str, term: str):
    """Synchronous wrapper used inside to_thread; caches raw Koha response per keyword."""
    key = _koha_key(field, term)
    with KOHA_LOCK:
        cached = KOHA_CACHE.get(key)
    if cached is not None:
        return cached

    res = search_books(field, term)  # may raise; let caller handle
    with KOHA_LOCK:
        KOHA_CACHE[key] = res
    return res


async def koha_multi_search_title(keywords: list[str]) -> list[dict]:
    """Query Koha concurrently for multiple keywords against title (with per-keyword cache)."""
    keywords = keywords[:8]  # avoid hammering Koha
    tasks = [asyncio.to_thread(cached_koha_search, "title", kw) for kw in keywords]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=KOHA_TIMEOUT_SECONDS + 4,
        )
    except asyncio.TimeoutError:
        logger.error("[Koha] multi-search timed out entirely.")
        return [
            {
                "answer": "Sorry, our book database is taking too long to respond. Please try again later."
            }
        ]

    merged = []
    errors = 0

    for idx, res in enumerate(results):
        keyword = keywords[idx]
        if isinstance(res, Exception):
            if isinstance(res, asyncio.TimeoutError):
                logger.error(f"[Koha] timeout for keyword '{keyword}'")
            else:
                logger.error(f"[Koha] error for keyword '{keyword}': {res}")
            errors += 1
            continue

        if isinstance(res, dict) and "error" in res:
            logger.error(f"[Koha] API error for keyword '{keyword}': {res['error']}")
            errors += 1
            continue

        if not res:
            logger.warning(f"[Koha] Empty result for keyword '{keyword}'")
            continue

        merged.extend(res)

    if not merged and errors >= len(keywords):
        return [
            {
                "answer": "Sorry, the library database service is currently unavailable or returned no results. Please try again later."
            }
        ]

    return merged


# ---------- Route ----------
@router.post("/search_books")
async def search_books_api(
    request: Request, chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()

        if not user_query:
            return JSONResponse(
                content={"error": "Query parameter is required"}, status_code=400
            )

        user_query_clean = clean_query_text(user_query)
        intent = classify_book_intent(user_query_clean)
        logger.info(f"[Intent] Detected intent: {intent}")

        try:
            history = await chat_session.get_history()
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            history = []

        history_text = "\n".join(
            f"{'Human' if msg['role']=='user' else 'AI'}: {msg['content']}"
            for msg in history
        )

        # ---------- ISBN Lookup ----------
        if intent == "isbn_lookup":
            match = re.search(
                r"isbn (of|for) (.+?)(?:\?|\.)?$", user_query_clean.lower()
            )
            book_title = match.group(2).strip() if match else user_query_clean

            try:
                raw_results = await asyncio.to_thread(search_books, "title", book_title)
            except Exception as e:
                logger.error(f"Error connecting to Koha: {e}")
                return JSONResponse(
                    content={
                        "answer": "Sorry, the library database is currently unavailable."
                    },
                    status_code=200,
                )

            # Ensure Koha returned a list of dicts
            if not isinstance(raw_results, list):
                logger.error(f"[Koha] Unexpected response format for ISBN lookup: {type(raw_results)} - {raw_results}")
                return JSONResponse(
                    content={"answer": "Sorry, I couldn’t process your ISBN request right now."},
                    status_code=200,
                )

            found = False
            for book in raw_results:
                if not isinstance(book, dict):
                    logger.warning(f"[Koha] Skipping unexpected result item: {book}")
                    continue

                title = replace_null(book.get("title"))
                isbn = replace_null(book.get("isbn"))

                if (
                    title
                    and isbn
                    and isbn.lower()
                    not in {"unknown", "not available", "no isbn", "isbn unavailable"}
                ):
                    try:
                        prompt = lookup_isbn_prompt(title, isbn)
                        bot_response = await generate_response(prompt)
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        bot_response = f"The ISBN of '{title}' is {isbn}."
                    found = True
                    break

            if not found:
                try:
                    prompt = lookup_isbn_not_found_prompt(book_title)
                    bot_response = await generate_response(prompt)
                except Exception as e:
                    logger.error(f"LLM not-found prompt error: {e}")
                    bot_response = "Sorry, I could not find the ISBN for that book."

            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", bot_response)

            return JSONResponse(
                content={"type": "isbn", "answer": bot_response},
                status_code=200,
            )
        # ---------- Book Recommendations ----------
        elif intent == "recommend":
            # Expand the topic into keywords
            expanded_keywords = await expand_query_with_llm(user_query_clean)
            logger.info(f"[Recommend] Expanded keywords: {expanded_keywords}")

            raw_results = await koha_multi_search_title(expanded_keywords)
            if not raw_results or (
                isinstance(raw_results, list) and "answer" in raw_results[0]
            ):
                return JSONResponse(
                    content={
                        "answer": (
                            raw_results[0]["answer"]
                            if raw_results
                            else "Sorry, no recommended books found."
                        )
                    },
                    status_code=200,
                )

            aggregated = {}
            seen = set()

            for book in raw_results:
                title = replace_null(book.get("title")).title()
                author = replace_null(book.get("author"))
                isbn = replace_null(book.get("isbn"))
                publisher = replace_null(book.get("publisher"))
                biblio_id = replace_null(book.get("biblio_id"))
                year = replace_null(book.get("year"))
                key = f"{title}|{author}"
                if key not in seen and title not in ["Unknown", "Not Available"]:
                    aggregated[key] = {"title": title, "author": author, "isbn": isbn, "publisher": publisher, "biblio_id": biblio_id, "year": year}
                    seen.add(key)
                if len(aggregated) >= 10:
                    break

            formatted_books = list(aggregated.values())

            if not formatted_books:
                return JSONResponse(
                    content={"answer": "Sorry, I couldn’t find any recommended books."},
                    status_code=200,
                )

            try:
                prompt = recommend_books_prompt(user_query_clean, history_text)
                logger.info(f"Prompt to LLM:\n{prompt}")
                bot_response = await generate_response(prompt)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                bot_response = (
                    "Here are some books from our collection you might find useful."
                )

            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", bot_response)

            return JSONResponse(
                content={ 
                    "response": [
                        {
                            "type": "recommendation",
                            "answer": bot_response,
                            "books": formatted_books,
                        }
                    ]
                },
                status_code=200,
            )
        # ---------- General Search ----------
        else:
            if not fuzzy_match_keywords(
                user_query_clean.split(), SEARCH_BOOKS_KEYWORDS, threshold=75
            ):
                return JSONResponse(
                    content={"error": "Query does not contain valid search keywords."},
                    status_code=400,
                )

            expanded_keywords = await expand_query_with_llm(user_query)
            logger.info(f"[Search] Expanded keywords: {expanded_keywords}")

            agg_key = _agg_key_from_keywords(expanded_keywords)
            with AGG_LOCK:
                cached_formatted = AGG_CACHE.get(agg_key)
            if cached_formatted is not None:
                formatted_books = cached_formatted
                logger.info(
                    f"[Search] Served from AGG_CACHE: {len(formatted_books)} books"
                )
            else:
                raw_results = await koha_multi_search_title(expanded_keywords)
                if not raw_results or (
                    isinstance(raw_results, list) and "answer" in raw_results[0]
                ):
                    return JSONResponse(
                        content={
                            "answer": (
                                raw_results[0]["answer"]
                                if raw_results
                                else "No books found."
                            )
                        },
                        status_code=200,
                    )

                aggregated = {}
                for book in raw_results:
                    isbn = replace_null(book.get("isbn"))
                    title = replace_null(book.get("title"))
                    author = replace_null(book.get("author"))
                    publisher = replace_null(book.get("publisher"))
                    year = replace_null(book.get("year"))
                    biblio_id = replace_null(book.get("biblio_id"))
                    if title in ["Unknown", "Not Available"]:
                        continue

                    key = (
                        f"isbn::{isbn}"
                        if isbn not in ["Unknown", "Not Available"]
                        else f"title::{title}|author::{author}"
                    )
                    if key in aggregated:
                        aggregated[key]["quantity_available"] += 1
                    else:
                        aggregated[key] = {
                            "title": title.strip(" ,;:"),
                            "author": author.strip(" ,;:"),
                            "isbn": isbn,
                            "publisher": publisher.strip(" ,;:"),
                            "quantity_available": 1,
                            "year": year,
                            "biblio_id": biblio_id,
                        }

                formatted_books = list(aggregated.values())
                with AGG_LOCK:
                    AGG_CACHE[agg_key] = formatted_books

            try:
                prompt = search_books_prompt(user_query_clean, history_text)
                bot_response = await generate_response(prompt)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                bot_response = "Here are the books I found that match your search."

            await chat_session.add_message("user", user_query)
            await chat_session.add_message("assistant", bot_response)

            return JSONResponse(
                content={
                    "response": [
                        {
                            "type": "booksearch",
                            "answer": bot_response,
                            "books": formatted_books,
                        }
                    ]
                },
                status_code=200,
            )

    except Exception as e:
        logger.error(f"Error in /search_books: {e}")
        return JSONResponse(
            content={"error": "Internal server error."}, status_code=500
        )
