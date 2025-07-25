import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncio

from utils.llm_client import generate_response
from utils.koha_client import search_books
from utils.sessions import ChatSession, get_chat_session
from utils.text_utils import replace_null, clean_query_text, fuzzy_match_keywords
from utils.prompt_templates import search_books_prompt

router = APIRouter()
logger = logging.getLogger("search_books_api")

SEARCH_BOOKS_KEYWORDS = {"search", "quantity", "title", "find", "lookup", "copies", "available"}
KOHA_TIMEOUT_SECONDS = 8  

@router.post("/search_books")
async def search_books_api(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "").lower().strip()

        if not user_query:
            return JSONResponse(content={"error": "Query parameter is required"}, status_code=400)
        if not fuzzy_match_keywords([user_query], SEARCH_BOOKS_KEYWORDS):
            return JSONResponse(content={"error": "Query does not contain valid search keywords."}, status_code=400)

        try:
            history = await chat_session.get_history()
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            history = []
        history_text = "\n".join([
            f"{'Human' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in history
        ])
        user_query_clean = clean_query_text(user_query)

        # --- Koha search with timeout ---
        try:
            koha_results = await asyncio.wait_for(
                asyncio.to_thread(search_books, "title", user_query_clean),
                timeout=KOHA_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error("Koha API timed out")
            return JSONResponse(
                content={"answer": "Sorry, our book database is taking too long to respond. Please try again later."},
                status_code=200
            )
        except Exception as api_exc:
            logger.error(f"Error connecting to Koha API: {api_exc}")
            return JSONResponse(
                content={"answer": "Sorry, the library database service is currently unavailable. Please try again soon."},
                status_code=200
            )

        if "error" in koha_results:
            return JSONResponse(content={"answer": koha_results["error"]}, status_code=200)
        if not koha_results:
            return JSONResponse(content={"answer": "No books found. Try refining your query."}, status_code=200)

        # --- Book result processing ---
        koha_books = {}
        for book in koha_results:
            isbn = book.get("isbn", "Unknown")
            title = book.get("title", "Unknown")
            author = book.get("author", "Unknown")

            if isbn == "Unknown" or title == "Unknown":
                continue

            if fuzzy_match_keywords(user_query_clean, title, author):
                if isbn in koha_books:
                    koha_books[isbn]["quantity"] += 1
                else:
                    koha_books[isbn] = {
                        "title": title,
                        "isbn": isbn,
                        "quantity": 1,
                        "publisher": book.get("publisher", "Unknown"),
                        "author": author,
                    }

        formatted_books = [
            f"Title: {b['title']}, Author: {b['author']}, ISBN: {b['isbn']}, Quantity: {b['quantity']}, Publisher: {b['publisher']}"
            for b in koha_books.values()
        ]

        try:
            prompt = search_books_prompt(user_query_clean, "\n".join(formatted_books), history_text)
            logger.info(f"Prompt to LLM:\n{prompt}")
            bot_response = await generate_response(prompt)
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            bot_response = "Sorry, I could not generate an answer right now."

        # --- Try to record chat; never block on error ---
        try:
            await chat_session.add_message("user", user_query)
        except Exception as e:
            logger.error(f"Error adding user message to chat session: {e}")
        try:
            await chat_session.add_message("assistant", bot_response)
        except Exception as e:
            logger.error(f"Error adding assistant message to chat session: {e}")

        return JSONResponse(content={
            "type": "booksearch",
            "answer": bot_response,
            "suggestion1": "You can check the availability of these books at our library.",
            "suggestion2": "Would you like to know more about any specific book?",
            "suggestion3": "I can help you find similar books if you're interested."
        }, status_code=200)

    except Exception as e:
        logger.error(f"Error occurred in /search_books: {e}")

        return JSONResponse(content={"error": "Internal server error."}, status_code=500)
