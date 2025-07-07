from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import logging
import re

from utils.sessions import ChatSession, get_chat_session
from utils.chroma_client import books_db
from utils.text_utils import replace_null
from utils.llm_client import generate_response  
from utils.prompt_templates import lookup_isbn_prompt, lookup_isbn_not_found_prompt
from utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language
)

router = APIRouter()
logger = logging.getLogger("lookup_isbn")

def clean_isbn(isbn):
    invalids = {"Unknown", "Not Available", "NO ISBN", "ISBN unavailable", ""}
    return isbn if isbn not in invalids else None

@router.post("/lookup_isbn")
async def lookup_isbn(
    request: Request,
    chat_session: ChatSession = Depends(get_chat_session)
):
    try:
        data = await request.json()
        user_query = data.get("query", "")

        user_lang = detect_language(user_query)
        translated_query = await translate_to_english(user_query, user_lang)

        # Try to extract book title from translated query
        match = re.search(r'isbn (of|for) (.+?)(?:\?|\.)?$', translated_query.lower())
        if match:
            book_title = match.group(2).strip()
        else:
            book_title = translated_query.strip()
            
        try:
            results = await asyncio.to_thread(
                books_db.similarity_search_with_score, book_title, k=3
            )
        except Exception as e:
            logger.error(f"Error running similarity search: {e}")
            results = []

        found = False
        for doc, _ in results:
            title = replace_null(doc.metadata.get("Title", "")).title()
            isbn = clean_isbn(replace_null(doc.metadata.get("ISBN", "")))
            if title and isbn:
                prompt = lookup_isbn_prompt(title, isbn)
                try:
                    raw_answer = await generate_response(prompt)
                except Exception as e:
                    logger.error(f"Error generating LLM response: {e}")
                    raw_answer = f"The ISBN of '{title}' is {isbn}."
                found = True
                break

        if not found:
            prompt = lookup_isbn_not_found_prompt(book_title)
            try:
                raw_answer = await generate_response(prompt)
            except Exception as e:
                logger.error(f"Error generating LLM not-found response: {e}")
                raw_answer = "Sorry, I could not find the ISBN for that book."

        translated_answer = await translate_to_user_language(raw_answer, user_lang)

        try:
            await chat_session.add_message("user", user_query)
        except Exception as e:
            logger.error(f"Error adding user message to chat session: {e}")
        try:
            await chat_session.add_message("assistant", translated_answer)
        except Exception as e:
            logger.error(f"Error adding assistant message to chat session: {e}")

        return JSONResponse(content={"answer": translated_answer}, status_code=200)

    except Exception as e:
        logger.error(f"Error in /lookup_isbn: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
